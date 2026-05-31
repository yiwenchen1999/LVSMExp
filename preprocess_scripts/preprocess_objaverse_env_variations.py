#!/usr/bin/env python3
"""
Preprocessing script to convert Objaverse data format to re10k format with environment map variations.

The script processes Objaverse data where:
- Each object has train/ and test/ folders
- test/ contains env_0/, env_1/, ..., env_4/, white_env_0/ folders (different scenes with same trajectory)
- Each env folder contains gt_{idx}.png images
- cameras.json contains camera parameters in Blender convention

For each input scene, creates n variations with different environment map rotations:
- Variations are named: {original_scene_name}_1, {original_scene_name}_2, ..., {original_scene_name}_n
- Each variation uses the same environment map but with different rotation angles
- Rotation angles are equally distributed from 0 to 2π around the z-axis (last axis)

Output format matches re10k:
- JSON files with scene_name and frames array
- Each frame has: image_path, fxfycxcy, w2c
- Images organized in images/{scene_name}/ folders
"""

import os
import json
import numpy as np
import argparse
from pathlib import Path
from PIL import Image
import shutil
import io
import tarfile
import re
import random
import torch
import torch.nn.functional as F
import cv2
try:
    import pyexr
    HAS_PYEXR = True
except ImportError:
    HAS_PYEXR = False
    print("Warning: pyexr not available, HDR EXR files cannot be read")


def _resolve_env_source(split_path, env_folder):
    """Return ('dir'|'tar'|None, source_path) for env folder data."""
    dir_path = os.path.join(split_path, env_folder)
    tar_path = os.path.join(split_path, f"{env_folder}.tar")
    if os.path.isdir(dir_path):
        return "dir", dir_path
    if os.path.isfile(tar_path):
        return "tar", tar_path
    return None, None


def _list_env_folders(split_path):
    """List env folders from either extracted dirs or compressed .tar files."""
    env_names = set()
    if not os.path.isdir(split_path):
        return []
    for item in os.listdir(split_path):
        item_path = os.path.join(split_path, item)
        if os.path.isdir(item_path) and (item.startswith("env_") or item.startswith("white_env_")):
            env_names.add(item)
        elif os.path.isfile(item_path) and item.endswith(".tar"):
            name = item[:-4]
            if name.startswith("env_") or name.startswith("white_env_"):
                env_names.add(name)
    return sorted(env_names)


def _list_numeric_gt_images(split_path, env_folder):
    """
    List (frame_idx, filename) for gt_*.png in a folder or tar.
    """
    source_type, source_path = _resolve_env_source(split_path, env_folder)
    if source_type is None:
        return []

    all_image_files = []
    if source_type == "dir":
        all_image_files = [f for f in os.listdir(source_path) if f.startswith("gt_") and f.endswith(".png")]
    else:
        with tarfile.open(source_path, "r") as tar:
            all_image_files = [
                os.path.basename(m.name)
                for m in tar.getmembers()
                if m.isfile() and os.path.basename(m.name).startswith("gt_") and os.path.basename(m.name).endswith(".png")
            ]

    image_files_with_idx = []
    for image_file in all_image_files:
        idx_str = image_file.replace("gt_", "").replace(".png", "")
        try:
            frame_idx = int(idx_str)
            image_files_with_idx.append((frame_idx, image_file))
        except ValueError:
            continue
    image_files_with_idx.sort(key=lambda x: x[0])
    return image_files_with_idx


def _read_image_size_from_source(split_path, env_folder, image_name):
    source_type, source_path = _resolve_env_source(split_path, env_folder)
    if source_type is None:
        raise FileNotFoundError(f"Missing source for {env_folder} under {split_path}")

    if source_type == "dir":
        image_path = os.path.join(source_path, image_name)
        with Image.open(image_path) as img:
            return img.size

    with tarfile.open(source_path, "r") as tar:
        members = [m for m in tar.getmembers() if m.isfile() and os.path.basename(m.name) == image_name]
        if not members:
            raise FileNotFoundError(f"{image_name} not found in tar {source_path}")
        member = members[0]
        fobj = tar.extractfile(member)
        if fobj is None:
            raise FileNotFoundError(f"Could not extract {image_name} from {source_path}")
        with Image.open(io.BytesIO(fobj.read())) as img:
            return img.size


def _read_rgb_image_from_source(split_path, env_folder, image_name):
    source_type, source_path = _resolve_env_source(split_path, env_folder)
    if source_type is None:
        return None

    if source_type == "dir":
        image_path = os.path.join(source_path, image_name)
        if not os.path.exists(image_path):
            return None
        return Image.open(image_path).convert("RGB")

    with tarfile.open(source_path, "r") as tar:
        members = [m for m in tar.getmembers() if m.isfile() and os.path.basename(m.name) == image_name]
        if not members:
            return None
        fobj = tar.extractfile(members[0])
        if fobj is None:
            return None
        return Image.open(io.BytesIO(fobj.read())).convert("RGB")


def _copy_image_from_source(split_path, env_folder, image_name, output_image_path):
    source_type, source_path = _resolve_env_source(split_path, env_folder)
    if source_type is None:
        raise FileNotFoundError(f"Missing source for {env_folder} under {split_path}")

    if source_type == "dir":
        shutil.copy2(os.path.join(source_path, image_name), output_image_path)
        return

    with tarfile.open(source_path, "r") as tar:
        members = [m for m in tar.getmembers() if m.isfile() and os.path.basename(m.name) == image_name]
        if not members:
            raise FileNotFoundError(f"{image_name} not found in tar {source_path}")
        fobj = tar.extractfile(members[0])
        if fobj is None:
            raise FileNotFoundError(f"Could not extract {image_name} from {source_path}")
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        with open(output_image_path, "wb") as out_f:
            out_f.write(fobj.read())


def _load_json_from_source(split_path, env_folder, json_names):
    """
    Load first existing json from source folder/tar.
    Returns dict or None.
    """
    source_type, source_path = _resolve_env_source(split_path, env_folder)
    if source_type is None:
        return None

    if source_type == "dir":
        for name in json_names:
            path = os.path.join(source_path, name)
            if os.path.exists(path):
                with open(path, "r") as f:
                    return json.load(f)
        return None

    with tarfile.open(source_path, "r") as tar:
        file_members = [m for m in tar.getmembers() if m.isfile()]
        member_by_name = {m.name: m for m in file_members}
        member_by_base = {os.path.basename(m.name): m for m in file_members}
        for name in json_names:
            member = member_by_name.get(name) or member_by_base.get(name)
            if member is not None:
                fobj = tar.extractfile(member)
                if fobj is not None:
                    return json.load(fobj)
        return None


def _extract_object_id_from_scene(scene_name):
    """
    Extract object id from base scene name:
      <object_id>_env_k or <object_id>_white_env_k.
    """
    m = re.match(r"^(.*?)_(?:white_env|env)_\d+$", scene_name)
    if m:
        return m.group(1)
    return scene_name.rsplit("_", 1)[0]


def _sample_shared_consecutive_indices(frame_index_sets, chunk_len, rng):
    """
    Sample one consecutive frame chunk shared by all scenes of the same object.
    Returns (selected_indices_set, start_idx) or (None, None) if unavailable.
    """
    if chunk_len is None or chunk_len <= 0:
        return None, None
    if not frame_index_sets:
        return None, None

    common_idx = set(frame_index_sets[0])
    for s in frame_index_sets[1:]:
        common_idx &= set(s)
    if len(common_idx) < chunk_len:
        return None, None

    common_sorted = sorted(common_idx)
    common_set = set(common_sorted)
    valid_starts = []
    for st in common_sorted:
        ok = True
        for off in range(chunk_len):
            if (st + off) not in common_set:
                ok = False
                break
        if ok:
            valid_starts.append(st)
    if not valid_starts:
        return None, None

    start_idx = rng.choice(valid_starts)
    selected = set(range(start_idx, start_idx + chunk_len))
    return selected, start_idx


def blender_to_opencv_c2w(c2w_blender):
    """
    Convert Blender c2w matrix to OpenCV c2w matrix.
    
    Blender convention: +X right, +Y forward, +Z up
    OpenCV convention: +X right, +Y down, +Z forward
    
    Transformation: Rotate by [1,0,0][0,-1,0][0,0,-1] on rotation part only
    Translation vector remains unchanged.
    
    Args:
        c2w_blender: 4x4 numpy array in Blender convention
        
    Returns:
        c2w_opencv: 4x4 numpy array in OpenCV convention
    """
    c2w_opencv = c2w_blender.copy()
    
    # Transformation matrix for rotation part only
    # [1,  0,  0]
    # [0, -1,  0]
    # [0,  0, -1]
    transform = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ])
    
    # Apply transformation to rotation part (3x3 top-left)
    # Multiply from the right: R_opencv = R_blender @ transform
    c2w_opencv[:3, :3] = c2w_blender[:3, :3] @ transform
    
    # Translation (last column, first 3 rows) remains unchanged
    # Already copied, no change needed
    
    return c2w_opencv


def fov_to_fxfycxcy(fov_degrees, image_width, image_height):
    """
    Convert field of view (FOV) to fxfycxcy intrinsic parameters.
    
    Args:
        fov_degrees: Field of view in degrees
        image_width: Image width in pixels
        image_height: Image height in pixels
        
    Returns:
        fxfycxcy: [fx, fy, cx, cy] array
    """
    fov_rad = np.radians(fov_degrees)
    fx = fy = (image_width / 2.0) / np.tan(fov_rad / 2.0)
    cx = image_width / 2.0
    cy = image_height / 2.0
    return [fx, fy, cx, cy]


def check_scene_broken(split_path, object_id):
    """
    Check if a scene is broken (no materials) by checking albedo mask and RGB images.
    
    Args:
        split_path: Path to the split folder (test/ or train/)
        object_id: Object ID
        
    Returns:
        bool: True if scene is broken, False otherwise
    """
    albedo_dir = os.path.join(split_path, 'albedo')
    if not os.path.exists(albedo_dir):
        return False  # No albedo folder, cannot check
    
    # Get first albedo image to check mask
    albedo_files = sorted([f for f in os.listdir(albedo_dir) if f.endswith('.png')])
    if not albedo_files:
        return False  # No albedo images
    
    first_albedo_path = os.path.join(albedo_dir, albedo_files[0])
    try:
        albedo_img = Image.open(first_albedo_path)
        if albedo_img.mode == 'RGBA':
            albedo_array = np.array(albedo_img)
            alpha_channel = albedo_array[:, :, 3]
            # Mask is where alpha is 0
            mask = (alpha_channel == 0)
            mask_ratio = np.sum(mask) / mask.size
            
            # If mask takes up less than 1/4 of image, mark as broken
            if mask_ratio < 0.25:
                # Check RGB images inside the mask
                # Find corresponding RGB image (gt_0.png for albedo_cam_0.png)
                cam_idx = albedo_files[0].replace('albedo_cam_', '').replace('.png', '')
                try:
                    cam_idx_int = int(cam_idx)
                    # Look for gt_{cam_idx}.png in any env source (folder or tar)
                    env_folders = _list_env_folders(split_path)
                    
                    for env_folder in env_folders:
                        rgb_img = _read_rgb_image_from_source(split_path, env_folder, f'gt_{cam_idx_int}.png')
                        if rgb_img is not None:
                            rgb_array = np.array(rgb_img)

                            # Get RGB values inside the mask
                            masked_rgb = rgb_array[mask]
                            if len(masked_rgb) > 0:
                                # Check if average color is all black (smaller than (1,1,1) in 0-255 scale)
                                avg_color = np.mean(masked_rgb, axis=0)
                                if np.all(avg_color < 1.0):
                                    return True  # Scene is broken
                            break
                except (ValueError, FileNotFoundError):
                    pass
        else:
            # If albedo doesn't have alpha channel, check if it's all black or very dark
            albedo_array = np.array(albedo_img.convert('RGB'))
            avg_color = np.mean(albedo_array, axis=(0, 1))
            if np.all(avg_color < 1.0):
                return True  # Scene is broken
    except Exception as e:
        print(f"Error checking scene {object_id}: {e}")
    
    return False


def generate_envir_map_dir(envmap_h, envmap_w):
    """Generate environment map directions and weights."""
    lat_step_size = np.pi / envmap_h
    lng_step_size = 2 * np.pi / envmap_w
    theta, phi = torch.meshgrid([
        torch.linspace(np.pi / 2 - 0.5 * lat_step_size, -np.pi / 2 + 0.5 * lat_step_size, envmap_h), 
        torch.linspace(np.pi - 0.5 * lng_step_size, -np.pi + 0.5 * lng_step_size, envmap_w)
    ], indexing='ij')
    
    sin_theta = torch.sin(torch.pi / 2 - theta)
    light_area_weight = 4 * np.pi * sin_theta / torch.sum(sin_theta)
    light_area_weight = light_area_weight.to(torch.float32)
    
    view_dirs = torch.stack([
        torch.cos(phi) * torch.cos(theta), 
        torch.sin(phi) * torch.cos(theta), 
        torch.sin(theta)
    ], dim=-1).view(-1, 3)
    
    return light_area_weight, view_dirs


def get_light(envir_map, incident_dir, hdr_weight=None, if_weighted=False):
    """Sample light from environment map given incident direction."""
    try:
        envir_map = envir_map.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        if hdr_weight is not None:
            hdr_weight = hdr_weight.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        incident_dir = incident_dir.clamp(-1, 1)
        theta = torch.arccos(incident_dir[:, 2]).reshape(-1)
        phi = torch.atan2(incident_dir[:, 1], incident_dir[:, 0]).reshape(-1)
        
        query_y = (theta / np.pi) * 2 - 1
        query_y = query_y.clamp(-1+1e-8, 1-1e-8)
        query_x = -phi / np.pi
        query_x = query_x.clamp(-1+1e-8, 1-1e-8)
        
        grid = torch.stack((query_x, query_y)).permute(1, 0).unsqueeze(0).unsqueeze(0).float()
        
        if if_weighted and hdr_weight is not None:
            weighted_envir_map = envir_map * hdr_weight
            light_rgbs = F.grid_sample(weighted_envir_map, grid, align_corners=True).squeeze().permute(1, 0).reshape(-1, 3)
            light_rgbs = light_rgbs / hdr_weight.reshape(-1, 1)
        else:
            light_rgbs = F.grid_sample(envir_map, grid, align_corners=True).squeeze().permute(1, 0).reshape(-1, 3)
        
        return light_rgbs
    except Exception as e:
        print(f"Error in get_light: {e}")
        return None


def read_hdr_exr(path):
    """Read HDR EXR file."""
    if not HAS_PYEXR:
        raise ImportError("pyexr is required to read EXR files")
    try:
        rgb = pyexr.read(path)
        if rgb is not None and rgb.shape[-1] == 4:
            rgb = rgb[:, :, :3]
        return rgb
    except Exception as e:
        print(f"Error reading HDR file {path}: {e}")
        return None


def read_hdr(path):
    """Read HDR file (supports both .hdr and .exr)."""
    if path.endswith('.exr'):
        return read_hdr_exr(path)
    else:
        try:
            with open(path, 'rb') as h:
                buffer_ = np.frombuffer(h.read(), np.uint8)
            bgr = cv2.imdecode(buffer_, cv2.IMREAD_UNCHANGED)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return rgb
        except Exception as e:
            print(f"Error reading HDR file {path}: {e}")
            return None


def rotate_and_preprocess_envir_map(envir_map, camera_pose, euler_rotation=None, light_area_weight=None, view_dirs=None):
    """
    Rotate and preprocess environment map based on euler rotation and camera pose.
    Returns HDR raw, LDR, and HDR processed versions.
    """
    try:
        # Convert to numpy if it's a tensor
        if isinstance(envir_map, torch.Tensor):
            envir_map_np = envir_map.cpu().numpy()
        else:
            envir_map_np = envir_map.copy()
        
        env_h, env_w = envir_map_np.shape[0], envir_map_np.shape[1]
        if light_area_weight is None or view_dirs is None:
            light_area_weight, view_dirs = generate_envir_map_dir(env_h, env_w)
        
        # Store original for raw version
        envir_map_raw = envir_map_np.copy()
        
        # Step 1: Apply euler rotation (horizontal roll) if provided
        if euler_rotation is not None:
            z_rotation = euler_rotation[2] if len(euler_rotation) >= 3 else 0.0
            rotation_angle_deg = np.degrees(z_rotation)
            shift = int((rotation_angle_deg / 360.0) * env_w)
            envir_map_np = np.roll(envir_map_np, shift, axis=1)
        
        # Convert to tensor
        envir_map_tensor = torch.from_numpy(envir_map_np).float()
        
        # Step 2: Apply camera pose rotation
        if camera_pose.shape == (4, 4):
            c2w_rotation = camera_pose[:3, :3]
            w2c_rotation = c2w_rotation.T
        else:
            w2c_rotation = camera_pose.T
        
        # Blender's convention
        axis_aligned_transform = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        axis_aligned_R = axis_aligned_transform @ w2c_rotation
        view_dirs_world = view_dirs @ torch.from_numpy(axis_aligned_R).float()
        
        # Apply rotation using get_light
        rotated_hdr_rgb = get_light(envir_map_tensor, view_dirs_world.clamp(-1, 1))
        if rotated_hdr_rgb is not None and rotated_hdr_rgb.numel() > 0:
            rotated_hdr_rgb = rotated_hdr_rgb.reshape(env_h, env_w, 3).cpu().numpy()
        else:
            rotated_hdr_rgb = envir_map_raw
        
        # HDR raw
        envir_map_hdr_raw = rotated_hdr_rgb
        
        # LDR (gamma correction)
        envir_map_ldr = rotated_hdr_rgb.clip(0, 1)
        envir_map_ldr = envir_map_ldr ** (1/2.2)
        
        # HDR processed (log transform)
        envir_map_hdr = np.log1p(10 * rotated_hdr_rgb)
        envir_map_hdr_rescaled = (envir_map_hdr / np.max(envir_map_hdr)).clip(0, 1)
        
        return envir_map_hdr_raw, envir_map_ldr, envir_map_hdr_rescaled
    except Exception as e:
        print(f"Error in rotate_and_preprocess_envir_map: {e}")
        return None, None, None


def load_scene_list(scene_list_path):
    """
    Load scene list from JSON file.
    
    Expected format:
    {
        "with_variations": ["scene1", "scene2", ...],  # Scenes to generate env variations for
        "without_variations": ["scene3", "scene4", ...]  # Scenes to process without variations
    }
    
    Also supports legacy formats:
    1. List of scene names: ["scene1", "scene2", ...] (all treated as with_variations)
    2. Dictionary with scene names as keys: {"scene1": {...}, "scene2": {...}, ...} (all treated as with_variations)
    
    Args:
        scene_list_path: Path to JSON file containing scene list
        
    Returns:
        tuple: (scenes_with_variations, scenes_without_variations) as sets, or (None, None) if file doesn't exist
    """
    if not scene_list_path or not os.path.exists(scene_list_path):
        return None, None
    
    try:
        with open(scene_list_path, 'r') as f:
            data = json.load(f)
        
        scenes_with_variations = set()
        scenes_without_variations = set()
        
        if isinstance(data, dict):
            # New format with two groups
            if "with_variations" in data:
                if isinstance(data["with_variations"], list):
                    scenes_with_variations = set(data["with_variations"])
                else:
                    print(f"Warning: 'with_variations' should be a list in {scene_list_path}")
            
            if "without_variations" in data:
                if isinstance(data["without_variations"], list):
                    scenes_without_variations = set(data["without_variations"])
                else:
                    print(f"Warning: 'without_variations' should be a list in {scene_list_path}")
            
            # If neither key exists, treat all keys as scene names (legacy dict format)
            if not scenes_with_variations and not scenes_without_variations:
                scenes_with_variations = set(data.keys())
        elif isinstance(data, list):
            # Legacy list format - all scenes treated as with_variations
            scenes_with_variations = set(data)
        else:
            print(f"Warning: Unsupported JSON format in {scene_list_path}, expected dict with 'with_variations' and 'without_variations' keys, or list")
            return None, None
        
        total_scenes = len(scenes_with_variations) + len(scenes_without_variations)
        print(f"Loaded {total_scenes} scenes from {scene_list_path}:")
        print(f"  - {len(scenes_with_variations)} scenes with variations")
        print(f"  - {len(scenes_without_variations)} scenes without variations")
        
        return scenes_with_variations, scenes_without_variations
    except Exception as e:
        print(f"Error loading scene list from {scene_list_path}: {e}")
        return None, None


def process_objaverse_scene(objaverse_root, object_id, output_root, split='test', hdri_dir=None, n_variations=1,
                            scenes_with_variations=None, scenes_without_variations=None,
                            consecutive_frames=0, frame_chunk_rng=None):
    """
    Process a single Objaverse object and convert all env scenes to re10k format with variations.
    
    Args:
        objaverse_root: Root directory of objaverse data (e.g., data_samples/sample_objaverse)
        object_id: Object ID folder name
        output_root: Root directory for output (e.g., data_samples/objaverse_processed)
        split: 'train' or 'test'
        hdri_dir: Directory containing HDR environment maps (optional)
        n_variations: Number of variations to create for each scene (default: 1)
        scenes_with_variations: Set of scene names to process with variations (if None, process all with variations)
        scenes_without_variations: Set of scene names to process without variations (if None, process all without variations)
        consecutive_frames: If > 0, sample this many consecutive frames per object and
            apply the same frame chunk to all scenes of the object.
        frame_chunk_rng: Random generator used for chunk start sampling.
        
    Returns:
        tuple: (processed_scenes_with_variations, processed_scenes_without_variations) as lists
    """
    object_path = os.path.join(objaverse_root, object_id)
    split_path = os.path.join(object_path, split)
    
    if not os.path.exists(split_path):
        print(f"Warning: {split_path} does not exist, skipping")
        return None
    
    # Check if scene is broken (no materials)
    if check_scene_broken(split_path, object_id):
        print(f"Scene {object_id} is broken (no materials), marking...")
        broken_file = os.path.join(object_path, 'broken.txt')
        with open(broken_file, 'w') as f:
            f.write("broken\n")
        return "broken"
    
    # Load cameras.json
    cameras_json_path = os.path.join(split_path, 'cameras.json')
    if not os.path.exists(cameras_json_path):
        print(f"Warning: {cameras_json_path} does not exist, skipping {object_id}")
        return None
    
    with open(cameras_json_path, 'r') as f:
        cameras_data = json.load(f)
    
    # Find all env sources (env_0, env_1, ..., white_env_0) from dirs or .tar files.
    env_folders = _list_env_folders(split_path)
    
    if not env_folders:
        print(f"Warning: No env folders found in {split_path}, skipping {object_id}")
        return
    
    if frame_chunk_rng is None:
        frame_chunk_rng = random

    # Build scene plans first so we can sample one shared frame chunk per object.
    scene_plans = []
    for env_folder in sorted(env_folders):
        image_files_with_idx = _list_numeric_gt_images(split_path, env_folder)
        if not image_files_with_idx:
            print(f"Warning: No valid gt_*.png images with numeric indices found for {object_id}/{env_folder}, skipping")
            continue
        image_files_with_idx.sort(key=lambda x: x[0])

        base_scene_name = f"{object_id}_{env_folder}"
        should_process_with_variations = False
        should_process_without_variations = False
        if scenes_with_variations is not None or scenes_without_variations is not None:
            if scenes_with_variations is not None and base_scene_name in scenes_with_variations:
                should_process_with_variations = True
            elif scenes_without_variations is not None and base_scene_name in scenes_without_variations:
                should_process_without_variations = True
            else:
                print(f"Skipping {base_scene_name}: not in scene list")
                continue
        else:
            should_process_with_variations = True

        scene_plans.append({
            "env_folder": env_folder,
            "base_scene_name": base_scene_name,
            "image_files_with_idx": image_files_with_idx,
            "should_process_with_variations": should_process_with_variations,
            "should_process_without_variations": should_process_without_variations,
        })

    if len(scene_plans) == 0:
        return None

    selected_frame_idx_set = None
    if consecutive_frames and int(consecutive_frames) > 0:
        chunk_len = int(consecutive_frames)
        frame_sets = [[idx for idx, _ in p["image_files_with_idx"]] for p in scene_plans]
        selected_frame_idx_set, chunk_start = _sample_shared_consecutive_indices(frame_sets, chunk_len, frame_chunk_rng)
        if selected_frame_idx_set is None:
            print(
                f"Warning: Cannot find shared consecutive chunk of {chunk_len} frames for object {object_id}; skipping object"
            )
            return None
        print(
            f"Object {object_id}: using shared frame chunk [{chunk_start}, {chunk_start + chunk_len - 1}] "
            f"({chunk_len} consecutive frames)"
        )

    # Process each planned scene, creating n_variations for each.
    processed_scenes_with_variations = []  # Track scenes processed with variations
    processed_scenes_without_variations = []  # Track scenes processed without variations

    for plan in scene_plans:
        env_folder = plan["env_folder"]
        base_scene_name = plan["base_scene_name"]
        should_process_with_variations = plan["should_process_with_variations"]
        should_process_without_variations = plan["should_process_without_variations"]
        image_files_with_idx = plan["image_files_with_idx"]

        if selected_frame_idx_set is not None:
            image_files_with_idx = [(idx, img) for idx, img in image_files_with_idx if idx in selected_frame_idx_set]
            image_files_with_idx.sort(key=lambda x: x[0])
            if len(image_files_with_idx) != int(consecutive_frames):
                print(
                    f"Warning: {base_scene_name} does not contain the sampled shared frame chunk, skipping this scene"
                )
                continue

        # Get image dimensions from first image
        try:
            image_width, image_height = _read_image_size_from_source(split_path, env_folder, image_files_with_idx[0][1])
        except Exception as e:
            print(f"Error reading first image for {object_id}/{env_folder}: {e}, skipping")
            continue

        # Determine number of variations to create
        if should_process_with_variations:
            actual_n_variations = n_variations
        else:
            actual_n_variations = 1  # No variations, just process once
        
        # Create output directories
        output_metadata_dir = os.path.join(output_root, split, 'metadata')
        os.makedirs(output_metadata_dir, exist_ok=True)
        
        # Load environment map info if available (needed to check if envmaps should exist).
        # For train split, prefer corresponding test source; otherwise fallback to current split.
        env_info = None
        if split == 'train':
            test_split_path = os.path.join(object_path, 'test')
            if os.path.exists(test_split_path):
                env_info = _load_json_from_source(test_split_path, env_folder, ['env.json', 'white_env.json'])
        if env_info is None:
            env_info = _load_json_from_source(split_path, env_folder, ['env.json', 'white_env.json'])
        
        # Load environment map if available (shared across all variations)
        env_map = None
        base_euler_rotation = None
        if env_info and hdri_dir:
            env_map_name = env_info.get('env_map', '')
            if env_map_name:
                # Try different filename variations (with/without extension, with _8k suffix, etc.)
                possible_names = [
                    env_map_name,  # Original name as-is
                    f"{env_map_name}.exr",  # Add .exr extension
                    f"{env_map_name}.hdr",  # Add .hdr extension
                    f"{env_map_name}_8k.exr",  # Add _8k.exr suffix
                    f"{env_map_name}_8k.hdr",  # Add _8k.hdr suffix
                ]
                
                env_map_path = None
                for name in possible_names:
                    test_path = os.path.join(hdri_dir, name)
                    if os.path.exists(test_path):
                        env_map_path = test_path
                        break
                
                if env_map_path:
                    env_map = read_hdr(env_map_path)
                    if env_map is not None:
                        env_map = torch.from_numpy(env_map).float()
                        base_euler_rotation = env_info.get('rotation_euler', None)
                    else:
                        print(f"Warning: Failed to read environment map {env_map_path}")
                else:
                    print(f"Warning: Environment map '{env_map_name}' not found in {hdri_dir} (tried: {', '.join(possible_names)})")
        
        # Generate environment map directions if needed (shared across all variations)
        light_area_weight = None
        view_dirs = None
        if env_map is not None:
            env_h, env_w = env_map.shape[0], env_map.shape[1]
            light_area_weight, view_dirs = generate_envir_map_dir(env_h, env_w)
        
        # Create variations of the scene (or just process once if without variations)
        for variation_idx in range(1, actual_n_variations + 1):
            # Create variation scene name
            if should_process_with_variations:
                scene_name = f"{base_scene_name}_{variation_idx}"
            else:
                # No suffix for scenes without variations
                scene_name = base_scene_name
            
            # Create output directories for this variation
            output_images_dir = os.path.join(output_root, split, 'images', scene_name)
            output_envmaps_dir = os.path.join(output_root, split, 'envmaps', scene_name)
            
            # Check if scene already exists and all files are present
            scene_exists = os.path.exists(output_images_dir) and os.path.exists(output_envmaps_dir)
            
            # Initialize skip_file_processing flag
            skip_file_processing = False
            
            if scene_exists:
                # Check if all expected files exist
                all_files_exist = True
                
                # Check if all image files exist
                for frame_idx, image_file in image_files_with_idx:
                    output_image_name = f"{frame_idx:05d}.png"
                    output_image_path = os.path.join(output_images_dir, output_image_name)
                    if not os.path.exists(output_image_path):
                        all_files_exist = False
                        break
                
                # Check if environment maps exist (if env_info is available and hdri_dir is provided)
                if all_files_exist and env_info and hdri_dir:
                    for frame_idx, image_file in image_files_with_idx:
                        output_envmap_hdr_name = f"{frame_idx:05d}_hdr.png"
                        output_envmap_ldr_name = f"{frame_idx:05d}_ldr.png"
                        output_envmap_hdr_path = os.path.join(output_envmaps_dir, output_envmap_hdr_name)
                        output_envmap_ldr_path = os.path.join(output_envmaps_dir, output_envmap_ldr_name)
                        if not (os.path.exists(output_envmap_hdr_path) and os.path.exists(output_envmap_ldr_path)):
                            all_files_exist = False
                            break
                
                if all_files_exist:
                    print(f"Skipping image/envmap processing for {scene_name}: all files already exist")
                    skip_file_processing = True
                else:
                    skip_file_processing = False
            
            # Create directories if they don't exist
            os.makedirs(output_images_dir, exist_ok=True)
            os.makedirs(output_envmaps_dir, exist_ok=True)
            
            # Calculate additional rotation angle for this variation
            # Rotation angles are equally distributed from 0 to 2π
            # variation_idx goes from 1 to n_variations, so we use (variation_idx - 1) to get 0 to n_variations-1
            additional_rotation = (variation_idx - 1) * (2 * np.pi / n_variations)
            
            # Create modified euler_rotation with additional rotation
            euler_rotation = None
            if base_euler_rotation is not None:
                euler_rotation = list(base_euler_rotation)
                if len(euler_rotation) >= 3:
                    # Add additional rotation to z-axis (last axis)
                    euler_rotation[2] = euler_rotation[2] + additional_rotation
                else:
                    # If euler_rotation doesn't have 3 elements, create a new one
                    euler_rotation = [0.0, 0.0, additional_rotation]
            else:
                # If no base rotation, just use the additional rotation
                euler_rotation = [0.0, 0.0, additional_rotation]
            
            # Process frames
            frames = []
            for frame_idx, image_file in image_files_with_idx:
                # Find corresponding camera in cameras.json using frame_idx
                if frame_idx < len(cameras_data):
                    camera_info = cameras_data[frame_idx]
                else:
                    print(f"Warning: Frame {frame_idx} not found in cameras.json (only {len(cameras_data)} cameras available), skipping")
                    continue
                
                # Convert c2w from Blender to OpenCV
                c2w_blender = np.array(camera_info['c2w'])
                c2w_opencv = blender_to_opencv_c2w(c2w_blender)
                
                # Convert to w2c (world-to-camera)
                w2c = np.linalg.inv(c2w_opencv)
                
                # Convert FOV to fxfycxcy
                fov = camera_info.get('fov', 30.0)  # Default to 30 if not specified
                fxfycxcy = fov_to_fxfycxcy(fov, image_width, image_height)
                
                # Determine output image path (used for JSON even if we skip copying)
                output_image_name = f"{frame_idx:05d}.png"
                output_image_path = os.path.join(output_images_dir, output_image_name)
                
                # Only copy image and process envmaps if not skipping
                if not skip_file_processing:
                    # Copy image to output directory with zero-padded name
                    _copy_image_from_source(split_path, env_folder, image_file, output_image_path)
                    
                    # Process environment map if available
                    if env_map is not None:
                        # Get camera pose (c2w) - use Blender format before OpenCV conversion
                        c2w_blender = np.array(camera_info['c2w'])
                        
                        # Rotate and preprocess environment map (using Blender format c2w)
                        # Use the modified euler_rotation with additional rotation
                        env_hdr_raw, env_ldr, env_hdr = rotate_and_preprocess_envir_map(
                            env_map, c2w_blender, euler_rotation=euler_rotation,
                            light_area_weight=light_area_weight, view_dirs=view_dirs
                        )
                        
                        if env_hdr_raw is not None:
                            # Save HDR and LDR versions separately
                            # HDR version: log transform and rescale
                            env_hdr_uint8 = np.uint8(env_hdr * 255)
                            env_hdr_img = Image.fromarray(env_hdr_uint8)
                            output_envmap_hdr_name = f"{frame_idx:05d}_hdr.png"
                            output_envmap_hdr_path = os.path.join(output_envmaps_dir, output_envmap_hdr_name)
                            env_hdr_img.save(output_envmap_hdr_path)
                            
                            # LDR version: gamma correction
                            env_ldr_uint8 = np.uint8(env_ldr * 255)
                            env_ldr_img = Image.fromarray(env_ldr_uint8)
                            output_envmap_ldr_name = f"{frame_idx:05d}_ldr.png"
                            output_envmap_ldr_path = os.path.join(output_envmaps_dir, output_envmap_ldr_name)
                            env_ldr_img.save(output_envmap_ldr_path)
                
                # Create absolute image path for the JSON file
                # This ensures the path works regardless of where the code is run from
                absolute_image_path = os.path.abspath(output_image_path)
                
                # Create frame entry
                frame = {
                    "image_path": absolute_image_path,
                    "fxfycxcy": fxfycxcy,
                    "w2c": w2c.tolist()
                }
                frames.append(frame)
            
            # Create scene JSON
            scene_data = {
                "scene_name": scene_name,
                "frames": frames
            }
            
            # Save scene JSON (always regenerate, even if files were skipped)
            output_json_path = os.path.join(output_metadata_dir, f"{scene_name}.json")
            with open(output_json_path, 'w') as f:
                json.dump(scene_data, f, indent=2)
            
            if skip_file_processing:
                if should_process_with_variations:
                    print(f"Regenerated JSON for {scene_name}: {len(frames)} frames (skipped file processing, variation {variation_idx}/{actual_n_variations})")
                else:
                    print(f"Regenerated JSON for {scene_name}: {len(frames)} frames (skipped file processing)")
            else:
                if should_process_with_variations:
                    print(f"Processed {scene_name}: {len(frames)} frames (variation {variation_idx}/{actual_n_variations}, rotation={np.degrees(additional_rotation):.2f}°)")
                else:
                    print(f"Processed {scene_name}: {len(frames)} frames (no variations)")
            
            # Add to appropriate list
            if should_process_with_variations:
                processed_scenes_with_variations.append(scene_name)
            else:
                processed_scenes_without_variations.append(scene_name)
    
    # Return lists of processed scenes
    return processed_scenes_with_variations, processed_scenes_without_variations


def create_full_list(output_root, split='test', broken_scenes=None, 
                    scenes_with_variations=None, scenes_without_variations=None):
    """
    Create two full_list.txt files: one for scenes with variations, one for scenes without.
    
    Args:
        output_root: Root directory for output
        split: 'train' or 'test'
        broken_scenes: List of broken scene names to exclude
        scenes_with_variations: Set of scene names that have variations (to identify which scenes belong to which list)
        scenes_without_variations: Set of scene names without variations
    """
    metadata_dir = os.path.join(output_root, split, 'metadata')
    if not os.path.exists(metadata_dir):
        print(f"Warning: {metadata_dir} does not exist")
        return
    
    json_files = sorted([f for f in os.listdir(metadata_dir) if f.endswith('.json')])
    
    if broken_scenes is None:
        broken_scenes = []
    
    # Separate scenes into with_variations and without_variations
    json_files_with_variations = []
    json_files_without_variations = []
    
    for json_file in json_files:
        scene_name = json_file.replace('.json', '')
        
        # Skip broken scenes
        if scene_name in broken_scenes:
            continue
        
        # Check if this scene has variations by checking if the base scene name is in the variations list
        # Scene names with variations are like: base_scene_name_1, base_scene_name_2, etc.
        # We need to extract the base scene name (remove _1, _2, etc. suffix)
        is_with_variations = False
        is_without_variations = False
        
        if scenes_with_variations is not None:
            # Try to find base scene name by checking if scene_name starts with a base name + variation suffix
            for base_name in scenes_with_variations:
                # Check if scene_name is base_name_1, base_name_2, etc.
                if scene_name.startswith(base_name + '_'):
                    # Extract the suffix part
                    suffix = scene_name[len(base_name) + 1:]
                    # Check if suffix is a number (variation index)
                    try:
                        variation_idx = int(suffix)
                        if variation_idx >= 1:  # Valid variation index
                            is_with_variations = True
                            break
                    except ValueError:
                        # Not a number, might be a different scene
                        pass
        
        if not is_with_variations and scenes_without_variations is not None:
            # Check if scene_name exactly matches a scene in without_variations
            # (scenes without variations don't have _1, _2 suffixes)
            if scene_name in scenes_without_variations:
                is_without_variations = True
        
        # If scene lists are provided but scene is not in either, skip it
        if (scenes_with_variations is not None or scenes_without_variations is not None):
            if not is_with_variations and not is_without_variations:
                continue
        
        # Add to appropriate list
        if is_with_variations:
            json_files_with_variations.append(json_file)
        else:
            json_files_without_variations.append(json_file)
    
    # Create full_list_with_variations.txt
    full_list_with_variations_path = os.path.join(output_root, split, 'full_list_with_variations.txt')
    with open(full_list_with_variations_path, 'w') as f:
        for json_file in json_files_with_variations:
            json_path = os.path.join(metadata_dir, json_file)
            f.write(f"{os.path.abspath(json_path)}\n")
    print(f"Created {full_list_with_variations_path} with {len(json_files_with_variations)} scenes (with variations)")
    
    # Create full_list_without_variations.txt
    full_list_without_variations_path = os.path.join(output_root, split, 'full_list_without_variations.txt')
    with open(full_list_without_variations_path, 'w') as f:
        for json_file in json_files_without_variations:
            json_path = os.path.join(metadata_dir, json_file)
            f.write(f"{os.path.abspath(json_path)}\n")
    print(f"Created {full_list_without_variations_path} with {len(json_files_without_variations)} scenes (without variations)")
    
    # Create full_list.txt that contains both kinds of scenes
    all_json_files = sorted(json_files_with_variations + json_files_without_variations)
    full_list_path = os.path.join(output_root, split, 'full_list.txt')
    with open(full_list_path, 'w') as f:
        for json_file in all_json_files:
            json_path = os.path.join(metadata_dir, json_file)
            f.write(f"{os.path.abspath(json_path)}\n")
    print(f"Created {full_list_path} with {len(all_json_files)} scenes (both with and without variations)")


def main():
    parser = argparse.ArgumentParser(description='Preprocess Objaverse data to re10k format with environment map variations')
    parser.add_argument('--input', '-i', required=True,
                       help='Input directory containing objaverse data (e.g., data_samples/sample_objaverse)')
    parser.add_argument('--output', '-o', required=True,
                       help='Output directory for processed data (e.g., data_samples/objaverse_processed)')
    parser.add_argument('--split', '-s', default='test', choices=['train', 'test'],
                       help='Split to process (default: test)')
    parser.add_argument('--object-id', type=str, default=None,
                       help='Process specific object ID only (default: process all)')
    parser.add_argument('--hdri-dir', type=str, default=None,
                       help='Directory containing HDR environment maps (e.g., data_samples/sample_hdris)')
    parser.add_argument('--n-variations', type=int, default=1,
                       help='Number of variations to create for each scene (default: 1). Each variation uses the same env map with different rotation angles (0 to 2π).')
    parser.add_argument('--test-run', action='store_true',
                       help='Test run: only process first 5 objects (default: False)')
    parser.add_argument('--max-objects', type=int, default=None,
                       help='Maximum number of objects to process (overrides --test-run if specified)')
    parser.add_argument('--scene-list', type=str, default=None,
                       help='Path to JSON file containing list of scenes to process. If provided, only scenes in this list will be processed. Supports list format ["scene1", "scene2"] or dict format {"scene1": {...}, "scene2": {...}}')
    parser.add_argument('--consecutive-frames', type=int, default=0,
                       help='If > 0, sample this many consecutive frames per object and use the same chunk for all scenes of that object.')
    parser.add_argument('--frame-chunk-seed', type=int, default=777,
                       help='Random seed used when sampling object-level consecutive frame chunks.')
    
    args = parser.parse_args()
    
    objaverse_root = args.input
    output_root = args.output
    
    if not os.path.exists(objaverse_root):
        print(f"Error: Input directory {objaverse_root} does not exist")
        return
    
    if args.n_variations < 1:
        print(f"Error: --n-variations must be >= 1, got {args.n_variations}")
        return
    if args.consecutive_frames < 0:
        print(f"Error: --consecutive-frames must be >= 0, got {args.consecutive_frames}")
        return
    
    # Load scene list if provided
    scenes_with_variations, scenes_without_variations = load_scene_list(args.scene_list)
    if args.scene_list and scenes_with_variations is None and scenes_without_variations is None:
        print(f"Error: Failed to load scene list from {args.scene_list}.")
        return
    
    # Find all object folders
    if args.object_id:
        object_ids = [args.object_id]
    elif args.scene_list and (scenes_with_variations is not None or scenes_without_variations is not None):
        requested_scenes = set()
        if scenes_with_variations:
            requested_scenes.update(scenes_with_variations)
        if scenes_without_variations:
            requested_scenes.update(scenes_without_variations)
        requested_object_ids = sorted({_extract_object_id_from_scene(scene_name) for scene_name in requested_scenes})
        missing_objects = [oid for oid in requested_object_ids if not os.path.isdir(os.path.join(objaverse_root, oid))]
        if missing_objects:
            print(f"Warning: {len(missing_objects)} objects from scene-list not found under input root (examples: {missing_objects[:5]})")
        object_ids = [oid for oid in requested_object_ids if os.path.isdir(os.path.join(objaverse_root, oid))]
        print(f"Scene-list restricted mode: processing {len(object_ids)} objects mapped from scene list")
    else:
        object_ids = [d for d in os.listdir(objaverse_root)
                     if os.path.isdir(os.path.join(objaverse_root, d))]
    
    # Apply test run or max objects limit
    original_count = len(object_ids)
    if args.max_objects is not None:
        object_ids = object_ids[:args.max_objects]
        print(f"Limiting to {args.max_objects} objects (from {original_count} total)")
    elif args.test_run:
        test_count = min(5, len(object_ids))
        object_ids = object_ids[:test_count]
        print(f"TEST RUN: Processing first {test_count} objects only (from {original_count} total)")
    
    print(f"Processing {len(object_ids)} objects with {args.n_variations} variations per scene...")
    if args.consecutive_frames > 0:
        print(
            f"Object-level frame chunk mode enabled: {args.consecutive_frames} consecutive frames "
            f"(seed={args.frame_chunk_seed})"
        )
    
    broken_scenes = []
    processed_scenes_with_variations = []
    processed_scenes_without_variations = []
    skipped_scenes = []  # Scenes that were skipped because files already exist
    
    frame_chunk_rng = random.Random(args.frame_chunk_seed)

    for object_id in sorted(object_ids):
        print(f"\nProcessing object: {object_id}")
        try:
            result = process_objaverse_scene(objaverse_root, object_id, output_root, 
                                            split=args.split, hdri_dir=args.hdri_dir,
                                            n_variations=args.n_variations, 
                                            scenes_with_variations=scenes_with_variations,
                                            scenes_without_variations=scenes_without_variations,
                                            consecutive_frames=args.consecutive_frames,
                                            frame_chunk_rng=frame_chunk_rng)
            if result == "broken":
                # Find all scenes for this object and mark them as broken
                split_path = os.path.join(objaverse_root, object_id, args.split)
                if os.path.exists(split_path):
                    env_folders = _list_env_folders(split_path)
                    for env_folder in env_folders:
                        base_scene_name = f"{object_id}_{env_folder}"
                        # Mark all variations as broken
                        for variation_idx in range(1, args.n_variations + 1):
                            scene_name = f"{base_scene_name}_{variation_idx}"
                            broken_scenes.append(scene_name)
            elif isinstance(result, tuple) and len(result) == 2:
                # result is a tuple: (processed_scenes_with_variations, processed_scenes_without_variations)
                processed_with_vars, processed_without_vars = result
                processed_scenes_with_variations.extend(processed_with_vars)
                processed_scenes_without_variations.extend(processed_without_vars)
            elif result is None:
                # No scenes were processed (maybe all were skipped or object had no valid scenes)
                pass
        except Exception as e:
            print(f"Error processing {object_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # Create full_list.txt files (excluding broken scenes)
    print(f"\nCreating full_list.txt files for {args.split} split...")
    create_full_list(output_root, split=args.split, broken_scenes=broken_scenes,
                    scenes_with_variations=scenes_with_variations,
                    scenes_without_variations=scenes_without_variations)
    
    # Save broken scenes list to a file
    if broken_scenes:
        broken_list_path = os.path.join(output_root, args.split, 'broken_scenes.txt')
        with open(broken_list_path, 'w') as f:
            for scene_name in sorted(broken_scenes):
                f.write(f"{scene_name}\n")
        print(f"Saved broken scenes list to {broken_list_path}")
    
    print(f"\nPreprocessing complete!")
    print(f"  - Processed {len(processed_scenes_with_variations)} scenes with variations")
    print(f"  - Processed {len(processed_scenes_without_variations)} scenes without variations")
    print(f"  - Skipped {len(skipped_scenes)} scenes (files already exist)")
    print(f"  - Marked {len(broken_scenes)} scenes as broken")


if __name__ == '__main__':
    main()

