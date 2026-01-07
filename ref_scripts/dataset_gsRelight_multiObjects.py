import os
import json
import math
from pathlib import Path
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2
import random
import torchvision.transforms as T
import torch.nn.functional as F
import pyexr
import sys
sys.path.append('/projects/vig/yiwenc/ResearchProjects/lightingDiffusion/3dgs/FLUX_finetune')
from gaffer_preprocess import get_light

def read_scene_list(scene_list_file):
    """
    Read scene names from a text file.
    Each line should contain one scene name.
    Removes '_flux' suffix if present to get the actual scene name.
    
    Args:
        scene_list_file: Path to the text file containing scene names
        
    Returns:
        List of scene names
    """
    scene_names = []
    try:
        with open(scene_list_file, 'r') as f:
            for line in f:
                scene_name = line.strip()
                if scene_name:  # Skip empty lines
                    # Remove '_flux' suffix if present
                    if scene_name.endswith('_flux'):
                        scene_name = scene_name[:-5]  # Remove '_flux'
                    scene_names.append(scene_name)
        print(f"Loaded {len(scene_names)} scene names from {scene_list_file}")
        return scene_names
    except FileNotFoundError:
        print(f"Scene list file not found: {scene_list_file}")
        return []
    except Exception as e:
        print(f"Error reading scene list file {scene_list_file}: {e}")
        return []

def generate_envir_map_dir(envmap_h, envmap_w):
    lat_step_size = np.pi / envmap_h
    lng_step_size = 2 * np.pi / envmap_w
    theta, phi = torch.meshgrid([torch.linspace(np.pi / 2 - 0.5 * lat_step_size, -np.pi / 2 + 0.5 * lat_step_size, envmap_h), 
                                torch.linspace(np.pi - 0.5 * lng_step_size, -np.pi + 0.5 * lng_step_size, envmap_w)], indexing='ij')

    sin_theta = torch.sin(torch.pi / 2 - theta)  # [envH, envW]
    light_area_weight = 4 * torch.pi * sin_theta / torch.sum(sin_theta)  # [envH, envW]
    assert 0 not in light_area_weight, "There shouldn't be light pixel that doesn't contribute"
    light_area_weight = light_area_weight.to(torch.float32).reshape(-1) # [envH * envW, ]


    view_dirs = torch.stack([   torch.cos(phi) * torch.cos(theta), 
                                torch.sin(phi) * torch.cos(theta), 
                                torch.sin(theta)], dim=-1).view(-1, 3)    # [envH * envW, 3]
    light_area_weight = light_area_weight.reshape(envmap_h, envmap_w)
    
    return light_area_weight, view_dirs

def read_hdr_exr(path):
    """Read HDR EXR file using pyexr"""
    try:
        # Use pyexr to read EXR file
        rgb = pyexr.read(path)
        # pyexr returns RGB format directly, no need for BGR to RGB conversion
        
        # Ensure we only have 3 channels (RGB), remove alpha channel if present
        if rgb is not None and rgb.shape[-1] == 4:
            rgb = rgb[:, :, :3]  # Take only RGB channels, remove alpha
            # print(f"Removed alpha channel from {path}, shape: {rgb.shape}")
        elif rgb is not None and rgb.shape[-1] != 3:
            # print(f"Unexpected number of channels in {path}: {rgb.shape[-1]}")
            pass
    except Exception as e:
        print(f"Error reading HDR file {path}: {e}")
        rgb = None
    return rgb

def get_light_from_envmap(envir_map, incident_dir, hdr_weight=None, if_weighted=False):
    """Sample light from environment map given incident direction"""
    try:
        envir_map = envir_map.permute(2, 0, 1).unsqueeze(0) # [1, 3, H, W]
        if hdr_weight is not None:
            hdr_weight = hdr_weight.unsqueeze(0).unsqueeze(0)   # [1, 1, H, W]
        incident_dir = incident_dir.clamp(-1, 1)
        theta = torch.arccos(incident_dir[:, 2]).reshape(-1) # top to bottom: 0 to pi
        phi = torch.atan2(incident_dir[:, 1], incident_dir[:, 0]).reshape(-1) # left to right: pi to -pi
        
        query_y = (theta / np.pi) * 2 - 1 # top to bottom: -1-> 1
        query_y = query_y.clamp(-1+10e-8, 1-10e-8)
        query_x = -phi / np.pi # left to right: -1 -> 1
        query_x = query_x.clamp(-1+10e-8, 1-10e-8)

        grid = torch.stack((query_x, query_y)).permute(1, 0).unsqueeze(0).unsqueeze(0).float() # [1, 1, N, 2]
        
        if if_weighted is False or hdr_weight is None:
            light_rgbs = F.grid_sample(envir_map, grid, align_corners=True).squeeze().permute(1, 0).reshape(-1, 3)
        else:
            weighted_envir_map = envir_map * hdr_weight
            light_rgbs = F.grid_sample(weighted_envir_map, grid, align_corners=True).squeeze().permute(1, 0).reshape(-1, 3)
            light_rgbs = light_rgbs / hdr_weight.reshape(-1, 1)
            
    except Exception as e:
        print(f"Error in get_light_from_envmap: {e}")
        light_rgbs = None
    return light_rgbs

def rotate_and_preprocess_envir_map(envir_map, camera_pose, euler_rotation=None, target_resolution=512, light_area_weight=None, view_dirs=None, split_env_map=False):
    """Rotate and preprocess environment map based on euler rotation and camera pose"""
    try:
        env_h, env_w = envir_map.shape[0], envir_map.shape[1]
        if light_area_weight is None or view_dirs is None:
            light_area_weight, view_dirs = generate_envir_map_dir(env_h, env_w)
        
        # Store original environment map for raw version (before rotation)
        envir_map_raw = envir_map.cpu().numpy()
        
        # Step 1: Apply euler rotation (horizontal roll for equirectangular HDRI) if provided
        if euler_rotation is not None:
            # Extract Z rotation (horizontal rotation angle in range [-pi, pi])
            z_rotation = euler_rotation[2] if len(euler_rotation) >= 3 else 0.0
            
            # Convert radians to degrees
            rotation_angle_deg = np.degrees(z_rotation)
            
            # Apply horizontal roll to the environment map (equirectangular projection)
            envir_map_np = envir_map.cpu().numpy()
            h, w = envir_map_np.shape[:2]
            
            # Convert degrees to pixel shift
            # Positive degrees = CCW from above (Blender-style +Z rotation)
            shift = int((rotation_angle_deg / 360.0) * w)
            
            # Roll image horizontally
            rotated_envir_map = np.roll(envir_map_np, shift, axis=1)
            
            # Convert back to tensor
            envir_map = torch.from_numpy(rotated_envir_map).float()
        
        # Use original view directions (no need to modify them for horizontal roll)
        view_dirs_euler = view_dirs
        
        # Step 2: Apply camera pose rotation (convert c2w to w2c)
        # Convert camera pose to rotation matrix (assuming it's 4x4 c2w)
        if camera_pose.shape == (4, 4):
            c2w_rotation = camera_pose[:3, :3]  # Extract rotation part from c2w
            # Convert c2w to w2c by taking transpose (for rotation matrices, transpose = inverse)
            w2c_rotation = c2w_rotation.T  # Convert c2w rotation to w2c
        else:
            # Assume input is already c2w rotation matrix
            w2c_rotation = camera_pose.T  # Convert c2w rotation to w2c
        
        # Blender's convention
        axis_aligned_transform = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        # Apply axis alignment transform to w2c rotation
        axis_aligned_R = axis_aligned_transform @ w2c_rotation  # [3, 3]
        view_dirs_world = view_dirs_euler @ axis_aligned_R  # [envH * envW, 3]
        
        # Apply rotation using get_light function
        try:
            rotated_hdr_rgb = get_light(envir_map, view_dirs_world.clamp(-1, 1))
            if rotated_hdr_rgb is not None and rotated_hdr_rgb.numel() > 0:
                # Ensure the tensor has the correct shape
                if rotated_hdr_rgb.shape[-1] == 3:
                    rotated_hdr_rgb = rotated_hdr_rgb.reshape(env_h, env_w, 3).cpu().numpy()
                else:
                    print(f"Unexpected shape from get_light: {rotated_hdr_rgb.shape}")
                    rotated_hdr_rgb = envir_map_raw
            else:
                # Fallback to original environment map if get_light fails
                rotated_hdr_rgb = envir_map_raw
        except Exception as e:
            print(f"Error in get_light, using original env map: {e}")
            rotated_hdr_rgb = envir_map_raw

        # Separate HDR and LDR following NeuralGaffer's approach
        # HDR raw (use rotated environment map, not original)
        envir_map_hdr_raw = rotated_hdr_rgb

        # LDR (gamma correction of rotated environment map)
        envir_map_ldr = rotated_hdr_rgb.clip(0, 1)
        envir_map_ldr = envir_map_ldr ** (1/2.2)
        
        # HDR processed (log transform of rotated environment map)
        envir_map_hdr = np.log1p(10 * rotated_hdr_rgb)
        envir_map_hdr_rescaled = (envir_map_hdr / np.max(envir_map_hdr)).clip(0, 1)
        
        # Convert to uint8 for PIL
        envir_map_ldr = np.uint8(envir_map_ldr * 255)
        envir_map_hdr = np.uint8(envir_map_hdr_rescaled * 255)
        
        envir_map_ldr = Image.fromarray(envir_map_ldr)
        envir_map_hdr = Image.fromarray(envir_map_hdr)
        
        # Handle split env map mode
        if split_env_map:
            # Check if env_w = 2 * env_h (expected for equirectangular maps)
            if env_w != 2 * env_h:
                print(f"Warning: Expected env_w=2*env_h, but got {env_w}x{env_h}. Proceeding with split anyway.")
            
            # Split the environment map into two halves
            mid_point = env_w // 2
            
            # Split LDR environment map
            envir_map_ldr_np = np.array(envir_map_ldr)
            envir_map_ldr_back = envir_map_ldr_np[:, :mid_point, :]  # First half (back)
            envir_map_ldr_front = envir_map_ldr_np[:, mid_point:, :]  # Second half (front)
            
            # Split HDR environment map
            envir_map_hdr_np = np.array(envir_map_hdr)
            envir_map_hdr_back = envir_map_hdr_np[:, :mid_point, :]  # First half (back)
            envir_map_hdr_front = envir_map_hdr_np[:, mid_point:, :]  # Second half (front)
            
            # Flip front half horizontally
            envir_map_ldr_front = np.fliplr(envir_map_ldr_front)
            envir_map_hdr_front = np.fliplr(envir_map_hdr_front)
            
            # Convert back to PIL Images
            envir_map_ldr_back = Image.fromarray(envir_map_ldr_back)
            envir_map_ldr_front = Image.fromarray(envir_map_ldr_front)
            envir_map_hdr_back = Image.fromarray(envir_map_hdr_back)
            envir_map_hdr_front = Image.fromarray(envir_map_hdr_front)
            
            # Resize each half to target resolution
            if target_resolution != env_h:
                envir_map_ldr_back = envir_map_ldr_back.resize((target_resolution, target_resolution), Image.BILINEAR)
                envir_map_ldr_front = envir_map_ldr_front.resize((target_resolution, target_resolution), Image.BILINEAR)
                envir_map_hdr_back = envir_map_hdr_back.resize((target_resolution, target_resolution), Image.BILINEAR)
                envir_map_hdr_front = envir_map_hdr_front.resize((target_resolution, target_resolution), Image.BILINEAR)
            
            return envir_map_ldr_back, envir_map_ldr_front, envir_map_hdr_back, envir_map_hdr_front
        
        else:
            # Original behavior: single environment map
            # Resize environment maps to target resolution if needed
            if target_resolution != env_h or target_resolution != env_w:
                # Use bilinear interpolation for smooth resizing
                envir_map_ldr = envir_map_ldr.resize((target_resolution, target_resolution), Image.BILINEAR)
                envir_map_hdr = envir_map_hdr.resize((target_resolution, target_resolution), Image.BILINEAR)
                
                # Also resize the raw HDR data
                envir_map_hdr_raw = cv2.resize(envir_map_hdr_raw, (target_resolution, target_resolution), interpolation=cv2.INTER_LINEAR)
            
            return envir_map_ldr, envir_map_hdr, envir_map_hdr_raw
        
    except Exception as e:
        print(f"Error in rotating and preprocessing envir_map: {e}")
        # Return default values with target resolution
        if split_env_map:
            # Return 4 default images for split mode
            default_img = Image.new('RGB', (target_resolution, target_resolution), (0, 0, 0))
            return default_img, default_img, default_img, default_img
        else:
            # Return 3 default values for normal mode
            envir_map_ldr = Image.new('RGB', (target_resolution, target_resolution), (0, 0, 0))
            envir_map_hdr = Image.new('RGB', (target_resolution, target_resolution), (0, 0, 0))
            envir_map_hdr_raw = np.zeros((target_resolution, target_resolution, 3))
            return envir_map_ldr, envir_map_hdr, envir_map_hdr_raw

class GSRelightDataset(Dataset):
    """
    Dataset for GSRelight training with new directory structure.
    Loads GT images from train/env_* folders (training) or test/env_* folders (validation) 
    and conditional latents from GSTrain/[latent_folder] or GSTest/[latent_folder].
    """
    
    def __init__(self,
                 gt_data_dir,
                 latent_folder_name='flux_latents_64',
                 latent_base_dir=None,  # Base latent directory like ../torchSplattingMod/result/01c9013483b6427fbc2f478e5e328810_flux_64
                 envmaps_dir='/projects/vig/Datasets/objaverse/envmaps_256/hdirs/',
                 image_transforms=None,
                 validation=False,
                 fix_sample=False,
                 resolution=512,
                 custom_euler_rotation=None,  # Custom euler rotation for inference experiments
                 split_env_map=False,  # Whether to split environment maps into front/back halves
                 scene_list_file=None,  # Path to file containing list of scene names
                 gt_data_base_dir='/projects/vig/Datasets/objaverse/hf-objaverse-v1/rendered_dense',  # Base directory for GT data
                 latent_data_base_dir='../torchSplattingMod/result',  # Base directory for latent data
                 denoising_mode=False,  # Whether to use denoising mode (always use white_env_0)
                 use_gpt_captions=False):  # Whether to use GPT captions (caption_gpt.txt) instead of original captions (caption.txt)
        """
        Args:
            gt_data_dir: Base directory containing GSTrain, GSTest, train/test folders (used when scene_list_file is None)
            latent_folder_name: Name of the latent folder (used only if latent_base_dir is None)
            latent_base_dir: Base directory for latents. Will try eval_train_step_50000/eval_test_step_50000 first, 
                            then fallback to eval_step_100000 or eval_step_50000 if not found
            envmaps_dir: Directory containing environment maps (.exr files)
            image_transforms: Image transformations
            validation: Whether this is validation dataset (uses GSTest instead of GSTrain)
            fix_sample: Whether to use fixed samples for logging
            resolution: Target resolution for images
            custom_euler_rotation: Custom euler rotation [x, y, z] for inference experiments (overrides dataset rotation)
            split_env_map: Whether to split environment maps into front/back halves (returns 4 env maps instead of 2)
            scene_list_file: Path to file containing list of scene names (one per line). If provided, loads data from multiple scenes.
            gt_data_base_dir: Base directory for GT data when using scene_list_file (default: /projects/vig/Datasets/objaverse/hf-objaverse-v1/rendered_dense)
            latent_data_base_dir: Base directory for latent data when using scene_list_file (default: ../torchSplattingMod/result)
            denoising_mode: Whether to use denoising mode (always use white_env_0 for GT images, default: False)
            use_gpt_captions: Whether to use GPT captions (caption_gpt.txt) instead of original captions (caption.txt), default: False
        """
        self.envmaps_dir = Path(envmaps_dir)
        self.fix_sample = fix_sample
        self.resolution = resolution
        self.validation = validation
        self.split_env_map = split_env_map
        self.latent_base_dir = Path(latent_base_dir) if latent_base_dir else None
        self.custom_euler_rotation = custom_euler_rotation
        self.scene_list_file = scene_list_file
        self.gt_data_base_dir = Path(gt_data_base_dir)
        self.latent_data_base_dir = Path(latent_data_base_dir)
        self.denoising_mode = denoising_mode
        self.use_gpt_captions = use_gpt_captions
        
        # Counter for logging first 5 objects during testing
        self.logged_objects = 0
        self.max_logged_objects = 5
        
        # Initialize scene data
        self.scene_data = []  # List to store data from all scenes
        self.scene_names = []  # List to store scene names
        
        # Initialize dataset based on whether we're using scene list or single scene
        if self.scene_list_file is not None:
            self._initialize_multi_scene_dataset()
        else:
            self._initialize_single_scene_dataset(gt_data_dir, latent_folder_name)
        
        print(f'============= GSRelight Dataset: {len(self.available_images)} valid samples =============')
        print(f'============= Using {"GSTest" if validation else "GSTrain"} data: {len(self.available_images)} samples =============')
        
        # Set up transforms
        if image_transforms is None:
            self.transforms = T.Compose([
            T.Resize((resolution, resolution),
                    interpolation=T.InterpolationMode.BICUBIC,
                    antialias=True),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])
        else:
            self.transforms = image_transforms

    def _initialize_single_scene_dataset(self, gt_data_dir, latent_folder_name):
        """Initialize dataset for a single scene (original behavior)"""
        self.gt_data_dir = Path(gt_data_dir)
        
        # (1) Load info.json from GSTrain or GSTest directory based on validation flag
        if self.validation:
            data_dir = self.gt_data_dir / 'GSTest'
        else:
            data_dir = self.gt_data_dir / 'GSTrain'
            
        info_path = data_dir / 'info.json'
        if not info_path.exists():
            raise FileNotFoundError(f"info.json not found in {data_dir}")
        
        with open(info_path, 'r') as f:
            self.info = json.load(f)
        
        # (2) Set up latent directory
        if self.latent_base_dir is not None:
            # Use latent_base_dir with eval_train_step_50000 or eval_test_step_50000 subfolders
            if self.validation:
                # Try eval_test_step_50000 first, then fallback to eval_step_100000 or eval_step_50000
                potential_dirs = [
                    self.latent_base_dir / 'eval_test_step_50000',
                    self.latent_base_dir / 'eval_step_100000',
                    self.latent_base_dir / 'eval_step_50000'
                ]
            else:
                # Try eval_train_step_50000 first, then fallback to eval_step_100000 or eval_step_50000
                potential_dirs = [
                    self.latent_base_dir / 'eval_train_step_50000',
                    self.latent_base_dir / 'eval_step_100000',
                    self.latent_base_dir / 'eval_step_50000',
                    self.latent_base_dir / latent_folder_name  # Also try the folder name directly
                ]
            
            # Find the first existing directory
            self.latent_dir = None
            for potential_dir in potential_dirs:
                if potential_dir.exists():
                    self.latent_dir = potential_dir
                    break
            
            if self.latent_dir is None:
                raise FileNotFoundError(f"None of the expected latent directories found in {self.latent_base_dir}. Tried: {[str(d) for d in potential_dirs]}")
        else:
            # Use default location in GSTrain/GSTest
            self.latent_dir = data_dir / latent_folder_name
            if not self.latent_dir.exists():
                raise FileNotFoundError(f"Latent directory not found: {self.latent_dir}")
        
        # (3) Get available environment folders from train or test directory
        if self.validation:
            # For validation, get environment folders from test directory
            env_dir = self.gt_data_dir / 'test'
            if not env_dir.exists():
                raise FileNotFoundError(f"Test directory not found: {env_dir}")
        else:
            # For training, get environment folders from train directory
            env_dir = self.gt_data_dir / 'train'
            if not env_dir.exists():
                raise FileNotFoundError(f"Train directory not found: {env_dir}")
        
        # Find all env_* folders in the appropriate directory
        self.env_folders = []
        for item in env_dir.iterdir():
            if item.is_dir() and item.name.startswith('env_'):
                self.env_folders.append(item.name)
        if self.denoising_mode:
            self.env_folders.append('white_env_0')
        # self.env_folders.append('white_env_0')
        
        
        if not self.env_folders:
            raise FileNotFoundError(f"No env_* folders found in {env_dir}")
        
        print(f'Found environment folders: {self.env_folders}')
        
        # (4) Get list of available images
        self.available_images = []
        for img_info in self.info['images']:
            rgb_file = img_info['rgb']
            if rgb_file:
                # Extract sample number from filename (e.g., gt_0.png -> 0)
                sample_num = rgb_file.split('_')[1].split('.')[0]
                
                # Try different naming conventions for conditional latents
                conditional_files = [
                    f"gt_{sample_num}.npy",  # Default format
                    f"sample_{int(sample_num):04d}.npy"  # Alternative format
                ]
                
                conditional_file = None
                for cf in conditional_files:
                    conditional_path = self.latent_dir / cf
                    if conditional_path.exists():
                        conditional_file = cf
                        break
                
                if conditional_file:
                        self.available_images.append({
                            'gt_image': rgb_file,
                            'conditional_latent': conditional_file,
                            'sample_num': sample_num,
                            'metadata': img_info,
                            'intrinsic': img_info.get('intrinsic', None),
                            'pose': img_info.get('pose', None),
                            'scene_name': None,  # No scene name for single scene mode
                            'latent_dir': self.latent_dir,
                            'caption': None  # No caption for single scene mode
                        })

    def _initialize_multi_scene_dataset(self):
        """Initialize dataset for multiple scenes from scene list file"""
        # Read scene names from file
        scene_names = read_scene_list(self.scene_list_file)
        if not scene_names:
            raise ValueError(f"No valid scene names found in {self.scene_list_file}")
        
        self.scene_names = scene_names
        self.available_images = []
        
        # Detect latent suffix based on keywords in scene_list_file name
        scene_list_filename = str(self.scene_list_file).lower()
        if "dino32" in scene_list_filename:
            latent_suffix = "_dino_32"
            print(f"Detected 'dino32' in scene list file, using latent suffix: {latent_suffix}")
        elif "flux64" in scene_list_filename:
            latent_suffix = "_flux_64"
            print(f"Detected 'flux64' in scene list file, using latent suffix: {latent_suffix}")
        else:
            # Default to flux_64 if no keyword is found
            latent_suffix = "_flux_64"
            print(f"No specific keyword found in scene list file, using default latent suffix: {latent_suffix}")
        
        # Process each scene
        for scene_name in scene_names:
            scene_name = scene_name.split('_')[0]
            print(f"Processing scene: {scene_name}")
            
            # Set up paths for this scene using detected latent suffix
            scene_gt_dir = self.gt_data_base_dir / scene_name
            scene_latent_dir = self.latent_data_base_dir / f"{scene_name}{latent_suffix}"
            
            # Check if scene directories exist
            if not scene_gt_dir.exists():
                print(f"Warning: GT directory not found for scene {scene_name}: {scene_gt_dir}")
                continue
            
            if not scene_latent_dir.exists():
                print(f"Warning: Latent directory not found for scene {scene_name}: {scene_latent_dir}")
                continue
            
            # Load info.json for this scene
            if self.validation:
                data_dir = scene_gt_dir / 'GSTest'
            else:
                data_dir = scene_gt_dir / 'GSTrain'
                
            info_path = data_dir / 'info.json'
            if not info_path.exists():
                print(f"Warning: info.json not found for scene {scene_name}: {info_path}")
                continue
            
            try:
                with open(info_path, 'r') as f:
                    scene_info = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load info.json for scene {scene_name}: {e}")
                continue
            
            # Set up latent directory for this scene
            if self.validation:
                # Try eval_test_step_50000 first, then fallback to eval_step_100000 or eval_step_50000
                potential_dirs = [
                    scene_latent_dir / 'eval_test_step_50000',
                    scene_latent_dir / 'eval_test_step_35000',
                    scene_latent_dir / 'eval_step_100000',
                    scene_latent_dir / 'eval_step_50000'
                ]
            else:
                # Try eval_train_step_50000 first, then fallback to eval_step_100000 or eval_step_50000
                potential_dirs = [
                    scene_latent_dir / 'eval_train_step_50000',
                    scene_latent_dir / 'eval_train_step_35000',
                    scene_latent_dir / 'eval_step_100000',
                    scene_latent_dir / 'eval_step_50000'
                ]
            
            # Find the first existing directory
            scene_latent_subdir = None
            for potential_dir in potential_dirs:
                if potential_dir.exists():
                    scene_latent_subdir = potential_dir
                    break
            
            if scene_latent_subdir is None:
                print(f"Warning: No latent subdirectory found for scene {scene_name}. Tried: {[str(d) for d in potential_dirs]}")
                continue
            
            # Get available environment folders for this scene
            if self.validation:
                env_dir = scene_gt_dir / 'test'
            else:
                env_dir = scene_gt_dir / 'train'
            
            if not env_dir.exists():
                print(f"Warning: Environment directory not found for scene {scene_name}: {env_dir}")
                continue
            
            # Find all env_* folders in the appropriate directory
            scene_env_folders = []
            for item in env_dir.iterdir():
                if item.is_dir() and item.name.startswith('env_'):
                    scene_env_folders.append(item.name)
            if self.denoising_mode:
                scene_env_folders.append('white_env_0')
            # scene_env_folders.append('white_env_0')
            
            if not scene_env_folders:
                print(f"Warning: No env_* folders found for scene {scene_name} in {env_dir}")
                continue
            
            # Store environment folders for this scene (we'll use the first scene's env folders for all scenes)
            if not hasattr(self, 'env_folders'):
                self.env_folders = scene_env_folders
                print(f'Found environment folders: {self.env_folders}')
            
            # Load caption for this scene
            caption = None
            if self.use_gpt_captions:
                # Try to load GPT caption first, fallback to original caption if not found
                caption_path = scene_gt_dir / 'caption_gpt.txt'
                if caption_path.exists():
                    try:
                        with open(caption_path, 'r', encoding='utf-8') as f:
                            caption = f.read().strip()
                    except Exception as e:
                        print(f"Warning: Failed to load GPT caption for scene {scene_name}: {e}")
                        caption = None
                
                # Fallback to original caption if GPT caption not found or failed to load
                if caption is None:
                    caption_path = scene_gt_dir / 'caption.txt'
                    if caption_path.exists():
                        try:
                            with open(caption_path, 'r', encoding='utf-8') as f:
                                caption = f.read().strip()
                        except Exception as e:
                            print(f"Warning: Failed to load original caption for scene {scene_name}: {e}")
            else:
                # Use original caption loading logic
                caption_path = scene_gt_dir / 'caption.txt'
                if caption_path.exists():
                    try:
                        with open(caption_path, 'r', encoding='utf-8') as f:
                            caption = f.read().strip()
                    except Exception as e:
                        print(f"Warning: Failed to load caption for scene {scene_name}: {e}")
            
            # Get list of available images for this scene
            scene_images = []
            for img_info in scene_info['images']:
                rgb_file = img_info['rgb']
                if rgb_file:
                    # Extract sample number from filename (e.g., gt_0.png -> 0)
                    sample_num = rgb_file.split('_')[1].split('.')[0]
                    
                    # Try different naming conventions for conditional latents
                    conditional_files = [
                        f"gt_{sample_num}.npy",  # Default format
                        f"sample_{int(sample_num):04d}.npy"  # Alternative format
                    ]
                    
                    conditional_file = None
                    for cf in conditional_files:
                        conditional_path = scene_latent_subdir / cf
                        if conditional_path.exists():
                            conditional_file = cf
                            break
                    
                    if conditional_file:
                        scene_images.append({
                            'gt_image': rgb_file,
                            'conditional_latent': conditional_file,
                            'sample_num': sample_num,
                            'metadata': img_info,
                            'intrinsic': img_info.get('intrinsic', None),
                            'pose': img_info.get('pose', None),
                            'scene_name': scene_name,
                            'latent_dir': scene_latent_subdir,
                            'gt_dir': scene_gt_dir,
                            'caption': caption
                        })
            
            print(f"Scene {scene_name}: {len(scene_images)} valid samples")
            self.available_images.extend(scene_images)
        
        print(f"Total scenes processed: {len(scene_names)}")
        print(f"Total samples across all scenes: {len(self.available_images)}")

    def __len__(self):
        return len(self.available_images)

    def load_im(self, path, color = (0, 0, 0, 255)):
        '''
        replace background pixel with random color in rendering
        '''
        try:
            img = plt.imread(path)
        except:
            print(path)
            sys.exit()
        if img.shape[-1] == 4:
            img[img[:, :, -1] == 0.] = color
        img = Image.fromarray(np.uint8(img[:, :, :3] * 255.))
        return img


    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Returns:
            dict with keys:
                - image: GT image tensor (3, H, W)
                - input_image: Same as image (for CLIP encoding)
                - conditional_latents: Pre-encoded conditional latents (4, 64, 64)
                - sample_num: Sample number for reference
                - env_info: Environment information from env.json
        """
        try:
            sample_info = self.available_images[idx]
            
            # (3) Select environment folder for GT images
            if self.denoising_mode:
                # In denoising mode, always use white_env_0
                selected_env = 'white_env_0'
                # Check if white_env_0 exists in available environments
                if selected_env not in self.env_folders:
                    # If white_env_0 doesn't exist, fall back to first available environment
                    selected_env = self.env_folders[0] if self.env_folders else 'env_0'
                    print(f"Warning: white_env_0 not found in available environments {self.env_folders}, using {selected_env}")
            elif self.fix_sample:
                # Use first environment for fixed samples (for logging)
                selected_env = self.env_folders[0]
            else:
                # Randomly select an environment
                selected_env = random.choice(self.env_folders)
            
            # Load GT image from train/[selected_env] or test/[selected_env] folder based on validation status
            if sample_info.get('scene_name') is not None:
                # Multi-scene mode: use the scene-specific GT directory
                if self.validation:
                    gt_dir = sample_info['gt_dir'] / 'test' / selected_env
                else:
                    gt_dir = sample_info['gt_dir'] / 'train' / selected_env
            else:
                # Single scene mode: use the original logic
                if self.validation:
                    gt_dir = self.gt_data_dir / 'test' / selected_env
                else:
                    gt_dir = self.gt_data_dir / 'train' / selected_env
                
            gt_image_path = gt_dir / sample_info['gt_image']
            
            if not gt_image_path.exists():
                # If the specific image doesn't exist in this env, try to find any gt_*.png
                gt_files = list(gt_dir.glob('gt_*.png'))
                if gt_files:
                    # Use the first available gt image
                    gt_image_path = gt_files[0]
                else:                
                    gt_dir = self.gt_data_dir / 'train' / 'env_0'
                    gt_image_path = gt_dir / sample_info['gt_image']
                    gt_files = list(gt_dir.glob('gt_*.png'))
                    if gt_files:
                        # Use the first available gt image
                        gt_image_path = gt_files[0]
                    else:
                        raise FileNotFoundError(f"No GT images found in {gt_dir}")
                    # raise FileNotFoundError(f"No GT images found in {gt_dir}")
            
            gt_image = self.load_im(gt_image_path)
            gt_image = self.transforms(gt_image)
            
            # Load conditional latents from the appropriate latent directory
            if sample_info.get('latent_dir') is not None:
                # Multi-scene mode: use the scene-specific latent directory
                conditional_path = sample_info['latent_dir'] / sample_info['conditional_latent']
            else:
                # Single scene mode: use the original logic
                conditional_path = self.latent_dir / sample_info['conditional_latent']
            
            conditional_latents = np.load(conditional_path)  # (4, 64, 64)
            conditional_latents = torch.from_numpy(conditional_latents).float()  # Convert to tensor
            
            # (4) Load environment information from test/[selected_env]/env.json
            if sample_info.get('scene_name') is not None:
                # Multi-scene mode: use the scene-specific GT directory
                test_env_dir = sample_info['gt_dir'] / 'test' / selected_env
            else:
                # Single scene mode: use the original logic
                test_env_dir = self.gt_data_dir / 'test' / selected_env
            
            env_json_path = test_env_dir / 'env.json'
            env_info = {}
            if env_json_path.exists():
                with open(env_json_path, 'r') as f:
                    env_info = json.load(f)
            
            # (5) Load and process environment map
            envmap_ldr = None
            envmap_hdr = None
            envmap_hdr_raw = None
            
            # Skip environment map loading in denoising mode
            if not self.denoising_mode and env_info and 'env_map' in env_info:
                env_map_name = env_info['env_map']
                # Try different possible filenames
                possible_paths = [
                    self.envmaps_dir / f"{env_map_name}.exr",
                    self.envmaps_dir / f"{env_map_name}_8k.exr",
                    self.envmaps_dir / f"{env_map_name}_4k.exr",
                    self.envmaps_dir / f"{env_map_name}_2k.exr",
                    self.envmaps_dir / f"{env_map_name}_1k.exr"
                ]
                
                envmap_path = None
                for path in possible_paths:
                    if path.exists():
                        envmap_path = path
                        break
                
                if envmap_path:
                    try:
                        # Read HDR environment map
                        envmap_hdr_data = read_hdr_exr(str(envmap_path))
                        if envmap_hdr_data is not None:
                            envmap_hdr_tensor = torch.from_numpy(envmap_hdr_data)
                            
                            # Get camera pose for rotation
                            camera_pose = np.array(sample_info['pose'])
                            
                            # Get euler rotation from environment info or use custom rotation
                            euler_rotation = None
                            if self.custom_euler_rotation is not None:
                                # Use custom euler rotation for inference experiments
                                euler_rotation = self.custom_euler_rotation
                            elif env_info and 'rotation_euler' in env_info:
                                # Use original euler rotation from dataset
                                euler_rotation = env_info['rotation_euler']
                            
                            # Rotate and preprocess environment map
                            if self.split_env_map:
                                envmap_ldr_back, envmap_ldr_front, envmap_hdr_back, envmap_hdr_front = rotate_and_preprocess_envir_map(
                                    envmap_hdr_tensor, camera_pose, euler_rotation=euler_rotation, target_resolution=self.resolution, split_env_map=True
                                )
                            else:
                                envmap_ldr, envmap_hdr, envmap_hdr_raw = rotate_and_preprocess_envir_map(
                                    envmap_hdr_tensor, camera_pose, euler_rotation=euler_rotation, target_resolution=self.resolution, split_env_map=False
                                )
                            
                            # Convert PIL images to tensors
                            if self.split_env_map:
                                # Handle split environment maps
                                if envmap_ldr_back is not None:
                                    envmap_ldr_back = self.transforms(envmap_ldr_back)
                                if envmap_ldr_front is not None:
                                    envmap_ldr_front = self.transforms(envmap_ldr_front)
                                if envmap_hdr_back is not None:
                                    envmap_hdr_back = self.transforms(envmap_hdr_back)
                                if envmap_hdr_front is not None:
                                    envmap_hdr_front = self.transforms(envmap_hdr_front)
                            else:
                                # Handle single environment maps
                                if envmap_ldr is not None:
                                    envmap_ldr = self.transforms(envmap_ldr)
                                if envmap_hdr is not None:
                                    envmap_hdr = self.transforms(envmap_hdr)
                                
                    except Exception as e:
                        print(f"Error loading environment map {envmap_path}: {e}")
                    
            # Provide default values for None environment maps to avoid DataLoader issues
            if self.split_env_map:
                # Handle split environment maps
                if envmap_ldr_back is None:
                    envmap_ldr_back = torch.zeros(3, self.resolution, self.resolution)
                if envmap_ldr_front is None:
                    envmap_ldr_front = torch.zeros(3, self.resolution, self.resolution)
                if envmap_hdr_back is None:
                    envmap_hdr_back = torch.zeros(3, self.resolution, self.resolution)
                if envmap_hdr_front is None:
                    envmap_hdr_front = torch.zeros(3, self.resolution, self.resolution)
                
                return {
                    'image': gt_image,  # GT image for training (3, H, W)
                    'input_image': gt_image,  # Input image for CLIP encoding (3, H, W)
                    'conditional_latents': conditional_latents,  # Pre-encoded conditional latents (4, 64, 64)
                    'sample_num': sample_info['sample_num'],
                    'env_info': env_info,  # Environment information
                    'selected_env': selected_env,  # Selected environment folder name
                    'intrinsic': sample_info['intrinsic'],  # Camera intrinsic matrix
                    'pose': sample_info['pose'],  # Camera pose (extrinsic matrix)
                    'scene_name': sample_info.get('scene_name'),  # Scene name (None for single scene mode)
                    'caption': sample_info.get('caption'),  # Image caption (None for single scene mode)
                    'envmap_ldr_back': envmap_ldr_back,  # LDR environment map back half (3, H, W)
                    'envmap_ldr_front': envmap_ldr_front,  # LDR environment map front half (3, H, W)
                    'envmap_hdr_back': envmap_hdr_back,  # HDR environment map back half (3, H, W)
                    'envmap_hdr_front': envmap_hdr_front  # HDR environment map front half (3, H, W)
                }
            else:
                # Handle single environment maps
                if envmap_ldr is None:
                    envmap_ldr = torch.zeros(3, self.resolution, self.resolution)
                if envmap_hdr is None:
                    envmap_hdr = torch.zeros(3, self.resolution, self.resolution)
                if envmap_hdr_raw is None:
                    envmap_hdr_raw = np.zeros((self.resolution, self.resolution, 3))
                
                return {
                    'image': gt_image,  # GT image for training (3, H, W)
                    'input_image': gt_image,  # Input image for CLIP encoding (3, H, W)
                    'conditional_latents': conditional_latents,  # Pre-encoded conditional latents (4, 64, 64)
                    'sample_num': sample_info['sample_num'],
                    'env_info': env_info,  # Environment information
                    'selected_env': selected_env,  # Selected environment folder name
                    'intrinsic': sample_info['intrinsic'],  # Camera intrinsic matrix
                    'pose': sample_info['pose'],  # Camera pose (extrinsic matrix)
                    'scene_name': sample_info.get('scene_name'),  # Scene name (None for single scene mode)
                    'caption': sample_info.get('caption'),  # Image caption (None for single scene mode)
                    'envmap_ldr': envmap_ldr,  # LDR environment map (3, H, W)
                    'envmap_hdr': envmap_hdr,  # HDR environment map (3, H, W)
                    'envmap_hdr_raw': envmap_hdr_raw  # Raw HDR environment map (H, W, 3)
                }

        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            return self.__getitem__(random.randint(0, len(self.available_images) - 1))

    def get_sample_by_number(self, sample_num):
        """
        Get a specific sample by its number.
        Useful for validation/logging with fixed samples.
        """
        for i, sample_info in enumerate(self.available_images):
            if sample_info['sample_num'] == str(sample_num):
                return self.__getitem__(i)
        return None

class GSRelightDataLoader:
    """
    DataLoader wrapper for GSRelight dataset.
    """
    
    def __init__(self, 
                 gt_data_dir,
                 latent_folder_name='flux_latents_64',
                 latent_base_dir=None,
                 envmaps_dir='/projects/vig/Datasets/objaverse/envmaps_256/hdirs/',
                 batch_size=4,
                 num_workers=4,
                 resolution=512,
                 scene_list_file=None,
                 gt_data_base_dir='/projects/vig/Datasets/objaverse/hf-objaverse-v1/rendered_dense',
                 latent_data_base_dir='../torchSplattingMod/result',
                 use_gpt_captions=False):
        self.gt_data_dir = gt_data_dir
        self.latent_folder_name = latent_folder_name
        self.latent_base_dir = latent_base_dir
        self.envmaps_dir = envmaps_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resolution = resolution
        self.scene_list_file = scene_list_file
        self.gt_data_base_dir = gt_data_base_dir
        self.latent_data_base_dir = latent_data_base_dir
        self.use_gpt_captions = use_gpt_captions

    def train_dataloader(self):
        """Get training dataloader"""
        dataset = GSRelightDataset(
            gt_data_dir=self.gt_data_dir,
            latent_folder_name=self.latent_folder_name,
            latent_base_dir=self.latent_base_dir,
            envmaps_dir=self.envmaps_dir,
            validation=False,
            resolution=self.resolution,
            scene_list_file=self.scene_list_file,
            gt_data_base_dir=self.gt_data_base_dir,
            latent_data_base_dir=self.latent_data_base_dir,
            use_gpt_captions=self.use_gpt_captions
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        """Get validation dataloader"""
        dataset = GSRelightDataset(
            gt_data_dir=self.gt_data_dir,
            latent_folder_name=self.latent_folder_name,
            latent_base_dir=self.latent_base_dir,
            envmaps_dir=self.envmaps_dir,
            validation=True,
            resolution=self.resolution,
            scene_list_file=self.scene_list_file,
            gt_data_base_dir=self.gt_data_base_dir,
            latent_data_base_dir=self.latent_data_base_dir,
            use_gpt_captions=self.use_gpt_captions
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def train_log_dataloader(self):
        """Get training log dataloader with fixed samples"""
        dataset = GSRelightDataset(
            gt_data_dir=self.gt_data_dir,
            latent_folder_name=self.latent_folder_name,
            latent_base_dir=self.latent_base_dir,
            envmaps_dir=self.envmaps_dir,
            validation=False,
            fix_sample=True,
            resolution=self.resolution,
            scene_list_file=self.scene_list_file,
            gt_data_base_dir=self.gt_data_base_dir,
            latent_data_base_dir=self.latent_data_base_dir,
            use_gpt_captions=self.use_gpt_captions
        )
        return DataLoader(
            dataset,
            batch_size=1,  # Use batch size 1 for logging
            shuffle=True,
            num_workers=1,
            pin_memory=True
        )

def test_dataset():
    """Test the dataset"""
    print("Testing GSRelight dataset...")
    
    # Test with real dataset using default latent location
    print("Testing with real dataset (default latent location)...")
    
    try:
        dataset = GSRelightDataset(
            gt_data_dir='/projects/vig/Datasets/objaverse/hf-objaverse-v1/rendered_dense/2893bff5223841a095b3b5453d46d3ad',
            latent_folder_name='flux_latents_64',
            resolution=512
        )
        
        print(f"✅ Dataset created successfully!")
        print(f"Dataset size: {len(dataset)}")
        
        # Get a sample
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"GT image shape: {sample['image'].shape}")
        print(f"Conditional latents shape: {sample['conditional_latents'].shape}")
        print(f"Sample number: {sample['sample_num']}")
        print(f"Selected environment: {sample['selected_env']}")
        print(f"Environment info: {sample['env_info']}")
        print(f"Camera intrinsic: {sample['intrinsic']}")
        print(f"Camera pose: {sample['pose']}")
        print(f"Environment map LDR shape: {sample['envmap_ldr'].shape if sample['envmap_ldr'] is not None else 'None'}")
        print(f"Environment map HDR shape: {sample['envmap_hdr'].shape if sample['envmap_hdr'] is not None else 'None'}")
        print(f"Environment map HDR raw shape: {sample['envmap_hdr_raw'].shape if sample['envmap_hdr_raw'] is not None else 'None'}")
        
        # Save visualization images
        import os
        os.makedirs('test_visualization', exist_ok=True)
        
        # Save GT image
        gt_image = sample['image']
        # Convert from [-1, 1] to [0, 1] and then to [0, 255]
        gt_image_vis = ((gt_image + 1) / 2).clamp(0, 1)
        gt_image_vis = (gt_image_vis * 255).byte().permute(1, 2, 0).cpu().numpy()
        gt_image_pil = Image.fromarray(gt_image_vis)
        gt_image_pil.save('test_visualization/gt_image.png')
        print("✅ Saved GT image to test_visualization/gt_image.png")
        
        # Save LDR environment map
        if sample['envmap_ldr'] is not None:
            envmap_ldr = sample['envmap_ldr']
            # Convert from [-1, 1] to [0, 1] and then to [0, 255]
            envmap_ldr_vis = ((envmap_ldr + 1) / 2).clamp(0, 1)
            envmap_ldr_vis = (envmap_ldr_vis * 255).byte().permute(1, 2, 0).cpu().numpy()
            envmap_ldr_pil = Image.fromarray(envmap_ldr_vis)
            envmap_ldr_pil.save('test_visualization/envmap_ldr.png')
            print("✅ Saved LDR environment map to test_visualization/envmap_ldr.png")
        
        # Save HDR environment map
        if sample['envmap_hdr'] is not None:
            envmap_hdr = sample['envmap_hdr']
            # Convert from [-1, 1] to [0, 1] and then to [0, 255]
            envmap_hdr_vis = ((envmap_hdr + 1) / 2).clamp(0, 1)
            envmap_hdr_vis = (envmap_hdr_vis * 255).byte().permute(1, 2, 0).cpu().numpy()
            envmap_hdr_pil = Image.fromarray(envmap_hdr_vis)
            envmap_hdr_pil.save('test_visualization/envmap_hdr.png')
            print("✅ Saved HDR environment map to test_visualization/envmap_hdr.png")
        
        # Save HDR raw data (normalize for visualization)
        if sample['envmap_hdr_raw'] is not None:
            envmap_hdr_raw = sample['envmap_hdr_raw']
            # Normalize HDR data for visualization
            envmap_hdr_raw_vis = np.clip(envmap_hdr_raw / np.max(envmap_hdr_raw), 0, 1)
            envmap_hdr_raw_vis = (envmap_hdr_raw_vis * 255).astype(np.uint8)
            envmap_hdr_raw_pil = Image.fromarray(envmap_hdr_raw_vis)
            envmap_hdr_raw_pil.save('test_visualization/envmap_hdr_raw.png')
            print("✅ Saved HDR raw environment map to test_visualization/envmap_hdr_raw.png")
        
        # Test batch loading
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(dataloader))
        print(f"Batch GT image shape: {batch['image'].shape}")
        print(f"Batch conditional latents shape: {batch['conditional_latents'].shape}")
        
        print("✅ Dataset test passed!")
        
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test with separate conditional latents directory
    print("\nTesting with latent_base_dir...")
    
    try:
        # Example: GT data from one location, latents from another
        dataset = GSRelightDataset(
            gt_data_dir='/projects/vig/Datasets/objaverse/hf-objaverse-v1/rendered_dense/5bda8477fb974ed19e2e1bd72597d03b',
            latent_base_dir='/projects/vig/yiwenc/ResearchProjects/lightingDiffusion/3dgs/torchSplattingMod/result/5bda8477fb974ed19e2e1bd72597d03b_flux_64',
            resolution=512
        )
        
        print(f"✅ Dataset with separate latents created successfully!")
        print(f"Dataset size: {len(dataset)}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"GT image shape: {sample['image'].shape}")
            print(f"Conditional latents shape: {sample['conditional_latents'].shape}")
            print(f"Selected environment: {sample['selected_env']}")
            print(f"Camera intrinsic: {sample['intrinsic']}")
            print(f"Camera pose: {sample['pose']}")
            print(f"Environment map LDR shape: {sample['envmap_ldr'].shape if sample['envmap_ldr'] is not None else 'None'}")
            print(f"Environment map HDR shape: {sample['envmap_hdr'].shape if sample['envmap_hdr'] is not None else 'None'}")
            print(f"Environment map HDR raw shape: {sample['envmap_hdr_raw'].shape if sample['envmap_hdr_raw'] is not None else 'None'}")
            
            # Save visualization images for separate latents test
            import os
            os.makedirs('test_visualization_separate', exist_ok=True)
            
            # Save GT image
            gt_image = sample['image']
            gt_image_vis = ((gt_image + 1) / 2).clamp(0, 1)
            gt_image_vis = (gt_image_vis * 255).byte().permute(1, 2, 0).cpu().numpy()
            gt_image_pil = Image.fromarray(gt_image_vis)
            gt_image_pil.save('test_visualization_separate/gt_image.png')
            print("✅ Saved GT image to test_visualization_separate/gt_image.png")
            
            # Save LDR environment map
            if sample['envmap_ldr'] is not None:
                envmap_ldr = sample['envmap_ldr']
                envmap_ldr_vis = ((envmap_ldr + 1) / 2).clamp(0, 1)
                envmap_ldr_vis = (envmap_ldr_vis * 255).byte().permute(1, 2, 0).cpu().numpy()
                envmap_ldr_pil = Image.fromarray(envmap_ldr_vis)
                envmap_ldr_pil.save('test_visualization_separate/envmap_ldr.png')
                print("✅ Saved LDR environment map to test_visualization_separate/envmap_ldr.png")
            
            # Save HDR environment map
            if sample['envmap_hdr'] is not None:
                envmap_hdr = sample['envmap_hdr']
                envmap_hdr_vis = ((envmap_hdr + 1) / 2).clamp(0, 1)
                envmap_hdr_vis = (envmap_hdr_vis * 255).byte().permute(1, 2, 0).cpu().numpy()
                envmap_hdr_pil = Image.fromarray(envmap_hdr_vis)
                envmap_hdr_pil.save('test_visualization_separate/envmap_hdr.png')
                print("✅ Saved HDR environment map to test_visualization_separate/envmap_hdr.png")
            
            # Save HDR raw data
            if sample['envmap_hdr_raw'] is not None:
                envmap_hdr_raw = sample['envmap_hdr_raw']
                envmap_hdr_raw_vis = np.clip(envmap_hdr_raw / np.max(envmap_hdr_raw), 0, 1)
                envmap_hdr_raw_vis = (envmap_hdr_raw_vis * 255).astype(np.uint8)
                envmap_hdr_raw_pil = Image.fromarray(envmap_hdr_raw_vis)
                envmap_hdr_raw_pil.save('test_visualization_separate/envmap_hdr_raw.png')
                print("✅ Saved HDR raw environment map to test_visualization_separate/envmap_hdr_raw.png")
        
        print("✅ Separate latents test passed!")
        
    except Exception as e:
        print(f"❌ Separate latents test failed: {e}")
        print("This is expected if the example directories don't exist")
    
    print("\nDataset structure requirements:")
    print("1. [gt_data_dir]/GSTrain/info.json - Contains image metadata for training (including camera poses)")
    print("2. [gt_data_dir]/GSTest/info.json - Contains image metadata for validation (including camera poses)")
    print("3. [gt_data_dir]/train/env_*/ - Contains GT images for training in different environments")
    print("4. [gt_data_dir]/test/env_*/ - Contains GT images for validation in different environments")
    print("5. [gt_data_dir]/test/env_*/env.json - Contains environment information")
    print("6. [latent_base_dir]/eval_train_step_50000/ or eval_test_step_50000/ - Contains .npy latent files (if specified)")
    print("   Fallback options: eval_step_100000/ or eval_step_50000/ if primary directories don't exist")
    print("   OR [gt_data_dir]/GSTrain/[latent_folder_name]/ - Default latent location")
    print("7. [envmaps_dir]/ - Contains .exr environment map files")
    
    print("\nExample usage:")
    print("# Using default latent location:")
    print("dataset = GSRelightDataset(")
    print("    gt_data_dir='/path/to/your/data',")
    print("    latent_folder_name='flux_latents_64',")
    print("    envmaps_dir='/path/to/envmaps',")
    print("    resolution=512")
    print(")")
    print()
    print("# Using latent_base_dir with eval_train_step_50000/eval_test_step_50000:")
    print("dataset = GSRelightDataset(")
    print("    gt_data_dir='/path/to/gt/data',")
    print("    latent_base_dir='../torchSplattingMod/result/01c9013483b6427fbc2f478e5e328810_flux_64',")
    print("    envmaps_dir='/path/to/envmaps',")
    print("    validation=False,  # Uses eval_train_step_50000")
    print("    resolution=512")
    print(")")
    print()
    print("val_dataset = GSRelightDataset(")
    print("    gt_data_dir='/path/to/gt/data',")
    print("    latent_base_dir='../torchSplattingMod/result/01c9013483b6427fbc2f478e5e328810_flux_64',")
    print("    envmaps_dir='/path/to/envmaps',")
    print("    validation=True,   # Uses eval_test_step_50000")
    print("    resolution=512")
    print(")")
    
    print("\nDataset test completed! ✅")

if __name__ == "__main__":
    test_dataset()
