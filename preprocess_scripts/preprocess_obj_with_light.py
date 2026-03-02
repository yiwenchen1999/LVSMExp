#!/usr/bin/env python3
"""
Preprocess objects-with-lighting dataset to re10k/objaverse_processed format.

Input structure (per scene):
  <scene>/test/inputs/
    camera_0000.txt  # K (3x3) | R (3x3) | t (3x1) | [W H C]
    image_0000.png
    mask_0000.png
    ...

Output structure:
  <output>/<split>/images/<scene>/<frame:05d>.png
  <output>/<split>/metadata/<scene>.json
  <output>/<split>/full_list.txt

Process:
  1. Apply mask (white background)
  2. Center-crop from source FOV to target FOV
  3. Resize to 512x512
  4. Generate metadata: w2c, fxfycxcy (target FOV=30)
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def parse_camera_file(camera_path: Path):
    """
    Parse camera_*.txt file.
    
    Format:
      Line 0-2: K (3x3 intrinsic matrix)
      Line 3-5: R (3x3 rotation matrix, cam-to-world)
      Line 6:   t (3x1 translation vector, cam position in world)
      Line 7:   [width, height, channels]
    
    Returns:
      K (3x3), R (3x3), t (3,), width, height
    """
    with open(camera_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    data = []
    for line in lines:
        parts = line.split()
        data.extend([float(x) for x in parts])
    
    K = np.array(data[0:9]).reshape(3, 3)
    R = np.array(data[9:18]).reshape(3, 3)
    t = np.array(data[18:21])
    width = int(data[21])
    height = int(data[22])
    
    return K, R, t, width, height


def intrinsics_to_fov(f: float, size: int) -> float:
    """Compute FOV (degrees) from focal length and image dimension."""
    return math.degrees(2.0 * math.atan(size / (2.0 * f)))


def fov_to_focal(fov_degrees: float, size: int) -> float:
    """Compute focal length from FOV (degrees) and image dimension."""
    fov_rad = math.radians(fov_degrees)
    return (size / 2.0) / math.tan(fov_rad / 2.0)


def fov_to_fxfycxcy(fov_degrees: float, image_width: int, image_height: int):
    """Convert horizontal FOV to [fx, fy, cx, cy]."""
    fov_rad = math.radians(fov_degrees)
    fx = fy = (image_width / 2.0) / math.tan(fov_rad / 2.0)
    cx = image_width / 2.0
    cy = image_height / 2.0
    return [fx, fy, cx, cy]


def apply_mask_white_background(image_path: Path, mask_path: Path) -> np.ndarray:
    """Apply mask to image, setting background to white (255)."""
    with Image.open(image_path) as img:
        rgb = np.array(img.convert("RGB"), dtype=np.uint8)
    with Image.open(mask_path) as m:
        mask = np.array(m.convert("L"), dtype=np.uint8) > 127
    
    if mask.shape[:2] != rgb.shape[:2]:
        raise ValueError(f"Mask shape {mask.shape[:2]} != image shape {rgb.shape[:2]}")
    
    out = rgb.copy()
    out[~mask] = 255
    return out


def center_crop_to_square_simple(
    rgb: np.ndarray, K: np.ndarray, target_size: int
) -> tuple[np.ndarray, list]:
    """
    Simple center crop to square (min(w,h)), then resize to target_size.
    Preserves original focal length ratios.
    
    Args:
        rgb: Input image (H, W, 3)
        K: Intrinsic matrix (3x3) with fx=K[0,0], fy=K[1,1]
        target_size: Output square size
    
    Returns:
        cropped_rgb: Cropped and resized image (target_size, target_size, 3)
        new_fxfycxcy: [fx, fy, cx, cy] for the output image
    """
    h, w = rgb.shape[:2]
    fx_src = K[0, 0]
    fy_src = K[1, 1]
    
    # Take the smaller dimension as crop size
    crop_size = min(w, h)
    
    # Center crop to square
    x0 = (w - crop_size) // 2
    y0 = (h - crop_size) // 2
    
    cropped = rgb[y0 : y0 + crop_size, x0 : x0 + crop_size]
    
    # Compute intrinsics after cropping
    # fx' = fx * (crop_size / w)
    # fy' = fy * (crop_size / h)
    fx_crop = fx_src * (crop_size / float(w))
    fy_crop = fy_src * (crop_size / float(h))
    
    # Resize to target size
    if crop_size != target_size:
        img = Image.fromarray(cropped)
        img = img.resize((target_size, target_size), Image.Resampling.BICUBIC)
        cropped = np.array(img, dtype=np.uint8)
        
        # Scale intrinsics proportionally
        scale = target_size / float(crop_size)
        fx_final = fx_crop * scale
        fy_final = fy_crop * scale
    else:
        fx_final = fx_crop
        fy_final = fy_crop
    
    cx_final = target_size / 2.0
    cy_final = target_size / 2.0
    
    return cropped, [fx_final, fy_final, cx_final, cy_final]


def center_crop_to_target_fov_square(
    rgb: np.ndarray, K: np.ndarray, target_fov_deg: float, target_size: int
) -> tuple[np.ndarray, list]:
    """
    Center-crop image to square with target FOV in both x and y directions.
    
    For square output with FOV=30° in both directions, we need fx=fy and correct crop size.
    
    Strategy:
      1. Since fx_src = fy_src (equal focal lengths), we can determine the crop size
         that gives us exactly target_fov_deg
      2. crop_size = 2 * fx_src * tan(target_fov_deg/2)
      3. Center crop to that square size
      4. Resize to target_size
    
    Args:
        rgb: Input image (H, W, 3)
        K: Intrinsic matrix (3x3) with fx=K[0,0], fy=K[1,1]
        target_fov_deg: Target FOV (same for both x and y)
        target_size: Output square size
    
    Returns:
        cropped_rgb: Cropped and resized image (target_size, target_size, 3)
        new_fxfycxcy: [fx, fy, cx, cy] for the output image (fx=fy for square)
    """
    h, w = rgb.shape[:2]
    fx_src = K[0, 0]
    fy_src = K[1, 1]
    
    # For square image with target FOV, compute required crop dimension
    # FOV = 2 * atan(size / (2 * f))
    # size = 2 * f * tan(FOV / 2)
    crop_size_for_fov = 2.0 * fx_src * math.tan(math.radians(target_fov_deg / 2.0))
    crop_size = int(round(crop_size_for_fov))
    
    # Ensure crop size fits within image bounds
    crop_size = min(crop_size, w, h)
    crop_size = max(1, crop_size)
    
    # Center crop to square
    x0 = (w - crop_size) // 2
    y0 = (h - crop_size) // 2
    
    cropped = rgb[y0 : y0 + crop_size, x0 : x0 + crop_size]
    
    # Compute intrinsics after crop (before resize)
    # For square crop from image with fx=fy, the new focal length is still fx_src
    # because we're just cropping, not changing the camera
    fx_crop = fx_src
    fy_crop = fy_src
    
    # Resize to target size
    if crop_size != target_size:
        img = Image.fromarray(cropped)
        img = img.resize((target_size, target_size), Image.Resampling.BICUBIC)
        cropped = np.array(img, dtype=np.uint8)
        
        # Scale intrinsics proportionally
        scale = target_size / float(crop_size)
        fx_final = fx_crop * scale
        fy_final = fy_crop * scale
    else:
        fx_final = fx_crop
        fy_final = fy_crop
    
    cx_final = target_size / 2.0
    cy_final = target_size / 2.0
    
    return cropped, [fx_final, fy_final, cx_final, cy_final]


def compute_w2c(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Compute world-to-camera matrix from camera-to-world R and t.
    
    c2w = [R | t]
          [0 | 1]
    w2c = inv(c2w) = [R^T | -R^T @ t]
                     [0   | 1      ]
    """
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = R.T
    w2c[:3, 3] = -R.T @ t
    return w2c


def process_scene(
    scene_dir: Path,
    output_root: Path,
    split: str,
    crop_mode: str,
    target_fov: float,
    target_size: int,
):
    """Process a single scene from objects-with-lighting dataset."""
    inputs_dir = scene_dir / split / "inputs"
    if not inputs_dir.exists():
        print(f"[Skip] {scene_dir.name}/{split}: inputs directory not found")
        return None
    
    camera_files = sorted(inputs_dir.glob("camera_*.txt"))
    if not camera_files:
        print(f"[Skip] {scene_dir.name}/{split}: no camera files found")
        return None
    
    scene_name = scene_dir.name
    out_images_dir = output_root / split / "images" / scene_name
    out_meta_dir = output_root / split / "metadata"
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_meta_dir.mkdir(parents=True, exist_ok=True)
    
    frames_out = []
    
    for cam_file in camera_files:
        frame_id = cam_file.stem.replace("camera_", "")
        image_file = inputs_dir / f"image_{frame_id}.png"
        mask_file = inputs_dir / f"mask_{frame_id}.png"
        
        if not image_file.exists() or not mask_file.exists():
            print(f"[Warn] {scene_name}/{split}: missing image or mask for {frame_id}, skipping")
            continue
        
        K, R, t, src_w, src_h = parse_camera_file(cam_file)
        
        rgb = apply_mask_white_background(image_file, mask_file)
        
        if crop_mode == "fov":
            rgb_out, fxfycxcy = center_crop_to_target_fov_square(rgb, K, target_fov, target_size)
        elif crop_mode == "square":
            rgb_out, fxfycxcy = center_crop_to_square_simple(rgb, K, target_size)
        else:
            raise ValueError(f"Unknown crop_mode: {crop_mode}")
        
        out_name = f"{int(frame_id):05d}.png"
        out_path = out_images_dir / out_name
        Image.fromarray(rgb_out).save(out_path)
        
        w2c = compute_w2c(R, t)
        
        frames_out.append(
            {
                "image_path": str(out_path.resolve()),
                "fxfycxcy": fxfycxcy,
                "w2c": w2c.tolist(),
            }
        )
    
    if not frames_out:
        print(f"[Skip] {scene_name}/{split}: no valid frames")
        return None
    
    scene_meta = {"scene_name": scene_name, "frames": frames_out}
    meta_path = out_meta_dir / f"{scene_name}.json"
    with open(meta_path, "w") as f:
        json.dump(scene_meta, f, indent=2)
    
    mode_desc = f"crop_mode={crop_mode}"
    if crop_mode == "fov":
        mode_desc += f", target_fov={target_fov}"
    print(
        f"[Done] {scene_name} ({split}): {len(frames_out)} frames, "
        f"{mode_desc}, target_size={target_size}"
    )
    return scene_name


def collect_scene_dirs(input_root: Path, split: str):
    """Collect scene directories containing split/inputs folder."""
    scene_dirs = []
    for d in sorted(input_root.iterdir()):
        if not d.is_dir():
            continue
        if (d / split / "inputs").exists():
            scene_dirs.append(d)
    return scene_dirs


def write_full_list(output_root: Path, split: str):
    """Write full_list.txt with absolute paths to all metadata JSON files."""
    meta_dir = output_root / split / "metadata"
    if not meta_dir.exists():
        return
    json_paths = sorted(meta_dir.glob("*.json"))
    full_list_path = output_root / split / "full_list.txt"
    with open(full_list_path, "w") as f:
        for p in json_paths:
            f.write(f"{p.resolve()}\n")
    print(f"[Done] Wrote {full_list_path} ({len(json_paths)} scenes)")


def parse_args():
    parser = argparse.ArgumentParser("Preprocess objects-with-lighting to objaverse_processed format")
    parser.add_argument(
        "--input-root",
        type=str,
        default="data_samples/obj_with_light",
        help="Root containing object folders (e.g., apple/, antman/)",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data_samples/obj_with_light_processed",
        help="Output root directory",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "train"],
        help="Which split to process (default: test)",
    )
    parser.add_argument(
        "--crop-mode",
        type=str,
        default="fov",
        choices=["fov", "square"],
        help="Crop mode: 'fov' (crop to target FOV, fx=fy), 'square' (simple center crop to square, preserve fx/fy ratio)",
    )
    parser.add_argument("--target-size", type=int, default=512, help="Output square image size")
    parser.add_argument(
        "--target-fov",
        type=float,
        default=30.0,
        help="Target horizontal FOV in degrees (only used when crop-mode=fov)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    
    if not input_root.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_root}")
    
    scene_dirs = collect_scene_dirs(input_root, args.split)
    if not scene_dirs:
        print(f"[Warn] No scenes found for split='{args.split}' under {input_root}")
        return
    
    print(f"[Info] Processing split='{args.split}' with {len(scene_dirs)} scene(s)")
    for scene_dir in scene_dirs:
        process_scene(
            scene_dir=scene_dir,
            output_root=output_root,
            split=args.split,
            crop_mode=args.crop_mode,
            target_fov=args.target_fov,
            target_size=args.target_size,
        )
    
    write_full_list(output_root, args.split)


if __name__ == "__main__":
    main()
