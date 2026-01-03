#!/usr/bin/env python3
"""
Preprocessing script to convert Objaverse data format to re10k format.

The script processes Objaverse data where:
- Each object has train/ and test/ folders
- test/ contains env_0/, env_1/, ..., env_4/, white_env_0/ folders (different scenes with same trajectory)
- Each env folder contains gt_{idx}.png images
- cameras.json contains camera parameters in Blender convention

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


def process_objaverse_scene(objaverse_root, object_id, output_root, split='test'):
    """
    Process a single Objaverse object and convert all env scenes to re10k format.
    
    Args:
        objaverse_root: Root directory of objaverse data (e.g., data_samples/sample_objaverse)
        object_id: Object ID folder name
        output_root: Root directory for output (e.g., data_samples/objaverse_processed)
        split: 'train' or 'test'
    """
    object_path = os.path.join(objaverse_root, object_id)
    split_path = os.path.join(object_path, split)
    
    if not os.path.exists(split_path):
        print(f"Warning: {split_path} does not exist, skipping")
        return
    
    # Load cameras.json
    cameras_json_path = os.path.join(split_path, 'cameras.json')
    if not os.path.exists(cameras_json_path):
        print(f"Warning: {cameras_json_path} does not exist, skipping {object_id}")
        return
    
    with open(cameras_json_path, 'r') as f:
        cameras_data = json.load(f)
    
    # Find all env folders (env_0, env_1, ..., env_4, white_env_0, etc.)
    env_folders = [d for d in os.listdir(split_path) 
                   if os.path.isdir(os.path.join(split_path, d)) 
                   and (d.startswith('env_') or d.startswith('white_env_'))]
    
    if not env_folders:
        print(f"Warning: No env folders found in {split_path}, skipping {object_id}")
        return
    
    # Process each env folder as a separate scene
    for env_folder in sorted(env_folders):
        env_path = os.path.join(split_path, env_folder)
        
        # Find all gt_*.png images and filter to only numeric indices
        all_image_files = [f for f in os.listdir(env_path) 
                          if f.startswith('gt_') and f.endswith('.png')]
        
        # Filter to only include files where the index is a number
        # Extract index from filename (gt_0.png -> 0) and check if it's numeric
        image_files_with_idx = []
        for image_file in all_image_files:
            # Extract the index part (between 'gt_' and '.png')
            idx_str = image_file.replace('gt_', '').replace('.png', '')
            try:
                frame_idx = int(idx_str)
                image_files_with_idx.append((frame_idx, image_file))
            except ValueError:
                # Skip files where index is not a number (e.g., gt_{idx}.png)
                continue
        
        if not image_files_with_idx:
            print(f"Warning: No valid gt_*.png images with numeric indices found in {env_path}, skipping")
            continue
        
        # Sort by frame index
        image_files_with_idx.sort(key=lambda x: x[0])
        
        # Get image dimensions from first image
        first_image_path = os.path.join(env_path, image_files_with_idx[0][1])
        try:
            img = Image.open(first_image_path)
            image_width, image_height = img.size
        except Exception as e:
            print(f"Error reading image {first_image_path}: {e}, skipping")
            continue
        
        # Create scene name: object_id_env_folder
        scene_name = f"{object_id}_{env_folder}"
        
        # Create output directories
        output_metadata_dir = os.path.join(output_root, split, 'metadata')
        output_images_dir = os.path.join(output_root, split, 'images', scene_name)
        os.makedirs(output_metadata_dir, exist_ok=True)
        os.makedirs(output_images_dir, exist_ok=True)
        
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
            
            # Copy image to output directory with zero-padded name
            input_image_path = os.path.join(env_path, image_file)
            output_image_name = f"{frame_idx:05d}.png"
            output_image_path = os.path.join(output_images_dir, output_image_name)
            
            # Copy image
            shutil.copy2(input_image_path, output_image_path)
            
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
        
        # Save scene JSON
        output_json_path = os.path.join(output_metadata_dir, f"{scene_name}.json")
        with open(output_json_path, 'w') as f:
            json.dump(scene_data, f, indent=2)
        
        print(f"Processed {scene_name}: {len(frames)} frames")


def create_full_list(output_root, split='test'):
    """
    Create full_list.txt file listing all scene JSON files.
    
    Args:
        output_root: Root directory for output
        split: 'train' or 'test'
    """
    metadata_dir = os.path.join(output_root, split, 'metadata')
    if not os.path.exists(metadata_dir):
        print(f"Warning: {metadata_dir} does not exist")
        return
    
    json_files = sorted([f for f in os.listdir(metadata_dir) if f.endswith('.json')])
    
    full_list_path = os.path.join(output_root, split, 'full_list.txt')
    with open(full_list_path, 'w') as f:
        for json_file in json_files:
            json_path = os.path.join(metadata_dir, json_file)
            # Write absolute path
            f.write(f"{os.path.abspath(json_path)}\n")
    
    print(f"Created {full_list_path} with {len(json_files)} scenes")


def main():
    parser = argparse.ArgumentParser(description='Preprocess Objaverse data to re10k format')
    parser.add_argument('--input', '-i', required=True,
                       help='Input directory containing objaverse data (e.g., data_samples/sample_objaverse)')
    parser.add_argument('--output', '-o', required=True,
                       help='Output directory for processed data (e.g., data_samples/objaverse_processed)')
    parser.add_argument('--split', '-s', default='test', choices=['train', 'test'],
                       help='Split to process (default: test)')
    parser.add_argument('--object-id', type=str, default=None,
                       help='Process specific object ID only (default: process all)')
    
    args = parser.parse_args()
    
    objaverse_root = args.input
    output_root = args.output
    
    if not os.path.exists(objaverse_root):
        print(f"Error: Input directory {objaverse_root} does not exist")
        return
    
    # Find all object folders
    if args.object_id:
        object_ids = [args.object_id]
    else:
        object_ids = [d for d in os.listdir(objaverse_root) 
                     if os.path.isdir(os.path.join(objaverse_root, d))]
    
    print(f"Processing {len(object_ids)} objects...")
    
    for object_id in sorted(object_ids):
        print(f"\nProcessing object: {object_id}")
        try:
            process_objaverse_scene(objaverse_root, object_id, output_root, split=args.split)
        except Exception as e:
            print(f"Error processing {object_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # Create full_list.txt
    print(f"\nCreating full_list.txt for {args.split} split...")
    create_full_list(output_root, split=args.split)
    
    print("\nPreprocessing complete!")


if __name__ == '__main__':
    main()

