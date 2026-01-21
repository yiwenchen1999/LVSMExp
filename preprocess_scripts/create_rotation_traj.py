#!/usr/bin/env python3
"""
Script to create rotation trajectory index JSON files for inference.

The script samples k consecutive frames from a scene, then selects n frames as input
and uses the remaining k-n frames as target.

Output format matches data/evaluation_index_re10k.json:
{
    "scene_name": {
        "context": [input_frame_indices],
        "target": [target_frame_indices]
    }
}
"""

import os
import json
import argparse
import random
import numpy as np
from pathlib import Path


def sample_consecutive_frames(num_frames, window_size, n_input, seed=None):
    """
    Sample k consecutive frames, then select n input frames from them,
    and use the remaining k-n frames as target.
    
    Args:
        num_frames: Total number of frames in the scene
        window_size: Number of consecutive frames to sample (k)
        n_input: Number of input frames to select from the window (n)
        seed: Random seed for reproducibility
        
    Returns:
        tuple: (input_indices, target_indices) or None if sampling fails
    """
    if seed is not None:
        random.seed(seed)
    
    # Check if we have enough frames
    if num_frames < window_size:
        return None
    
    if n_input >= window_size:
        return None  # Need at least one target frame
    
    # Randomly select a starting position for the consecutive window
    max_start = num_frames - window_size
    window_start = random.randint(0, max_start)
    
    # Get consecutive frame indices
    consecutive_indices = list(range(window_start, window_start + window_size))
    
    # Randomly select n_input frames from consecutive_indices as input
    input_indices = sorted(random.sample(consecutive_indices, n_input))
    
    # Remaining frames are target
    target_indices = sorted([idx for idx in consecutive_indices if idx not in input_indices])
    
    return input_indices, target_indices


def create_rotation_traj_index(full_list_path, output_path, window_size, n_input, 
                               seed=42, max_scenes=None):
    """
    Create rotation trajectory index JSON file from a full_list.txt.
    
    Args:
        full_list_path: Path to full_list.txt containing scene JSON paths
        output_path: Path to save rotation trajectory index JSON
        window_size: Number of consecutive frames to sample (k)
        n_input: Number of input frames to select from the window (n)
        seed: Random seed for reproducibility
        max_scenes: Maximum number of scenes to process (optional)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Read full_list.txt
    with open(full_list_path, 'r') as f:
        scene_json_paths = [line.strip() for line in f if line.strip()]
    
    if max_scenes is not None and max_scenes > 0:
        if len(scene_json_paths) > max_scenes:
            print(f"Limiting to {max_scenes} scenes (from {len(scene_json_paths)} total).")
            # Randomly sample scenes if we have more than max_scenes
            # Use fixed seed for reproducibility of scene selection
            scene_json_paths = sorted(random.sample(scene_json_paths, max_scenes))
        else:
            print(f"Requested {max_scenes} scenes, but only {len(scene_json_paths)} available.")
    
    rotation_traj_index = {}
    
    print(f"Processing {len(scene_json_paths)} scenes...")
    successful = 0
    failed = 0
    
    for scene_json_path in scene_json_paths:
        try:
            # Load scene JSON
            with open(scene_json_path, 'r') as f:
                scene_data = json.load(f)
            
            scene_name = scene_data['scene_name']
            frames = scene_data['frames']
            num_frames = len(frames)
            
            # Sample consecutive frames
            result = sample_consecutive_frames(num_frames, window_size, n_input, seed=seed)
            
            if result is None:
                # Skip scenes that don't have enough frames
                rotation_traj_index[scene_name] = None
                failed += 1
                print(f"  Skipped {scene_name}: insufficient frames ({num_frames} < {window_size}) or n_input ({n_input}) >= window_size ({window_size})")
                continue
            
            input_indices, target_indices = result
            
            rotation_traj_index[scene_name] = {
                "context": input_indices,
                "target": target_indices
            }
            successful += 1
            
        except Exception as e:
            print(f"  Error processing {scene_json_path}: {e}")
            # Extract scene name from path if possible
            scene_name = Path(scene_json_path).stem
            rotation_traj_index[scene_name] = None
            failed += 1
    
    # Save rotation trajectory index
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(rotation_traj_index, f, indent=2)
    
    print(f"\nRotation trajectory index created: {output_path}")
    print(f"  Successful: {successful}")
    print(f"  Failed/Skipped: {failed}")
    print(f"  Total: {len(rotation_traj_index)}")


def main():
    parser = argparse.ArgumentParser(description='Create rotation trajectory index JSON file')
    parser.add_argument('--full-list', '-f', required=True,
                       help='Path to full_list.txt file')
    parser.add_argument('--output', '-o', required=True,
                       help='Output path for rotation trajectory index JSON')
    parser.add_argument('--window-size', '-k', type=int, required=True,
                       help='Number of consecutive frames to sample (k)')
    parser.add_argument('--n-input', '-n', type=int, required=True,
                       help='Number of input frames to select from the window (n)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--max-scenes', type=int, default=None,
                       help='Maximum number of scenes to include in the index (default: all)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.n_input >= args.window_size:
        parser.error(f"n_input ({args.n_input}) must be less than window_size ({args.window_size})")
    
    print(f"Creating rotation trajectory index with:")
    print(f"  Window size (k): {args.window_size}")
    print(f"  Input frames (n): {args.n_input}")
    print(f"  Target frames: {args.window_size - args.n_input}")
    print(f"  Random seed: {args.seed}")
    if args.max_scenes:
        print(f"  Max scenes: {args.max_scenes}")
    
    create_rotation_traj_index(
        args.full_list,
        args.output,
        args.window_size,
        args.n_input,
        args.seed,
        args.max_scenes
    )


if __name__ == '__main__':
    main()

