#!/usr/bin/env python3
"""
Script to create evaluation index JSON files for inference.

The script randomly samples input and target frame indices from each scene,
ensuring the sampling window is smaller than 4*(n_input + n_target).

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


def sample_frames(num_frames, n_input, n_target, max_window_size=None):
    """
    Randomly sample input and target frame indices from a scene.
    
    Args:
        num_frames: Total number of frames in the scene
        n_input: Number of input frames to sample
        n_target: Number of target frames to sample
        max_window_size: Maximum window size (default: 4*(n_input+n_target))
        
    Returns:
        tuple: (input_indices, target_indices) or None if sampling fails
    """
    if max_window_size is None:
        max_window_size = 4 * (n_input + n_target)
    
    # Check if we have enough frames
    if num_frames < max_window_size:
        max_window_size = num_frames
    
    if num_frames < n_input + n_target:
        return None
    
    # Randomly select a window start position
    max_start = num_frames - max_window_size
    if max_start < 0:
        window_start = 0
        window_size = num_frames
    else:
        window_start = random.randint(0, max_start)
        window_size = max_window_size
    
    # Create frame indices within the window
    window_indices = list(range(window_start, window_start + window_size))
    
    # Randomly sample input and target frames from the window
    if len(window_indices) < n_input + n_target:
        return None
    
    # Sample without replacement
    all_sampled = random.sample(window_indices, n_input + n_target)
    input_indices = sorted(all_sampled[:n_input])
    target_indices = sorted(all_sampled[n_input:])
    
    return input_indices, target_indices


def create_evaluation_index(full_list_path, output_path, n_input, n_target, 
                            max_window_size=None, seed=42, max_scenes=None):
    """
    Create evaluation index JSON file from a full_list.txt.
    
    Args:
        full_list_path: Path to full_list.txt containing scene JSON paths
        output_path: Path to save evaluation index JSON
        n_input: Number of input frames
        n_target: Number of target frames
        max_window_size: Maximum window size (default: 4*(n_input+n_target))
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
    
    evaluation_index = {}
    
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
            
            # Sample frames
            result = sample_frames(num_frames, n_input, n_target, max_window_size)
            
            if result is None:
                # Skip scenes that don't have enough frames
                evaluation_index[scene_name] = None
                failed += 1
                print(f"  Skipped {scene_name}: insufficient frames ({num_frames})")
                continue
            
            input_indices, target_indices = result
            
            evaluation_index[scene_name] = {
                "context": input_indices,
                "target": target_indices
            }
            successful += 1
            
        except Exception as e:
            print(f"  Error processing {scene_json_path}: {e}")
            # Extract scene name from path if possible
            scene_name = Path(scene_json_path).stem
            evaluation_index[scene_name] = None
            failed += 1
    
    # Save evaluation index
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(evaluation_index, f, indent=2)
    
    print(f"\nEvaluation index created: {output_path}")
    print(f"  Successful: {successful}")
    print(f"  Failed/Skipped: {failed}")
    print(f"  Total: {len(evaluation_index)}")


def main():
    parser = argparse.ArgumentParser(description='Create evaluation index JSON file')
    parser.add_argument('--full-list', '-f', required=True,
                       help='Path to full_list.txt file')
    parser.add_argument('--output', '-o', required=True,
                       help='Output path for evaluation index JSON')
    parser.add_argument('--n-input', type=int, default=2,
                       help='Number of input frames (default: 2)')
    parser.add_argument('--n-target', type=int, default=6,
                       help='Number of target frames (default: 6)')
    parser.add_argument('--max-window-size', type=int, default=None,
                       help='Maximum window size (default: 4*(n_input+n_target))')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--max-scenes', type=int, default=None,
                       help='Maximum number of scenes to include in the index (default: all)')
    
    args = parser.parse_args()
    
    if args.max_window_size is None:
        args.max_window_size = 1 * (args.n_input + args.n_target)
    
    print(f"Creating evaluation index with:")
    print(f"  Input frames: {args.n_input}")
    print(f"  Target frames: {args.n_target}")
    print(f"  Max window size: {args.max_window_size}")
    print(f"  Random seed: {args.seed}")
    if args.max_scenes:
        print(f"  Max scenes: {args.max_scenes}")
    
    create_evaluation_index(
        args.full_list,
        args.output,
        args.n_input,
        args.n_target,
        args.max_window_size,
        args.seed,
        args.max_scenes
    )


if __name__ == '__main__':
    main()

