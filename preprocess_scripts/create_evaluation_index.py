#!/usr/bin/env python3
"""
Script to create evaluation index JSON files for inference.

The script samples input and target frame indices from each scene using the same
strategy as data/dataset_scene.py:
1. Randomly selects a frame distance between min_frame_dist and max_frame_dist
2. Picks a start_frame and end_frame as anchors separated by this distance
3. Samples remaining frames from the range between start and end
4. First n_input frames become context, rest become targets

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


def sample_frames(num_frames, n_input, n_target, min_frame_dist=25, max_frame_dist=100):
    """
    Sample input and target frame indices using the same strategy as dataset_scene.py.
    
    This strategy:
    1. Randomly selects a frame distance (min_frame_dist to max_frame_dist)
    2. Picks a start_frame and end_frame as anchors
    3. Samples remaining frames from the range between start and end
    4. First n_input frames become context, rest become targets
    
    Args:
        num_frames: Total number of frames in the scene
        n_input: Number of input frames to sample
        n_target: Number of target frames to sample
        min_frame_dist: Minimum distance between start and end frame (default: 25)
        max_frame_dist: Maximum distance between start and end frame (default: 100)
        
    Returns:
        tuple: (input_indices, target_indices) or None if sampling fails
    """
    num_views = n_input + n_target
    
    # Check if we have enough frames
    if num_frames < num_views:
        return None
    
    # Adjust max_frame_dist based on available frames
    max_frame_dist = min(num_frames - 1, max_frame_dist)
    
    if max_frame_dist <= min_frame_dist:
        return None
    
    # Randomly select frame distance
    frame_dist = random.randint(min_frame_dist, max_frame_dist)
    
    if num_frames <= frame_dist:
        return None
    
    # Randomly select start frame
    start_frame = random.randint(0, num_frames - frame_dist - 1)
    end_frame = start_frame + frame_dist
    
    # Check if we have enough frames between start and end
    num_samples_needed = num_views - 2
    available_range_size = end_frame - start_frame - 1
    
    if available_range_size < num_samples_needed:
        return None
    
    # Sample frames between start and end
    sampled_frames = random.sample(range(start_frame + 1, end_frame), num_samples_needed)
    
    # Combine: start_frame, end_frame, and sampled frames
    image_indices = sorted([start_frame, end_frame] + sampled_frames)
    
    # Split into input and target
    input_indices = image_indices[:n_input]
    target_indices = image_indices[n_input:]
    
    return input_indices, target_indices


def create_evaluation_index(full_list_path, output_path, n_input, n_target, 
                            min_frame_dist=25, max_frame_dist=100, seed=42, max_scenes=None):
    """
    Create evaluation index JSON file from a full_list.txt.
    
    Args:
        full_list_path: Path to full_list.txt containing scene JSON paths
        output_path: Path to save evaluation index JSON
        n_input: Number of input frames
        n_target: Number of target frames
        min_frame_dist: Minimum distance between start and end frame (default: 25)
        max_frame_dist: Maximum distance between start and end frame (default: 100)
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
            
            # Sample frames using same strategy as dataset_scene.py
            result = sample_frames(num_frames, n_input, n_target, min_frame_dist, max_frame_dist)
            
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
    parser.add_argument('--min-frame-dist', type=int, default=25,
                       help='Minimum distance between start and end frame (default: 25)')
    parser.add_argument('--max-frame-dist', type=int, default=100,
                       help='Maximum distance between start and end frame (default: 100)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--max-scenes', type=int, default=None,
                       help='Maximum number of scenes to include in the index (default: all)')
    
    args = parser.parse_args()
    
    print(f"Creating evaluation index with:")
    print(f"  Input frames: {args.n_input}")
    print(f"  Target frames: {args.n_target}")
    print(f"  Min frame distance: {args.min_frame_dist}")
    print(f"  Max frame distance: {args.max_frame_dist}")
    print(f"  Random seed: {args.seed}")
    if args.max_scenes:
        print(f"  Max scenes: {args.max_scenes}")
    
    create_evaluation_index(
        args.full_list,
        args.output,
        args.n_input,
        args.n_target,
        args.min_frame_dist,
        args.max_frame_dist,
        args.seed,
        args.max_scenes
    )


if __name__ == '__main__':
    main()

