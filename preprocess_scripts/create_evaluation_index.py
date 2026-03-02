#!/usr/bin/env python3
"""
Script to create evaluation index JSON files for inference.

The script samples input and target frame indices from each scene using the same
strategy as data/dataset_scene.py:
1. Randomly selects a frame distance between min_frame_dist and max_frame_dist
2. Picks a start_frame and end_frame as anchors separated by this distance
3. Samples remaining frames from the range between start and end
4. First n_input frames become context, rest become targets

For scenes with the same object prefix (e.g., wooden_table_02_env_0, wooden_table_02_rgb_pl_0),
the same input and target indices are used to ensure consistent camera views across
different lighting conditions.

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
from collections import defaultdict


def extract_object_prefix(scene_name):
    """
    Extract the object prefix from a scene name.
    
    Examples:
        wooden_table_02_env_0 -> wooden_table_02
        wooden_table_02_rgb_pl_0 -> wooden_table_02
        wooden_table_02_white_env_0 -> wooden_table_02
        
    Args:
        scene_name: Scene name (e.g., "wooden_table_02_env_0")
        
    Returns:
        str: Object prefix (e.g., "wooden_table_02")
    """
    # Look for common lighting condition suffixes
    split_tags = ["_white_env_", "_env_", "_white_pl_", "_rgb_pl_"]
    for tag in split_tags:
        idx = scene_name.rfind(tag)
        if idx != -1:
            return scene_name[:idx]
    # Fallback: return the scene name as-is
    return scene_name


def sample_frames(num_frames, n_input, n_target, min_frame_dist=25, max_frame_dist=100):
    """
    Sample input and target frame indices using the same strategy as dataset_scene.py.
    
    This strategy:
    1. Randomly selects a frame distance (min_frame_dist to max_frame_dist)
    2. Picks a start_frame and end_frame as anchors
    3. Samples remaining frames from the range between start and end
    4. Randomly selects which frames are context vs targets (not just first n_input)
    
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
    
    # Randomly select which indices are context vs targets
    # This ensures context frames are distributed throughout the sequence, not just the first few
    context_positions = sorted(random.sample(range(num_views), n_input))
    input_indices = sorted([image_indices[i] for i in context_positions])
    target_indices = sorted([image_indices[i] for i in range(num_views) if i not in context_positions])
    
    return input_indices, target_indices


def create_evaluation_index(full_list_path, output_path, n_input, n_target, 
                            min_frame_dist=25, max_frame_dist=100, seed=42, max_scenes=None):
    """
    Create evaluation index JSON file from a full_list.txt.
    
    Scenes with the same object prefix (e.g., wooden_table_02_env_0, wooden_table_02_rgb_pl_0)
    will use the same input and target indices for consistent camera views across lighting conditions.
    
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
    
    # Group scenes by object prefix
    scenes_by_object = defaultdict(list)
    scene_data_cache = {}
    
    print(f"Loading and grouping {len(scene_json_paths)} scenes...")
    for scene_json_path in scene_json_paths:
        try:
            # Load scene JSON
            with open(scene_json_path, 'r') as f:
                scene_data = json.load(f)
            
            scene_name = scene_data['scene_name']
            scene_data_cache[scene_name] = scene_data
            
            # Extract object prefix and group scenes
            object_prefix = extract_object_prefix(scene_name)
            scenes_by_object[object_prefix].append(scene_name)
            
        except Exception as e:
            print(f"  Error loading {scene_json_path}: {e}")
            scene_name = Path(scene_json_path).stem
            scene_data_cache[scene_name] = None
            object_prefix = extract_object_prefix(scene_name)
            scenes_by_object[object_prefix].append(scene_name)
    
    print(f"Found {len(scenes_by_object)} unique objects")
    
    # Sample indices per object (all scenes with same object prefix use same indices)
    evaluation_index = {}
    successful = 0
    failed = 0
    object_indices_cache = {}
    
    print(f"\nProcessing scenes...")
    for object_prefix, scene_names in scenes_by_object.items():
        # Sample indices once per object
        if object_prefix not in object_indices_cache:
            # Use the first valid scene to determine frame count and sample indices
            sampled_result = None
            reference_scene = None
            
            for scene_name in scene_names:
                scene_data = scene_data_cache.get(scene_name)
                if scene_data is None:
                    continue
                    
                frames = scene_data.get("frames", [])
                num_frames = len(frames)
                
                # Try to sample frames
                result = sample_frames(num_frames, n_input, n_target, min_frame_dist, max_frame_dist)
                
                if result is not None:
                    sampled_result = result
                    reference_scene = scene_name
                    break
            
            if sampled_result is not None:
                object_indices_cache[object_prefix] = sampled_result
                print(f"  Object '{object_prefix}': sampled indices from '{reference_scene}'")
            else:
                object_indices_cache[object_prefix] = None
                print(f"  Object '{object_prefix}': insufficient frames in all variants")
        
        # Apply the same indices to all scenes with this object prefix
        cached_indices = object_indices_cache[object_prefix]
        
        for scene_name in scene_names:
            scene_data = scene_data_cache.get(scene_name)
            
            if scene_data is None or cached_indices is None:
                evaluation_index[scene_name] = None
                failed += 1
                continue
            
            frames = scene_data.get("frames", [])
            num_frames = len(frames)
            input_indices, target_indices = cached_indices
            
            # Verify that the cached indices are valid for this scene
            max_index = max(input_indices + target_indices)
            if max_index >= num_frames:
                print(f"    Warning: '{scene_name}' has only {num_frames} frames, but needs index {max_index}")
                evaluation_index[scene_name] = None
                failed += 1
                continue
            
            evaluation_index[scene_name] = {
                "context": input_indices,
                "target": target_indices
            }
            successful += 1
    
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
    print(f"  Unique objects: {len(scenes_by_object)}")


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

