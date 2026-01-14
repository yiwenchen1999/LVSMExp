#!/usr/bin/env python3
"""
Script to remove broken scenes from full_list.txt.
Scenes that are listed in broken_scene.txt will be removed from full_list.txt.
"""

import os
import argparse


def extract_scene_name(line):
    """
    Extract scene name from a line in broken_scene.txt or full_list.txt.
    
    For broken_scene.txt: "scene_name\tblack:..." -> "scene_name"
    For full_list.txt: "/path/to/metadata/scene_name.json" -> "scene_name"
    
    Args:
        line: Line from the file
        
    Returns:
        scene_name: Extracted scene name
    """
    line = line.strip()
    if not line:
        return None
    
    # For broken_scene.txt: scene name is before first tab or space
    if '\t' in line:
        scene_name = line.split('\t')[0]
    elif ' ' in line:
        scene_name = line.split(' ')[0]
    else:
        scene_name = line
    
    # For full_list.txt: extract scene name from JSON path
    # Format: "/path/to/metadata/scene_name.json" or "scene_name.json"
    if scene_name.endswith('.json'):
        # Remove .json extension
        scene_name = scene_name[:-5]
        # Extract just the filename if it's a full path
        if '/' in scene_name:
            scene_name = scene_name.split('/')[-1]
    
    return scene_name.strip()


def load_broken_scenes(broken_scene_path):
    """
    Load broken scene names from broken_scene.txt.
    
    Args:
        broken_scene_path: Path to broken_scene.txt
        
    Returns:
        set: Set of broken scene names
    """
    if not os.path.exists(broken_scene_path):
        print(f"Warning: broken_scene.txt not found at {broken_scene_path}")
        return set()
    
    broken_scenes = set()
    with open(broken_scene_path, 'r') as f:
        for line in f:
            scene_name = extract_scene_name(line)
            if scene_name:
                broken_scenes.add(scene_name)
    
    print(f"Loaded {len(broken_scenes)} broken scenes from {broken_scene_path}")
    return broken_scenes


def remove_broken_scenes_from_full_list(full_list_path, broken_scenes, output_path=None, backup=True):
    """
    Remove broken scenes from full_list.txt.
    
    Args:
        full_list_path: Path to full_list.txt
        broken_scenes: Set of broken scene names to remove
        output_path: Path to save the cleaned full_list.txt (default: overwrite original)
        backup: Whether to create a backup of the original file (default: True)
    """
    if not os.path.exists(full_list_path):
        print(f"Error: full_list.txt not found at {full_list_path}")
        return
    
    # Read all lines from full_list.txt
    with open(full_list_path, 'r') as f:
        all_lines = f.readlines()
    
    print(f"Loaded {len(all_lines)} scenes from {full_list_path}")
    
    # Filter out broken scenes
    cleaned_lines = []
    removed_scenes = []
    
    for line in all_lines:
        scene_name = extract_scene_name(line)
        if scene_name and scene_name in broken_scenes:
            removed_scenes.append(scene_name)
        else:
            cleaned_lines.append(line)
    
    print(f"Removed {len(removed_scenes)} broken scenes from full_list.txt")
    if removed_scenes:
        print(f"Removed scenes: {', '.join(removed_scenes[:10])}{'...' if len(removed_scenes) > 10 else ''}")
    
    # Create backup if requested
    if backup:
        backup_path = full_list_path + '.backup'
        with open(backup_path, 'w') as f:
            f.writelines(all_lines)
        print(f"Created backup at {backup_path}")
    
    # Determine output path
    if output_path is None:
        output_path = full_list_path
    else:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Write cleaned full_list.txt
    with open(output_path, 'w') as f:
        f.writelines(cleaned_lines)
    
    print(f"Cleaned full_list.txt saved to {output_path}")
    print(f"Original: {len(all_lines)} scenes, Cleaned: {len(cleaned_lines)} scenes")


def main():
    parser = argparse.ArgumentParser(description='Remove broken scenes from full_list.txt')
    parser.add_argument('--broken-scene', '-b', required=True,
                       help='Path to broken_scene.txt file')
    parser.add_argument('--full-list', '-f', required=True,
                       help='Path to full_list.txt file')
    parser.add_argument('--output', '-o', default=None,
                       help='Output path for cleaned full_list.txt (default: overwrite original)')
    parser.add_argument('--no-backup', action='store_true',
                       help='Do not create a backup of the original full_list.txt')
    
    args = parser.parse_args()
    
    # Load broken scenes
    broken_scenes = load_broken_scenes(args.broken_scene)
    
    if not broken_scenes:
        print("No broken scenes found. Exiting.")
        return
    
    # Remove broken scenes from full_list.txt
    remove_broken_scenes_from_full_list(
        args.full_list,
        broken_scenes,
        output_path=args.output,
        backup=not args.no_backup
    )
    
    print("\nDone!")


if __name__ == '__main__':
    main()

