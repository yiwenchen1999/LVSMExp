#!/usr/bin/env python3
"""
Utility script to analyze inference metadata summary.

Usage:
    python utils/analyze_inference_metadata.py <path_to_inference_metadata_summary.json>
"""

import json
import sys
from collections import defaultdict

def analyze_metadata(metadata_path):
    """Analyze inference metadata summary."""
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print("=" * 80)
    print(f"Inference Metadata Analysis")
    print("=" * 80)
    print(f"Total scenes: {len(metadata)}")
    print()
    
    # Group by object prefix
    objects = defaultdict(list)
    for scene_name in metadata.keys():
        # Extract object prefix
        for tag in ["_white_env_", "_env_", "_white_pl_", "_rgb_pl_"]:
            idx = scene_name.rfind(tag)
            if idx != -1:
                obj_prefix = scene_name[:idx]
                objects[obj_prefix].append(scene_name)
                break
    
    print(f"Unique objects: {len(objects)}")
    print()
    
    # Check consistency: same object should use same view indices
    print("Checking view index consistency across lighting conditions...")
    inconsistent = []
    for obj_prefix, scene_names in objects.items():
        if len(scene_names) <= 1:
            continue
        
        # Get context and target indices from first scene
        first_scene = scene_names[0]
        ref_context = metadata[first_scene]["context_view_indices"]
        ref_target = metadata[first_scene]["target_view_indices"]
        
        # Check all other scenes with same object
        for scene_name in scene_names[1:]:
            context = metadata[scene_name]["context_view_indices"]
            target = metadata[scene_name]["target_view_indices"]
            
            if context != ref_context or target != ref_target:
                inconsistent.append((obj_prefix, scene_names))
                break
    
    if inconsistent:
        print(f"❌ Found {len(inconsistent)} objects with inconsistent view indices:")
        for obj_prefix, scene_names in inconsistent[:5]:  # Show first 5
            print(f"  - {obj_prefix} ({len(scene_names)} scenes)")
    else:
        print(f"✓ All objects have consistent view indices across lighting conditions!")
    print()
    
    # Analyze relit scene distribution
    print("Relit scene distribution:")
    relit_scenes = defaultdict(int)
    null_relit = 0
    for scene_name, data in metadata.items():
        relit_scene = data.get("relit_scene_name")
        if relit_scene is None:
            null_relit += 1
        else:
            relit_scenes[relit_scene] += 1
    
    print(f"  Scenes with relit supervision: {len(metadata) - null_relit}")
    print(f"  Scenes without relit supervision: {null_relit}")
    print(f"  Unique relit scenes used: {len(relit_scenes)}")
    print()
    
    # Sample analysis
    print("Sample metadata (first 3 scenes):")
    for i, (scene_name, data) in enumerate(list(metadata.items())[:3]):
        print(f"\n{i+1}. {scene_name}")
        print(f"   Context: {data['context_view_indices']}")
        print(f"   Target:  {data['target_view_indices']}")
        print(f"   Relit:   {data.get('relit_scene_name', 'N/A')}")
    
    print()
    print("=" * 80)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python utils/analyze_inference_metadata.py <path_to_inference_metadata_summary.json>")
        sys.exit(1)
    
    metadata_path = sys.argv[1]
    analyze_metadata(metadata_path)
