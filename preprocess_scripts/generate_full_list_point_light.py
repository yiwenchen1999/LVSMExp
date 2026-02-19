#!/usr/bin/env python3
"""
Generate full_list_point_light.txt from full_list.txt.

1. Add all scenes lit by point lights (white_pl_*, rgb_pl_*).
2. Add white_env-lit scenes (white_env_*) that share an object_id with at least
   one point-light scene.
"""

import argparse
import os
from typing import Optional


def _is_point_light_scene(scene_name: str) -> bool:
    return "_white_pl_" in scene_name or "_rgb_pl_" in scene_name


def _is_white_env_scene(scene_name: str) -> bool:
    return "_white_env_" in scene_name


def _extract_object_id(scene_name: str) -> Optional[str]:
    split_tags = ["_white_env_", "_env_", "_white_pl_", "_rgb_pl_"]
    for tag in split_tags:
        idx = scene_name.rfind(tag)
        if idx != -1:
            return scene_name[:idx]
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate full_list_point_light.txt from full_list.txt"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to full_list.txt",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Path to output file. Default: same dir as input, named full_list_point_light.txt",
    )
    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    point_light_paths = []
    white_env_paths_by_object = {}  # object_id -> list of paths

    for line in lines:
        # scene_name is basename without .json
        basename = os.path.basename(line.rstrip("/"))
        scene_name = basename.replace(".json", "") if basename.endswith(".json") else basename

        if _is_point_light_scene(scene_name):
            point_light_paths.append(line)
            continue

        if _is_white_env_scene(scene_name):
            obj_id = _extract_object_id(scene_name)
            if obj_id is not None:
                if obj_id not in white_env_paths_by_object:
                    white_env_paths_by_object[obj_id] = []
                white_env_paths_by_object[obj_id].append(line)

    object_ids_with_point_light = set()
    for line in point_light_paths:
        basename = os.path.basename(line.rstrip("/"))
        scene_name = basename.replace(".json", "") if basename.endswith(".json") else basename
        obj_id = _extract_object_id(scene_name)
        if obj_id is not None:
            object_ids_with_point_light.add(obj_id)

    # White-env scenes to add: those whose object_id has at least one point-light scene
    white_env_to_add = []
    for obj_id in object_ids_with_point_light:
        if obj_id in white_env_paths_by_object:
            white_env_to_add.extend(white_env_paths_by_object[obj_id])

    sublist = point_light_paths + white_env_to_add
    sublist = sorted(set(sublist))  # deduplicate and sort

    output_path = args.output
    if output_path is None:
        output_dir = os.path.dirname(input_path)
        output_path = os.path.join(output_dir, "full_list_point_light.txt")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w") as f:
        for path in sublist:
            f.write(path + "\n")

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Total scenes: {len(sublist)}")
    print(f"  - Point-light scenes: {len(point_light_paths)}")
    print(f"  - White-env scenes (shared object_id): {len(white_env_to_add)}")


if __name__ == "__main__":
    main()
