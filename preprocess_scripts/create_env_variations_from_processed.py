#!/usr/bin/env python3
"""
Create environment map variations from already-processed re10k format data.

Takes processed data (output of preprocess_objaverse.py) and creates N variations
for each env-lit scene by horizontally rolling the per-view environment maps.

This is equivalent to rotating the original HDRI around the vertical axis when
all cameras share the same elevation angle, because the horizontal roll in the
camera-frame env map directly corresponds to a world-space azimuthal rotation.

Output is written to a SEPARATE directory (--output-root) so the original data
stays untouched. Images are symlinked (or copied with --copy-images) to avoid
duplication. The output directory is self-contained and can be used directly with
exp_rotate_env.py.

Output naming follows the convention expected by exp_rotate_env.py:
  {original_scene_name}_{variation_idx}
  e.g., Camera_01_env_0 -> Camera_01_env_0_1, Camera_01_env_0_2, ...

Usage:
  python create_env_variations_from_processed.py \
      --data-root /data/polyhaven_lvsm \
      --output-root /data/polyhaven_lvsm_env_variations \
      --split test \
      --n-variations 12

The output directory will contain:
  {output_root}/{split}/
    metadata/       -> variation scene JSONs (image_path points to original images)
    envmaps/        -> rolled env map PNGs
    images/         -> symlinks to original image directories
    full_list.txt   -> all variation scenes
"""

import os
import re
import json
import argparse
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed


def is_env_scene(scene_name):
    """Check if a scene is env-lit (not white_env, not point_light, not already a variation)."""
    return bool(re.search(r'(?<!white)_env_\d+$', scene_name))


def process_single_variation(args_tuple):
    """
    Worker function: create one variation of one scene.
    Rolls all LDR/HDR env maps horizontally and writes a new scene JSON.
    """
    (scene_name, variation_idx, n_variations,
     src_envmaps_base, out_metadata_dir, out_envmaps_base,
     scene_json_data, copy_images, src_images_base, out_images_base) = args_tuple

    var_scene_name = f"{scene_name}_{variation_idx}"
    var_envmaps_dir = os.path.join(out_envmaps_base, var_scene_name)
    src_envmaps_dir = os.path.join(src_envmaps_base, scene_name)

    # Skip if already fully done
    var_json_path = os.path.join(out_metadata_dir, f"{var_scene_name}.json")
    if os.path.exists(var_envmaps_dir) and os.path.exists(var_json_path):
        existing = set(os.listdir(var_envmaps_dir))
        expected_ldr = {f for f in os.listdir(src_envmaps_dir) if f.endswith('_ldr.png')}
        expected_hdr = {f for f in os.listdir(src_envmaps_dir) if f.endswith('_hdr.png')}
        if expected_ldr.issubset(existing) and expected_hdr.issubset(existing):
            return var_scene_name, "skipped"

    os.makedirs(var_envmaps_dir, exist_ok=True)

    rotation_angle = (variation_idx - 1) * (2 * np.pi / n_variations)

    env_files = sorted(os.listdir(src_envmaps_dir))
    ldr_files = [f for f in env_files if f.endswith('_ldr.png')]
    hdr_files = [f for f in env_files if f.endswith('_hdr.png')]

    # Determine shift from the first file's width
    sample = np.array(Image.open(os.path.join(src_envmaps_dir, ldr_files[0])))
    env_w = sample.shape[1]
    shift = int(round((rotation_angle / (2 * np.pi)) * env_w))

    for fname in ldr_files + hdr_files:
        src = os.path.join(src_envmaps_dir, fname)
        dst = os.path.join(var_envmaps_dir, fname)
        if shift == 0:
            shutil.copy2(src, dst)
        else:
            img = np.array(Image.open(src))
            rolled = np.roll(img, shift, axis=1)
            Image.fromarray(rolled).save(dst)

    # Link or copy the images directory for this variation
    src_img_dir = os.path.join(src_images_base, scene_name)
    out_img_dir = os.path.join(out_images_base, var_scene_name)
    if os.path.isdir(src_img_dir) and not os.path.exists(out_img_dir):
        if copy_images:
            shutil.copytree(src_img_dir, out_img_dir)
        else:
            os.symlink(os.path.abspath(src_img_dir), out_img_dir)

    # Scene JSON: same frames (same image paths / cameras), just rename scene
    var_data = dict(scene_json_data)
    var_data["scene_name"] = var_scene_name
    with open(var_json_path, 'w') as f:
        json.dump(var_data, f, indent=2)

    deg = np.degrees(rotation_angle)
    return var_scene_name, f"created (rotation={deg:.1f}°, shift={shift}px)"


def create_env_variations(data_root, output_root, split, n_variations,
                          scene_filter=None, workers=4, copy_images=False):
    """
    Main entry: iterate over all env-lit scenes and create variations in output_root.

    Args:
        data_root:     Root of the source processed dataset
        output_root:   Root of the output directory for variations
        split:         'train' or 'test'
        n_variations:  Number of variations per scene (equally spaced 0..2pi)
        scene_filter:  Optional set of scene names to process (None = all)
        workers:       Number of parallel workers
        copy_images:   If True, copy image dirs; otherwise symlink them
    """
    src_metadata_dir = os.path.join(data_root, split, 'metadata')
    src_envmaps_base = os.path.join(data_root, split, 'envmaps')
    src_images_base = os.path.join(data_root, split, 'images')

    if not os.path.isdir(src_metadata_dir):
        raise FileNotFoundError(f"Source metadata dir not found: {src_metadata_dir}")
    if not os.path.isdir(src_envmaps_base):
        raise FileNotFoundError(f"Source envmaps dir not found: {src_envmaps_base}")

    # Create output directories
    out_metadata_dir = os.path.join(output_root, split, 'metadata')
    out_envmaps_base = os.path.join(output_root, split, 'envmaps')
    out_images_base = os.path.join(output_root, split, 'images')
    os.makedirs(out_metadata_dir, exist_ok=True)
    os.makedirs(out_envmaps_base, exist_ok=True)
    os.makedirs(out_images_base, exist_ok=True)

    json_files = sorted(f for f in os.listdir(src_metadata_dir) if f.endswith('.json'))

    # Collect env scenes that actually have envmaps
    scenes_to_process = []
    for jf in json_files:
        name = jf[:-5]
        if not is_env_scene(name):
            continue
        if scene_filter and name not in scene_filter:
            continue
        env_dir = os.path.join(src_envmaps_base, name)
        if not os.path.isdir(env_dir):
            print(f"  [skip] {name}: no envmaps directory")
            continue
        ldr_count = sum(1 for f in os.listdir(env_dir) if f.endswith('_ldr.png'))
        if ldr_count == 0:
            print(f"  [skip] {name}: no LDR env maps found")
            continue
        scenes_to_process.append(name)

    print(f"Found {len(scenes_to_process)} env-lit scenes to create {n_variations} variations each.")
    total_jobs = len(scenes_to_process) * n_variations

    # Pre-load all scene JSONs (they are small)
    scene_jsons = {}
    for name in scenes_to_process:
        with open(os.path.join(src_metadata_dir, f"{name}.json")) as f:
            scene_jsons[name] = json.load(f)

    # Build work items
    work_items = []
    for name in scenes_to_process:
        for vi in range(1, n_variations + 1):
            work_items.append((
                name, vi, n_variations,
                src_envmaps_base, out_metadata_dir, out_envmaps_base,
                scene_jsons[name], copy_images, src_images_base, out_images_base,
            ))

    created_scenes = []
    skipped = 0

    if workers <= 1:
        for item in work_items:
            var_name, status = process_single_variation(item)
            created_scenes.append(var_name)
            if status == "skipped":
                skipped += 1
            else:
                print(f"  {var_name}: {status}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(process_single_variation, item): item for item in work_items}
            done_count = 0
            for future in as_completed(futures):
                done_count += 1
                var_name, status = future.result()
                created_scenes.append(var_name)
                if status == "skipped":
                    skipped += 1
                else:
                    print(f"  [{done_count}/{total_jobs}] {var_name}: {status}")

    print(f"\nDone: {len(created_scenes)} variation scenes ({skipped} already existed, {len(created_scenes) - skipped} newly created)")
    return sorted(created_scenes)


def write_full_list(output_root, split, variation_scenes):
    """Write full_list.txt in the output directory containing all variation scenes."""
    out_metadata_dir = os.path.join(output_root, split, 'metadata')
    full_list_path = os.path.join(output_root, split, 'full_list.txt')
    with open(full_list_path, 'w') as f:
        for name in sorted(variation_scenes):
            json_path = os.path.abspath(os.path.join(out_metadata_dir, f"{name}.json"))
            f.write(f"{json_path}\n")
    print(f"Wrote {full_list_path} ({len(variation_scenes)} scenes)")


def main():
    parser = argparse.ArgumentParser(
        description="Create env-map variations from processed re10k data by horizontal rolling."
    )
    parser.add_argument(
        '--data-root', '-d', required=True,
        help='Root of source processed dataset (e.g., /data/polyhaven_lvsm)')
    parser.add_argument(
        '--output-root', '-o', required=True,
        help='Root of output directory for variations (e.g., /data/polyhaven_lvsm_env_variations)')
    parser.add_argument(
        '--split', '-s', default='test', choices=['train', 'test'],
        help='Split to process (default: test)')
    parser.add_argument(
        '--n-variations', '-n', type=int, default=12,
        help='Number of variations per scene (default: 12)')
    parser.add_argument(
        '--workers', '-w', type=int, default=8,
        help='Number of parallel workers (default: 8)')
    parser.add_argument(
        '--copy-images', action='store_true',
        help='Copy image directories instead of symlinking (default: symlink)')
    parser.add_argument(
        '--scene-list', type=str, default=None,
        help='Path to a full_list.txt file to select which scenes to process. '
             'Each line should be an absolute path to a scene JSON. '
             'Only env-lit scenes in this list will be processed. (default: all env scenes)')
    parser.add_argument(
        '--scene-filter', type=str, default=None,
        help='Comma-separated scene names to process (default: all env scenes)')

    args = parser.parse_args()

    if args.n_variations < 1:
        parser.error("--n-variations must be >= 1")

    scene_filter = None
    if args.scene_list:
        scene_filter = set()
        with open(args.scene_list, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                name = os.path.basename(line)
                if name.endswith('.json'):
                    name = name[:-5]
                scene_filter.add(name)
        print(f"Loaded {len(scene_filter)} scenes from {args.scene_list}")
    if args.scene_filter:
        extra = set(s.strip() for s in args.scene_filter.split(','))
        scene_filter = (scene_filter | extra) if scene_filter else extra

    print(f"Source:       {args.data_root}")
    print(f"Output:       {args.output_root}")
    print(f"Split:        {args.split}")
    print(f"Variations:   {args.n_variations}")
    print(f"Workers:      {args.workers}")
    print(f"Images:       {'copy' if args.copy_images else 'symlink'}")
    if scene_filter:
        print(f"Scene filter: {scene_filter}")
    print()

    variation_scenes = create_env_variations(
        args.data_root, args.output_root, args.split, args.n_variations,
        scene_filter=scene_filter, workers=args.workers,
        copy_images=args.copy_images,
    )

    write_full_list(args.output_root, args.split, variation_scenes)

    print("\nAll done. Use full_list.txt with exp_rotate_env.py:")
    print(f"  training.dataset_path = {os.path.join(args.output_root, args.split, 'full_list.txt')}")


if __name__ == '__main__':
    main()
