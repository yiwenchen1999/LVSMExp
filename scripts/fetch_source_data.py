#!/usr/bin/env python3
"""
Fetch the original source images/envmaps for each scene in a metadata folder.

For each metadata JSON the script copies:
  1) input images    – from  <dataset_root>/images/<scene_name>/<idx>.png
  2) envmaps         – from  <dataset_root>/envmaps/<relit_scene_name>/<idx>_ldr.png  (+ _hdr.png)
  3) target images   – from  <dataset_root>/images/<relit_scene_name>/<idx>.png

Output layout (one sub-folder per scene):
  <out_dir>/<scene_name>/input_images/00006.png  …
  <out_dir>/<scene_name>/envmaps/00006_ldr.png   …
  <out_dir>/<scene_name>/target_images/00006.png  …

Usage:
  python scripts/fetch_source_data.py \
      --metadata_dir  result_previews/polyhaven_dense_inference_same_pose_flat/metadata \
      --dataset_root  /path/to/polyhaven_lvsm/test \
      --out_dir       result_previews/polyhaven_dense_inference_same_pose_flat/source_data
"""

import argparse
import json
import os
import shutil
from pathlib import Path


def copy_if_exists(src: str, dst: str) -> bool:
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False


def process_one(metadata_path: str, dataset_root: str, out_dir: str):
    with open(metadata_path, "r") as f:
        meta = json.load(f)

    scene_name = meta["scene_name"]
    relit_scene_name = meta.get("relit_scene_name")
    context_indices = meta.get("context_view_indices", [])
    target_indices = meta.get("target_view_indices", [])
    all_indices = sorted(set(context_indices + target_indices))

    scene_out = os.path.join(out_dir, scene_name)

    copied = {"input_images": 0, "envmaps": 0, "target_images": 0}
    missing = {"input_images": [], "envmaps": [], "target_images": []}

    # (1) Input images from scene_name
    for idx in all_indices:
        fname = f"{idx:05d}.png"
        src = os.path.join(dataset_root, "images", scene_name, fname)
        dst = os.path.join(scene_out, "input_images", fname)
        if copy_if_exists(src, dst):
            copied["input_images"] += 1
        else:
            missing["input_images"].append(src)

    if relit_scene_name:
        # (2) Envmaps from relit_scene_name
        for idx in all_indices:
            for suffix in ("_ldr.png", "_hdr.png"):
                fname = f"{idx:05d}{suffix}"
                src = os.path.join(dataset_root, "envmaps", relit_scene_name, fname)
                dst = os.path.join(scene_out, "envmaps", fname)
                if copy_if_exists(src, dst):
                    copied["envmaps"] += 1
                else:
                    missing["envmaps"].append(src)

        # (3) Target images from relit_scene_name
        for idx in all_indices:
            fname = f"{idx:05d}.png"
            src = os.path.join(dataset_root, "images", relit_scene_name, fname)
            dst = os.path.join(scene_out, "target_images", fname)
            if copy_if_exists(src, dst):
                copied["target_images"] += 1
            else:
                missing["target_images"].append(src)

    return scene_name, copied, missing


def main():
    parser = argparse.ArgumentParser(description="Fetch source images/envmaps from dataset for each metadata entry")
    parser.add_argument("--metadata_dir", required=True, help="Directory containing metadata JSON files")
    parser.add_argument("--dataset_root", required=True,
                        help="Root of the dataset (e.g. .../test), containing images/ and envmaps/ folders")
    parser.add_argument("--out_dir", required=True, help="Output directory for fetched files")
    args = parser.parse_args()

    metadata_dir = Path(args.metadata_dir)
    json_files = sorted(metadata_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {metadata_dir}")
        return

    total_copied = {"input_images": 0, "envmaps": 0, "target_images": 0}
    total_missing = {"input_images": 0, "envmaps": 0, "target_images": 0}

    for jf in json_files:
        scene_name, copied, missing_files = process_one(str(jf), args.dataset_root, args.out_dir)
        for k in total_copied:
            total_copied[k] += copied[k]
            total_missing[k] += len(missing_files[k])
        if any(missing_files[k] for k in missing_files):
            n_miss = sum(len(v) for v in missing_files.values())
            print(f"  {scene_name}: {n_miss} file(s) not found")

    print(f"\nDone. Processed {len(json_files)} scenes → {args.out_dir}")
    for k in total_copied:
        print(f"  {k}: {total_copied[k]} copied, {total_missing[k]} missing")


if __name__ == "__main__":
    main()
