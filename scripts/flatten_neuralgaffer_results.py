#!/usr/bin/env python3
"""
Flatten NeuralGaffer nested relighting results into a flat dir with combined images.
Each scene has input_image/, gt_image/, pred_image/, target_envmap_ldr/ subdirs.
Combines them into one image per scene: each row = input | gt | pred | envmap for one view.

Output layout:
  <out_dir>/combined/<scene_name>.png   - joint image per scene
  <out_dir>/inputs/<scene_name>.png     - input grid (all views)
  <out_dir>/gt/<scene_name>.png
  <out_dir>/pred/<scene_name>.png
  <out_dir>/envmaps/<scene_name>.png

Usage:
  python scripts/flatten_neuralgaffer_results.py result_previews/NeuralGaffer/polyhaven_relighting_results [out_dir]
"""

import os
import sys
from pathlib import Path

from PIL import Image


SUBdirs = ["input_image", "gt_image", "pred_image", "target_envmap_ldr"]
PADDING = 4
ROW_PADDING = 2
REF_SIZE = (256, 256)


def load_or_blank(path: Path, ref_w: int, ref_h: int) -> Image.Image:
    if path and path.exists():
        img = Image.open(path).convert("RGB")
        if img.size != (ref_w, ref_h):
            img = img.resize((ref_w, ref_h), Image.Resampling.LANCZOS)
        return img
    return Image.new("RGB", (ref_w, ref_h), (128, 128, 128))


def combine_scene(scene_dir: Path, out_dir: Path) -> bool:
    scene_name = scene_dir.name
    view_files = {}
    for sub in SUBdirs:
        d = scene_dir / sub
        if not d.is_dir():
            continue
        for f in d.glob("*.png"):
            view_id = f.stem
            if view_id not in view_files:
                view_files[view_id] = {}
            view_files[view_id][sub] = f

    if not view_files:
        return False

    views = sorted(view_files.keys())
    ref_h, ref_w = REF_SIZE
    row_imgs = []

    for vid in views:
        imgs = []
        for sub in SUBdirs:
            path = view_files.get(vid, {}).get(sub)
            imgs.append(load_or_blank(path, ref_w, ref_h))

        if PADDING > 0:
            pad = Image.new("RGB", (PADDING, ref_h), (255, 255, 255))
            row = Image.new("RGB", (ref_w * 4 + PADDING * 3, ref_h))
            x = 0
            for i, im in enumerate(imgs):
                row.paste(im, (x, 0))
                x += ref_w
                if i < 3:
                    row.paste(pad, (x, 0))
                    x += PADDING
        else:
            row = Image.new("RGB", (ref_w * 4, ref_h))
            for i, im in enumerate(imgs):
                row.paste(im, (i * ref_w, 0))
        row_imgs.append(row)

    n_rows = len(row_imgs)
    total_h = sum(r.height for r in row_imgs) + max(0, n_rows - 1) * ROW_PADDING
    total_w = row_imgs[0].width if row_imgs else 0
    combined = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    y = 0
    for i, r in enumerate(row_imgs):
        combined.paste(r, (0, y))
        y += r.height + ROW_PADDING

    combined_path = out_dir / "combined" / f"{scene_name}.png"
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(combined_path)

    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python flatten_neuralgaffer_results.py <source_dir> [out_dir]")
        sys.exit(1)
    source_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else source_dir.parent / f"{source_dir.name}_flat"

    if not source_dir.is_dir():
        print(f"Error: {source_dir} is not a directory")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "combined").mkdir(parents=True, exist_ok=True)

    count = 0
    for scene_dir in sorted(source_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        if combine_scene(scene_dir, out_dir):
            count += 1

    print(f"Created {count} combined images in {out_dir}/combined/")


if __name__ == "__main__":
    main()
