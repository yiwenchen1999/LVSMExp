#!/usr/bin/env python3
"""
Split 3x8 grid images in result_previews/lighting-progressive (each tile 256x256).
Row 0 = input views 1-8, Row 1 = gt 1-8, Row 2 = pred 1-8.
Regroup output into: input-view1, input-view2, ..., input-view8, gt_1, ..., gt_8, pred_1, ..., pred_8.
"""
import os
import argparse
from PIL import Image

TILE_W, TILE_H = 256, 256
ROWS, COLS = 3, 8


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", default="result_previews/lighting-progressive", nargs="?",
                        help="Directory containing 3x8 grid PNGs")
    parser.add_argument("-o", "--output-dir", default=None,
                        help="Output root directory (default: input_dir + '_split')")
    args = parser.parse_args()
    input_dir = os.path.abspath(args.input_dir)
    output_dir = args.output_dir or (input_dir.rstrip("/") + "_split")
    output_dir = os.path.abspath(output_dir)

    # Create output folders: input-view1..8, gt_1..8, pred_1..8
    subdirs = [f"input-view{i}" for i in range(1, COLS + 1)] + \
              [f"gt_{i}" for i in range(1, COLS + 1)] + \
              [f"pred_{i}" for i in range(1, COLS + 1)]
    for d in subdirs:
        os.makedirs(os.path.join(output_dir, d), exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.lower().endswith(".png")]
    for fn in files:
        path = os.path.join(input_dir, fn)
        base = os.path.splitext(fn)[0]
        try:
            im = Image.open(path)
        except Exception as e:
            print(f"Skip {fn}: {e}")
            continue
        w, h = im.size
        if w != COLS * TILE_W or h != ROWS * TILE_H:
            print(f"Skip {fn}: size {w}x{h} (expected {COLS * TILE_W}x{ROWS * TILE_H})")
            continue
        for row in range(ROWS):
            for col in range(COLS):
                x = col * TILE_W
                y = row * TILE_H
                tile = im.crop((x, y, x + TILE_W, y + TILE_H))
                if row == 0:
                    subdir = f"input-view{col + 1}"
                elif row == 1:
                    subdir = f"gt_{col + 1}"
                else:
                    subdir = f"pred_{col + 1}"
                out_path = os.path.join(output_dir, subdir, f"{base}.png")
                tile.save(out_path)
    print(f"Done. Output: {output_dir}")
    print(f"Processed {len(files)} images -> {len(subdirs)} folders.")


if __name__ == "__main__":
    main()
