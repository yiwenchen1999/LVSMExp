#!/usr/bin/env python3
"""
Split horizontally concatenated supervision strips into per-view folders and merged PNGs.

See --help and the "strip layout" group for the tile ordering (matches
utils/metric_utils.visualize_intermediate_results multi-pass layout).
"""

from __future__ import annotations

import argparse
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from pathlib import Path
from typing import List, Sequence

from PIL import Image

TILE_W = TILE_H = 256

DEFAULT_ROOT = Path("result_previews/progressive_stability/finetuned_ckpt")


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="flatten_progressive_supervision",
        description=(
            "Read each supervision_*.jpg/png (a single row of fixed-size tiles), split by view, "
            "and write per-view folders plus merged relit sequences."
        ),
        epilog=(
            "Tile order in the strip (per view block): "
            "prelit, then (gt_relit, pred_relit) for each pass. "
            "Blocks are laid out as view_1 | view_2 | … | view_n left to right."
        ),
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    io = parser.add_argument_group("input / output")
    io.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=DEFAULT_ROOT,
        metavar="DIR",
        help=(
            "Directory to process: either a run root containing iter_* subfolders, "
            "or a single iter_* folder."
        ),
    )

    layout = parser.add_argument_group(
        "strip layout",
        "Must match training/inference visualization (tile size and number of views).",
    )
    layout.add_argument(
        "-n",
        "--num-views",
        "--n-views",
        type=int,
        default=8,
        metavar="N",
        dest="num_views",
        help="Number of views n: the strip is split into n equal blocks of tiles.",
    )
    layout.add_argument(
        "--tile-w",
        type=int,
        default=TILE_W,
        metavar="PX",
        help="Width of each square tile in pixels.",
    )
    layout.add_argument(
        "--tile-h",
        type=int,
        default=TILE_H,
        metavar="PX",
        help="Height of each square tile in pixels.",
    )

    behavior = parser.add_argument_group("behavior")
    behavior.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse strips and print counts only; do not write PNG files.",
    )

    return parser


def parse_args(argv: Sequence[str] | None = None) -> Namespace:
    return build_parser().parse_args(argv)


def stack_horizontal(images: List[Image.Image]) -> Image.Image | None:
    if not images:
        return None
    images = [im.convert("RGB") for im in images]
    w, h = images[0].size
    out = Image.new("RGB", (w * len(images), h))
    for i, im in enumerate(images):
        out.paste(im, (i * w, 0))
    return out


def split_strip(path: Path, n_views: int, tile_w: int, tile_h: int) -> tuple[list[Image.Image], int, int]:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    if h < tile_h:
        raise ValueError(f"{path}: height {h} < tile {tile_h}")
    # Use top tile_h rows if image is taller (e.g. debug grids).
    if h > tile_h:
        im = im.crop((0, 0, w, tile_h))
        h = tile_h
    if w % tile_w != 0:
        raise ValueError(f"{path}: width {w} not divisible by tile width {tile_w}")
    num_tiles = w // tile_w
    if num_tiles % n_views != 0:
        raise ValueError(
            f"{path}: {num_tiles} tiles not divisible by n_views={n_views} "
            f"(width={w}, tile_w={tile_w})"
        )
    m = num_tiles // n_views
    tiles: list[Image.Image] = []
    for t in range(num_tiles):
        x0 = t * tile_w
        tiles.append(im.crop((x0, 0, x0 + tile_w, tile_h)))
    return tiles, m, n_views


def process_iter_dir(iter_dir: Path, n_views: int, tile_w: int, tile_h: int, dry_run: bool) -> None:
    sup_files = sorted(iter_dir.glob("supervision_*.jpg")) + sorted(iter_dir.glob("supervision_*.png"))
    if not sup_files:
        return
    for sup_path in sup_files:
        tiles, m, _ = split_strip(sup_path, n_views, tile_w, tile_h)
        # Per view: m tiles = 1 prelit + (m-1)/2 pairs of (gt, pred)
        if m < 1:
            continue
        n_pairs = (m - 1) // 2
        if (m - 1) != 2 * n_pairs:
            raise ValueError(
                f"{sup_path}: per-view tile count m={m} expected 1 + 2*k (prelit + gt/pred pairs)"
            )
        if dry_run:
            print(f"[dry-run] {sup_path} -> {n_views} views, m={m}, steps={n_pairs}")
            continue

        for v in range(n_views):
            base = v * m
            pre = tiles[base + 0]
            gts = [tiles[base + 1 + 2 * s] for s in range(n_pairs)]
            preds = [tiles[base + 2 + 2 * s] for s in range(n_pairs)]

            view_sub = iter_dir / f"view_{v + 1}"
            view_sub.mkdir(parents=True, exist_ok=True)
            pre.save(view_sub / "prelit.png")
            for s, im in enumerate(gts):
                im.save(view_sub / f"relit_gt_{s:02d}.png")
            for s, im in enumerate(preds):
                im.save(view_sub / f"relit_pred_{s:02d}.png")

            pre.save(iter_dir / f"view{v + 1}_prelit.png")
            gt_stack = stack_horizontal(gts)
            pred_stack = stack_horizontal(preds)
            if gt_stack is not None:
                gt_stack.save(iter_dir / f"view{v + 1}_relit_gt.png")
            if pred_stack is not None:
                pred_stack.save(iter_dir / f"view{v + 1}_relit_pred.png")


def main() -> None:
    args = parse_args()
    root: Path = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    if root.name.startswith("iter_"):
        iter_dirs = [root]
    else:
        iter_dirs = sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith("iter_"))
    if not iter_dirs:
        raise SystemExit(f"No iter_* subdirectories under {root}")

    for d in iter_dirs:
        process_iter_dir(d, args.num_views, args.tile_w, args.tile_h, args.dry_run)

    if args.dry_run:
        print(f"Dry-run done. Scanned {len(iter_dirs)} iter_* folders under {root}.")
    else:
        print(f"Done. Processed {len(iter_dirs)} iter_* folders under {root}.")


if __name__ == "__main__":
    main()
