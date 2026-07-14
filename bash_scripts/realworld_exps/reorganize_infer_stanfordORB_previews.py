#!/usr/bin/env python3
"""
Reorganize infer_stanfordORB_512_editor previews into "flat" and "single_image".

Each iter_* folder contains (per scene):
  input_<scene>.jpg        : 16 context views side by side (n_in * tile_w x tile_h)
  supervision_<scene>.jpg  : per target view, 3 tiles laid out as input | gt | pred
                             (target.image | target.relit_images | render).

Outputs (under <run_dir>):
  flat/<scene>_<iter>.jpg
      The supervision strip (input | gt | pred). For multiple target views the
      per-view input|gt|pred rows are stacked vertically.
  single_image/<scene>/<iter>/
      input.jpg, gt.jpg, pred.jpg                (single target view)
      view_XX_input.jpg / _gt.jpg / _pred.jpg    (multiple target views)
      context_view_XX.jpg                        (split input context views)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_RUN = _REPO_ROOT / "result_previews/infer_stanfordORB_512_editor"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        default=_DEFAULT_RUN,
        help=f"Run folder containing raw iter_* data (default: {_DEFAULT_RUN}).",
    )
    p.add_argument(
        "--raw-subdir",
        default="raw",
        help="Sub-folder under run_dir holding iter_* dirs. Use '' if iter_* sit directly in run_dir.",
    )
    p.add_argument("--dry-run", action="store_true", help="Print planned outputs only; write nothing.")
    return p.parse_args()


def split_strip(path: Path, tile_w: int) -> List[Image.Image]:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    if w % tile_w != 0:
        raise ValueError(f"{path}: width {w} not divisible by tile_w {tile_w}")
    return [im.crop((i * tile_w, 0, (i + 1) * tile_w, h)) for i in range(w // tile_w)]


def split_supervision(path: Path) -> tuple[List[Image.Image], int, int]:
    """Square tiles (tile_w == height). Per target view = 3 tiles: input | gt | pred."""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    tile_w = h
    if w % tile_w != 0:
        raise ValueError(f"{path}: supervision width {w} not divisible by tile size {tile_w}")
    n_tiles = w // tile_w
    if n_tiles % 3 != 0:
        raise ValueError(f"{path}: expected tile count multiple of 3 (input|gt|pred), got {n_tiles}")
    tiles = [im.crop((i * tile_w, 0, (i + 1) * tile_w, h)) for i in range(n_tiles)]
    return tiles, tile_w, n_tiles // 3


def vstack(images: List[Image.Image]) -> Image.Image:
    w = max(im.width for im in images)
    hsum = sum(im.height for im in images)
    out = Image.new("RGB", (w, hsum), (255, 255, 255))
    y = 0
    for im in images:
        out.paste(im, ((w - im.width) // 2, y))
        y += im.height
    return out


def hstack(images: List[Image.Image]) -> Image.Image:
    h = max(im.height for im in images)
    wsum = sum(im.width for im in images)
    out = Image.new("RGB", (wsum, h), (255, 255, 255))
    x = 0
    for im in images:
        out.paste(im, (x, (h - im.height) // 2))
        x += im.width
    return out


def scene_stem(iter_dir: Path) -> str | None:
    for p in sorted(iter_dir.glob("supervision_*.jpg")) + sorted(iter_dir.glob("supervision_*.png")):
        name = p.name
        if "_part" in name:  # skip chunked multi-part strips
            continue
        return name[len("supervision_") : -len(p.suffix)]
    return None


def find(iter_dir: Path, prefix: str, stem: str) -> Path | None:
    for ext in (".jpg", ".png"):
        cand = iter_dir / f"{prefix}_{stem}{ext}"
        if cand.is_file():
            return cand
    return None


def process_iter(iter_dir: Path, flat_dir: Path, single_root: Path, dry_run: bool) -> bool:
    stem = scene_stem(iter_dir)
    if stem is None:
        return False
    sup_path = find(iter_dir, "supervision", stem)
    sup_tiles, tile_w, n_views = split_supervision(sup_path)

    per_view_rows = []
    for v in range(n_views):
        b = v * 3
        per_view_rows.append(hstack([sup_tiles[b], sup_tiles[b + 1], sup_tiles[b + 2]]))
    flat_img = per_view_rows[0] if n_views == 1 else vstack(per_view_rows)

    flat_out = flat_dir / f"{stem}_{iter_dir.name}.jpg"
    if dry_run:
        print(f"[dry-run] {iter_dir.name}/{stem}: tgt_views={n_views} -> {flat_out}")
    else:
        flat_dir.mkdir(parents=True, exist_ok=True)
        flat_img.save(flat_out, quality=95)

    out_iter = single_root / stem / iter_dir.name
    if not dry_run:
        out_iter.mkdir(parents=True, exist_ok=True)
    for v in range(n_views):
        b = v * 3
        prefix = "" if n_views == 1 else f"view_{v + 1:02d}_"
        triplet = {"input": sup_tiles[b], "gt": sup_tiles[b + 1], "pred": sup_tiles[b + 2]}
        for label, tile in triplet.items():
            out_p = out_iter / f"{prefix}{label}.jpg"
            if not dry_run:
                tile.save(out_p, quality=95)

    inp_path = find(iter_dir, "input", stem)
    if inp_path is not None:
        ctx_tiles = split_strip(inp_path, tile_w)
        for i, tile in enumerate(ctx_tiles):
            out_p = out_iter / f"context_view_{i:02d}.jpg"
            if not dry_run:
                tile.save(out_p, quality=95)
    return True


def main() -> None:
    args = parse_args()
    run_dir: Path = args.run_dir.resolve()
    src_dir = run_dir / args.raw_subdir if args.raw_subdir else run_dir
    if not src_dir.is_dir():
        raise SystemExit(f"Not a directory: {src_dir}")

    iter_dirs = sorted(d for d in src_dir.iterdir() if d.is_dir() and d.name.startswith("iter_"))
    if not iter_dirs:
        raise SystemExit(f"No iter_* subdirectories under {src_dir}")

    flat_dir = run_dir / "flat"
    single_root = run_dir / "single_image"

    n_ok = 0
    for it in iter_dirs:
        if process_iter(it, flat_dir, single_root, args.dry_run):
            n_ok += 1
        else:
            print(f"  {it.name}: no supervision strip; skip")

    verb = "Would process" if args.dry_run else "Processed"
    print(f"{verb} {n_ok}/{len(iter_dirs)} iter_* folders.")
    if not args.dry_run:
        print(f"flat/        -> {flat_dir}")
        print(f"single_image -> {single_root}")


if __name__ == "__main__":
    main()
