#!/usr/bin/env python3
"""
Reorganize resolution comparison previews (infer_256 / infer_512):

  infer_* / iter_* / input_<scene>.jpg       — N context views stitched horizontally (default N=4; use --num-input-views)
  infer_* / iter_* / supervision_<scene>.jpg — M target views × (gt | relit_gt | pred) per view

Outputs under the same parent as infer_*:

  (1) infer_*_flattened / <scene_stem>_iter_XXXXXXXX.jpg
      One 4-row montage per (iter, scene): row1 input strip (scaled), row2 gt, row3 relit_gt,
      row4 pred — all files live directly under infer_*_flattened (no iter_* subfolders).

  (2) single_image / infer_* / <scene_stem> / iter_XXXXXXXX /
      Split tiles: input_view_00..(N-1), view_01_* … (names adjusted to n_views),
      plus envldr_view_01..N when envldr_<scene>.jpg exists.

Usage (default --base is repo/result_previews/resolution_comparisons):
  bash bash_scripts/img_quality_refinement/reorganize_resolution_comparison_previews.sh
  python bash_scripts/img_quality_refinement/reorganize_resolution_comparison_previews.py \\
    --base result_previews/resolution_comparisons --infer infer_256 infer_512
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_BASE = _REPO_ROOT / "result_previews/resolution_comparisons"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base",
        type=Path,
        default=_DEFAULT_BASE,
        help=f"Directory containing infer_256, infer_512, … (default: {_DEFAULT_BASE})",
    )
    p.add_argument(
        "--infer",
        nargs="*",
        default=["infer_256", "infer_512"],
        metavar="DIR",
        help="Subdirectory names under --base to process (default: infer_256 infer_512).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned outputs only.",
    )
    p.add_argument(
        "--num-input-views",
        type=int,
        default=4,
        metavar="N",
        help="Number of context views in the horizontal input strip (default: 4; mesh-gen 2-view uses 2).",
    )
    return p.parse_args()


def split_input_strip(path: Path, num_input_views: int) -> tuple[list[Image.Image], int, int]:
    """N context views stitched horizontally; tile_w = W // N."""
    if num_input_views < 1:
        raise ValueError(f"num_input_views must be >= 1, got {num_input_views}")
    im = Image.open(path).convert("RGB")
    w, h = im.size
    if w % num_input_views != 0:
        raise ValueError(f"{path}: input width {w} not divisible by num_input_views={num_input_views}")
    tw = w // num_input_views
    tiles = []
    for i in range(num_input_views):
        tiles.append(im.crop((i * tw, 0, (i + 1) * tw, h)))
    return tiles, tw, h


def split_supervision_strip(path: Path, tile_w: int, tile_h: int) -> tuple[list[Image.Image], int]:
    """
    Returns flat tile list and n_views where each view contributes 3 tiles:
    gt, relit, pred (see utils/metric_utils.visualize_intermediate_results).
    """
    im = Image.open(path).convert("RGB")
    w, h = im.size
    if h != tile_h:
        im = im.crop((0, 0, w, tile_h))
        w, h = im.size
    if w % tile_w != 0:
        raise ValueError(f"{path}: supervision width {w} not divisible by tile_w {tile_w}")
    n_tiles = w // tile_w
    if n_tiles % 3 != 0:
        raise ValueError(f"{path}: expected tile count multiple of 3 (gt|relit|pred per view), got {n_tiles}")
    n_views = n_tiles // 3
    tiles = []
    for t in range(n_tiles):
        x0 = t * tile_w
        tiles.append(im.crop((x0, 0, x0 + tile_w, tile_h)))
    return tiles, n_views


def split_envldr_strip(path: Path, n_views: int, tile_h_ref: int) -> list[Image.Image]:
    """
    Split envldr strip into per-target-view envmaps.
    Each view is expected to be one LDR env image (typically 512x256).
    """
    im = Image.open(path).convert("RGB")
    w, h = im.size
    if h != tile_h_ref:
        # Keep only top rows if there is accidental extra padding.
        im = im.crop((0, 0, w, tile_h_ref))
        w, h = im.size
    if n_views <= 0:
        raise ValueError(f"{path}: invalid n_views={n_views}")
    if w % n_views != 0:
        raise ValueError(f"{path}: width {w} not divisible by n_views {n_views}")
    tile_w = w // n_views
    tiles = []
    for i in range(n_views):
        x0 = i * tile_w
        tiles.append(im.crop((x0, 0, x0 + tile_w, h)))
    return tiles


def hstack(images: list[Image.Image]) -> Image.Image:
    if not images:
        raise ValueError("hstack: empty")
    images = [im.convert("RGB") for im in images]
    h = images[0].height
    wsum = sum(im.width for im in images)
    out = Image.new("RGB", (wsum, h))
    x = 0
    for im in images:
        out.paste(im, (x, 0))
        x += im.width
    return out


def vstack(images: list[Image.Image]) -> Image.Image:
    if not images:
        raise ValueError("vstack: empty")
    images = [im.convert("RGB") for im in images]
    w = max(im.width for im in images)
    hsum = sum(im.height for im in images)
    out = Image.new("RGB", (w, hsum))
    y = 0
    for im in images:
        x_off = (w - im.width) // 2
        out.paste(im, (x_off, y))
        y += im.height
    return out


def flattened_jpg_name(scene_stem: str, iter_dir_name: str) -> str:
    """Filesystem-safe base name: scene first, then iter folder tag."""
    safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in scene_stem)
    return f"{safe}_{iter_dir_name}.jpg"


def scene_stems(iter_dir: Path) -> list[str]:
    stems = []
    for p in sorted(iter_dir.glob("input_*.jpg")) + sorted(iter_dir.glob("input_*.png")):
        stem = p.name[len("input_") : -len(p.suffix)]
        sup_jpg = iter_dir / f"supervision_{stem}.jpg"
        sup_png = iter_dir / f"supervision_{stem}.png"
        if sup_jpg.exists() or sup_png.exists():
            stems.append(stem)
    return stems


def build_scene_column(
    iter_dir: Path,
    stem: str,
    num_input_views: int,
) -> tuple[Image.Image, int, int, int]:
    """Returns (4-row RGB image, tile_w, tile_h, n_views)."""
    inp_path = next(p for p in (iter_dir / f"input_{stem}.jpg", iter_dir / f"input_{stem}.png") if p.exists())
    sup_path = next(
        p for p in (iter_dir / f"supervision_{stem}.jpg", iter_dir / f"supervision_{stem}.png") if p.exists()
    )

    in_tiles, tw_in, th_in = split_input_strip(inp_path, num_input_views)
    sup_tiles, n_views = split_supervision_strip(sup_path, tw_in, th_in)

    row1 = hstack(in_tiles)
    gts, relits, preds = [], [], []
    for v in range(n_views):
        base = v * 3
        gts.append(sup_tiles[base + 0])
        relits.append(sup_tiles[base + 1])
        preds.append(sup_tiles[base + 2])

    row2 = hstack(gts)
    row3 = hstack(relits)
    row4 = hstack(preds)
    target_w = row2.width
    if row1.width != target_w:
        row1 = row1.resize((target_w, th_in), Image.Resampling.LANCZOS)

    col = vstack([row1, row2, row3, row4])
    return col, tw_in, th_in, n_views


def write_single_scene(
    iter_dir: Path,
    stem: str,
    out_scene_iter: Path,
    n_views_ref: int,
    num_input_views: int,
    dry_run: bool,
) -> None:
    inp_path = next(p for p in (iter_dir / f"input_{stem}.jpg", iter_dir / f"input_{stem}.png") if p.exists())
    sup_path = next(
        p for p in (iter_dir / f"supervision_{stem}.jpg", iter_dir / f"supervision_{stem}.png") if p.exists()
    )

    in_tiles, tw_in, th_in = split_input_strip(inp_path, num_input_views)
    sup_tiles, n_views = split_supervision_strip(sup_path, tw_in, th_in)
    if n_views != n_views_ref:
        raise ValueError(f"{iter_dir}: view count mismatch for {stem}: {n_views} vs {n_views_ref}")

    if dry_run:
        return

    out_scene_iter.mkdir(parents=True, exist_ok=True)
    for i, tile in enumerate(in_tiles):
        tile.save(out_scene_iter / f"input_view_{i:02d}.jpg", quality=95)

    for v in range(n_views):
        b = v * 3
        sup_tiles[b + 0].save(out_scene_iter / f"view_{v + 1:02d}_gt.jpg", quality=95)
        sup_tiles[b + 1].save(out_scene_iter / f"view_{v + 1:02d}_relit_gt.jpg", quality=95)
        sup_tiles[b + 2].save(out_scene_iter / f"view_{v + 1:02d}_relit_pred.jpg", quality=95)

    env_path = None
    for cand in (iter_dir / f"envldr_{stem}.jpg", iter_dir / f"envldr_{stem}.png"):
        if cand.exists():
            env_path = cand
            break
    if env_path is not None:
        env_tiles = split_envldr_strip(env_path, n_views, th_in)
        for v, tile in enumerate(env_tiles):
            tile.save(out_scene_iter / f"envldr_view_{v + 1:02d}.jpg", quality=95)


def process_infer_dir(base: Path, infer_name: str, dry_run: bool, num_input_views: int) -> None:
    infer_dir = base / infer_name
    if not infer_dir.is_dir():
        print(f"Skip (missing): {infer_dir}")
        return

    flat_dir = base / f"{infer_name}_flattened"
    single_root = base / "single_image" / infer_name
    if not dry_run:
        flat_dir.mkdir(parents=True, exist_ok=True)

    iter_dirs = sorted(d for d in infer_dir.iterdir() if d.is_dir() and d.name.startswith("iter_"))
    if not iter_dirs:
        print(f"No iter_* under {infer_dir}")
        return

    for it in iter_dirs:
        stems = scene_stems(it)
        if not stems:
            print(f"  {it.name}: no paired input/supervision; skip")
            continue

        n_views: int | None = None
        for stem in stems:
            col, _, _, nv = build_scene_column(it, stem, num_input_views)
            if n_views is None:
                n_views = nv
            elif nv != n_views:
                raise ValueError(f"{it}: inconsistent n_views for {stem}: {nv} vs {n_views}")
            out_flat = flat_dir / flattened_jpg_name(stem, it.name)
            if dry_run:
                print(f"[dry-run] would write {out_flat} ({col.size[0]}×{col.size[1]})")
            else:
                col.save(out_flat, quality=95)
                print(f"Wrote {out_flat}")
        assert n_views is not None

        for stem in stems:
            out_si = single_root / stem / it.name
            write_single_scene(it, stem, out_si, int(n_views), num_input_views, dry_run)

    if not dry_run:
        print(f"Flattened grids: {flat_dir}")
        print(f"Single-tile output root: {single_root}")


def main() -> None:
    args = parse_args()
    base = args.base.resolve()
    if not base.is_dir():
        raise SystemExit(f"Not a directory: {base}")

    for name in args.infer:
        print(f"=== {name} ===")
        process_infer_dir(base, name, args.dry_run, args.num_input_views)

    print("Done.")


if __name__ == "__main__":
    main()
