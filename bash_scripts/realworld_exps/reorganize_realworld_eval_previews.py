#!/usr/bin/env python3
"""
Reorganize realworld eval previews under result_previews/realworld_eval.

Expected per-iter files:
  input_<scene>.jpg/png
  supervision_<scene>.jpg/png
  envldr_<scene>.jpg/png   (optional, legacy)
  target_envldr_<scene>.png, target_envhdr_<scene>.png   (optional; from metric_utils)
  context_envldr_<scene>.png, context_envhdr_<scene>.png   (optional)
  camera_poses_<scene>.json   (optional)

Outputs:
  <run_name>_flattened/<scene>_iter_XXXXXXXX.jpg
  single_image/<run_name>/<scene>/iter_XXXXXXXX/...
      input_view_XX.jpg, input512_view_XX.png (if input512_* exists)
      Also: camera_poses.json, camera_context/target_view_XX.json;
      target_envldr_view_XX.jpg, target_envhdr_view_XX.png, context_* similarly (when source strips exist).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_BASE = _REPO_ROOT / "result_previews/realworld_eval"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base",
        type=Path,
        default=_DEFAULT_BASE,
        help=f"Directory containing run folders (default: {_DEFAULT_BASE})",
    )
    p.add_argument(
        "--infer",
        nargs="*",
        default=["test_relight_stanfordORB"],
        metavar="DIR",
        help="Run folder names under --base to process.",
    )
    p.add_argument("--dry-run", action="store_true", help="Print planned outputs only.")
    return p.parse_args()


def infer_input_view_count(path: Path) -> int:
    """Infer number of input views from strip width/height ratio."""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    if h <= 0:
        raise ValueError(f"{path}: invalid image height {h}")
    ratio = w / h
    n_views = int(round(ratio))
    if n_views <= 0:
        raise ValueError(f"{path}: invalid inferred n_views={n_views} from ratio={ratio:.4f}")
    return n_views


def split_input_strip(path: Path, expected_n_views: int | None = None) -> tuple[list[Image.Image], int]:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    n_inputs = expected_n_views if expected_n_views is not None else infer_input_view_count(path)
    if n_inputs <= 0:
        raise ValueError(f"{path}: invalid n_inputs={n_inputs}")
    if w % n_inputs != 0:
        raise ValueError(f"{path}: input width {w} not divisible by n_inputs {n_inputs}")
    tile_w = w // n_inputs
    tiles = []
    for i in range(n_inputs):
        tiles.append(im.crop((i * tile_w, 0, (i + 1) * tile_w, h)))
    return tiles, h


def split_supervision_strip(path: Path) -> tuple[list[Image.Image], int, int]:
    """
    Split supervision strip into [gt, relit_gt, relit_pred] tiles per target view.
    Uses image height as tile size (works for square rendered tiles in this project).
    """
    im = Image.open(path).convert("RGB")
    w, h = im.size
    tile_w = h
    if w % tile_w != 0:
        raise ValueError(f"{path}: supervision width {w} not divisible by tile_w {tile_w}")
    n_tiles = w // tile_w
    if n_tiles % 3 != 0:
        raise ValueError(f"{path}: expected tile count multiple of 3, got {n_tiles}")
    n_views = n_tiles // 3
    tiles = []
    for t in range(n_tiles):
        x0 = t * tile_w
        tiles.append(im.crop((x0, 0, x0 + tile_w, h)))
    return tiles, tile_w, n_views


def split_envldr_strip(path: Path, n_views: int) -> list[Image.Image]:
    im = Image.open(path).convert("RGB")
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


def first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.is_file():
            return p
    return None


def write_camera_splits(iter_dir: Path, stem: str, out_scene_iter: Path) -> None:
    """Copy full camera_poses JSON and write per-view context/target JSON slices."""
    src = iter_dir / f"camera_poses_{stem}.json"
    if not src.is_file():
        return
    shutil.copy2(src, out_scene_iter / "camera_poses.json")
    data = json.loads(src.read_text(encoding="utf-8"))
    scene_name = str(data.get("scene_name", ""))
    for i, entry in enumerate(data.get("context_views", []), start=1):
        payload = {**entry, "scene_name": scene_name, "role": "context"}
        (out_scene_iter / f"camera_context_view_{i:02d}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
    for i, entry in enumerate(data.get("target_views", []), start=1):
        payload = {**entry, "scene_name": scene_name, "role": "target"}
        (out_scene_iter / f"camera_target_view_{i:02d}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )


def write_split_env_strips(
    iter_dir: Path,
    stem: str,
    out_scene_iter: Path,
    n_input: int,
    n_target: int,
) -> None:
    """Split metric_utils env strips (LDR/HDR, context/target) into per-view files."""
    specs: list[tuple[str, int, str]] = [
        ("target_envldr", n_target, ".jpg"),
        ("target_envhdr", n_target, ".png"),
        ("context_envldr", n_input, ".jpg"),
        ("context_envhdr", n_input, ".png"),
    ]
    for name_prefix, n_parts, ext in specs:
        if n_parts <= 0:
            continue
        path = first_existing(
            [iter_dir / f"{name_prefix}_{stem}.png", iter_dir / f"{name_prefix}_{stem}.jpg"]
        )
        if path is None:
            continue
        try:
            tiles = split_envldr_strip(path, n_parts)
        except ValueError as e:
            print(f"Warning: skip {path}: {e}")
            continue
        for v, tile in enumerate(tiles, start=1):
            out_p = out_scene_iter / f"{name_prefix}_view_{v:02d}{ext}"
            if ext == ".png":
                tile.save(out_p, compress_level=6)
            else:
                tile.save(out_p, quality=95)


def hstack(images: list[Image.Image]) -> Image.Image:
    images = [im.convert("RGB") for im in images]
    wsum = sum(im.width for im in images)
    h = images[0].height
    out = Image.new("RGB", (wsum, h))
    x = 0
    for im in images:
        out.paste(im, (x, 0))
        x += im.width
    return out


def vstack(images: list[Image.Image]) -> Image.Image:
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


def paired_paths(iter_dir: Path, stem: str) -> tuple[Path, Path]:
    inp = next(p for p in (iter_dir / f"input_{stem}.jpg", iter_dir / f"input_{stem}.png") if p.exists())
    sup = next(p for p in (iter_dir / f"supervision_{stem}.jpg", iter_dir / f"supervision_{stem}.png") if p.exists())
    return inp, sup


def optional_input512_path(iter_dir: Path, stem: str) -> Path | None:
    for p in (iter_dir / f"input512_{stem}.jpg", iter_dir / f"input512_{stem}.png"):
        if p.exists():
            return p
    return None


def build_scene_column(iter_dir: Path, stem: str) -> tuple[Image.Image, int, int]:
    inp_path, sup_path = paired_paths(iter_dir, stem)
    sup_tiles, _, n_views = split_supervision_strip(sup_path)
    in_tiles, _ = split_input_strip(inp_path)

    row1 = hstack(in_tiles)
    gts, relits, preds = [], [], []
    for v in range(n_views):
        b = v * 3
        gts.append(sup_tiles[b + 0])
        relits.append(sup_tiles[b + 1])
        preds.append(sup_tiles[b + 2])
    row2 = hstack(gts)
    row3 = hstack(relits)
    row4 = hstack(preds)

    if row1.width != row2.width:
        row1 = row1.resize((row2.width, row1.height), Image.Resampling.LANCZOS)
    return vstack([row1, row2, row3, row4]), len(in_tiles), n_views


def write_single_scene(iter_dir: Path, stem: str, out_scene_iter: Path, dry_run: bool) -> None:
    inp_path, sup_path = paired_paths(iter_dir, stem)
    sup_tiles, _, n_views = split_supervision_strip(sup_path)
    in_tiles, _ = split_input_strip(inp_path)
    if dry_run:
        return

    out_scene_iter.mkdir(parents=True, exist_ok=True)
    for i, tile in enumerate(in_tiles):
        tile.save(out_scene_iter / f"input_view_{i:02d}.jpg", quality=95)

    input512_path = optional_input512_path(iter_dir, stem)
    if input512_path is not None:
        try:
            in512_tiles, _ = split_input_strip(input512_path, expected_n_views=len(in_tiles))
        except ValueError as e:
            print(f"Warning: skip {input512_path}: {e}")
            in512_tiles = []
        for i, tile in enumerate(in512_tiles):
            out_p = out_scene_iter / f"input512_view_{i:02d}.png"
            tile.save(out_p, compress_level=6)

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
        for v, tile in enumerate(split_envldr_strip(env_path, n_views)):
            tile.save(out_scene_iter / f"envldr_view_{v + 1:02d}.jpg", quality=95)

    write_split_env_strips(iter_dir, stem, out_scene_iter, len(in_tiles), n_views)
    write_camera_splits(iter_dir, stem, out_scene_iter)


def process_run_dir(base: Path, run_name: str, dry_run: bool) -> None:
    run_dir = base / run_name
    if not run_dir.is_dir():
        print(f"Skip (missing): {run_dir}")
        return

    flat_dir = base / f"{run_name}_flattened"
    single_root = base / "single_image" / run_name
    if not dry_run:
        flat_dir.mkdir(parents=True, exist_ok=True)

    iter_dirs = sorted(d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("iter_"))
    if not iter_dirs:
        print(f"No iter_* under {run_dir}")
        return

    for it in iter_dirs:
        stems = scene_stems(it)
        if not stems:
            print(f"  {it.name}: no paired input/supervision; skip")
            continue
        for stem in stems:
            col, n_in, n_tgt = build_scene_column(it, stem)
            out_flat = flat_dir / flattened_jpg_name(stem, it.name)
            if dry_run:
                print(f"[dry-run] {it.name}/{stem}: in_views={n_in}, tgt_views={n_tgt} -> {out_flat}")
            else:
                col.save(out_flat, quality=95)
                print(f"Wrote {out_flat}")
            write_single_scene(it, stem, single_root / stem / it.name, dry_run)

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
        process_run_dir(base, name, args.dry_run)
    print("Done.")


if __name__ == "__main__":
    main()
