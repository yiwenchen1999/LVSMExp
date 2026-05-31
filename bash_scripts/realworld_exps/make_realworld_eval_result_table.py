#!/usr/bin/env python3
"""
Build a result table for realworld eval samples listed in scene_id.txt.

Layout (one ROW per sample):
- col1: 4 sampled input views stitched into a 2x2 grid (single block)
- col2: gt relit
- col3: pred relit

Source: result_previews/realworld_eval/single_image/<run_name>/<scene>/<iter>/
  input_view_XX.jpg (16 total)
  view_YY_relit_gt.jpg / view_YY_relit_pred.jpg

The individual single images actually used (4 chosen inputs + gt + pred per
sample) are also copied into a separate folder for convenience.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = REPO_ROOT / "result_previews" / "realworld_eval" / "single_image" / "infer_stanfordORB_512_editor"
DEFAULT_LIST = REPO_ROOT / "result_previews" / "realworld_eval" / "infer_stanfordORB_512_editor_flattened" / "scene_id.txt"
DEFAULT_OUTPUT = REPO_ROOT / "result_previews" / "realworld_eval" / "result_table_infer_stanfordORB_512_editor.jpg"

CELL = 384
MARGIN_X = 28
MARGIN_Y = 24
GAP_X = 16
GAP_Y = 16
ROW_LABEL_W = 0
COL_LABEL_H = 84

COL_NAMES = ["input views\n(4 of 16)", "gt relit", "pred relit"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, default=DEFAULT_BASE)
    p.add_argument("--list", type=Path, default=DEFAULT_LIST)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--rows-per-image", type=int, default=3, help="Samples per output image.")
    return p.parse_args()


def font_or_default(size: int) -> ImageFont.ImageFont:
    for fp in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(fp, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def read_scene_entries(path: Path) -> list[tuple[str, str]]:
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    out: list[tuple[str, str]] = []
    for ln in lines:
        if "_iter_" not in ln:
            raise ValueError(f"Invalid line in {path}: {ln}")
        scene, it = ln.rsplit("_iter_", 1)
        out.append((scene, f"iter_{it}"))
    return out


def list_input_views(iter_dir: Path) -> list[Path]:
    views = sorted(iter_dir.glob("input_view_*.jpg")) + sorted(iter_dir.glob("input_view_*.png"))
    if not views:
        raise FileNotFoundError(f"No input_view_* in {iter_dir}")
    return views


def list_target_view_ids(iter_dir: Path) -> list[int]:
    ids: list[int] = []
    for i in range(1, 100):
        if (iter_dir / f"view_{i:02d}_relit_pred.jpg").exists() or (iter_dir / f"view_{i:02d}_relit_pred.png").exists():
            ids.append(i)
    if not ids:
        raise FileNotFoundError(f"No view_XX_relit_pred in {iter_dir}")
    return ids


def choose_target_view(scene: str, iter_name: str, ids: list[int], seed: int) -> int:
    rng = random.Random(f"{seed}:{scene}:{iter_name}")
    return rng.choice(ids)


def choose_inputs(scene: str, iter_name: str, in_paths: list[Path], seed: int, k: int = 4) -> list[Path]:
    if len(in_paths) <= k:
        return in_paths
    rng = random.Random(f"inputs:{seed}:{scene}:{iter_name}")
    return sorted(rng.sample(in_paths, k), key=lambda p: p.name)


def load_rgb(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")


def fit_square(im: Image.Image, size: int = CELL) -> Image.Image:
    return ImageOps.fit(im.convert("RGB"), (size, size), Image.Resampling.LANCZOS)


def input_grid_block(chosen: list[Path]) -> Image.Image:
    """Stitch up to 4 input views into a 2x2 grid filling one CELL block."""
    tile = CELL // 2
    canvas = Image.new("RGB", (CELL, CELL), "white")
    for idx, p in enumerate(chosen[:4]):
        im = ImageOps.fit(load_rgb(p), (tile, tile), Image.Resampling.LANCZOS)
        x = (idx % 2) * tile
        y = (idx // 2) * tile
        canvas.paste(im, (x, y))
    return canvas


def existing_or_alt(iter_dir: Path, name_jpg: str) -> Path:
    p = iter_dir / name_jpg
    if p.exists():
        return p
    return iter_dir / name_jpg.replace(".jpg", ".png")


def main() -> None:
    args = parse_args()
    base = args.base.resolve()
    entries = read_scene_entries(args.list.resolve())
    if not entries:
        raise SystemExit("No scenes in list.")

    col_font = font_or_default(32)
    n_cols = len(COL_NAMES)

    blocks_dir = args.output.parent / f"{args.output.stem}_single_images"
    blocks_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows_per_image = max(1, args.rows_per_image)
    chunks = [entries[i : i + rows_per_image] for i in range(0, len(entries), rows_per_image)]

    info_lines: list[str] = []
    for part_idx, chunk in enumerate(chunks, start=1):
        n_rows = len(chunk)
        w = MARGIN_X * 2 + ROW_LABEL_W + n_cols * CELL + (n_cols - 1) * GAP_X
        h = MARGIN_Y * 2 + COL_LABEL_H + n_rows * CELL + (n_rows - 1) * GAP_Y
        canvas = Image.new("RGB", (w, h), "white")
        draw = ImageDraw.Draw(canvas)

        # Column headers.
        for c, label in enumerate(COL_NAMES):
            x = MARGIN_X + ROW_LABEL_W + c * (CELL + GAP_X)
            lines = label.split("\n")
            line_heights, line_widths = [], []
            for ln in lines:
                bb = draw.textbbox((0, 0), ln, font=col_font)
                line_widths.append(bb[2] - bb[0])
                line_heights.append(bb[3] - bb[1])
            line_gap = 6
            total_h = sum(line_heights) + line_gap * (len(lines) - 1)
            cur_y = MARGIN_Y + (COL_LABEL_H - total_h) // 2
            for ln, tw, th in zip(lines, line_widths, line_heights):
                draw.text((x + (CELL - tw) // 2, cur_y), ln, fill="black", font=col_font)
                cur_y += th + line_gap

        for r, (scene, iter_name) in enumerate(chunk):
            global_idx = (part_idx - 1) * rows_per_image + r + 1
            it_dir = base / scene / iter_name
            if not it_dir.is_dir():
                raise FileNotFoundError(f"Missing sample dir: {it_dir}")

            in_paths = list_input_views(it_dir)
            chosen = choose_inputs(scene, iter_name, in_paths, args.seed, k=4)
            view_ids = list_target_view_ids(it_dir)
            picked = choose_target_view(scene, iter_name, view_ids, args.seed)
            vt = f"{picked:02d}"

            gt_path = existing_or_alt(it_dir, f"view_{vt}_relit_gt.jpg")
            pred_path = existing_or_alt(it_dir, f"view_{vt}_relit_pred.jpg")

            blocks = [
                input_grid_block(chosen),
                fit_square(load_rgb(gt_path)),
                fit_square(load_rgb(pred_path)),
            ]

            y = MARGIN_Y + COL_LABEL_H + r * (CELL + GAP_Y)
            for c, im in enumerate(blocks):
                x = MARGIN_X + ROW_LABEL_W + c * (CELL + GAP_X)
                canvas.paste(im, (x, y))

            # Save the individual single images actually used.
            safe_scene = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in scene)
            sample_dir = blocks_dir / f"object{global_idx}_{safe_scene}_{iter_name}_view{picked:02d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            for i, p in enumerate(chosen[:4], start=1):
                load_rgb(p).save(sample_dir / f"input_{i:02d}_{p.stem}{p.suffix.lower()}", quality=95)
            load_rgb(gt_path).save(sample_dir / f"gt_relit_view{picked:02d}.jpg", quality=95)
            load_rgb(pred_path).save(sample_dir / f"pred_relit_view{picked:02d}.jpg", quality=95)

            info_lines.append(
                f"object{global_idx} {scene}_{iter_name}: target_view={picked:02d}, "
                f"inputs={[p.stem for p in chosen[:4]]}"
            )

        out_path = args.output if len(chunks) == 1 else (
            args.output.parent / f"{args.output.stem}_part{part_idx}{args.output.suffix}"
        )
        canvas.save(out_path, quality=95)
        print(f"Saved table : {out_path}")

    print(f"Single imgs : {blocks_dir}")
    for ln in info_lines:
        print(f"  {ln}")


if __name__ == "__main__":
    main()
