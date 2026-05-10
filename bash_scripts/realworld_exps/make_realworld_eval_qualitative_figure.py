#!/usr/bin/env python3
"""
Build qualitative figure for realworld eval scenes listed in demo_scene.txt.

Layout:
- each scene is one column
- row1: input views (if >4 inputs, sample 4 and make a 2x2 grid)
- row2: gt relit
- row3: pred relit
- row4: envmap condition
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = REPO_ROOT / "result_previews" / "realworld_eval" / "single_image"
DEFAULT_LIST = REPO_ROOT / "result_previews" / "realworld_eval" / "demo_scene.txt"
DEFAULT_OUTPUT = REPO_ROOT / "result_previews" / "realworld_eval" / "qualitative_8scene_4row.jpg"

CELL = 384
MARGIN_X = 28
MARGIN_Y = 24
GAP_X = 16
GAP_Y = 16
ROW_LABEL_W = 250
COL_LABEL_H = 108

ROW_NAMES = ["input_views\n(16 total)", "gt relit results", "pred relit results", "env map"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, default=DEFAULT_BASE)
    p.add_argument("--list", type=Path, default=DEFAULT_LIST)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--seed", type=int, default=777)
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


def pad_square(im: Image.Image, size: int = CELL) -> Image.Image:
    im = im.convert("RGB")
    w, h = im.size
    scale = min(size / w, size / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    im = im.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(im, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def input_block(iter_dir: Path, scene: str, iter_name: str, seed: int) -> Image.Image:
    in_paths = list_input_views(iter_dir)
    chosen = choose_inputs(scene, iter_name, in_paths, seed, k=4)
    tile = CELL // 2
    canvas = Image.new("RGB", (CELL, CELL), "white")
    for idx, p in enumerate(chosen[:4]):
        im = ImageOps.fit(load_rgb(p), (tile, tile), Image.Resampling.LANCZOS)
        x = (idx % 2) * tile
        y = (idx // 2) * tile
        canvas.paste(im, (x, y))
    return canvas


def locate_iter_dir(base: Path, scene: str, iter_name: str) -> Path:
    # demo_scene.txt convention: first 4 from stanfordORB, last 4 from obj-with-light.
    for run in ("test_relight_stanfordORB", "test_relight_obj-with-light"):
        d = base / run / scene / iter_name
        if d.is_dir():
            return d
    raise FileNotFoundError(f"Cannot locate scene {scene} {iter_name} under known runs")


def existing_or_alt(iter_dir: Path, name_jpg: str) -> Path:
    p = iter_dir / name_jpg
    if p.exists():
        return p
    return iter_dir / name_jpg.replace(".jpg", ".png")


def build_column(
    base: Path,
    scene: str,
    iter_name: str,
    seed: int,
    env_override: Path | None = None,
) -> tuple[list[Image.Image], int]:
    it_dir = locate_iter_dir(base, scene, iter_name)
    view_ids = list_target_view_ids(it_dir)
    picked = choose_target_view(scene, iter_name, view_ids, seed)
    vt = f"{picked:02d}"

    row1 = input_block(it_dir, scene, iter_name, seed)
    row2 = fit_square(load_rgb(existing_or_alt(it_dir, f"view_{vt}_relit_gt.jpg")))
    row3 = fit_square(load_rgb(existing_or_alt(it_dir, f"view_{vt}_relit_pred.jpg")))

    if env_override is not None:
        row4 = pad_square(load_rgb(env_override))
    else:
        env_path = existing_or_alt(it_dir, f"envldr_view_{vt}.jpg")
        if not env_path.exists():
            env_path = existing_or_alt(it_dir, "envldr_view_01.jpg")
        row4 = pad_square(load_rgb(env_path))
    return [row1, row2, row3, row4], picked


def main() -> None:
    args = parse_args()
    base = args.base.resolve()
    entries = read_scene_entries(args.list.resolve())
    if not entries:
        raise SystemExit("No scenes in list.")

    col_font = font_or_default(36)
    row_font = font_or_default(34)

    w = MARGIN_X * 2 + ROW_LABEL_W + len(entries) * CELL + (len(entries) - 1) * GAP_X
    h = MARGIN_Y * 2 + COL_LABEL_H + len(ROW_NAMES) * CELL + (len(ROW_NAMES) - 1) * GAP_Y
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)

    for r, label in enumerate(ROW_NAMES):
        y = MARGIN_Y + COL_LABEL_H + r * (CELL + GAP_Y)
        if "\n" in label:
            lines = label.split("\n")
            line_heights = []
            line_widths = []
            for ln in lines:
                bb = draw.textbbox((0, 0), ln, font=row_font)
                line_widths.append(bb[2] - bb[0])
                line_heights.append(bb[3] - bb[1])
            line_gap = 6
            total_h = sum(line_heights) + line_gap * (len(lines) - 1)
            cur_y = y + (CELL - total_h) // 2
            for ln, tw, th in zip(lines, line_widths, line_heights):
                lx = MARGIN_X + (ROW_LABEL_W - tw) // 2
                draw.text((lx, cur_y), ln, fill="black", font=row_font)
                cur_y += th + line_gap
        else:
            bb = draw.textbbox((0, 0), label, font=row_font)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
            draw.text((MARGIN_X + (ROW_LABEL_W - tw) // 2, y + (CELL - th) // 2), label, fill="black", font=row_font)

    info_lines = []
    env_override_by_col = {
        5: REPO_ROOT
        / "data_samples/processed_obj_with_light_objaverse_like_testenv_venv_flip_after_process/test/envmaps/apple/00008_hdr.png",
        6: REPO_ROOT
        / "data_samples/processed_obj_with_light_objaverse_like_testenv_venv_flip_after_process/test/envmaps/apple/00007_hdr.png",
        7: REPO_ROOT
        / "data_samples/processed_obj_with_light_objaverse_like_testenv_venv_flip_after_process/test/envmaps/apple/00006_hdr.png",
    }
    blocks_dir = args.output.parent / f"{args.output.stem}_blocks"
    blocks_dir.mkdir(parents=True, exist_ok=True)
    for c, (scene, iter_name) in enumerate(entries):
        col_id = c + 1
        env_override = env_override_by_col.get(col_id)
        blocks, view_id = build_column(base, scene, iter_name, args.seed, env_override=env_override)
        x = MARGIN_X + ROW_LABEL_W + c * (CELL + GAP_X)
        title = f"object{col_id}"
        bb = draw.textbbox((0, 0), title, font=col_font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        draw.text((x + (CELL - tw) // 2, MARGIN_Y + (COL_LABEL_H - th) // 2), title, fill="black", font=col_font)

        for r, im in enumerate(blocks):
            y = MARGIN_Y + COL_LABEL_H + r * (CELL + GAP_Y)
            canvas.paste(im, (x, y))
            safe_scene = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in scene)
            row_tag = ROW_NAMES[r].replace("\n", "_").replace(" ", "_").replace("(", "").replace(")", "")
            block_name = f"object{col_id}_{safe_scene}_{iter_name}_view{view_id:02d}_{row_tag}.jpg"
            im.save(blocks_dir / block_name, quality=95)
        info_lines.append(f"{scene}_{iter_name}: target_view={view_id:02d}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, quality=95)
    print(f"Saved figure: {args.output}")
    print(f"Saved single blocks: {blocks_dir}")
    print("Picked target views:")
    for ln in info_lines:
        print(f"  {ln}")


if __name__ == "__main__":
    main()
