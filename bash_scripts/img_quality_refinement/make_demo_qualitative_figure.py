#!/usr/bin/env python3
"""
Build qualitative demo figure from candidate scene/iter list.

Row per candidate, 5 columns:
  1) 4 input views as a 2x2 grid (fit into 512x512 block)
  2) relit_gt from infer_256 (random one target view)
  3) relit_pred from infer_256 (same view id as col 2)
  4) relit_pred from infer_512 (same view id as col 2)
  5) envldr from infer_256 same view, padded to square with white bars

Candidate line format:
  <scene_stem>_iter_<8-digit-iter>
e.g. brass_vase_04_env_1_iter_00000348
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = REPO_ROOT / "result_previews" / "resolution_comparisons"
DEFAULT_CANDIDATES = DEFAULT_BASE / "demo" / "candidates.txt"
DEFAULT_OUTPUT = DEFAULT_BASE / "demo" / "qualitative_candidates_5col.jpg"

BLOCK = 512
GAP_X = 16
GAP_Y = 16
MARGIN = 24
CAPTION_H = 84
CAPTION_FONT_SIZE = 42
ROW_LABEL_W = 170
ROW_LABEL_FONT_SIZE = 40
COLUMN_CAPTIONS = [
    "input views",
    "gt relight results",
    "512x512 results",
    "256x256 results",
    "env map",
]


def load_caption_font(size: int) -> ImageFont.ImageFont:
    # Try common TrueType fonts first for larger, cleaner captions.
    for fp in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(fp, size=size)
        except Exception:
            continue
    # Fallback (small bitmap font) if no TTF is available.
    return ImageFont.load_default()


def load_row_label_font(size: int) -> ImageFont.ImageFont:
    return load_caption_font(size)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, default=DEFAULT_BASE)
    p.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--seed", type=int, default=777, help="Base seed for deterministic random view choice.")
    p.add_argument(
        "--first-view",
        type=int,
        default=None,
        help="If set, force the first candidate row to use this target view id (1-based).",
    )
    p.add_argument(
        "--view-overrides",
        type=str,
        default="",
        help='Comma-separated row:view overrides, 1-based rows. Example: "1:5,3:8".',
    )
    return p.parse_args()


def read_candidates(path: Path) -> list[tuple[str, str]]:
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    out: list[tuple[str, str]] = []
    for ln in lines:
        if "_iter_" not in ln:
            raise ValueError(f"Invalid candidate line (missing _iter_): {ln}")
        scene, it = ln.rsplit("_iter_", 1)
        out.append((scene, f"iter_{it}"))
    return out


def parse_view_overrides(spec: str) -> dict[int, int]:
    out: dict[int, int] = {}
    if not spec.strip():
        return out
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"Invalid override '{token}', expected row:view")
        row_s, view_s = token.split(":", 1)
        row = int(row_s)
        view = int(view_s)
        if row < 1 or view < 1:
            raise ValueError(f"Invalid override '{token}', row/view must be >= 1")
        out[row] = view
    return out


def load_rgb(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")


def make_input_block(scene_dir_256: Path) -> Image.Image:
    imgs = [load_rgb(scene_dir_256 / f"input_view_{i:02d}.jpg") for i in range(4)]
    tile = BLOCK // 2
    canvas = Image.new("RGB", (BLOCK, BLOCK), "white")
    for i, im in enumerate(imgs):
        im = ImageOps.fit(im, (tile, tile), Image.Resampling.LANCZOS)
        x = (i % 2) * tile
        y = (i // 2) * tile
        canvas.paste(im, (x, y))
    return canvas


def make_input_block_no_resize(scene_dir_256: Path) -> Image.Image:
    """Build a 2x2 input grid with original tile sizes (no resize)."""
    imgs = [load_rgb(scene_dir_256 / f"input_view_{i:02d}.jpg") for i in range(4)]
    w, h = imgs[0].size
    canvas = Image.new("RGB", (w * 2, h * 2), "white")
    for i, im in enumerate(imgs):
        if im.size != (w, h):
            raise ValueError(f"Input views in {scene_dir_256} have inconsistent sizes.")
        x = (i % 2) * w
        y = (i // 2) * h
        canvas.paste(im, (x, y))
    return canvas


def list_available_views(scene_dir_256: Path) -> list[int]:
    out: list[int] = []
    for i in range(1, 100):
        p = scene_dir_256 / f"view_{i:02d}_relit_pred.jpg"
        if p.exists():
            out.append(i)
    if not out:
        raise RuntimeError(f"No relit_pred views found in {scene_dir_256}")
    return out


def pick_view(scene: str, iter_name: str, candidates: list[int], base_seed: int) -> int:
    rng = random.Random(f"{base_seed}:{scene}:{iter_name}")
    return rng.choice(candidates)


def square_with_white_pad(im: Image.Image, block: int = BLOCK) -> Image.Image:
    im = im.convert("RGB")
    w, h = im.size
    if w == block and h == block:
        return im
    # Fit width first for envmap (usually 512x256), then pad top/bottom with white.
    if w != block:
        new_h = max(1, int(round(h * block / w)))
        im = im.resize((block, new_h), Image.Resampling.LANCZOS)
        w, h = im.size
    canvas = Image.new("RGB", (block, block), "white")
    x = (block - w) // 2
    y = (block - h) // 2
    canvas.paste(im, (x, y))
    return canvas


def relit_block(path: Path) -> Image.Image:
    return ImageOps.fit(load_rgb(path), (BLOCK, BLOCK), Image.Resampling.LANCZOS)


def build_row(
    base: Path,
    scene: str,
    iter_name: str,
    seed: int,
    forced_view: int | None = None,
) -> tuple[Image.Image, list[Image.Image], int]:
    d256 = base / "single_image" / "infer_256" / scene / iter_name
    d512 = base / "single_image" / "infer_512" / scene / iter_name

    col1 = make_input_block(d256)
    col1_raw = make_input_block_no_resize(d256)
    views = list_available_views(d256)
    if forced_view is not None:
        if forced_view not in views:
            raise ValueError(f"Requested forced view {forced_view} not available for {scene}/{iter_name}")
        view = forced_view
    else:
        view = pick_view(scene, iter_name, views, seed)
    view_tag = f"{view:02d}"

    gt_512 = load_rgb(d512 / f"view_{view_tag}_relit_gt.jpg")
    pred_512 = load_rgb(d512 / f"view_{view_tag}_relit_pred.jpg")
    pred_256 = load_rgb(d256 / f"view_{view_tag}_relit_pred.jpg")

    # Use GT relit from 512 branch (same sampled target view).
    col2 = ImageOps.fit(gt_512, (BLOCK, BLOCK), Image.Resampling.LANCZOS)
    col3 = ImageOps.fit(pred_512, (BLOCK, BLOCK), Image.Resampling.LANCZOS)
    col4 = ImageOps.fit(pred_256, (BLOCK, BLOCK), Image.Resampling.LANCZOS)

    env = load_rgb(d256 / f"envldr_view_{view_tag}.jpg")
    col5 = square_with_white_pad(env, BLOCK)

    row = Image.new("RGB", (BLOCK * 5 + GAP_X * 4, BLOCK), "white")
    x = 0
    for im in (col1, col2, col3, col4, col5):
        row.paste(im, (x, 0))
        x += BLOCK + GAP_X
    raw_blocks = [col1_raw, gt_512, pred_512, pred_256, env]
    return row, raw_blocks, view


def main() -> None:
    args = parse_args()
    base = args.base.resolve()
    candidates = read_candidates(args.candidates.resolve())
    row_view_override = parse_view_overrides(args.view_overrides)

    blocks_dir = args.output.parent / f"{args.output.stem}_blocks"
    blocks_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, (scene, iter_name) in enumerate(candidates):
        row_id = idx + 1
        forced_view = row_view_override.get(row_id)
        if forced_view is None and row_id == 1 and args.first_view is not None:
            forced_view = args.first_view
        row_img, raw_blocks, picked_view = build_row(base, scene, iter_name, args.seed, forced_view=forced_view)
        rows.append(row_img)

        safe_scene = "".join(c if (c.isalnum() or c in "._-") else "_" for c in scene)
        block_suffixes = ["input_views", "gt_relight_512", "result_512", "result_256", "env_map"]
        for col_id, (blk, suffix) in enumerate(zip(raw_blocks, block_suffixes), start=1):
            blk.save(
                blocks_dir / f"row{row_id}_col{col_id}_{safe_scene}_{iter_name}_view{picked_view:02d}_{suffix}.jpg",
                quality=95,
            )
    width = ROW_LABEL_W + rows[0].width + MARGIN * 2
    height = CAPTION_H + len(rows) * BLOCK + max(0, len(rows) - 1) * GAP_Y + MARGIN * 2
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = load_caption_font(CAPTION_FONT_SIZE)
    row_font = load_row_label_font(ROW_LABEL_FONT_SIZE)

    col_x = MARGIN + ROW_LABEL_W
    for caption in COLUMN_CAPTIONS:
        bbox = draw.textbbox((0, 0), caption, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = col_x + (BLOCK - tw) // 2
        ty = MARGIN + (CAPTION_H - th) // 2
        draw.text((tx, ty), caption, fill="black", font=font)
        col_x += BLOCK + GAP_X

    y = MARGIN + CAPTION_H
    for idx, r in enumerate(rows, start=1):
        # Left-side row labels: object1 / object2 / object3 ...
        label = f"object{idx}"
        lb = draw.textbbox((0, 0), label, font=row_font)
        ltw = lb[2] - lb[0]
        lth = lb[3] - lb[1]
        lx = MARGIN + (ROW_LABEL_W - ltw) // 2
        ly = y + (BLOCK - lth) // 2
        draw.text((lx, ly), label, fill="black", font=row_font)

        canvas.paste(r, (MARGIN + ROW_LABEL_W, y))
        y += BLOCK + GAP_Y

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, quality=95)
    print(f"Saved: {args.output}")
    print(f"Saved blocks (no resize): {blocks_dir}")


if __name__ == "__main__":
    main()

