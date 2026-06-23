#!/usr/bin/env python3
"""
Reorganize scene video previews when supervision layout is:
  input | gt | pred  (repeated per frame), with no separate input_<scene>.jpg.

Input layout:
  <base>/iter_XXXXXXXX/
    supervision_<scene>_part000_frames0000-0024.jpg
    supervision_<scene>_part001_frames0025-0049.jpg
    (or single supervision_<scene>.jpg fallback)

Output:
  <base>_flattened/<scene>_iter_XXXXXXXX.jpg
    3-row montage: input strip, gt strip, pred strip.

  <base>_single_image/<scene>/iter_XXXXXXXX/
    frame_0000/view_input.jpg, view_gt.jpg, view_pred.jpg
    frame_0001/...
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_BASE = _REPO_ROOT / "result_previews/polyhaven_all"

_PART_RE = re.compile(
    r"^supervision_(?P<stem>.+)_part(?P<part>\d{3})_frames(?P<start>\d{4})-(?P<end>\d{4})\.(jpg|jpeg|png)$"
)
_SINGLE_RE = re.compile(r"^supervision_(?P<stem>.+)\.(jpg|jpeg|png)$")


class SkipScene(Exception):
    pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base",
        type=Path,
        default=_DEFAULT_BASE,
        help=f"Directory containing iter_* folders (default: {_DEFAULT_BASE})",
    )
    p.add_argument(
        "--iters",
        nargs="*",
        default=None,
        metavar="ITER",
        help="Only process selected iter folder names.",
    )
    p.add_argument("--no-flattened", action="store_true", help="Skip flattened montage output.")
    p.add_argument("--no-single-image", action="store_true", help="Skip single-image split output.")
    p.add_argument("--dry-run", action="store_true", help="Print planned operations only.")
    return p.parse_args()


def safe_name(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in s)


def discover_stems(iter_dir: Path) -> list[str]:
    stems: set[str] = set()
    for p in iter_dir.iterdir():
        if not p.is_file():
            continue
        m = _PART_RE.match(p.name)
        if m:
            stems.add(m.group("stem"))
            continue
        m = _SINGLE_RE.match(p.name)
        if m:
            stems.add(m.group("stem"))
    return sorted(stems)


def open_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def supervision_parts(iter_dir: Path, stem: str) -> list[tuple[int, int, Path]]:
    parts: list[tuple[int, int, int, Path]] = []
    single: Path | None = None
    for p in sorted(iter_dir.iterdir()):
        if not p.is_file():
            continue
        m = _PART_RE.match(p.name)
        if m and m.group("stem") == stem:
            parts.append((int(m.group("part")), int(m.group("start")), int(m.group("end")), p))
            continue
        m = _SINGLE_RE.match(p.name)
        if m and m.group("stem") == stem:
            single = p
    if parts:
        parts.sort(key=lambda x: x[0])
        expect_start = 0
        for idx, (part_idx, start, end, _) in enumerate(parts):
            if part_idx != idx:
                raise SkipScene(f"{stem}: part index mismatch {part_idx} != {idx}")
            if start != expect_start:
                raise SkipScene(f"{stem}: frame gap start {start} != expected {expect_start}")
            if end < start:
                raise SkipScene(f"{stem}: invalid frame range {start}-{end}")
            expect_start = end + 1
        return [(s, e, p) for (_, s, e, p) in parts]
    if single is not None:
        return [(0, -1, single)]
    raise SkipScene(f"{stem}: missing supervision image")


def split_supervision_chunk(path: Path) -> list[tuple[Image.Image, Image.Image, Image.Image]]:
    """
    Split one chunk where each frame contributes 3 square tiles: input|gt|pred.
    """
    im = open_rgb(path)
    w, h = im.size
    tile = h
    if tile <= 0 or w % tile != 0:
        raise SkipScene(f"{path.name}: width {w} not divisible by tile {tile}")
    n_tiles = w // tile
    if n_tiles % 3 != 0:
        raise SkipScene(f"{path.name}: tile count {n_tiles} is not a multiple of 3 (input|gt|pred)")
    out: list[tuple[Image.Image, Image.Image, Image.Image]] = []
    for i in range(n_tiles // 3):
        x0 = i * 3 * tile
        inp = im.crop((x0, 0, x0 + tile, h))
        gt = im.crop((x0 + tile, 0, x0 + 2 * tile, h))
        pred = im.crop((x0 + 2 * tile, 0, x0 + 3 * tile, h))
        out.append((inp, gt, pred))
    return out


def load_frames(parts: list[tuple[int, int, Path]]) -> list[tuple[Image.Image, Image.Image, Image.Image]]:
    frames: list[tuple[Image.Image, Image.Image, Image.Image]] = []
    for start, end, path in parts:
        chunk = split_supervision_chunk(path)
        if end >= 0:
            expected = end - start + 1
            if len(chunk) != expected:
                raise SkipScene(
                    f"{path.name}: frame count {len(chunk)} != expected {expected} from filename"
                )
        frames.extend(chunk)
    return frames


def hstack(images: list[Image.Image]) -> Image.Image:
    if not images:
        raise SkipScene("hstack empty images")
    h = images[0].height
    w = sum(im.width for im in images)
    out = Image.new("RGB", (w, h))
    x = 0
    for im in images:
        out.paste(im, (x, 0))
        x += im.width
    return out


def vstack(images: list[Image.Image]) -> Image.Image:
    if not images:
        raise SkipScene("vstack empty images")
    w = max(im.width for im in images)
    h = sum(im.height for im in images)
    out = Image.new("RGB", (w, h))
    y = 0
    for im in images:
        out.paste(im, ((w - im.width) // 2, y))
        y += im.height
    return out


def process_scene(
    iter_dir: Path,
    stem: str,
    out_flat_dir: Path,
    out_single_root: Path,
    do_flattened: bool,
    do_single: bool,
    dry_run: bool,
) -> bool:
    try:
        parts = supervision_parts(iter_dir, stem)
        frames = load_frames(parts)
    except SkipScene as exc:
        print(f"  {iter_dir.name}/{stem}: {exc}; skip")
        return False

    if not frames:
        print(f"  {iter_dir.name}/{stem}: zero frames; skip")
        return False

    if dry_run:
        print(
            f"  [dry-run] {iter_dir.name}/{stem}: frames={len(frames)} -> "
            f"flattened={do_flattened}, single={do_single}"
        )
        return True

    if do_flattened:
        row_input = hstack([fr[0] for fr in frames])
        row_gt = hstack([fr[1] for fr in frames])
        row_pred = hstack([fr[2] for fr in frames])
        montage = vstack([row_input, row_gt, row_pred])
        out_flat = out_flat_dir / f"{safe_name(stem)}_{iter_dir.name}.jpg"
        montage.save(out_flat, quality=95)

    if do_single:
        out_iter_dir = out_single_root / safe_name(stem) / iter_dir.name
        out_iter_dir.mkdir(parents=True, exist_ok=True)
        for i, (inp, gt, pred) in enumerate(frames):
            fd = out_iter_dir / f"frame_{i:04d}"
            fd.mkdir(parents=True, exist_ok=True)
            inp.save(fd / "view_input.jpg", quality=95)
            gt.save(fd / "view_gt.jpg", quality=95)
            pred.save(fd / "view_pred.jpg", quality=95)

    print(f"  {iter_dir.name}/{stem}: frames={len(frames)} done")
    return True


def main() -> None:
    args = parse_args()
    base = args.base.resolve()
    if not base.is_dir():
        raise SystemExit(f"Not a directory: {base}")

    do_flattened = not args.no_flattened
    do_single = not args.no_single_image
    if not do_flattened and not do_single:
        raise SystemExit("Nothing to do: both flattened and single outputs disabled")

    out_flat_dir = base.parent / f"{base.name}_flattened"
    out_single_root = base.parent / f"{base.name}_single_image"
    if not args.dry_run:
        if do_flattened:
            out_flat_dir.mkdir(parents=True, exist_ok=True)
        if do_single:
            out_single_root.mkdir(parents=True, exist_ok=True)

    iter_dirs = sorted(d for d in base.iterdir() if d.is_dir() and d.name.startswith("iter_"))
    if args.iters:
        wanted = set(args.iters)
        iter_dirs = [d for d in iter_dirs if d.name in wanted]
    if not iter_dirs:
        raise SystemExit(f"No iter_* directories under {base}")

    ok_count = 0
    skip_count = 0
    for iter_dir in iter_dirs:
        stems = discover_stems(iter_dir)
        if not stems:
            print(f"{iter_dir.name}: no supervision files; skip")
            skip_count += 1
            continue
        for stem in stems:
            ok = process_scene(
                iter_dir=iter_dir,
                stem=stem,
                out_flat_dir=out_flat_dir,
                out_single_root=out_single_root,
                do_flattened=do_flattened,
                do_single=do_single,
                dry_run=args.dry_run,
            )
            ok_count += int(ok)
            skip_count += int(not ok)

    print(f"\nProcessed: {ok_count}, Skipped: {skip_count}")
    if not args.dry_run:
        if do_flattened:
            print(f"Flattened output: {out_flat_dir}")
        if do_single:
            print(f"Single-image output: {out_single_root}")
    print("Done.")


if __name__ == "__main__":
    main()
