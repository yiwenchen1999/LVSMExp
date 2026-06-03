#!/usr/bin/env python3
"""
Reorganize failurecases previews where supervision layout is:
  gt_prelit | gt_relit | pred_relit  (repeated per frame).

Input layout:
  result_previews/videos/failurecases/iter_XXXXXXXX/
    input_<scene>.jpg
    supervision_<scene>_part000_frames0000-0024.jpg
    supervision_<scene>_part001_frames0025-0049.jpg
    (or single supervision_<scene>.jpg fallback)
    camera_poses_<scene>.json (optional)

Output:
  result_previews/videos/failurecases_flattened/<scene>_iter_XXXXXXXX.jpg
    4-row montage: input strip (scaled), gt_prelit strip, gt_relit strip, pred_relit strip.

  result_previews/videos/failurecases_single_image/<scene>/iter_XXXXXXXX/
    input_view_00..N-1.jpg
    camera_poses.json + camera_context_view_XX.json (if source exists)
    frame_0000/view_gt_prelit.jpg, view_gt_relit.jpg, view_pred_relit.jpg,
               camera_target.json (if source exists)
    frame_0001/...
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from PIL import Image


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_BASE = _REPO_ROOT / "result_previews/videos/failurecases"

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
        "--num-input-views",
        type=int,
        default=10,
        help="Input strip view count. Use 0 to infer from width/height ratio (default: 10).",
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


def split_input(path: Path, n_views: int) -> list[Image.Image]:
    im = open_rgb(path)
    w, h = im.size
    n = n_views
    if n == 0:
        if h <= 0:
            raise SkipScene(f"{path.name}: invalid height {h}")
        n = int(round(w / h))
    if n <= 0:
        raise SkipScene(f"{path.name}: invalid input view count {n}")
    if w % n != 0:
        raise SkipScene(f"{path.name}: width {w} not divisible by n_views {n}")
    tw = w // n
    return [im.crop((i * tw, 0, (i + 1) * tw, h)) for i in range(n)]


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
    Split chunk where each frame contributes 3 square tiles:
      gt_prelit | gt_relit | pred_relit
    """
    im = open_rgb(path)
    w, h = im.size
    tile = h
    if tile <= 0 or w % tile != 0:
        raise SkipScene(f"{path.name}: width {w} not divisible by tile {tile}")
    n_tiles = w // tile
    if n_tiles % 3 != 0:
        raise SkipScene(
            f"{path.name}: tile count {n_tiles} is not divisible by 3 "
            "(expected gt_prelit|gt_relit|pred_relit triples)"
        )
    out: list[tuple[Image.Image, Image.Image, Image.Image]] = []
    for i in range(n_tiles // 3):
        x0 = i * 3 * tile
        gt_prelit = im.crop((x0, 0, x0 + tile, h))
        gt_relit = im.crop((x0 + tile, 0, x0 + 2 * tile, h))
        pred_relit = im.crop((x0 + 2 * tile, 0, x0 + 3 * tile, h))
        out.append((gt_prelit, gt_relit, pred_relit))
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


def write_camera_jsons(iter_dir: Path, stem: str, out_iter_dir: Path, frame_dirs: list[Path]) -> None:
    src = iter_dir / f"camera_poses_{stem}.json"
    if not src.is_file():
        return
    try:
        data = json.loads(src.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"  Warning: cannot parse {src.name}: {exc}")
        return
    (out_iter_dir / "camera_poses.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

    scene_name = str(data.get("scene_name", ""))
    for i, entry in enumerate(data.get("context_views", []), start=1):
        payload = {**entry, "scene_name": scene_name, "role": "context"}
        (out_iter_dir / f"camera_context_view_{i:02d}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    target_views = data.get("target_views", [])
    for i, entry in enumerate(target_views):
        if i >= len(frame_dirs):
            break
        payload = {**entry, "scene_name": scene_name, "role": "target"}
        (frame_dirs[i] / "camera_target.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )


def process_scene(
    iter_dir: Path,
    stem: str,
    out_flat_dir: Path,
    out_single_root: Path,
    n_input_views: int,
    do_flattened: bool,
    do_single: bool,
    dry_run: bool,
) -> bool:
    input_path = None
    for p in (iter_dir / f"input_{stem}.jpg", iter_dir / f"input_{stem}.png"):
        if p.is_file():
            input_path = p
            break
    if input_path is None:
        print(f"  {iter_dir.name}/{stem}: missing input strip; skip")
        return False

    try:
        input_tiles = split_input(input_path, n_input_views)
        parts = supervision_parts(iter_dir, stem)
        frames = load_frames(parts)
    except SkipScene as exc:
        print(f"  {iter_dir.name}/{stem}: {exc}; skip")
        return False

    if not frames:
        print(f"  {iter_dir.name}/{stem}: zero reconstructed frames; skip")
        return False

    if dry_run:
        print(
            f"  [dry-run] {iter_dir.name}/{stem}: input_views={len(input_tiles)}, "
            f"frames={len(frames)} -> flattened={do_flattened}, single={do_single}"
        )
        return True

    if do_flattened:
        row_input = hstack(input_tiles)
        row_gt_prelit = hstack([fr[0] for fr in frames])
        row_gt_relit = hstack([fr[1] for fr in frames])
        row_pred_relit = hstack([fr[2] for fr in frames])
        if row_input.width != row_gt_prelit.width:
            row_input = row_input.resize((row_gt_prelit.width, row_input.height), Image.Resampling.LANCZOS)
        montage = vstack([row_input, row_gt_prelit, row_gt_relit, row_pred_relit])
        out_flat = out_flat_dir / f"{safe_name(stem)}_{iter_dir.name}.jpg"
        montage.save(out_flat, quality=95)

    if do_single:
        out_iter_dir = out_single_root / safe_name(stem) / iter_dir.name
        out_iter_dir.mkdir(parents=True, exist_ok=True)
        for i, tile in enumerate(input_tiles):
            tile.save(out_iter_dir / f"input_view_{i:02d}.jpg", quality=95)

        frame_dirs: list[Path] = []
        for i, (gt_prelit, gt_relit, pred_relit) in enumerate(frames):
            fd = out_iter_dir / f"frame_{i:04d}"
            fd.mkdir(parents=True, exist_ok=True)
            frame_dirs.append(fd)
            gt_prelit.save(fd / "view_gt_prelit.jpg", quality=95)
            gt_relit.save(fd / "view_gt_relit.jpg", quality=95)
            pred_relit.save(fd / "view_pred_relit.jpg", quality=95)

        write_camera_jsons(iter_dir, stem, out_iter_dir, frame_dirs)

    print(f"  {iter_dir.name}/{stem}: input_views={len(input_tiles)}, frames={len(frames)} done")
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
                n_input_views=args.num_input_views,
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
