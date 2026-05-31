#!/usr/bin/env python3
"""
Reorganize polyhaven video previews under result_previews/videos/polyhaven.

Each iter_* directory holds one scene rendered as a long video sequence. The
supervision strip produced by utils.metric_utils.visualize_intermediate_results
is too wide for a single JPEG, so it is saved as multiple ordered chunks:

  supervision_<scene>_part000_frames0000-0024.jpg
  supervision_<scene>_part001_frames0025-0049.jpg
  ...
  (or a single supervision_<scene>.jpg when the sequence is short enough)

Per-iter files:
  input_<scene>.jpg                 — N context views stitched horizontally
  supervision_<scene>_part*.jpg     — F frames, each (gt | relit_gt | relit_pred)
  context_envldr_<scene>.png        — N context env maps (optional)
  context_envhdr_<scene>.png        — N context env maps (optional)
  target_envldr_<scene>.png         — F target env maps (optional)
  target_envhdr_<scene>.png         — F target env maps (optional)
  camera_poses_<scene>.json         — context_views + target_views (optional)

Outputs (default --base result_previews/videos/polyhaven):

  (1) <base>_flattened / <scene>_iter_XXXXXXXX.jpg
      One 4-row montage per (scene, iter): row1 input strip (scaled),
      row2 gt, row3 relit_gt, row4 relit_pred — across all reconstructed frames.

  (2) <base>_single_image / <scene> / iter_XXXXXXXX /
        input_view_00..(N-1).jpg
        context_envldr_view_01..N.jpg / context_envhdr_view_01..N.png (when present)
        camera_poses.json + camera_context_view_XX.json (when present)
        frame_0000 / view_gt.jpg, view_relit_gt.jpg, view_relit_pred.jpg,
                     target_envldr.jpg, target_envhdr.png (when present),
                     camera_target.json (when present)
        frame_0001 / ...

Incomplete iters (missing input, missing/garbled supervision chunks,
non-contiguous frame ranges, indivisible widths) print a warning and are
skipped without aborting the run.

Usage:
  bash bash_scripts/video_making/reorganize_polyhaven_video_previews.sh --dry-run
  python bash_scripts/video_making/reorganize_polyhaven_video_previews.py \
    --base result_previews/videos/polyhaven
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_BASE = _REPO_ROOT / "result_previews/videos/polyhaven"

# supervision_<stem>_part<NNN>_frames<SSSS>-<EEEE>.<ext>
_PART_RE = re.compile(
    r"^supervision_(?P<stem>.+)_part(?P<part>\d{3})_frames(?P<start>\d{4})-(?P<end>\d{4})\.(?P<ext>jpg|jpeg|png)$"
)
# supervision_<stem>.<ext>  (single-chunk fallback)
_SINGLE_RE = re.compile(r"^supervision_(?P<stem>.+)\.(?P<ext>jpg|jpeg|png)$")

# Each target frame contributes 3 tiles: gt | relit_gt | relit_pred.
_TILES_PER_FRAME = 3


class SkipScene(Exception):
    """Raised to skip a single (iter, scene) without aborting the run."""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--base",
        type=Path,
        default=_DEFAULT_BASE,
        help=f"Directory containing iter_* folders (default: {_DEFAULT_BASE})",
    )
    p.add_argument(
        "--num-input-views",
        type=int,
        default=0,
        metavar="N",
        help="Number of context views in the input strip. 0 = infer from aspect ratio (default: 0).",
    )
    p.add_argument(
        "--iters",
        nargs="*",
        default=None,
        metavar="ITER",
        help="Restrict to these iter_* folder names (default: all).",
    )
    p.add_argument(
        "--no-flattened",
        action="store_true",
        help="Skip the flattened montage output.",
    )
    p.add_argument(
        "--no-single-image",
        action="store_true",
        help="Skip the per-frame single_image output.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned outputs only; write nothing.",
    )
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Parsing / frame-sequence reconstruction
# --------------------------------------------------------------------------- #

def discover_scene_stems(iter_dir: Path) -> list[str]:
    """All scene stems that have at least one supervision image in this iter."""
    stems: set[str] = set()
    for p in sorted(iter_dir.iterdir()):
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


def ordered_supervision_parts(iter_dir: Path, stem: str) -> list[tuple[int, int, Path]]:
    """
    Return supervision chunks for `stem` as (start_frame, end_frame_inclusive, path),
    ordered by part index, with contiguity validated. Raises SkipScene on problems.

    A single un-chunked supervision_<stem>.<ext> is treated as one chunk whose
    frame span is resolved later from the image width.
    """
    parts: list[tuple[int, int, int, Path]] = []  # (part_idx, start, end, path)
    single: Path | None = None
    for p in sorted(iter_dir.iterdir()):
        if not p.is_file():
            continue
        m = _PART_RE.match(p.name)
        if m and m.group("stem") == stem:
            parts.append(
                (int(m.group("part")), int(m.group("start")), int(m.group("end")), p)
            )
            continue
        m = _SINGLE_RE.match(p.name)
        if m and m.group("stem") == stem:
            single = p

    if parts:
        parts.sort(key=lambda t: t[0])
        # Validate part indices are 0..K-1 and frame ranges are contiguous.
        expected_start = 0
        for idx, (part_idx, start, end, path) in enumerate(parts):
            if part_idx != idx:
                raise SkipScene(f"{stem}: non-sequential part index {part_idx} (expected {idx})")
            if start != expected_start:
                raise SkipScene(
                    f"{stem}: frame gap at part{part_idx:03d} (start {start}, expected {expected_start})"
                )
            if end < start:
                raise SkipScene(f"{stem}: bad frame range {start}-{end} at part{part_idx:03d}")
            expected_start = end + 1
        return [(start, end, path) for (_, start, end, path) in parts]

    if single is not None:
        # Span resolved from width downstream; mark with sentinel end=-1.
        return [(0, -1, single)]

    raise SkipScene(f"{stem}: no supervision image found")


def _open_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def split_supervision_chunk(path: Path) -> list[tuple[Image.Image, Image.Image, Image.Image]]:
    """
    Split one supervision chunk into per-frame (gt, relit_gt, relit_pred) tiles.
    Tile is square with side == image height. Raises SkipScene on bad geometry.
    """
    im = _open_rgb(path)
    w, h = im.size
    tile = h
    if tile <= 0 or w % tile != 0:
        raise SkipScene(f"{path.name}: width {w} not divisible by tile height {h}")
    n_tiles = w // tile
    if n_tiles % _TILES_PER_FRAME != 0:
        raise SkipScene(
            f"{path.name}: tile count {n_tiles} not a multiple of {_TILES_PER_FRAME} (gt|relit|pred)"
        )
    frames: list[tuple[Image.Image, Image.Image, Image.Image]] = []
    for f in range(n_tiles // _TILES_PER_FRAME):
        base = f * _TILES_PER_FRAME * tile
        gt = im.crop((base, 0, base + tile, h))
        relit = im.crop((base + tile, 0, base + 2 * tile, h))
        pred = im.crop((base + 2 * tile, 0, base + 3 * tile, h))
        frames.append((gt, relit, pred))
    return frames


def load_all_frames(parts: list[tuple[int, int, Path]]) -> list[tuple[Image.Image, Image.Image, Image.Image]]:
    """Concatenate per-chunk frame tiles in order into a single frame list."""
    frames: list[tuple[Image.Image, Image.Image, Image.Image]] = []
    for start, end, path in parts:
        chunk = split_supervision_chunk(path)
        if end >= 0:
            expected = end - start + 1
            if len(chunk) != expected:
                raise SkipScene(
                    f"{path.name}: holds {len(chunk)} frames but name claims {expected} ({start}-{end})"
                )
        frames.extend(chunk)
    return frames


def split_horizontal_strip(path: Path, n_parts: int, crop_height: int | None = None) -> list[Image.Image]:
    """Split a horizontal strip into n_parts equal-width tiles."""
    if n_parts <= 0:
        raise SkipScene(f"{path.name}: invalid n_parts={n_parts}")
    im = _open_rgb(path)
    w, h = im.size
    if crop_height is not None and h != crop_height:
        im = im.crop((0, 0, w, crop_height))
        w, h = im.size
    if w % n_parts != 0:
        raise SkipScene(f"{path.name}: width {w} not divisible by n_parts {n_parts}")
    tw = w // n_parts
    return [im.crop((i * tw, 0, (i + 1) * tw, h)) for i in range(n_parts)]


def infer_num_input_views(path: Path) -> int:
    im = _open_rgb(path)
    w, h = im.size
    if h <= 0:
        raise SkipScene(f"{path.name}: invalid height {h}")
    n = int(round(w / h))
    if n <= 0:
        raise SkipScene(f"{path.name}: invalid inferred view count {n}")
    return n


def first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.is_file():
            return p
    return None


# --------------------------------------------------------------------------- #
# Montage helpers
# --------------------------------------------------------------------------- #

def hstack(images: list[Image.Image]) -> Image.Image:
    if not images:
        raise SkipScene("hstack: empty image list")
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
        raise SkipScene("vstack: empty image list")
    w = max(im.width for im in images)
    hsum = sum(im.height for im in images)
    out = Image.new("RGB", (w, hsum))
    y = 0
    for im in images:
        out.paste(im, ((w - im.width) // 2, y))
        y += im.height
    return out


def safe_name(stem: str) -> str:
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in stem)


# --------------------------------------------------------------------------- #
# Per-scene processing
# --------------------------------------------------------------------------- #

def resolve_input_path(iter_dir: Path, stem: str) -> Path | None:
    return first_existing([iter_dir / f"input_{stem}.jpg", iter_dir / f"input_{stem}.png"])


def build_flattened(
    input_tiles: list[Image.Image],
    frames: list[tuple[Image.Image, Image.Image, Image.Image]],
) -> Image.Image:
    row_input = hstack(input_tiles)
    row_gt = hstack([f[0] for f in frames])
    row_relit = hstack([f[1] for f in frames])
    row_pred = hstack([f[2] for f in frames])
    if row_input.width != row_gt.width:
        row_input = row_input.resize((row_gt.width, row_input.height), Image.Resampling.LANCZOS)
    return vstack([row_input, row_gt, row_relit, row_pred])


def write_camera_splits(iter_dir: Path, stem: str, out_iter: Path, frame_dirs: list[Path]) -> None:
    src = iter_dir / f"camera_poses_{stem}.json"
    if not src.is_file():
        return
    try:
        data = json.loads(src.read_text(encoding="utf-8"))
    except (ValueError, OSError) as e:
        print(f"  Warning: cannot parse {src.name}: {e}")
        return
    (out_iter / "camera_poses.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    scene_name = str(data.get("scene_name", ""))
    for i, entry in enumerate(data.get("context_views", []), start=1):
        payload = {**entry, "scene_name": scene_name, "role": "context"}
        (out_iter / f"camera_context_view_{i:02d}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
    for i, entry in enumerate(data.get("target_views", [])):
        if i >= len(frame_dirs):
            break
        payload = {**entry, "scene_name": scene_name, "role": "target"}
        (frame_dirs[i] / "camera_target.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )


def write_single_image(
    iter_dir: Path,
    stem: str,
    out_iter: Path,
    input_tiles: list[Image.Image],
    frames: list[tuple[Image.Image, Image.Image, Image.Image]],
) -> None:
    out_iter.mkdir(parents=True, exist_ok=True)

    for i, tile in enumerate(input_tiles):
        tile.save(out_iter / f"input_view_{i:02d}.jpg", quality=95)

    # Context env maps (one per input view).
    n_input = len(input_tiles)
    for prefix, ext in (("context_envldr", ".jpg"), ("context_envhdr", ".png")):
        path = first_existing([iter_dir / f"{prefix}_{stem}.png", iter_dir / f"{prefix}_{stem}.jpg"])
        if path is None:
            continue
        try:
            tiles = split_horizontal_strip(path, n_input)
        except SkipScene as e:
            print(f"  Warning: skip {prefix} for {stem}: {e}")
            continue
        for v, tile in enumerate(tiles, start=1):
            out_p = out_iter / f"{prefix}_view_{v:02d}{ext}"
            tile.save(out_p, compress_level=6) if ext == ".png" else tile.save(out_p, quality=95)

    # Per-frame tiles.
    n_frames = len(frames)
    frame_dirs: list[Path] = []
    for f_idx, (gt, relit, pred) in enumerate(frames):
        fd = out_iter / f"frame_{f_idx:04d}"
        fd.mkdir(parents=True, exist_ok=True)
        frame_dirs.append(fd)
        gt.save(fd / "view_gt.jpg", quality=95)
        relit.save(fd / "view_relit_gt.jpg", quality=95)
        pred.save(fd / "view_relit_pred.jpg", quality=95)

    # Target env maps (one per frame) sliced into each frame dir.
    for prefix, ext in (("target_envldr", ".jpg"), ("target_envhdr", ".png")):
        path = first_existing([iter_dir / f"{prefix}_{stem}.png", iter_dir / f"{prefix}_{stem}.jpg"])
        if path is None:
            continue
        try:
            tiles = split_horizontal_strip(path, n_frames)
        except SkipScene as e:
            print(f"  Warning: skip {prefix} for {stem}: {e}")
            continue
        for f_idx, tile in enumerate(tiles):
            out_p = frame_dirs[f_idx] / f"{prefix.replace('target_', '')}{ext}"
            tile.save(out_p, compress_level=6) if ext == ".png" else tile.save(out_p, quality=95)

    write_camera_splits(iter_dir, stem, out_iter, frame_dirs)


def process_scene(
    iter_dir: Path,
    stem: str,
    flat_dir: Path,
    single_root: Path,
    num_input_views: int,
    do_flat: bool,
    do_single: bool,
    dry_run: bool,
) -> bool:
    """Process one (iter, scene). Returns True on success, False if skipped."""
    input_path = resolve_input_path(iter_dir, stem)
    if input_path is None:
        print(f"  {iter_dir.name}/{stem}: missing input_ strip; skip")
        return False

    try:
        n_input = num_input_views if num_input_views > 0 else infer_num_input_views(input_path)
        input_tiles = split_horizontal_strip(input_path, n_input)
        parts = ordered_supervision_parts(iter_dir, stem)
        frames = load_all_frames(parts)
    except SkipScene as e:
        print(f"  {iter_dir.name}/{stem}: {e}; skip")
        return False

    if not frames:
        print(f"  {iter_dir.name}/{stem}: zero frames reconstructed; skip")
        return False

    if dry_run:
        print(
            f"  [dry-run] {iter_dir.name}/{stem}: input_views={len(input_tiles)}, "
            f"frames={len(frames)} -> flat={do_flat}, single_image={do_single}"
        )
        return True

    if do_flat:
        try:
            montage = build_flattened(input_tiles, frames)
            out_flat = flat_dir / f"{safe_name(stem)}_{iter_dir.name}.jpg"
            montage.save(out_flat, quality=95)
        except (SkipScene, ValueError, OSError) as e:
            print(f"  Warning: flattened montage failed for {iter_dir.name}/{stem}: {e}")

    if do_single:
        write_single_image(iter_dir, stem, single_root / safe_name(stem) / iter_dir.name, input_tiles, frames)

    print(f"  {iter_dir.name}/{stem}: input_views={len(input_tiles)}, frames={len(frames)} done")
    return True


def main() -> None:
    args = parse_args()
    base = args.base.resolve()
    if not base.is_dir():
        raise SystemExit(f"Not a directory: {base}")

    do_flat = not args.no_flattened
    do_single = not args.no_single_image
    if not do_flat and not do_single:
        raise SystemExit("Nothing to do: both --no-flattened and --no-single-image set.")

    flat_dir = base.parent / f"{base.name}_flattened"
    single_root = base.parent / f"{base.name}_single_image"
    if not args.dry_run:
        if do_flat:
            flat_dir.mkdir(parents=True, exist_ok=True)
        if do_single:
            single_root.mkdir(parents=True, exist_ok=True)

    iter_dirs = sorted(d for d in base.iterdir() if d.is_dir() and d.name.startswith("iter_"))
    if args.iters:
        wanted = set(args.iters)
        iter_dirs = [d for d in iter_dirs if d.name in wanted]
    if not iter_dirs:
        raise SystemExit(f"No iter_* directories under {base}")

    n_ok = 0
    n_skip = 0
    for it in iter_dirs:
        stems = discover_scene_stems(it)
        if not stems:
            print(f"{it.name}: no supervision images; skip")
            n_skip += 1
            continue
        for stem in stems:
            ok = process_scene(
                it, stem, flat_dir, single_root,
                args.num_input_views, do_flat, do_single, args.dry_run,
            )
            n_ok += int(ok)
            n_skip += int(not ok)

    print(f"\nProcessed {n_ok} scene(s), skipped {n_skip}.")
    if not args.dry_run:
        if do_flat:
            print(f"Flattened montages: {flat_dir}")
        if do_single:
            print(f"Single-image root:  {single_root}")
    print("Done.")


if __name__ == "__main__":
    main()
