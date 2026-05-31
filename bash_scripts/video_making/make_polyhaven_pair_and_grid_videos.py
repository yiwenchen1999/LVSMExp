#!/usr/bin/env python3
"""
Build side-by-side object pairs and one random multi-video grid.

Input:
  result_previews/videos/polyhaven_obj_rot_video/*.mp4

Pair outputs:
  result_previews/videos/polyhaven_obj_rot_video_pairs/<obj_id>.mp4
  (for each object with at least two videos)

Big grid output (default 12 objects => 24 videos):
  result_previews/videos/polyhaven_obj_rot_video_pairs/random12_objects_24videos_grid.mp4
  plus manifest:
  result_previews/videos/polyhaven_obj_rot_video_pairs/random12_objects_24videos_manifest.txt
"""

from __future__ import annotations

import argparse
import random
import re
import subprocess
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_INPUT_ROOT = _REPO_ROOT / "result_previews/videos/polyhaven_obj_rot_video"
_DEFAULT_OUTPUT_ROOT = _REPO_ROOT / "result_previews/videos/polyhaven_obj_rot_video_pairs"

_ITER_RE = re.compile(r"^(?P<base>.+)_iter_\d+$")
_OBJ_RE = re.compile(r"^(?P<obj>.+?)_(?:white_)?env_\d+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=_DEFAULT_INPUT_ROOT,
        help=f"Directory containing object rotation videos (default: {_DEFAULT_INPUT_ROOT})",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_DEFAULT_OUTPUT_ROOT,
        help=f"Directory for pair videos and final grid video (default: {_DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument("--fps", type=float, default=24.0, help="Output fps (default: 24)")
    parser.add_argument(
        "--tile-size",
        type=int,
        default=512,
        help="Cell size for final 4x4 grid video (default: 512)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for selecting objects (default: random each run)",
    )
    parser.add_argument(
        "--num-objects",
        type=int,
        default=12,
        help="How many objects to randomly sample for the final merged video (default: 12)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print planned operations.",
    )
    return parser.parse_args()


def run_ffmpeg(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def object_id_from_video(path: Path) -> str | None:
    stem = path.stem
    m = _ITER_RE.match(stem)
    if not m:
        return None
    base = m.group("base")
    m2 = _OBJ_RE.match(base)
    if not m2:
        return None
    return m2.group("obj")


def group_videos_by_object(input_root: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for p in sorted(input_root.glob("*.mp4")):
        obj = object_id_from_video(p)
        if obj is None:
            continue
        groups.setdefault(obj, []).append(p)
    return groups


def pick_video_pairs(groups: dict[str, list[Path]]) -> dict[str, tuple[Path, Path]]:
    pairs: dict[str, tuple[Path, Path]] = {}
    for obj, vids in sorted(groups.items()):
        if len(vids) < 2:
            continue
        vids_sorted = sorted(vids)
        pairs[obj] = (vids_sorted[0], vids_sorted[1])
    return pairs


def make_pair_video(v1: Path, v2: Path, out_path: Path, fps: float, overwrite: bool) -> None:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(v1),
        "-i",
        str(v2),
        "-filter_complex",
        (
            f"[0:v]setpts=PTS-STARTPTS,fps={fps}[a];"
            f"[1:v]setpts=PTS-STARTPTS,fps={fps}[b];"
            "[a][b]hstack=inputs=2:shortest=1[v]"
        ),
        "-map",
        "[v]",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    run_ffmpeg(cmd)


def choose_grid_shape(n_videos: int) -> tuple[int, int]:
    """Pick rows/cols factorization close to square (cols >= rows)."""
    if n_videos <= 0:
        raise ValueError(f"Invalid video count: {n_videos}")
    best_rows, best_cols = 1, n_videos
    best_gap = best_cols - best_rows
    r = 1
    while r * r <= n_videos:
        if n_videos % r == 0:
            c = n_videos // r
            gap = c - r
            if gap < best_gap:
                best_rows, best_cols, best_gap = r, c, gap
        r += 1
    return best_rows, best_cols


def make_grid_video(
    videos: list[Path],
    out_path: Path,
    fps: float,
    tile_size: int,
    overwrite: bool,
) -> None:
    n_videos = len(videos)
    if n_videos <= 0:
        raise ValueError("Need at least 1 video for merged output")
    rows, cols = choose_grid_shape(n_videos)

    input_args: list[str] = []
    for v in videos:
        input_args.extend(["-i", str(v)])

    prep = []
    for i in range(n_videos):
        prep.append(
            f"[{i}:v]setpts=PTS-STARTPTS,fps={fps},scale={tile_size}:{tile_size}[v{i}]"
        )
    layout = []
    for r in range(rows):
        for c in range(cols):
            layout.append(f"{c * tile_size}_{r * tile_size}")
    xstack_inputs = "".join(f"[v{i}]" for i in range(n_videos))
    filter_complex = (
        ";".join(prep)
        + ";"
        + f"{xstack_inputs}xstack=inputs={n_videos}:layout={'|'.join(layout)}:fill=black:shortest=1[v]"
    )

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        *input_args,
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    run_ffmpeg(cmd)


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()

    if not input_root.is_dir():
        raise SystemExit(f"Input directory not found: {input_root}")
    if args.fps <= 0:
        raise SystemExit(f"Invalid --fps={args.fps}; must be > 0")
    if args.tile_size <= 0:
        raise SystemExit(f"Invalid --tile-size={args.tile_size}; must be > 0")
    if args.num_objects <= 0:
        raise SystemExit(f"Invalid --num-objects={args.num_objects}; must be > 0")

    groups = group_videos_by_object(input_root)
    pairs = pick_video_pairs(groups)
    if len(pairs) < args.num_objects:
        raise SystemExit(f"Need at least {args.num_objects} object pairs, found {len(pairs)}")

    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    # 1) Build all side-by-side pair videos.
    pair_done = 0
    for obj, (v1, v2) in sorted(pairs.items()):
        out_path = output_root / f"{obj}.mp4"
        if args.dry_run:
            print(f"[dry-run] pair {obj}: {v1.name} + {v2.name} -> {out_path.name}")
            pair_done += 1
            continue
        if out_path.exists() and not args.overwrite:
            print(f"Skip existing pair video: {out_path.name}")
            pair_done += 1
            continue
        make_pair_video(v1, v2, out_path, args.fps, args.overwrite)
        print(f"Wrote pair video: {out_path}")
        pair_done += 1

    # 2) Randomly sample objects and build one synchronized merged grid.
    rng = random.Random(args.seed)
    sampled_objs = sorted(rng.sample(list(pairs.keys()), args.num_objects))
    sampled_videos: list[Path] = []
    for obj in sampled_objs:
        sampled_videos.extend(list(pairs[obj]))
    n_videos = len(sampled_videos)

    grid_out = output_root / f"random{args.num_objects}_objects_{n_videos}videos_grid.mp4"
    manifest = output_root / f"random{args.num_objects}_objects_{n_videos}videos_manifest.txt"

    if args.dry_run:
        print("\n[dry-run] sampled objects:")
        for obj in sampled_objs:
            a, b = pairs[obj]
            print(f"  - {obj}: {a.name} | {b.name}")
        print(f"[dry-run] final grid -> {grid_out}")
        print(f"[dry-run] manifest  -> {manifest}")
        print(f"\nPair videos planned: {pair_done}")
        return

    if grid_out.exists() and not args.overwrite:
        print(f"Skip existing grid video: {grid_out}")
    else:
        make_grid_video(sampled_videos, grid_out, args.fps, args.tile_size, args.overwrite)
        print(f"Wrote grid video: {grid_out}")

    lines = [
        f"seed={args.seed}\n",
        f"fps={args.fps}\n",
        f"tile_size={args.tile_size}\n",
        f"num_objects={args.num_objects}\n",
        f"num_videos={n_videos}\n",
        "sampled_objects:\n",
    ]
    for obj in sampled_objs:
        a, b = pairs[obj]
        lines.append(f"- {obj}: {a.name} | {b.name}\n")
    manifest.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote manifest: {manifest}")
    print(f"\nPair videos handled: {pair_done}")


if __name__ == "__main__":
    main()
