#!/usr/bin/env python3
"""
1) Stitch scene video pairs side-by-side (grouped by scene id prefix).
2) Randomly sample 6 stitched scene-pair videos and merge them synchronously
   into one 3-column x 2-row grid video.

Input:
  result_previews/scene_vid_pred_video/*.mp4

Outputs:
  result_previews/scene_vid_pair_video/<scene_id>.mp4
  result_previews/scene_vid_pair_video/random6_scenes_3x2_grid.mp4
  result_previews/scene_vid_pair_video/random6_scenes_3x2_manifest.txt
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_INPUT_ROOT = _REPO_ROOT / "result_previews/scene_vid_pred_video"
_DEFAULT_OUTPUT_ROOT = _REPO_ROOT / "result_previews/scene_vid_pair_video"

_ITER_RE = re.compile(r"^(?P<base>.+)_iter_\d+$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root", type=Path, default=_DEFAULT_INPUT_ROOT)
    p.add_argument("--output-root", type=Path, default=_DEFAULT_OUTPUT_ROOT)
    p.add_argument("--fps", type=float, default=24.0, help="Output fps (default: 24)")
    p.add_argument(
        "--pair-width",
        type=int,
        default=1024,
        help="Width of each pair tile in final 3x2 merged video (default: 1024)",
    )
    p.add_argument(
        "--pair-height",
        type=int,
        default=512,
        help="Height of each pair tile in final 3x2 merged video (default: 512)",
    )
    p.add_argument(
        "--grid-gap",
        type=int,
        default=24,
        help="White gap (pixels) between scene-pair tiles in 3x2 merged video (default: 24).",
    )
    p.add_argument("--seed", type=int, default=None, help="Random seed for sampling 6 scenes.")
    p.add_argument(
        "--min-duration-sec",
        type=float,
        default=4.0,
        help="Only sample pair videos whose duration is >= this threshold (default: 4.0).",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def run_ffmpeg(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def probe_duration_sec(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        str(path),
    ]
    out = subprocess.check_output(cmd).decode("utf-8")
    data = json.loads(out)
    return float(data["format"]["duration"])


def group_by_scene_id(input_root: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for p in sorted(input_root.glob("*.mp4")):
        m = _ITER_RE.match(p.stem)
        if not m:
            continue
        scene_id = m.group("base").split("_")[0]
        groups.setdefault(scene_id, []).append(p)
    return groups


def pick_pairs(groups: dict[str, list[Path]]) -> dict[str, tuple[Path, Path]]:
    pairs: dict[str, tuple[Path, Path]] = {}
    for scene_id, vids in sorted(groups.items()):
        if len(vids) < 2:
            continue
        vids_sorted = sorted(vids)
        pairs[scene_id] = (vids_sorted[0], vids_sorted[1])
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


def make_grid_3x2(
    pair_videos: list[Path],
    out_path: Path,
    fps: float,
    pair_width: int,
    pair_height: int,
    grid_gap: int,
    overwrite: bool,
) -> None:
    if len(pair_videos) != 6:
        raise ValueError(f"Need exactly 6 pair videos, got {len(pair_videos)}")

    input_args: list[str] = []
    for p in pair_videos:
        input_args.extend(["-i", str(p)])

    prep = []
    for i in range(6):
        prep.append(
            f"[{i}:v]setpts=PTS-STARTPTS,fps={fps},scale={pair_width}:{pair_height}[v{i}]"
        )
    layout = [
        "0_0",
        f"{pair_width + grid_gap}_0",
        f"{2 * (pair_width + grid_gap)}_0",
        f"0_{pair_height + grid_gap}",
        f"{pair_width + grid_gap}_{pair_height + grid_gap}",
        f"{2 * (pair_width + grid_gap)}_{pair_height + grid_gap}",
    ]
    xstack_inputs = "".join(f"[v{i}]" for i in range(6))
    filter_complex = (
        ";".join(prep)
        + ";"
        + f"{xstack_inputs}xstack=inputs=6:layout={'|'.join(layout)}:fill=white:shortest=1[v]"
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
    in_root = args.input_root.resolve()
    out_root = args.output_root.resolve()

    if not in_root.is_dir():
        raise SystemExit(f"Input root not found: {in_root}")
    if args.fps <= 0:
        raise SystemExit(f"Invalid --fps={args.fps}; must be > 0")
    if args.pair_width <= 0 or args.pair_height <= 0:
        raise SystemExit("--pair-width and --pair-height must be > 0")
    if args.grid_gap < 0:
        raise SystemExit("--grid-gap must be >= 0")
    if args.min_duration_sec <= 0:
        raise SystemExit("--min-duration-sec must be > 0")

    groups = group_by_scene_id(in_root)
    pairs = pick_pairs(groups)
    if len(pairs) < 6:
        raise SystemExit(f"Need at least 6 scene pairs, found {len(pairs)}")

    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)

    # 1) Create all pair videos first.
    pair_count = 0
    for scene_id, (v1, v2) in sorted(pairs.items()):
        out_path = out_root / f"{scene_id}.mp4"
        if args.dry_run:
            print(f"[dry-run] pair {scene_id}: {v1.name} + {v2.name} -> {out_path.name}")
            pair_count += 1
            continue
        if out_path.exists() and not args.overwrite:
            print(f"Skip existing pair: {out_path.name}")
            pair_count += 1
            continue
        make_pair_video(v1, v2, out_path, args.fps, args.overwrite)
        print(f"Wrote pair video: {out_path}")
        pair_count += 1

    # 2) Randomly sample 6 scene ids and merge their pair videos into 3x2.
    rng = random.Random(args.seed)
    eligible_scene_ids: list[str] = []
    skipped_short: list[tuple[str, float]] = []
    for scene_id in sorted(pairs.keys()):
        pair_path = out_root / f"{scene_id}.mp4"
        if args.dry_run:
            # In dry-run we still probe existing pair videos to know eligibility.
            # If pair file does not exist yet in dry-run, skip duration filtering for it.
            if not pair_path.is_file():
                eligible_scene_ids.append(scene_id)
                continue
        dur = probe_duration_sec(pair_path)
        if dur + 1e-6 >= args.min_duration_sec:
            eligible_scene_ids.append(scene_id)
        else:
            skipped_short.append((scene_id, dur))

    if len(eligible_scene_ids) < 6:
        raise SystemExit(
            f"Need at least 6 eligible pair videos with duration >= {args.min_duration_sec}s, "
            f"found {len(eligible_scene_ids)}"
        )

    sampled_scene_ids = sorted(rng.sample(eligible_scene_ids, 6))
    sampled_pair_paths = [out_root / f"{scene_id}.mp4" for scene_id in sampled_scene_ids]

    grid_path = out_root / "random6_scenes_3x2_grid.mp4"
    manifest = out_root / "random6_scenes_3x2_manifest.txt"

    if args.dry_run:
        if skipped_short:
            print("\n[dry-run] short pair videos excluded:")
            for scene_id, dur in skipped_short:
                print(f"  - {scene_id}: {dur:.3f}s < {args.min_duration_sec:g}s")
        print("\n[dry-run] sampled scene ids:")
        for scene_id in sampled_scene_ids:
            a, b = pairs[scene_id]
            print(f"  - {scene_id}: {a.name} | {b.name}")
        print(f"[dry-run] grid -> {grid_path}")
        print(f"[dry-run] manifest -> {manifest}")
        print(f"\nPair videos planned: {pair_count}")
        return

    if grid_path.exists() and not args.overwrite:
        print(f"Skip existing grid: {grid_path}")
    else:
        make_grid_3x2(
            pair_videos=sampled_pair_paths,
            out_path=grid_path,
            fps=args.fps,
            pair_width=args.pair_width,
            pair_height=args.pair_height,
            grid_gap=args.grid_gap,
            overwrite=args.overwrite,
        )
        print(f"Wrote merged 3x2 grid video: {grid_path}")

    lines = [
        f"seed={args.seed}\n",
        f"fps={args.fps}\n",
        f"pair_width={args.pair_width}\n",
        f"pair_height={args.pair_height}\n",
        f"grid_gap={args.grid_gap}\n",
        f"min_duration_sec={args.min_duration_sec}\n",
        "sampled_scene_ids:\n",
    ]
    for scene_id in sampled_scene_ids:
        a, b = pairs[scene_id]
        lines.append(f"- {scene_id}: {a.name} | {b.name}\n")
    manifest.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote manifest: {manifest}")
    print(f"\nPair videos handled: {pair_count}")


if __name__ == "__main__":
    main()
