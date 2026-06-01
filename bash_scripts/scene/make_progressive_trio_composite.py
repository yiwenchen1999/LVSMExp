#!/usr/bin/env python3
"""
Build progressive trio composites and merge with existing 6-scene grid.

Pipeline:
1) From result_previews/scene_vid_pred_video/progressive_trios, group videos by scene id,
   each group containing combined_1, combined_2, combined_3.
2) Stitch each group left-to-right in strict order:
      combined_1 | combined_2 | combined_3
3) Merge the 2 stitched trio-group videos into one 2-column x 1-row video,
   with white gap between groups.
4) Put this trio-row video BELOW the existing 6-scene 3x2 grid video, with a larger
   white gap separating top and bottom blocks.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_TRIO_DIR = _REPO_ROOT / "result_previews/scene_vid_pred_video/progressive_trios"
_DEFAULT_OUT_DIR = _REPO_ROOT / "result_previews/scene_vid_pair_video/progressive_trios"
_DEFAULT_TOP_GRID = _REPO_ROOT / "result_previews/scene_vid_pair_video/random6_scenes_3x2_grid.mp4"
_GROUP_RE = re.compile(r"^(?P<scene>[0-9a-fA-F]+)_combined_(?P<idx>[123])_iter_\d+\.mp4$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trio-dir", type=Path, default=_DEFAULT_TRIO_DIR)
    p.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT_DIR)
    p.add_argument("--top-grid-video", type=Path, default=_DEFAULT_TOP_GRID)
    p.add_argument("--fps", type=float, default=24.0, help="Output fps (default: 24)")
    p.add_argument(
        "--pair-gap",
        type=int,
        default=24,
        help="White gap (px) between the 2 trio groups in 2-col row (default: 24).",
    )
    p.add_argument(
        "--block-gap",
        type=int,
        default=72,
        help="White gap (px) between top 6-scene grid and bottom trio row (default: 72).",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def run_ffmpeg(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def ffprobe_wh(path: Path) -> tuple[int, int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        str(path),
    ]
    data = json.loads(subprocess.check_output(cmd).decode("utf-8"))
    st = data["streams"][0]
    return int(st["width"]), int(st["height"])


def discover_trio_groups(trio_dir: Path) -> dict[str, dict[int, Path]]:
    groups: dict[str, dict[int, Path]] = {}
    for p in sorted(trio_dir.glob("*.mp4")):
        m = _GROUP_RE.match(p.name)
        if not m:
            continue
        scene = m.group("scene")
        idx = int(m.group("idx"))
        groups.setdefault(scene, {})[idx] = p
    return groups


def make_one_trio_row(v1: Path, v2: Path, v3: Path, out_path: Path, fps: float, overwrite: bool) -> None:
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
        "-i",
        str(v3),
        "-filter_complex",
        (
            f"[0:v]setpts=PTS-STARTPTS,fps={fps}[a];"
            f"[1:v]setpts=PTS-STARTPTS,fps={fps}[b];"
            f"[2:v]setpts=PTS-STARTPTS,fps={fps}[c];"
            "[a][b][c]hstack=inputs=3:shortest=1[v]"
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


def make_two_trio_row(
    left: Path, right: Path, out_path: Path, fps: float, pair_gap: int, overwrite: bool
) -> None:
    lw, lh = ffprobe_wh(left)
    rw, rh = ffprobe_wh(right)
    # Keep left size; scale right to same height for stable side-by-side composition.
    if rh != lh:
        scaled_rw = max(2, int(round((rw * lh / rh) / 2) * 2))
    else:
        scaled_rw = rw

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(left),
        "-i",
        str(right),
        "-filter_complex",
        (
            f"[0:v]setpts=PTS-STARTPTS,fps={fps}[l];"
            f"[1:v]setpts=PTS-STARTPTS,fps={fps},scale={scaled_rw}:{lh}[r];"
            f"[l][r]xstack=inputs=2:layout=0_0|{lw + pair_gap}_0:fill=white:shortest=1[v]"
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


def merge_top_bottom(
    top: Path,
    bottom: Path,
    out_path: Path,
    fps: float,
    block_gap: int,
    overwrite: bool,
) -> None:
    top_w, top_h = ffprobe_wh(top)
    bot_w, bot_h = ffprobe_wh(bottom)

    # Keep top block unchanged. If bottom block is wider, scale down to fit top width.
    if bot_w > top_w:
        scaled_w = top_w
        scaled_h = max(2, int(round((bot_h * top_w / bot_w) / 2) * 2))
    else:
        scaled_w = bot_w
        scaled_h = bot_h

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(top),
        "-i",
        str(bottom),
        "-filter_complex",
        (
            f"[0:v]setpts=PTS-STARTPTS,fps={fps}[top];"
            f"[1:v]setpts=PTS-STARTPTS,fps={fps},scale={scaled_w}:{scaled_h}[bot_s];"
            f"[bot_s]pad={top_w}:{scaled_h}:({top_w}-iw)/2:0:white[bot];"
            f"[top][bot]xstack=inputs=2:layout=0_0|0_{top_h + block_gap}:fill=white:shortest=1[v]"
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


def main() -> None:
    args = parse_args()
    trio_dir = args.trio_dir.resolve()
    out_dir = args.out_dir.resolve()
    top_grid = args.top_grid_video.resolve()

    if not trio_dir.is_dir():
        raise SystemExit(f"trio directory not found: {trio_dir}")
    if not top_grid.is_file():
        raise SystemExit(f"top grid video not found: {top_grid}")
    if args.fps <= 0:
        raise SystemExit("--fps must be > 0")
    if args.pair_gap < 0 or args.block_gap < 0:
        raise SystemExit("--pair-gap and --block-gap must be >= 0")

    groups = discover_trio_groups(trio_dir)
    full_groups = []
    for scene_id, d in sorted(groups.items()):
        if all(k in d for k in (1, 2, 3)):
            full_groups.append((scene_id, d[1], d[2], d[3]))
    if len(full_groups) < 2:
        raise SystemExit(f"need at least 2 full trio groups, found {len(full_groups)}")
    if len(full_groups) > 2:
        # keep deterministic behavior
        full_groups = full_groups[:2]

    if args.dry_run:
        print("Trio groups used:")
        for scene_id, c1, c2, c3 in full_groups:
            print(f"- {scene_id}: {c1.name} | {c2.name} | {c3.name}")
        print(f"top grid: {top_grid}")
        print("No files written in dry-run.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    trio_rows: list[Path] = []
    for scene_id, c1, c2, c3 in full_groups:
        out = out_dir / f"{scene_id}_combined123_row.mp4"
        make_one_trio_row(c1, c2, c3, out, args.fps, args.overwrite)
        trio_rows.append(out)
        print(f"Wrote trio row: {out}")

    two_trio = out_dir / "two_scene_trios_2col_1row.mp4"
    make_two_trio_row(trio_rows[0], trio_rows[1], two_trio, args.fps, args.pair_gap, args.overwrite)
    print(f"Wrote 2-group trio row: {two_trio}")

    final_out = out_dir / "random6_grid_with_trios_bottom.mp4"
    merge_top_bottom(top_grid, two_trio, final_out, args.fps, args.block_gap, args.overwrite)
    print(f"Wrote final composite: {final_out}")

    manifest = out_dir / "trio_composite_manifest.txt"
    lines = [
        f"fps={args.fps}\n",
        f"pair_gap={args.pair_gap}\n",
        f"block_gap={args.block_gap}\n",
        f"top_grid={top_grid}\n",
        "trio_groups:\n",
    ]
    for scene_id, c1, c2, c3 in full_groups:
        lines.append(f"- {scene_id}: {c1.name} | {c2.name} | {c3.name}\n")
    lines.append(f"two_trio_row={two_trio.name}\n")
    lines.append(f"final_composite={final_out.name}\n")
    manifest.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote manifest: {manifest}")


if __name__ == "__main__":
    main()
