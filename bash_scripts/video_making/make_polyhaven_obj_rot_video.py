#!/usr/bin/env python3
"""
Create per-scene rotation videos from polyhaven single-image frames.

Input layout (from reorganize_polyhaven_video_previews.py):
  result_previews/videos/polyhaven_single_image/<scene>/<iter>/frame_XXXX/view_relit_pred.jpg

Output:
  result_previews/videos/polyhaven_obj_rot_video/<scene>_<iter>.mp4
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_INPUT_ROOT = _REPO_ROOT / "result_previews/videos/polyhaven_single_image"
_DEFAULT_OUTPUT_ROOT = _REPO_ROOT / "result_previews/videos/polyhaven_obj_rot_video"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=_DEFAULT_INPUT_ROOT,
        help=f"Root with scene/iter/frame directories (default: {_DEFAULT_INPUT_ROOT})",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_DEFAULT_OUTPUT_ROOT,
        help=f"Output root for generated videos (default: {_DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=24.0,
        help="Output video FPS (default: 24.0)",
    )
    parser.add_argument(
        "--iters",
        nargs="*",
        default=None,
        metavar="ITER",
        help="Only process specific iter names (e.g., iter_00000001).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output mp4 files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned jobs only; write nothing.",
    )
    return parser.parse_args()


def list_frame_images(iter_dir: Path) -> list[Path]:
    frames: list[Path] = []
    for frame_dir in sorted(p for p in iter_dir.iterdir() if p.is_dir() and p.name.startswith("frame_")):
        image_path = frame_dir / "view_relit_pred.jpg"
        if image_path.is_file():
            frames.append(image_path)
        else:
            raise FileNotFoundError(f"Missing frame image: {image_path}")
    return frames


def write_concat_list(paths: list[Path], tmp_list: Path) -> None:
    # ffmpeg concat demuxer requires: file '/abs/path'
    lines = []
    for p in paths:
        escaped = str(p.resolve()).replace("'", "'\\''")
        lines.append(f"file '{escaped}'\n")
    tmp_list.write_text("".join(lines), encoding="utf-8")


def render_video(frame_paths: list[Path], output_path: Path, fps: float, overwrite: bool) -> None:
    tmp_list = output_path.with_suffix(".frames.txt")
    write_concat_list(frame_paths, tmp_list)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(tmp_list),
        "-vf",
        f"fps={fps}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True)
    finally:
        if tmp_list.exists():
            tmp_list.unlink()


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()

    if not input_root.is_dir():
        raise SystemExit(f"Input root not found: {input_root}")
    if args.fps <= 0:
        raise SystemExit(f"Invalid --fps={args.fps}; must be > 0")

    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    target_iters = set(args.iters or [])
    processed = 0
    skipped = 0

    scene_dirs = [p for p in sorted(input_root.iterdir()) if p.is_dir()]
    if not scene_dirs:
        raise SystemExit(f"No scene directories under {input_root}")

    for scene_dir in scene_dirs:
        iter_dirs = [p for p in sorted(scene_dir.iterdir()) if p.is_dir() and p.name.startswith("iter_")]
        if target_iters:
            iter_dirs = [p for p in iter_dirs if p.name in target_iters]
        if not iter_dirs:
            continue

        for iter_dir in iter_dirs:
            video_name = f"{scene_dir.name}_{iter_dir.name}.mp4"
            output_path = output_root / video_name
            if output_path.exists() and not args.overwrite:
                print(f"Skip existing: {output_path}")
                skipped += 1
                continue

            try:
                frame_paths = list_frame_images(iter_dir)
            except FileNotFoundError as exc:
                print(f"Warning: {scene_dir.name}/{iter_dir.name}: {exc}; skip")
                skipped += 1
                continue

            if not frame_paths:
                print(f"Warning: {scene_dir.name}/{iter_dir.name}: no frames; skip")
                skipped += 1
                continue

            if args.dry_run:
                print(
                    f"[dry-run] {scene_dir.name}/{iter_dir.name}: "
                    f"{len(frame_paths)} frames -> {output_path}"
                )
                processed += 1
                continue

            render_video(frame_paths, output_path, args.fps, args.overwrite)
            print(f"Wrote {output_path} ({len(frame_paths)} frames @ {args.fps:g} fps)")
            processed += 1

    print(f"\nProcessed: {processed}, Skipped: {skipped}")
    if not args.dry_run:
        print(f"Output root: {output_root}")


if __name__ == "__main__":
    main()
