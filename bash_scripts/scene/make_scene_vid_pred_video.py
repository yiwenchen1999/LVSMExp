#!/usr/bin/env python3
"""
Create per-scene prediction videos from scene_vid single-image frames.

Input:
  result_previews/scene_vid_single_image/<scene>/<iter>/frame_XXXX/view_pred.jpg

Output:
  result_previews/scene_vid_pred_video/<scene>_<iter>.mp4

Default fps is 12 (half of previous 24-fps set).
Supports slow-motion by frame duplication while keeping fps unchanged.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_INPUT_ROOT = _REPO_ROOT / "result_previews/scene_vid_single_image"
_DEFAULT_OUTPUT_ROOT = _REPO_ROOT / "result_previews/scene_vid_pred_video"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root", type=Path, default=_DEFAULT_INPUT_ROOT)
    p.add_argument("--output-root", type=Path, default=_DEFAULT_OUTPUT_ROOT)
    p.add_argument("--fps", type=float, default=12.0, help="Output fps (default: 12.0)")
    p.add_argument(
        "--slowdown",
        type=float,
        default=1.0,
        help=(
            "Slow-motion factor with unchanged fps. "
            "2.0 means 2x slower playback by duplicating each frame twice."
        ),
    )
    p.add_argument("--iters", nargs="*", default=None, metavar="ITER")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def list_frame_images(iter_dir: Path) -> list[Path]:
    frames: list[Path] = []
    for d in sorted(p for p in iter_dir.iterdir() if p.is_dir() and p.name.startswith("frame_")):
        pred = d / "view_pred.jpg"
        if pred.is_file():
            frames.append(pred)
        else:
            raise FileNotFoundError(f"Missing frame image: {pred}")
    return frames


def write_concat_file(images: list[Path], txt_path: Path, frame_repeat: int) -> None:
    if frame_repeat < 1:
        raise ValueError(f"frame_repeat must be >= 1, got {frame_repeat}")
    lines = []
    for p in images:
        escaped = str(p.resolve()).replace("'", "'\\''")
        for _ in range(frame_repeat):
            lines.append(f"file '{escaped}'\n")
    txt_path.write_text("".join(lines), encoding="utf-8")


def render_video(
    images: list[Path],
    out_path: Path,
    fps: float,
    overwrite: bool,
    frame_repeat: int,
) -> None:
    concat_txt = out_path.with_suffix(".frames.txt")
    write_concat_file(images, concat_txt, frame_repeat=frame_repeat)
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
        str(concat_txt),
        "-vf",
        f"fps={fps}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True)
    finally:
        if concat_txt.exists():
            concat_txt.unlink()


def main() -> None:
    args = parse_args()
    in_root = args.input_root.resolve()
    out_root = args.output_root.resolve()

    if not in_root.is_dir():
        raise SystemExit(f"Input root not found: {in_root}")
    if args.fps <= 0:
        raise SystemExit(f"Invalid --fps={args.fps}; must be > 0")
    if args.slowdown < 1:
        raise SystemExit(f"Invalid --slowdown={args.slowdown}; must be >= 1")
    frame_repeat = int(round(args.slowdown))
    if abs(frame_repeat - args.slowdown) > 1e-6:
        raise SystemExit(
            f"--slowdown must be an integer-like value for duplication mode, got {args.slowdown}"
        )

    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)

    target_iters = set(args.iters or [])
    processed = 0
    skipped = 0

    scene_dirs = [p for p in sorted(in_root.iterdir()) if p.is_dir()]
    if not scene_dirs:
        raise SystemExit(f"No scene directories under {in_root}")

    for scene_dir in scene_dirs:
        iter_dirs = [p for p in sorted(scene_dir.iterdir()) if p.is_dir() and p.name.startswith("iter_")]
        if target_iters:
            iter_dirs = [p for p in iter_dirs if p.name in target_iters]
        for iter_dir in iter_dirs:
            out_name = f"{scene_dir.name}_{iter_dir.name}.mp4"
            out_path = out_root / out_name

            if out_path.exists() and not args.overwrite:
                print(f"Skip existing: {out_path}")
                skipped += 1
                continue

            try:
                images = list_frame_images(iter_dir)
            except FileNotFoundError as exc:
                print(f"Warning: {scene_dir.name}/{iter_dir.name}: {exc}; skip")
                skipped += 1
                continue

            if not images:
                print(f"Warning: {scene_dir.name}/{iter_dir.name}: zero frames; skip")
                skipped += 1
                continue

            if args.dry_run:
                print(
                    f"[dry-run] {scene_dir.name}/{iter_dir.name}: "
                    f"{len(images)} frames x repeat {frame_repeat} -> {out_path}"
                )
                processed += 1
                continue

            render_video(images, out_path, args.fps, args.overwrite, frame_repeat=frame_repeat)
            print(
                f"Wrote {out_path} "
                f"({len(images)} src frames, repeat {frame_repeat}, @ {args.fps:g} fps)"
            )
            processed += 1

    print(f"\nProcessed: {processed}, Skipped: {skipped}")
    if not args.dry_run:
        print(f"Output root: {out_root}")
    print("Done.")


if __name__ == "__main__":
    main()
