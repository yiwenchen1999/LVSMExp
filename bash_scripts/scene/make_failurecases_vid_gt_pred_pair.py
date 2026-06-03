#!/usr/bin/env python3
"""
Merge failurecases gt/pred relit videos into side-by-side synced pair videos.

Input:
  result_previews/videos/failurecases_gt_relit_video/*.mp4
  result_previews/videos/failurecases_pred_relit_video/*.mp4

Output:
  result_previews/videos/failurecases_gt_pred_pair_video/*.mp4

Each output is horizontally stacked as: [gt_relit | pred_relit].
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_GT_ROOT = _REPO_ROOT / "result_previews/videos/failurecases_gt_relit_video"
_DEFAULT_PRED_ROOT = _REPO_ROOT / "result_previews/videos/failurecases_pred_relit_video"
_DEFAULT_OUT_ROOT = _REPO_ROOT / "result_previews/videos/failurecases_gt_pred_pair_video"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt-root", type=Path, default=_DEFAULT_GT_ROOT)
    p.add_argument("--pred-root", type=Path, default=_DEFAULT_PRED_ROOT)
    p.add_argument("--output-root", type=Path, default=_DEFAULT_OUT_ROOT)
    p.add_argument("--fps", type=float, default=24.0, help="Output fps (default: 24.0)")
    p.add_argument(
        "--iters",
        nargs="*",
        default=None,
        metavar="ITER",
        help="Only process files whose stem ends with one of these iter names.",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def run_ffmpeg(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def keep_file(stem: str, iters: set[str]) -> bool:
    if not iters:
        return True
    for it in iters:
        if stem.endswith(it):
            return True
    return False


def merge_pair(gt_path: Path, pred_path: Path, out_path: Path, fps: float, overwrite: bool) -> None:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(gt_path),
        "-i",
        str(pred_path),
        "-filter_complex",
        (
            f"[0:v]setpts=PTS-STARTPTS,fps={fps}[gt];"
            f"[1:v]setpts=PTS-STARTPTS,fps={fps}[pred];"
            "[gt][pred]hstack=inputs=2:shortest=1[v]"
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
    gt_root = args.gt_root.resolve()
    pred_root = args.pred_root.resolve()
    out_root = args.output_root.resolve()

    if not gt_root.is_dir():
        raise SystemExit(f"GT root not found: {gt_root}")
    if not pred_root.is_dir():
        raise SystemExit(f"Pred root not found: {pred_root}")
    if args.fps <= 0:
        raise SystemExit(f"Invalid --fps={args.fps}; must be > 0")

    target_iters = set(args.iters or [])

    gt_map = {p.stem: p for p in sorted(gt_root.glob("*.mp4")) if keep_file(p.stem, target_iters)}
    pred_map = {p.stem: p for p in sorted(pred_root.glob("*.mp4")) if keep_file(p.stem, target_iters)}
    common_stems = sorted(set(gt_map.keys()) & set(pred_map.keys()))
    if not common_stems:
        raise SystemExit("No matched gt/pred video pairs found.")

    gt_only = sorted(set(gt_map.keys()) - set(pred_map.keys()))
    pred_only = sorted(set(pred_map.keys()) - set(gt_map.keys()))
    if gt_only:
        print("Warning: GT-only files (no pred match):")
        for stem in gt_only:
            print(f"  - {stem}")
    if pred_only:
        print("Warning: Pred-only files (no gt match):")
        for stem in pred_only:
            print(f"  - {stem}")

    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    for stem in common_stems:
        out_path = out_root / f"{stem}.mp4"
        if out_path.exists() and not args.overwrite:
            print(f"Skip existing: {out_path}")
            skipped += 1
            continue

        if args.dry_run:
            print(f"[dry-run] {gt_map[stem].name} + {pred_map[stem].name} -> {out_path.name}")
            processed += 1
            continue

        merge_pair(gt_map[stem], pred_map[stem], out_path, fps=args.fps, overwrite=args.overwrite)
        print(f"Wrote {out_path}")
        processed += 1

    print(f"\nProcessed: {processed}, Skipped: {skipped}")
    if not args.dry_run:
        print(f"Output root: {out_root}")
    print("Done.")


if __name__ == "__main__":
    main()
