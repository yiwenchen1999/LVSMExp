#!/usr/bin/env python3
"""
Reorganize nested inference results into flat folders by file type.
Usage: python scripts/rearrange_result_previews.py <source_dir>
Output: <source_dir>_flat/ with inputs/, gt_vs_pred/, metadata/, metrics/
"""

import os
import shutil
import sys
from pathlib import Path


def rearrange_results(source_dir: str, out_dir: str | None = None) -> str:
    source = Path(source_dir)
    if not source.is_dir():
        raise NotADirectoryError(f"Source does not exist or is not a directory: {source_dir}")

    if out_dir is None:
        out_dir = source.parent / f"{source.name}_flat"
    out = Path(out_dir)

    dirs = {
        "inputs": out / "inputs",
        "gt_vs_pred": out / "gt_vs_pred",
        "metadata": out / "metadata",
        "metrics": out / "metrics",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    file_map = {
        "input.png": "inputs",
        "gt_vs_pred.png": "gt_vs_pred",
        "metadata.json": "metadata",
        "metrics.json": "metrics",
    }

    count = {k: 0 for k in dirs}
    for scene_dir in sorted(source.iterdir()):
        if not scene_dir.is_dir():
            continue
        scene_name = scene_dir.name
        for src_name, subdir in file_map.items():
            src = scene_dir / src_name
            if not src.exists():
                continue
            ext = src.suffix
            dst = dirs[subdir] / f"{scene_name}{ext}"
            shutil.copy2(src, dst)
            count[subdir] += 1

    print(f"Created: {out}")
    for k, n in count.items():
        print(f"  {k}: {n} files")
    return str(out)


def main():
    if len(sys.argv) < 2:
        print("Usage: python rearrange_result_previews.py <source_dir> [out_dir]")
        sys.exit(1)
    source_dir = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else None
    rearrange_results(source_dir, out_dir)


if __name__ == "__main__":
    main()
