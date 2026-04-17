#!/usr/bin/env python3
"""
Compute PSNR between relit_gt_* and relit_pred_* in flattened iter_*/view_* folders.

For each sequence index i (00, 01, ...), aggregates PSNR across all (iter, view) pairs,
reports mean/std per index, writes a CSV, and plots mean PSNR vs. sequence index.

Use --from-csv to skip images and redraw the plot from an existing CSV
(columns: sequence_index, mean_psnr, std_psnr, n_pairs).
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

GT_RE = re.compile(r"^relit_gt_(\d+)\.png$", re.IGNORECASE)
PRED_RE = re.compile(r"^relit_pred_(\d+)\.png$", re.IGNORECASE)

DEFAULT_ROOT = Path("result_previews/progressive_stability/finetuned_ckpt")
PSNR_CAP = 100.0
CSV_COLUMNS = ("sequence_index", "mean_psnr", "std_psnr", "n_pairs")


def build_parser() -> ArgumentParser:
    p = ArgumentParser(
        prog="psnr_relit_sequence",
        description=__doc__,
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    io = p.add_argument_group("input / output")
    io.add_argument(
        "--from-csv",
        type=Path,
        default=None,
        metavar="PATH",
        dest="from_csv",
        help=(
            "Load sequence_index / mean_psnr / std_psnr / n_pairs from this CSV and only draw the plot "
            "(no image scan). Default PNG/CSV output names follow -o / --basename; "
            "if omitted, PNG is written next to the CSV using the CSV stem."
        ),
    )
    io.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=None,
        metavar="DIR",
        help="Run root with iter_* subfolders, or a single iter_* folder (ignored when --from-csv is set).",
    )
    io.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Directory for CSV and plot. From images: default <root>/relit_psnr_metrics. "
            "From CSV: default is the CSV's parent when only --from-csv is passed."
        ),
    )
    io.add_argument(
        "--basename",
        type=str,
        default=None,
        metavar="NAME",
        help=(
            "Base filename for <NAME>.csv and <NAME>.png (no extension). "
            "Default: relit_sequence_psnr when scanning images; CSV stem when using --from-csv."
        ),
    )

    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Only write CSV; skip the figure.",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI for the PNG.",
    )
    return p


def parse_args(argv: Sequence[str] | None = None) -> Namespace:
    return build_parser().parse_args(argv)


def collect_iter_dirs(root: Path) -> List[Path]:
    if root.name.startswith("iter_"):
        return [root]
    return sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith("iter_"))


def load_rgb(path: Path) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    return np.asarray(im, dtype=np.uint8)


def try_load_pair(gt_path: Path, pred_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        gt = load_rgb(gt_path)
        pr = load_rgb(pred_path)
    except Exception as e:
        print(f"Warning: skip pair (read error): {gt_path.name} / {pred_path.name}: {e}", file=sys.stderr)
        return None
    return gt, pr


def psnr_uint8(gt: np.ndarray, pred: np.ndarray) -> float:
    """PSNR for uint8 HxWx3; identical images -> capped PSNR."""
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: {gt.shape} vs {pred.shape}")
    a = gt.astype(np.float64)
    b = pred.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse <= 0:
        return PSNR_CAP
    return float(min(PSNR_CAP, 10.0 * np.log10(255.0**2 / mse)))


def index_map_from_glob(view_dir: Path, pattern: re.Pattern) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for p in view_dir.glob("*.png"):
        m = pattern.match(p.name)
        if m:
            out[int(m.group(1))] = p
    return out


def gather_psnr_by_step(root: Path) -> tuple[DefaultDict[int, List[float]], int]:
    """Returns psnr_by_step (lists of values) and total pair count."""
    psnr_by_step: DefaultDict[int, List[float]] = defaultdict(list)
    iter_dirs = collect_iter_dirs(root)
    if not iter_dirs:
        raise SystemExit(f"No iter_* directories under {root}")

    for iter_dir in iter_dirs:
        view_dirs = sorted(
            d for d in iter_dir.iterdir() if d.is_dir() and d.name.startswith("view_")
        )
        for view_dir in view_dirs:
            gt_map = index_map_from_glob(view_dir, GT_RE)
            pred_map = index_map_from_glob(view_dir, PRED_RE)
            common = sorted(set(gt_map.keys()) & set(pred_map.keys()))
            for idx in common:
                loaded = try_load_pair(gt_map[idx], pred_map[idx])
                if loaded is None:
                    continue
                gt, pr = loaded
                try:
                    p = psnr_uint8(gt, pr)
                except ValueError as e:
                    print(f"Warning: skip pair {view_dir.name} idx {idx}: {e}", file=sys.stderr)
                    continue
                psnr_by_step[idx].append(p)

    total = sum(len(v) for v in psnr_by_step.values())
    return psnr_by_step, total


def load_psnr_csv(path: Path) -> tuple[List[int], np.ndarray, np.ndarray, np.ndarray]:
    """Read CSV written by this script; returns sorted steps and parallel arrays."""
    rows: list[tuple[int, float, float, int]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"Empty or invalid CSV: {path}")
        fields = {c.strip() for c in reader.fieldnames}
        missing = set(CSV_COLUMNS) - fields
        if missing:
            raise SystemExit(f"CSV {path} missing columns {sorted(missing)}; expected {CSV_COLUMNS}")
        for row in reader:
            r = {k.strip(): v for k, v in row.items()}
            rows.append(
                (
                    int(r["sequence_index"]),
                    float(r["mean_psnr"]),
                    float(r["std_psnr"]),
                    int(r["n_pairs"]),
                )
            )
    if not rows:
        raise SystemExit(f"No data rows in {path}")
    rows.sort(key=lambda r: r[0])
    steps = [r[0] for r in rows]
    means = np.array([r[1] for r in rows])
    stds = np.array([r[2] for r in rows])
    counts = np.array([r[3] for r in rows])
    return steps, means, stds, counts


def write_csv(
    path: Path,
    steps: List[int],
    means: np.ndarray,
    stds: np.ndarray,
    counts: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(CSV_COLUMNS))
        for i, m, s, c in zip(steps, means, stds, counts):
            w.writerow([i, f"{m:.6f}", f"{s:.6f}", int(c)])


def plot_curve(
    out_png: Path,
    steps: List[int],
    means: np.ndarray,
    stds: np.ndarray,
    dpi: int,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    x = np.array(steps, dtype=float)
    ax.plot(x, means, marker="o", linewidth=1.5, markersize=5, color="#1f77b4")
    ax.fill_between(x, means - stds, means + stds, color="#1f77b4", alpha=0.2, linewidth=0)
    ax.set_xlabel("Sequence index i (relit_gt_i vs relit_pred_i)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Mean PSNR vs. sequence index (±1 std over pairs)")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x)
    fig.savefig(out_png, dpi=dpi, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if args.from_csv is not None:
        csv_in = args.from_csv.expanduser().resolve()
        if not csv_in.is_file():
            raise SystemExit(f"Not a file: {csv_in}")

        steps, means, stds, counts = load_psnr_csv(csv_in)

        out_dir = args.output_dir
        if out_dir is None:
            out_dir = csv_in.parent
        else:
            out_dir = out_dir.resolve()
        base = args.basename if args.basename is not None else csv_in.stem
        png_path = out_dir / f"{base}.png"

        total_pairs = int(np.sum(counts))
        weighted_mean = float(np.sum(means * counts) / np.sum(counts)) if total_pairs else 0.0
        print(
            f"Loaded {csv_in} ({len(steps)} sequence indices, {total_pairs} pairs total "
            f"from n_pairs column)."
        )
        print(f"Weighted mean PSNR (by n_pairs): {weighted_mean:.4f} dB")

        if args.no_plot:
            return

        plot_curve(png_path, steps, means, stds, args.dpi)
        print(f"Wrote {png_path}")
        return

    root = (args.root or DEFAULT_ROOT).resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = root / "relit_psnr_metrics"
    else:
        out_dir = out_dir.resolve()
    base = args.basename if args.basename is not None else "relit_sequence_psnr"
    csv_path = out_dir / f"{base}.csv"
    png_path = out_dir / f"{base}.png"

    psnr_by_step, total_pairs = gather_psnr_by_step(root)
    if total_pairs == 0:
        raise SystemExit(
            f"No relit_gt_*.png / relit_pred_*.png pairs found under {root} (expected iter_*/view_*/*)."
        )

    steps = sorted(psnr_by_step.keys())
    means = np.array([np.mean(psnr_by_step[s]) for s in steps])
    stds = np.array([np.std(psnr_by_step[s], ddof=0) for s in steps])
    counts = np.array([len(psnr_by_step[s]) for s in steps])

    write_csv(csv_path, steps, means, stds, counts)
    print(f"Wrote {csv_path} ({len(steps)} sequence indices, {total_pairs} pairs total).")

    overall = float(np.mean([p for lst in psnr_by_step.values() for p in lst]))
    print(f"Overall mean PSNR (all pairs): {overall:.4f} dB")

    if not args.no_plot:
        plot_curve(png_path, steps, means, stds, args.dpi)
        print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
