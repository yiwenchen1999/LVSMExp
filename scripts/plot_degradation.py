#!/usr/bin/env python3
"""
Plot degradation curves from image-space and token-space CSV files.

Usage:
    python scripts/plot_degradation.py \\
        --image-csv experiments/degradation_exp/image_space/degrade_avg_image_space.csv \\
        --token-csv experiments/degradation_exp/token_space/degrade_avg_token_space.csv \\
        -o experiments/degradation_exp/degrade_comparison.png

If --image-csv or --token-csv is a directory, the script will look for the
default CSV filename inside that directory.
"""

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_avg_csv(path):
    """Return dict of lists: step, avg_psnr, avg_ssim, avg_lpips."""
    data = {"step": [], "avg_psnr": [], "avg_ssim": [], "avg_lpips": []}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["step"].append(int(row["step"]))
            data["avg_psnr"].append(float(row["avg_psnr"]))
            data["avg_ssim"].append(float(row["avg_ssim"]))
            data["avg_lpips"].append(float(row["avg_lpips"]))
    return data


def resolve_csv(path, default_name):
    if os.path.isdir(path):
        return os.path.join(path, default_name)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image-csv", required=True,
                        help="Path to image-space avg CSV (or its parent directory)")
    parser.add_argument("--token-csv", required=True,
                        help="Path to token-space avg CSV (or its parent directory)")
    parser.add_argument("-o", "--output", default="degrade_comparison.png",
                        help="Output PNG path (default: degrade_comparison.png)")
    args = parser.parse_args()

    img_csv = resolve_csv(args.image_csv, "degrade_avg_image_space.csv")
    tok_csv = resolve_csv(args.token_csv, "degrade_avg_token_space.csv")

    for p in [img_csv, tok_csv]:
        if not os.path.isfile(p):
            print(f"ERROR: CSV not found: {p}", file=sys.stderr)
            sys.exit(1)

    img = read_avg_csv(img_csv)
    tok = read_avg_csv(tok_csv)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    fig.suptitle("Iterative Editing Degradation: Image Space vs Token Space", fontsize=13)

    metrics = [
        ("avg_psnr",  "PSNR (dB) ↑",  True),
        ("avg_ssim",  "SSIM ↑",        True),
        ("avg_lpips", "LPIPS ↓",       False),
    ]

    for ax, (key, ylabel, higher_better) in zip(axes, metrics):
        ax.plot(img["step"], img[key], "o-", label="Image Space", markersize=3, linewidth=1.5)
        ax.plot(tok["step"], tok[key], "s--", label="Token Space", markersize=3, linewidth=1.5)
        ax.set_xlabel("Iteration Step")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(0, max(len(img["step"]), len(tok["step"])), max(1, len(img["step"]) // 10)))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Plot saved to {args.output}")

    print("\n--- Summary ---")
    for label, data in [("Image Space", img), ("Token Space", tok)]:
        if data["step"]:
            print(f"  {label}:")
            print(f"    Step  0: PSNR={data['avg_psnr'][0]:.2f}  SSIM={data['avg_ssim'][0]:.4f}  LPIPS={data['avg_lpips'][0]:.4f}")
            print(f"    Step {data['step'][-1]:2d}: PSNR={data['avg_psnr'][-1]:.2f}  SSIM={data['avg_ssim'][-1]:.4f}  LPIPS={data['avg_lpips'][-1]:.4f}")
            delta_psnr = data["avg_psnr"][-1] - data["avg_psnr"][0]
            delta_ssim = data["avg_ssim"][-1] - data["avg_ssim"][0]
            delta_lpips = data["avg_lpips"][-1] - data["avg_lpips"][0]
            print(f"    Delta:   PSNR={delta_psnr:+.2f}  SSIM={delta_ssim:+.4f}  LPIPS={delta_lpips:+.4f}")


if __name__ == "__main__":
    main()
