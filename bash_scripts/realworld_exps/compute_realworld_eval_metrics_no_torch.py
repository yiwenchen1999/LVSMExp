#!/usr/bin/env python3
"""
Compute PSNR and SSIM for relit_gt vs relit_pred under:
  result_previews/realworld_eval/single_image/<run_name>/...

No torch dependency (numpy-only implementation).
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_SINGLE_IMAGE = _REPO_ROOT / "result_previews/realworld_eval/single_image"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base",
        type=Path,
        default=_DEFAULT_SINGLE_IMAGE,
        help=f"Root containing run folders (default: {_DEFAULT_SINGLE_IMAGE})",
    )
    p.add_argument(
        "--infer",
        nargs="*",
        default=["test_relight_stanfordORB"],
        metavar="NAME",
        help="Subfolders under --base to scan.",
    )
    return p.parse_args()


def load_image01(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0)


def compute_psnr_np(gt: np.ndarray, pred: np.ndarray) -> float:
    mse = float(np.mean((gt - pred) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return float(-10.0 * np.log10(mse))


def gaussian_kernel(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    ax = np.arange(-(size // 2), size // 2 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax, indexing="xy")
    k = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    k /= np.sum(k)
    return k.astype(np.float32)


def conv2_reflect(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    pad = kernel.shape[0] // 2
    padded = np.pad(img, ((pad, pad), (pad, pad)), mode="reflect")
    windows = sliding_window_view(padded, kernel.shape)
    return np.einsum("ijkl,kl->ij", windows, kernel, optimize=True)


def compute_ssim_np(gt: np.ndarray, pred: np.ndarray) -> float:
    """
    SSIM with Gaussian window (11, sigma=1.5), averaged over RGB channels.
    Matches the settings used by utils.metric_utils (win_size=11, gaussian_weights=True).
    """
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: {gt.shape} vs {pred.shape}")

    k = gaussian_kernel(11, 1.5)
    c1 = (0.01 * 1.0) ** 2
    c2 = (0.03 * 1.0) ** 2

    vals = []
    for ch in range(gt.shape[2]):
        x = gt[:, :, ch]
        y = pred[:, :, ch]

        mu_x = conv2_reflect(x, k)
        mu_y = conv2_reflect(y, k)
        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x2 = conv2_reflect(x * x, k) - mu_x2
        sigma_y2 = conv2_reflect(y * y, k) - mu_y2
        sigma_xy = conv2_reflect(x * y, k) - mu_xy

        sigma_x2 = np.maximum(sigma_x2, 0.0)
        sigma_y2 = np.maximum(sigma_y2, 0.0)

        num = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
        den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
        ssim_map = num / np.maximum(den, 1e-12)
        vals.append(float(np.mean(ssim_map)))

    return float(np.mean(vals))


def pred_path_for(gt_path: Path) -> Path | None:
    stem = gt_path.stem
    if not stem.endswith("_relit_gt"):
        return None
    pred = gt_path.with_name(stem.replace("_relit_gt", "_relit_pred") + gt_path.suffix)
    if pred.is_file():
        return pred
    for ext in (".jpg", ".png", ".jpeg"):
        alt = gt_path.with_name(stem.replace("_relit_gt", "_relit_pred") + ext)
        if alt.is_file():
            return alt
    return None


def collect_pairs(infer_dir: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for pattern in ("view_*_relit_gt.jpg", "view_*_relit_gt.png"):
        for gt in sorted(infer_dir.rglob(pattern)):
            pr = pred_path_for(gt)
            if pr is not None:
                pairs.append((gt, pr))
    return pairs


def scene_iter_from_gt(gt_path: Path, infer_dir: Path) -> tuple[str, str]:
    rel = gt_path.relative_to(infer_dir)
    return rel.parts[0], rel.parts[1]


def process_infer_folder(infer_dir: Path, out_csv: Path, out_json: Path) -> None:
    pairs = collect_pairs(infer_dir)
    if not pairs:
        print(f"No relit_gt / relit_pred pairs under {infer_dir}")
        return

    rows: list[dict[str, object]] = []
    for gp, pp in pairs:
        gt = load_image01(gp)
        pr = load_image01(pp)
        if gt.shape != pr.shape:
            raise ValueError(f"Shape mismatch {gp} vs {pp}: {gt.shape} vs {pr.shape}")

        psnr_v = compute_psnr_np(gt, pr)
        ssim_v = compute_ssim_np(gt, pr)
        scene, it_name = scene_iter_from_gt(gp, infer_dir)
        rows.append(
            {
                "scene": scene,
                "iter": it_name,
                "view": gp.stem,
                "psnr": float(psnr_v),
                "ssim": float(ssim_v),
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scene", "iter", "view", "psnr", "ssim"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    psnrs = [float(r["psnr"]) for r in rows if np.isfinite(float(r["psnr"]))]
    ssims = [float(r["ssim"]) for r in rows]
    summary: dict[str, object] = {
        "infer_dir": str(infer_dir),
        "num_pairs": len(rows),
        "psnr_mean": float(statistics.mean(psnrs)) if psnrs else None,
        "psnr_std": float(statistics.pstdev(psnrs)) if len(psnrs) > 1 else 0.0,
        "ssim_mean": float(statistics.mean(ssims)) if ssims else None,
        "ssim_std": float(statistics.pstdev(ssims)) if len(ssims) > 1 else 0.0,
        "metric_backend": "numpy_fallback",
    }
    out_json.write_text(json.dumps(summary, indent=2))

    print(f"Wrote {out_csv} ({len(rows)} rows)")
    print(f"Wrote {out_json}")
    print(f"  PSNR mean={summary['psnr_mean']:.4f}  SSIM mean={summary['ssim_mean']:.4f}")


def main() -> None:
    args = parse_args()
    base = args.base.resolve()
    if not base.is_dir():
        raise SystemExit(f"Not a directory: {base}")

    for name in args.infer:
        infer_dir = base / name
        if not infer_dir.is_dir():
            print(f"Skip (missing): {infer_dir}")
            continue
        out_csv = base / f"metrics_relit_{name}.csv"
        out_json = base / f"metrics_relit_{name}_summary.json"
        print(f"=== {name} ===")
        process_infer_folder(infer_dir, out_csv, out_json)

    print("Done.")


if __name__ == "__main__":
    main()
