#!/usr/bin/env python3
"""
Compute PSNR, SSIM, LPIPS between GT (with alpha) and lightSwitch predictions.
- Both images are composited on black background (GT alpha applied to pred as well).
- Default: whole image (metrics over full HxW; black bg regions match).
- Optional --masked: compute only over foreground pixels.
- Optional --tone-map: optimal linear scale for pred on foreground.

Local run (use venv from shortcut.sh):
  source /Users/yiwenchen/Desktop/ResearchProjects/scripts/venv/bin/activate
  cd /path/to/LVSMExp
  python scripts/eval_gt_vs_lightswitch_masked.py gt_dir pred_dir [--tone-map]

Requires: numpy, PIL. For SSIM: scikit-image. For LPIPS: torch, lpips.
"""

import os
import sys
import math
import argparse
import numpy as np
from PIL import Image

try:
    import torch
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False

try:
    from skimage.metrics import structural_similarity
    HAS_SSIM = True
except ImportError:
    HAS_SSIM = False


def psnr_single(img1, img2, mask=None):
    """img1, img2: float [H,W,C] in [0,1]. mask: [H,W] or [H,W,1] foreground=1."""
    if mask is not None:
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        m = mask > 0.5
        if np.sum(m) == 0:
            return float("nan")
        diff = (img1 - img2) ** 2
        mse = np.sum(diff[m]) / (np.sum(m) * img1.shape[2])
    else:
        mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-12:
        return 50.0
    return float(10 * math.log10(1.0 / mse))


def ssim_single(img1, img2, mask=None):
    """SSIM; if mask given, crop to foreground bbox and compute SSIM on that region."""
    if not HAS_SSIM:
        return float("nan")
    if mask is not None and mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask is not None and np.sum(mask > 0.5) > 0:
        ys, xs = np.where(mask > 0.5)
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        img1 = img1[y0:y1, x0:x1]
        img2 = img2[y0:y1, x0:x1]
        if min(img1.shape[0], img1.shape[1]) < 11:
            return float("nan")
    # skimage: channel_axis=2 for HWC, data_range=1.0 for [0,1]
    ssim = structural_similarity(
        img1, img2, win_size=11, channel_axis=2, data_range=1.0, gaussian_weights=True
    )
    return float(ssim)


def optimal_scale_fg(gt_fg, pred_fg, mask):
    """Scalar s minimizing ||gt - s*pred||^2 over foreground. mask: [H,W,1]."""
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    m = mask > 0.5
    gt_m = gt_fg[m]
    pr_m = pred_fg[m]
    if pr_m.size == 0 or np.sum(pr_m * pr_m) < 1e-12:
        return 1.0
    s = float(np.sum(gt_m * pr_m) / np.sum(pr_m * pr_m))
    return float(np.clip(s, 0.1, 10.0))


def load_gt_rgba(path):
    im = Image.open(path)
    arr = np.array(im)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr, np.full_like(arr, 255)], axis=-1)
    elif arr.shape[-1] == 3:
        arr = np.concatenate([arr, np.full((*arr.shape[:2], 1), 255)], axis=-1)
    rgba = arr.astype(np.float32) / 255.0
    return rgba


def load_pred_rgb(path):
    im = Image.open(path).convert("RGB")
    return np.array(im).astype(np.float32) / 255.0


def eval_one_scene(gt_dir, pred_dir, tone_map=False, masked=False, lpips_fn=None):
    """
    Evaluate one scene pair. Returns dict with keys: scene_name, n, per_image, avg_psnr, avg_ssim, avg_lpips.
    """
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.lower().endswith(".png")])
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.lower().endswith(".png")])
    common = sorted(set(gt_files) & set(pred_files))
    if not common:
        return None

    results = []
    for fn in common:
        gt_path = os.path.join(gt_dir, fn)
        pred_path = os.path.join(pred_dir, fn)
        try:
            gt_rgba = load_gt_rgba(gt_path)
            pred_rgb = load_pred_rgb(pred_path)
        except Exception as e:
            continue
        h_gt, w_gt = gt_rgba.shape[:2]
        if pred_rgb.shape[0] != h_gt or pred_rgb.shape[1] != w_gt:
            pred_rgb = np.array(
                Image.fromarray((pred_rgb * 255).astype(np.uint8)).resize((w_gt, h_gt), Image.LANCZOS)
            ).astype(np.float32) / 255.0

        alpha = gt_rgba[:, :, 3:4]
        gt_black = gt_rgba[:, :, :3] * alpha
        pred_black = pred_rgb * alpha

        if tone_map:
            scale = optimal_scale_fg(gt_black, pred_black, alpha)
            pred_black = np.clip(pred_black * scale, 0.0, 1.0)

        use_mask = masked
        psnr = psnr_single(gt_black, pred_black, alpha if use_mask else None)
        ssim = ssim_single(gt_black, pred_black, alpha if use_mask else None)
        lpips_val = None
        if lpips_fn is not None:
            a = torch.from_numpy(gt_black).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1
            b = torch.from_numpy(pred_black).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1
            with torch.no_grad():
                lpips_val = float(lpips_fn(a, b).item())

        results.append({"file": fn, "psnr": psnr, "ssim": ssim, "lpips": lpips_val})

    psnrs = [r["psnr"] for r in results if math.isfinite(r["psnr"])]
    ssims = [r["ssim"] for r in results if math.isfinite(r["ssim"])]
    lpipss = [r["lpips"] for r in results if r.get("lpips") is not None]
    avg_psnr = sum(psnrs) / len(psnrs) if psnrs else None
    avg_ssim = sum(ssims) / len(ssims) if ssims else None
    avg_lpips = sum(lpipss) / len(lpipss) if lpipss else None

    return {
        "n": len(results),
        "per_image": results,
        "avg_psnr": avg_psnr,
        "avg_ssim": avg_ssim,
        "avg_lpips": avg_lpips,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gt_dir", help="GT directory or gt_samples root (with --all)")
    parser.add_argument("pred_dir", help="Pred directory or lightSwitch root (with --all)")
    parser.add_argument("--all", action="store_true", help="Run for all scene pairs under gt_dir and pred_dir")
    parser.add_argument("--tone-map", action="store_true", help="Apply optimal linear scale to pred on foreground")
    parser.add_argument("--masked", action="store_true", help="Compute metrics only over foreground (default: whole image)")
    parser.add_argument("--no-lpips", action="store_true", help="Skip LPIPS")
    parser.add_argument("-o", "--output", default=None, help="Save per-scene and overall metrics to JSON (with --all)")
    args = parser.parse_args()

    gt_root = os.path.abspath(args.gt_dir)
    pred_root = os.path.abspath(args.pred_dir)
    if not os.path.isdir(gt_root):
        print(f"GT dir not found: {gt_root}")
        sys.exit(1)
    if not os.path.isdir(pred_root):
        print(f"Pred dir not found: {pred_root}")
        sys.exit(1)

    lpips_fn = None
    if HAS_LPIPS and not args.no_lpips:
        lpips_fn = lpips.LPIPS(net="alex")
        lpips_fn.eval()

    if args.all:
        # Discover scene names: subdirs present in both gt_root and pred_root
        gt_subdirs = [d for d in os.listdir(gt_root) if os.path.isdir(os.path.join(gt_root, d))]
        pred_subdirs = [d for d in os.listdir(pred_root) if os.path.isdir(os.path.join(pred_root, d))]
        scene_names = sorted(set(gt_subdirs) & set(pred_subdirs))
        if not scene_names:
            print("No common scene subdirs between GT and pred roots.")
            sys.exit(1)
        print(f"Mode: whole image (black bg). Tone mapping: {'on' if args.tone_map else 'off'}")
        print(f"Found {len(scene_names)} scene pairs: {scene_names}")
        print()

        all_results = {}
        agg_psnr, agg_ssim, agg_lpips = [], [], []
        for name in scene_names:
            gt_dir = os.path.join(gt_root, name)
            pred_dir = os.path.join(pred_root, name)
            out = eval_one_scene(gt_dir, pred_dir, tone_map=args.tone_map, masked=args.masked, lpips_fn=lpips_fn)
            if out is None:
                print(f"  [skip] {name}: no common PNGs")
                continue
            all_results[name] = {
                "n": out["n"],
                "avg_psnr": out["avg_psnr"],
                "avg_ssim": out["avg_ssim"],
                "avg_lpips": out["avg_lpips"],
            }
            if out["avg_psnr"] is not None:
                agg_psnr.append(out["avg_psnr"])
            if out["avg_ssim"] is not None:
                agg_ssim.append(out["avg_ssim"])
            if out["avg_lpips"] is not None:
                agg_lpips.append(out["avg_lpips"])
            ps = out["avg_psnr"] or 0
            ss = f"{out['avg_ssim']:.4f}" if out["avg_ssim"] is not None else "N/A"
            lp = f"{out['avg_lpips']:.4f}" if out["avg_lpips"] is not None else "N/A"
            print(f"  {name}: n={out['n']}  PSNR={ps:.2f}  SSIM={ss}  LPIPS={lp}")

        print()
        n_scenes = len(agg_psnr)
        overall_psnr = sum(agg_psnr) / n_scenes if agg_psnr else None
        overall_ssim = sum(agg_ssim) / len(agg_ssim) if agg_ssim else None
        overall_lpips = sum(agg_lpips) / len(agg_lpips) if agg_lpips else None
        print("Overall (mean over scene averages):")
        print(f"  PSNR:  {overall_psnr:.4f}" if overall_psnr is not None else "  PSNR:  N/A")
        print(f"  SSIM:  {overall_ssim:.4f}" if overall_ssim is not None else "  SSIM:  N/A")
        print(f"  LPIPS: {overall_lpips:.6f}" if overall_lpips is not None else "  LPIPS: N/A")

        if args.output:
            import json
            with open(args.output, "w") as f:
                json.dump({
                    "scenes": all_results,
                    "overall": {"psnr": overall_psnr, "ssim": overall_ssim, "lpips": overall_lpips, "n_scenes": n_scenes},
                }, f, indent=2)
            print(f"\nSaved to {args.output}")
        return

    # Single-scene mode
    gt_files = sorted([f for f in os.listdir(gt_root) if f.lower().endswith(".png")])
    pred_files = sorted([f for f in os.listdir(pred_root) if f.lower().endswith(".png")])
    common = sorted(set(gt_files) & set(pred_files))
    if not common:
        print("No common PNG filenames between GT and pred dirs.")
        sys.exit(1)

    out = eval_one_scene(gt_root, pred_root, tone_map=args.tone_map, masked=args.masked, lpips_fn=lpips_fn)
    results = out["per_image"]
    avg_psnr, avg_ssim, avg_lpips = out["avg_psnr"], out["avg_ssim"], out["avg_lpips"]
    n = len(results)

    print(f"GT dir:  {gt_root}")
    print(f"Pred dir: {pred_root}")
    print(f"Mode: {'masked (foreground only)' if args.masked else 'whole image (black bg)'}")
    print(f"Tone mapping (linear scale): {'on' if args.tone_map else 'off'}")
    print(f"Common images: {n}")
    print()
    print("Per-image (first 10):")
    for r in results[:10]:
        lp = f"{r['lpips']:.4f}" if r.get("lpips") is not None else "N/A"
        print(f"  {r['file']}: PSNR={r['psnr']:.2f}  SSIM={r['ssim']:.4f}  LPIPS={lp}")
    if n > 10:
        print(f"  ... and {n - 10} more")
    print()
    print("Averages:")
    print(f"  PSNR:  {avg_psnr:.4f}" if avg_psnr is not None else "  PSNR:  N/A")
    print(f"  SSIM:  {avg_ssim:.4f}" if avg_ssim is not None else "  SSIM:  N/A")
    print(f"  LPIPS: {avg_lpips:.6f}" if avg_lpips is not None else "  LPIPS: N/A")


if __name__ == "__main__":
    main()
