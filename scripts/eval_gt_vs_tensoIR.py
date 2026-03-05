#!/usr/bin/env python3
"""
Compute PSNR, SSIM, LPIPS between GT (with alpha, composited on white bg) and TensoIR predictions (*_nobg.png).
- GT transparent background -> white; compare with pred files ending in _nobg.png.
- Match: pred 00069_nobg.png <-> GT 00069.png

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



def psnr_single(img1, img2):
    """img1, img2: float [H,W,C] in [0,1]."""
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-12:
        return 50.0
    return float(10 * math.log10(1.0 / mse))


def ssim_single(img1, img2):
    if not HAS_SSIM:
        return float("nan")
    ssim = structural_similarity(
        img1, img2, win_size=11, channel_axis=2, data_range=1.0, gaussian_weights=True
    )
    return float(ssim)


def load_gt_rgba(path):
    im = Image.open(path)
    arr = np.array(im)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr, np.full_like(arr, 255)], axis=-1)
    elif arr.shape[-1] == 3:
        arr = np.concatenate([arr, np.full((*arr.shape[:2], 1), 255)], axis=-1)
    return arr.astype(np.float32) / 255.0


def to_white_bg(rgba):
    """rgba: [H,W,4] float [0,1]. Returns RGB on white bg."""
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3:4]
    return (rgb * alpha + (1.0 - alpha)).astype(np.float32)


def load_pred_rgb(path):
    im = Image.open(path).convert("RGB")
    return np.array(im).astype(np.float32) / 255.0


def eval_one_scene(gt_dir, pred_dir, pred_suffix="_nobg.png", lpips_fn=None):
    """Match pred *{pred_suffix} to GT {base}.png. Returns dict with n, per_image, avg_psnr, avg_ssim, avg_lpips."""
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith(pred_suffix)])
    if not pred_files:
        return None

    results = []
    for pred_fn in pred_files:
        base = pred_fn[: -len(pred_suffix)]
        gt_fn = base + ".png"
        gt_path = os.path.join(gt_dir, gt_fn)
        pred_path = os.path.join(pred_dir, pred_fn)
        if not os.path.isfile(gt_path):
            continue
        try:
            gt_rgba = load_gt_rgba(gt_path)
            gt_white = to_white_bg(gt_rgba)
            pred_rgb = load_pred_rgb(pred_path)
        except Exception:
            continue

        h_gt, w_gt = gt_white.shape[:2]
        if pred_rgb.shape[0] != h_gt or pred_rgb.shape[1] != w_gt:
            pred_rgb = np.array(
                Image.fromarray((pred_rgb * 255).astype(np.uint8)).resize((w_gt, h_gt), Image.LANCZOS)
            ).astype(np.float32) / 255.0

        psnr = psnr_single(gt_white, pred_rgb)
        ssim = ssim_single(gt_white, pred_rgb)
        lpips_val = None
        if lpips_fn is not None:
            a = torch.from_numpy(gt_white).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1
            b = torch.from_numpy(pred_rgb).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1
            with torch.no_grad():
                lpips_val = float(lpips_fn(a, b).item())

        results.append({"file": pred_fn, "gt_file": gt_fn, "psnr": psnr, "ssim": ssim, "lpips": lpips_val})

    if not results:
        return None

    psnrs = [r["psnr"] for r in results if math.isfinite(r["psnr"])]
    ssims = [r["ssim"] for r in results if math.isfinite(r["ssim"])]
    lpipss = [r["lpips"] for r in results if r.get("lpips") is not None]
    return {
        "n": len(results),
        "per_image": results,
        "avg_psnr": sum(psnrs) / len(psnrs) if psnrs else None,
        "avg_ssim": sum(ssims) / len(ssims) if ssims else None,
        "avg_lpips": sum(lpipss) / len(lpipss) if lpipss else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gt_dir", help="GT directory (e.g. eval/gt_samples/ceramic_vase_02_env_1)")
    parser.add_argument("tensoIR_dir", help="TensoIR directory (e.g. eval/TensoIR/ceramic_vase_02_env_1)")
    parser.add_argument("--all", action="store_true", help="Run for all scene pairs under gt_dir and tensoIR_dir")
    parser.add_argument("--pred-suffix", default="_nobg.png", help="Pred file suffix (default: _nobg.png; use _nogb.png if needed)")
    parser.add_argument("--no-lpips", action="store_true", help="Skip LPIPS")
    parser.add_argument("-o", "--output", default=None, help="Save JSON (with --all)")
    args = parser.parse_args()

    gt_root = os.path.abspath(args.gt_dir)
    tensoIR_root = os.path.abspath(args.tensoIR_dir)
    if not os.path.isdir(gt_root):
        print(f"GT dir not found: {gt_root}")
        sys.exit(1)
    if not os.path.isdir(tensoIR_root):
        print(f"TensoIR dir not found: {tensoIR_root}")
        sys.exit(1)

    pred_suffix = args.pred_suffix if args.pred_suffix.endswith(".png") else args.pred_suffix + ".png"
    lpips_fn = None
    if HAS_LPIPS and not args.no_lpips:
        lpips_fn = lpips.LPIPS(net="alex")
        lpips_fn.eval()

    if args.all:
        gt_subdirs = [d for d in os.listdir(gt_root) if os.path.isdir(os.path.join(gt_root, d))]
        pred_subdirs = [d for d in os.listdir(tensoIR_root) if os.path.isdir(os.path.join(tensoIR_root, d))]
        scene_names = sorted(set(gt_subdirs) & set(pred_subdirs))
        if not scene_names:
            print("No common scene subdirs.")
            sys.exit(1)
        print(f"GT: white bg. Pred: *{pred_suffix}")
        print(f"Found {len(scene_names)} scene pairs: {scene_names}")
        print()

        all_results = {}
        agg_psnr, agg_ssim, agg_lpips = [], [], []
        for name in scene_names:
            gt_dir = os.path.join(gt_root, name)
            pred_dir = os.path.join(tensoIR_root, name)
            out = eval_one_scene(gt_dir, pred_dir, pred_suffix=pred_suffix, lpips_fn=lpips_fn)
            if out is None:
                print(f"  [skip] {name}: no matching *{pred_suffix} files")
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
        print("Overall:")
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

    out = eval_one_scene(gt_root, tensoIR_root, pred_suffix=pred_suffix, lpips_fn=lpips_fn)
    if out is None:
        print(f"No matching *{pred_suffix} files found.")
        sys.exit(1)
    results = out["per_image"]
    print(f"GT dir (white bg): {gt_root}")
    print(f"TensoIR dir (*{pred_suffix}): {tensoIR_root}")
    print(f"Common pairs: {len(results)}")
    print()
    for r in results[:15]:
        lp = f"{r['lpips']:.4f}" if r.get("lpips") is not None else "N/A"
        print(f"  {r['gt_file']} <-> {r['file']}: PSNR={r['psnr']:.2f}  SSIM={r['ssim']:.4f}  LPIPS={lp}")
    if len(results) > 15:
        print(f"  ... and {len(results) - 15} more")
    print()
    print("Averages:")
    print(f"  PSNR:  {out['avg_psnr']:.4f}" if out["avg_psnr"] else "  PSNR:  N/A")
    print(f"  SSIM:  {out['avg_ssim']:.4f}" if out["avg_ssim"] else "  SSIM:  N/A")
    print(f"  LPIPS: {out['avg_lpips']:.6f}" if out["avg_lpips"] else "  LPIPS: N/A")


if __name__ == "__main__":
    main()
