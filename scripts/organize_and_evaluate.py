#!/usr/bin/env python3
"""
Organize and evaluate experiment results:
  1) Split gt_vs_pred grids into individual per-view images (row0=unlitGT, row1=relitGT, row2=pred)
  2) Extract lightSwitch_3d renders at target indices from nvs_metadata
  3) Compute PSNR / SSIM / LPIPS of lightSwitch results vs GT (black background)
  4) Flatten everything into easy-to-view directories
"""

import os
import json
import math
import shutil
import numpy as np
from PIL import Image

try:
    import torch
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False
    print("[WARN] torch/lpips not available – LPIPS will be skipped")


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT_VS_PRED_DIR   = os.path.join(BASE, "result_previews/polyhaven_dense_inference_flat/gt_vs_pred")
NVS_META_PATH    = os.path.join(BASE, "result_previews/nvs_meatadata.json")
LIGHTSWITCH_DIR  = os.path.join(BASE, "result_previews/lightSwitch_3d")
GT_RELIT_DIR     = os.path.join(BASE, "result_previews/gt_relit")
OUT_DIR          = os.path.join(BASE, "result_previews/organized_eval")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ssim_single(img1, img2, C1=6.5025, C2=58.5225):
    """Compute SSIM between two HxWxC float32 arrays in [0,1]."""
    from scipy.ndimage import uniform_filter
    mu1 = uniform_filter(img1, size=11)
    mu2 = uniform_filter(img2, size=11)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = uniform_filter(img1 * img1, size=11) - mu1_sq
    sigma2_sq = uniform_filter(img2 * img2, size=11) - mu2_sq
    sigma12   = uniform_filter(img1 * img2, size=11) - mu12
    num   = (2 * mu12 + C1) * (2 * sigma12 + C2)
    denom = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = num / denom
    return float(np.mean(ssim_map))


def psnr_single(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return float('inf')
    return float(10 * math.log10(1.0 / mse))


def to_black_bg(rgba_arr):
    """RGBA uint8 -> RGB float32 with black background."""
    rgba = rgba_arr.astype(np.float32) / 255.0
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3:4]
    return (rgb * alpha).astype(np.float32)


def to_alpha_masked(rgb_arr, alpha_arr):
    """Apply alpha mask: set transparent pixels to 0 (keep RGB elsewhere)."""
    rgb = rgb_arr.astype(np.float32) / 255.0
    alpha = alpha_arr.astype(np.float32) / 255.0
    if alpha.ndim == 2:
        alpha = alpha[:, :, None]
    return (rgb * alpha).astype(np.float32)


def to_white_bg(rgba_arr):
    """RGBA uint8 -> RGB float32 with white background."""
    rgba = rgba_arr.astype(np.float32) / 255.0
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3:4]
    return (rgb * alpha + (1.0 - alpha)).astype(np.float32)


def optimal_scale_fg(gt_fg, pred_fg, mask):
    """
    Solve for scalar s that minimises ||gt - s*pred||^2 over foreground pixels.
    mask: float [H,W,1] with 1 for foreground.
    Returns s (clamped to [0.1, 10]).
    """
    gt_m = gt_fg[mask[..., 0] > 0.5]
    pr_m = pred_fg[mask[..., 0] > 0.5]
    if pr_m.size == 0 or np.sum(pr_m * pr_m) < 1e-12:
        return 1.0
    s = float(np.sum(gt_m * pr_m) / np.sum(pr_m * pr_m))
    return float(np.clip(s, 0.1, 10.0))


def compute_metrics_suite(gt_rgba, ls_rgb_f32, alpha_f32, lpips_fn):
    """
    Compute metrics in 3 modes:
      1) black_bg       – raw, no scale
      2) black_bg_scaled – optimal foreground scale, black bg
      3) white_bg_scaled – optimal foreground scale, white bg
    gt_rgba: float32 [H,W,4] in [0,1]
    ls_rgb_f32: float32 [H,W,3] in [0,1]  (lightSwitch, already black bg)
    alpha_f32: float32 [H,W,1] in [0,1]
    Returns dict with keys for each mode.
    """
    gt_black = gt_rgba[:, :, :3] * alpha_f32
    ls_black = ls_rgb_f32  # already black bg from lightSwitch

    # Mode 1: black bg, raw
    m1 = _compute_one(gt_black, ls_black, lpips_fn)

    # Optimal scale on foreground
    scale = optimal_scale_fg(gt_black, ls_black, alpha_f32)

    ls_scaled = np.clip(ls_black * scale, 0.0, 1.0)

    # Mode 2: black bg, scaled
    m2 = _compute_one(gt_black, ls_scaled, lpips_fn)

    # Mode 3: white bg, scaled
    gt_white = gt_rgba[:, :, :3] * alpha_f32 + (1.0 - alpha_f32)
    ls_white = ls_scaled * alpha_f32 + (1.0 - alpha_f32)
    m3 = _compute_one(gt_white, ls_white, lpips_fn)

    return {
        "black_bg": m1,
        "black_bg_scaled": {**m2, "scale": scale},
        "white_bg_scaled": {**m3, "scale": scale},
    }


def _compute_one(img_a, img_b, lpips_fn):
    p = psnr_single(img_a, img_b)
    s = ssim_single(img_a, img_b)
    result = {"psnr": p, "ssim": s}
    if lpips_fn is not None:
        a_t = torch.from_numpy(img_a).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1
        b_t = torch.from_numpy(img_b).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1
        with torch.no_grad():
            result["lpips"] = float(lpips_fn(a_t, b_t).item())
    return result


# ---------------------------------------------------------------------------
# 1) Split gt_vs_pred grids
# ---------------------------------------------------------------------------

def split_gt_vs_pred(nvs_meta):
    out_base = os.path.join(OUT_DIR, "polyhaven_split")
    os.makedirs(out_base, exist_ok=True)

    count = 0
    for fname in sorted(os.listdir(GT_VS_PRED_DIR)):
        if not fname.endswith(".png"):
            continue
        scene_name = fname[:-4]
        meta = nvs_meta.get(scene_name)
        if meta is None:
            print(f"  [skip] {scene_name}: not in nvs_metadata")
            continue
        target_indices = meta["target"]

        img = Image.open(os.path.join(GT_VS_PRED_DIR, fname))
        w, h = img.size
        n_cols = len(target_indices)
        n_rows = 3
        cell_w = w // n_cols
        cell_h = h // n_rows
        row_names = ["unlitGT", "relitGT", "pred"]

        scene_out = os.path.join(out_base, scene_name)
        os.makedirs(scene_out, exist_ok=True)

        for col_i, view_idx in enumerate(target_indices):
            for row_i, row_name in enumerate(row_names):
                x0 = col_i * cell_w
                y0 = row_i * cell_h
                cell = img.crop((x0, y0, x0 + cell_w, y0 + cell_h))
                cell.save(os.path.join(scene_out, f"{view_idx:05d}_{row_name}.png"))
        count += 1

    print(f"[1] Split {count} gt_vs_pred grids -> {out_base}")


# ---------------------------------------------------------------------------
# 2) & 3) & 4)  Extract lightSwitch, compute metrics, organize flat
# ---------------------------------------------------------------------------

def _safe_mean(vals):
    valid = [v for v in vals if math.isfinite(v)]
    return sum(valid) / len(valid) if valid else None


def process_lightswitch(nvs_meta):
    ls_flat_dir     = os.path.join(OUT_DIR, "lightSwitch_flat")
    gt_flat_dir     = os.path.join(OUT_DIR, "gt_relit_flat")
    metrics_dir     = os.path.join(OUT_DIR, "lightSwitch_metrics")
    os.makedirs(ls_flat_dir, exist_ok=True)
    os.makedirs(gt_flat_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    lpips_fn = None
    if HAS_LPIPS:
        lpips_fn = lpips.LPIPS(net='alex')
        lpips_fn.eval()

    MODES = ["black_bg", "black_bg_scaled", "white_bg_scaled"]
    # accumulators: mode -> metric_name -> [values]
    agg = {m: {"psnr": [], "ssim": [], "lpips": []} for m in MODES}

    for input_scene in sorted(os.listdir(LIGHTSWITCH_DIR)):
        input_scene_dir = os.path.join(LIGHTSWITCH_DIR, input_scene)
        if not os.path.isdir(input_scene_dir):
            continue
        for target_scene in sorted(os.listdir(input_scene_dir)):
            render_dir = os.path.join(
                input_scene_dir, target_scene, "train", "ours_40000", "renders"
            )
            if not os.path.isdir(render_dir):
                continue

            meta = nvs_meta.get(input_scene)
            if meta is None:
                print(f"  [skip] {input_scene}: not in nvs_metadata")
                continue
            target_indices = meta["target"]

            gt_relit_scene_dir = os.path.join(GT_RELIT_DIR, target_scene)
            has_gt = os.path.isdir(gt_relit_scene_dir)

            pair_name = f"{input_scene}_TO_{target_scene}"
            pair_ls_dir = os.path.join(ls_flat_dir, pair_name)
            pair_gt_dir = os.path.join(gt_flat_dir, pair_name)
            os.makedirs(pair_ls_dir, exist_ok=True)
            if has_gt:
                os.makedirs(pair_gt_dir, exist_ok=True)

            per_view_metrics = []

            for view_idx in target_indices:
                ls_path = os.path.join(render_dir, f"{view_idx:05d}.png")
                if not os.path.exists(ls_path):
                    print(f"  [miss] lightSwitch render: {ls_path}")
                    continue

                ls_img = Image.open(ls_path)

                if has_gt:
                    gt_path = os.path.join(gt_relit_scene_dir, f"{view_idx:05d}.png")
                    if not os.path.exists(gt_path):
                        print(f"  [miss] GT relit: {gt_path}")
                        ls_img.save(os.path.join(pair_ls_dir, f"{view_idx:05d}.png"))
                        continue

                    gt_img = Image.open(gt_path)
                    gt_arr = np.array(gt_img)
                    if gt_arr.shape[2] == 4:
                        alpha_u8 = gt_arr[:, :, 3]
                    else:
                        alpha_u8 = np.full(gt_arr.shape[:2], 255, dtype=np.uint8)

                    gt_rgba = np.zeros((*gt_arr.shape[:2], 4), dtype=np.float32)
                    gt_rgba[:, :, :3] = gt_arr[:, :, :3].astype(np.float32) / 255.0
                    gt_rgba[:, :, 3] = alpha_u8.astype(np.float32) / 255.0
                    alpha_f32 = gt_rgba[:, :, 3:4]

                    ls_arr = np.array(ls_img.convert("RGB"))

                    # Resize lightSwitch if needed
                    if gt_arr.shape[:2] != ls_arr.shape[:2]:
                        h_gt, w_gt = gt_arr.shape[:2]
                        ls_arr = np.array(
                            Image.fromarray(ls_arr).resize((w_gt, h_gt), Image.LANCZOS)
                        )

                    ls_rgb_f32 = ls_arr.astype(np.float32) / 255.0

                    # Compute 3-mode metrics
                    modes = compute_metrics_suite(gt_rgba, ls_rgb_f32, alpha_f32, lpips_fn)
                    vm = {"view": view_idx}
                    for mode_name in MODES:
                        vm[mode_name] = modes[mode_name]
                        for mk in ("psnr", "ssim", "lpips"):
                            val = modes[mode_name].get(mk)
                            if val is not None and math.isfinite(val):
                                agg[mode_name][mk].append(val)
                    per_view_metrics.append(vm)

                    # Save GT (black bg)
                    gt_black = (gt_rgba[:, :, :3] * alpha_f32 * 255).clip(0, 255).astype(np.uint8)
                    Image.fromarray(gt_black).save(os.path.join(pair_gt_dir, f"{view_idx:05d}.png"))

                    # Save lightSwitch with transparent bg (RGBA, masked by GT alpha)
                    ls_masked = ls_rgb_f32 * alpha_f32
                    ls_rgba_save = np.zeros((*ls_arr.shape[:2], 4), dtype=np.uint8)
                    ls_rgba_save[:, :, :3] = (ls_masked * 255).clip(0, 255).astype(np.uint8)
                    ls_rgba_save[:, :, 3] = alpha_u8
                    Image.fromarray(ls_rgba_save, "RGBA").save(
                        os.path.join(pair_ls_dir, f"{view_idx:05d}.png")
                    )
                else:
                    ls_img.save(os.path.join(pair_ls_dir, f"{view_idx:05d}.png"))

            # Per-pair summary
            if per_view_metrics:
                pair_summary = {"pair": pair_name}
                for mode_name in MODES:
                    vals_p = [v[mode_name]["psnr"] for v in per_view_metrics if mode_name in v]
                    vals_s = [v[mode_name]["ssim"] for v in per_view_metrics if mode_name in v]
                    vals_l = [v[mode_name].get("lpips") for v in per_view_metrics
                              if mode_name in v and v[mode_name].get("lpips") is not None]
                    pair_summary[mode_name] = {
                        "psnr": _safe_mean(vals_p),
                        "ssim": _safe_mean(vals_s),
                        "lpips": _safe_mean(vals_l),
                    }
                    if mode_name == "black_bg_scaled":
                        scales = [v[mode_name].get("scale", 1.0) for v in per_view_metrics if mode_name in v]
                        pair_summary[mode_name]["avg_scale"] = _safe_mean(scales)

                with open(os.path.join(metrics_dir, f"{pair_name}.json"), "w") as f:
                    json.dump({"summary": pair_summary, "per_view": per_view_metrics}, f, indent=2)

                bb = pair_summary["black_bg"]
                bs = pair_summary["black_bg_scaled"]
                wb = pair_summary["white_bg_scaled"]
                print(f"  [{pair_name}]")
                print(f"    black_bg        : PSNR={bb['psnr']:.2f}  SSIM={bb['ssim']:.4f}  LPIPS={bb.get('lpips','N/A')}")
                print(f"    black_bg_scaled : PSNR={bs['psnr']:.2f}  SSIM={bs['ssim']:.4f}  LPIPS={bs.get('lpips','N/A')}  scale={bs.get('avg_scale','?'):.3f}")
                print(f"    white_bg_scaled : PSNR={wb['psnr']:.2f}  SSIM={wb['ssim']:.4f}  LPIPS={wb.get('lpips','N/A')}")

    # Overall summary
    print("\n" + "=" * 60)
    overall = {}
    for mode_name in MODES:
        n = len(agg[mode_name]["psnr"])
        m_psnr = _safe_mean(agg[mode_name]["psnr"])
        m_ssim = _safe_mean(agg[mode_name]["ssim"])
        m_lpips = _safe_mean(agg[mode_name]["lpips"])
        overall[mode_name] = {"n_views": n, "psnr": m_psnr, "ssim": m_ssim, "lpips": m_lpips}
        print(f"  {mode_name:20s} ({n:3d} views): PSNR={m_psnr:.4f}  SSIM={m_ssim:.4f}  LPIPS={m_lpips:.6f}" if m_psnr else f"  {mode_name}: no data")

    with open(os.path.join(metrics_dir, "_overall.json"), "w") as f:
        json.dump(overall, f, indent=2)


# ---------------------------------------------------------------------------

def main():
    with open(NVS_META_PATH) as f:
        nvs_meta = json.load(f)

    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 50)
    print("[1] Splitting gt_vs_pred grids ...")
    split_gt_vs_pred(nvs_meta)

    print("\n" + "=" * 50)
    print("[2-4] Processing lightSwitch_3d, computing metrics ...")
    process_lightswitch(nvs_meta)

    print(f"\nAll results saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
