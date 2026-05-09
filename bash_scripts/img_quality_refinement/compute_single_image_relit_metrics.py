#!/usr/bin/env python3
"""
Compute PSNR, SSIM, and LPIPS between paired relit_gt / relit_pred tiles under
result_previews/resolution_comparisons/single_image/<infer_256|infer_512>/...

Uses utils.metric_utils (same as training/inference metrics).

Outputs per resolution (separate CSV + JSON summary):
  <single_image_root>/metrics_relit_infer_256.csv
  <single_image_root>/metrics_relit_infer_256_summary.json
  (and infer_512)

CSV columns: scene, iter, view, psnr, ssim, lpips (lpips column empty if --skip-lpips)
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.metric_utils import compute_psnr, compute_ssim  # noqa: E402

_DEFAULT_SINGLE_IMAGE = _REPO_ROOT / "result_previews/resolution_comparisons/single_image"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base",
        type=Path,
        default=_DEFAULT_SINGLE_IMAGE,
        help=f"Root containing infer_256/, infer_512/ (default: {_DEFAULT_SINGLE_IMAGE})",
    )
    p.add_argument(
        "--infer",
        nargs="*",
        default=["infer_256", "infer_512"],
        metavar="NAME",
        help="Subfolders under --base to scan (default: infer_256 infer_512).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="torch device for LPIPS / tensors (default: cuda if available else cpu).",
    )
    p.add_argument(
        "--skip-lpips",
        action="store_true",
        help="Do not compute or load LPIPS (faster; CSV/JSON use empty / null for lpips).",
    )
    return p.parse_args()


def pil_to_tensor01(im: Image.Image) -> torch.Tensor:
    arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


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
            if pr is None:
                print(f"Warning: missing pred for {gt}", file=sys.stderr)
                continue
            pairs.append((gt, pr))
    return pairs


def scene_iter_from_gt(gt_path: Path, infer_dir: Path) -> tuple[str, str]:
    rel = gt_path.relative_to(infer_dir)
    return rel.parts[0], rel.parts[1]


def process_infer_folder(
    infer_dir: Path,
    out_csv: Path,
    out_json: Path,
    device: torch.device,
    skip_lpips: bool,
) -> None:
    pairs = collect_pairs(infer_dir)
    if not pairs:
        print(f"No relit_gt / relit_pred pairs under {infer_dir}", file=sys.stderr)
        return

    compute_lpips_fn = None
    if not skip_lpips:
        from utils.metric_utils import compute_lpips as compute_lpips_fn  # noqa: E402

    rows: list[dict[str, object]] = []
    for gp, pp in pairs:
        t_g = pil_to_tensor01(Image.open(gp)).unsqueeze(0).to(device)
        t_p = pil_to_tensor01(Image.open(pp)).unsqueeze(0).to(device)
        if t_g.shape != t_p.shape:
            raise ValueError(f"Shape mismatch {gp} vs {pp}: {t_g.shape} vs {t_p.shape}")

        psnr_v = compute_psnr(t_g, t_p)[0]
        ssim_v = compute_ssim(t_g, t_p)[0]
        if skip_lpips:
            lp_val: float | str = ""
        else:
            assert compute_lpips_fn is not None
            lp_v = compute_lpips_fn(t_g, t_p)
            if lp_v.ndim > 0:
                lp_v = lp_v[0]
            lp_val = float(lp_v)

        scene, it_name = scene_iter_from_gt(gp, infer_dir)
        rows.append(
            {
                "scene": scene,
                "iter": it_name,
                "view": gp.stem,
                "psnr": float(psnr_v),
                "ssim": float(ssim_v),
                "lpips": lp_val,
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scene", "iter", "view", "psnr", "ssim", "lpips"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    psnrs = [float(r["psnr"]) for r in rows if np.isfinite(float(r["psnr"]))]
    lpips_v = [float(r["lpips"]) for r in rows if r["lpips"] != ""]
    ssims = [float(r["ssim"]) for r in rows]

    summary: dict[str, object] = {
        "infer_dir": str(infer_dir),
        "num_pairs": len(rows),
        "psnr_mean": float(statistics.mean(psnrs)) if psnrs else None,
        "psnr_std": float(statistics.pstdev(psnrs)) if len(psnrs) > 1 else 0.0,
        "ssim_mean": float(statistics.mean(ssims)) if ssims else None,
        "ssim_std": float(statistics.pstdev(ssims)) if len(ssims) > 1 else 0.0,
        "device": str(device),
        "lpips_skipped": skip_lpips,
    }
    if not skip_lpips:
        summary["lpips_mean"] = float(statistics.mean(lpips_v)) if lpips_v else None
        summary["lpips_std"] = float(statistics.pstdev(lpips_v)) if len(lpips_v) > 1 else 0.0
    else:
        summary["lpips_mean"] = None
        summary["lpips_std"] = None

    out_json.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_csv} ({len(rows)} rows)")
    print(f"Wrote {out_json}")
    if skip_lpips:
        print(
            f"  PSNR mean={summary['psnr_mean']:.4f}  SSIM mean={summary['ssim_mean']:.4f}  "
            "(LPIPS skipped)"
        )
    else:
        print(
            f"  PSNR mean={summary['psnr_mean']:.4f}  SSIM mean={summary['ssim_mean']:.4f}  "
            f"LPIPS mean={summary['lpips_mean']:.4f}"
        )


def main() -> None:
    args = parse_args()
    base = args.base.resolve()
    if not base.is_dir():
        raise SystemExit(f"Not a directory: {base}")

    device = torch.device(args.device)

    for name in args.infer:
        infer_dir = base / name
        if not infer_dir.is_dir():
            print(f"Skip (missing): {infer_dir}", file=sys.stderr)
            continue
        out_csv = base / f"metrics_relit_{name}.csv"
        out_json = base / f"metrics_relit_{name}_summary.json"
        print(f"=== {name} ===")
        process_infer_folder(infer_dir, out_csv, out_json, device, args.skip_lpips)

    print("Done.")


if __name__ == "__main__":
    main()
