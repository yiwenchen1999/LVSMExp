"""Shared utilities for iterative editing degradation tests."""

import os
import json
import shutil
import numpy as np
import PIL
import torch
from PIL import Image
from utils.metric_utils import compute_psnr, compute_ssim, compute_lpips


def load_envmap_image(path):
    """Load a single envmap image (LDR or HDR) as [3, H, W] float tensor."""
    img = PIL.Image.open(path)
    img.load()
    arr = np.array(img) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=2)
    elif arr.shape[2] == 4:
        rgb, alpha = arr[:, :, :3], arr[:, :, 3:4]
        arr = rgb * alpha + (1.0 - alpha)
    else:
        arr = arr[:, :, :3]
    return torch.from_numpy(arr).permute(2, 0, 1).float()


def collect_envmaps_and_gt(dataset, scene_path, scene_name, image_indices, num_input_views, device,
                          exclude_white_env=False):
    """
    Collect all envmap-type lighting variations for the same object.

    Args:
        dataset: Dataset instance (must have _extract_object_id and preprocess_frames)
        scene_path: Path to the current scene's JSON file
        scene_name: Name of the current scene
        image_indices: List of frame indices (context views) to load GT images at
        num_input_views: Number of input (context) views to load GT for
        device: torch device
        exclude_white_env: If True, skip candidates whose name contains 'white_env'

    Returns:
        Sorted list of (env_ldr, env_hdr, gt_images, candidate_name) tuples.
        - env_ldr:  [1, num_input_views, 3, eh, ew] (broadcast to all views)
        - env_hdr:  [1, num_input_views, 3, eh, ew]
        - gt_images: [1, num_input_views, 3, h, w]
        - candidate_name: str
    """
    object_id = dataset._extract_object_id(scene_name)
    base_dir = os.path.dirname(os.path.dirname(scene_path))
    metadata_dir = os.path.join(base_dir, "metadata")

    context_indices = image_indices[:num_input_views]

    all_json_files = sorted(f for f in os.listdir(metadata_dir) if f.endswith(".json"))
    candidates = []
    for json_file in all_json_files:
        candidate_name = json_file[:-5]
        if candidate_name == scene_name:
            continue
        if dataset._extract_object_id(candidate_name) != object_id:
            continue
        if exclude_white_env and "white_env" in candidate_name:
            continue
        envmaps_dir = os.path.join(base_dir, "envmaps", candidate_name)
        if not os.path.exists(envmaps_dir):
            continue
        ldr_files = [f for f in os.listdir(envmaps_dir) if f.endswith("_ldr.png")]
        if not ldr_files:
            continue
        candidates.append(candidate_name)

    candidates.sort()

    result = []
    for candidate_name in candidates:
        # candidate_name = scene_name
        envmaps_dir = os.path.join(base_dir, "envmaps", candidate_name)
        ldr_files = sorted(f for f in os.listdir(envmaps_dir) if f.endswith("_ldr.png"))
        env_idx = int(ldr_files[0].split("_")[0])

        ldr_path = os.path.join(envmaps_dir, f"{env_idx:05d}_ldr.png")
        hdr_path = os.path.join(envmaps_dir, f"{env_idx:05d}_hdr.png")
        if not os.path.exists(hdr_path):
            continue

        env_ldr = load_envmap_image(ldr_path)  # [3, eh, ew]
        env_hdr = load_envmap_image(hdr_path)

        # Broadcast to [1, num_input_views, 3, eh, ew]
        env_ldr = env_ldr.unsqueeze(0).unsqueeze(0).expand(1, num_input_views, -1, -1, -1).clone().to(device)
        env_hdr = env_hdr.unsqueeze(0).unsqueeze(0).expand(1, num_input_views, -1, -1, -1).clone().to(device)

        # Load GT images at context view frame indices
        candidate_json_path = os.path.join(metadata_dir, candidate_name + ".json")
        try:
            with open(candidate_json_path, "r") as f:
                candidate_data = json.load(f)
        except Exception:
            print(f"  Skipping {candidate_name}: cannot read JSON")
            continue

        candidate_frames = candidate_data.get("frames", [])
        if len(candidate_frames) <= max(context_indices):
            print(f"  Skipping {candidate_name}: not enough frames "
                  f"({len(candidate_frames)} < {max(context_indices) + 1})")
            continue

        gt_frames_chosen = [candidate_frames[ic] for ic in context_indices]
        gt_image_paths = [frm["image_path"] for frm in gt_frames_chosen]

        if not all(os.path.exists(p) for p in gt_image_paths):
            print(f"  Skipping {candidate_name}: missing GT image files")
            continue

        gt_images, _, _ = dataset.preprocess_frames(gt_frames_chosen, gt_image_paths)
        gt_images = gt_images.unsqueeze(0).to(device)  # [1, v, 3, h, w]

        result.append((env_ldr, env_hdr, gt_images, candidate_name))

    return result


def save_step(step, rendered, gt_images, env_name, scene_dir, num_views, save_images=True):
    """
    Compute metrics for one iteration step, optionally saving images.

    Returns:
        dict with summary metrics for this step.
    """
    rendered_f = rendered[0].detach().float()
    gt_f = gt_images[0].detach().float()

    if save_images:
        step_dir = os.path.join(scene_dir, f"step_{step:03d}")
        os.makedirs(step_dir, exist_ok=True)

        rendered_cpu = rendered_f.cpu()
        gt_cpu = gt_f.cpu()

        for vi in range(num_views):
            img = (rendered_cpu[vi].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(img).save(os.path.join(step_dir, f"rendered_v{vi}.png"))

            gt_img = (gt_cpu[vi].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(gt_img).save(os.path.join(step_dir, f"gt_relit_v{vi}.png"))

    psnr_vals = compute_psnr(gt_f, rendered_f)
    lpips_vals = compute_lpips(gt_f, rendered_f)
    ssim_vals = compute_ssim(gt_f, rendered_f)

    metrics = {
        "step": step,
        "envmap_name": env_name,
        "psnr": float(psnr_vals.mean()),
        "ssim": float(ssim_vals.mean()),
        "lpips": float(lpips_vals.mean()),
    }

    if save_images:
        metrics["per_view"] = [
            {
                "view": vi,
                "psnr": float(psnr_vals[vi]),
                "ssim": float(ssim_vals[vi]),
                "lpips": float(lpips_vals[vi]),
            }
            for vi in range(num_views)
        ]
        with open(os.path.join(step_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    return metrics


def create_flattened_views(scene_dir, num_steps, num_views, entry_id=None, scene_name=None, env_name=None):
    """
    Create flattened organization of rendered images for easy browsing.

    1) Per-scene: flattened/v{i}/step_XXX.png — grouped by view
    2) If entry_id/scene_name/env_name provided: also link to global flattened_all/ at parent
    """
    flat_dir = os.path.join(scene_dir, "flattened")
    for vi in range(num_views):
        view_dir = os.path.join(flat_dir, f"v{vi}")
        os.makedirs(view_dir, exist_ok=True)
        for step in range(num_steps):
            src = os.path.join(scene_dir, f"step_{step:03d}", f"rendered_v{vi}.png")
            dst = os.path.join(view_dir, f"step_{step:03d}.png")
            if os.path.exists(src):
                if os.path.exists(dst) or os.path.islink(dst):
                    os.remove(dst)
                try:
                    os.symlink(os.path.abspath(src), dst)
                except OSError:
                    shutil.copy2(src, dst)

    # Global flattened folder: all samples in one flat directory
    if entry_id is not None and scene_name is not None and env_name is not None:
        parent = os.path.dirname(scene_dir)
        flat_all_dir = os.path.join(parent, "flattened_all")
        os.makedirs(flat_all_dir, exist_ok=True)
        safe_scene = scene_name.replace("/", "_")[:64]
        safe_env = env_name.replace("/", "_")[:64]
        prefix = f"{entry_id:04d}_{safe_scene}_{safe_env}"
        for vi in range(num_views):
            for step in range(num_steps):
                src = os.path.join(scene_dir, f"step_{step:03d}", f"rendered_v{vi}.png")
                dst = os.path.join(flat_all_dir, f"{prefix}_step{step:03d}_v{vi}.png")
                if os.path.exists(src):
                    try:
                        if os.path.exists(dst) or os.path.islink(dst):
                            os.remove(dst)
                        os.symlink(os.path.abspath(src), dst)
                    except OSError:
                        shutil.copy2(src, dst)
