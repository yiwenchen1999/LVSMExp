"""
Token-space iterative editing degradation test.

Supports single-scene (backward compat) and multi-scene modes.
At each iteration:
  1. Edit *previous* latent tokens with the next envmap (no re-encoding)
  2. Render at context views (for saving / metrics only)
  3. Carry the edited tokens forward to the next iteration
"""

import importlib
import os
import json
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from PIL import Image

from setup import init_config, init_distributed
from utils.degrade_test_utils import (
    collect_envmaps_and_gt,
    save_step,
    create_flattened_views,
)

# ---------------------------------------------------------------------------
# Setup (config, DDP, model, dataset) — identical to inference_editor.py
# ---------------------------------------------------------------------------

config = init_config()
os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))

ddp_info = init_distributed(seed=777)
dist.barrier()

torch.backends.cuda.matmul.allow_tf32 = config.training.use_tf32
torch.backends.cudnn.allow_tf32 = config.training.use_tf32
amp_dtype_mapping = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "tf32": torch.float32,
}

config.inference.same_pose = True

dataset_name = config.training.get("dataset_name", "data.dataset.Dataset")
mod, cls = dataset_name.rsplit(".", 1)
Dataset = importlib.import_module(mod).__dict__[cls]
dataset = Dataset(config)

dist.barrier()

mod, cls = config.model.class_name.rsplit(".", 1)
LVSM = importlib.import_module(mod).__dict__[cls]
model = LVSM(config).to(ddp_info.device)

checkpoint_dir = config.training.get("checkpoint_dir", "")
has_checkpoint = False
if checkpoint_dir:
    if os.path.isdir(checkpoint_dir):
        has_checkpoint = any(f.endswith(".pt") for f in os.listdir(checkpoint_dir))
    elif os.path.isfile(checkpoint_dir) and checkpoint_dir.endswith(".pt"):
        has_checkpoint = True

if has_checkpoint:
    result = model.load_ckpt(checkpoint_dir)
    if result is not None:
        print(f"Loaded checkpoint from {checkpoint_dir}")
    else:
        print(f"Warning: Failed to load checkpoint from {checkpoint_dir}, trying LVSM_checkpoint_dir...")
        has_checkpoint = False

if not has_checkpoint and config.training.get("LVSM_checkpoint_dir", ""):
    lvsm_dir = config.training.LVSM_checkpoint_dir
    print(f"Initializing LatentSceneEditor from Images2LatentScene at {lvsm_dir}")
    result = model.init_from_LVSM(lvsm_dir)
    if result is None:
        print("Warning: Failed to initialize from LVSM checkpoint")
    else:
        print("Successfully initialized from Images2LatentScene")
elif not has_checkpoint:
    print("Warning: No checkpoint found — starting from random initialization")

model = DDP(model, device_ids=[ddp_info.local_rank])

if ddp_info.is_main_process:
    import lpips  # noqa: F401
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

dist.barrier()
model.eval()

# ---------------------------------------------------------------------------
# Degradation test parameters
# ---------------------------------------------------------------------------

num_scenes = config.inference.get("degrade_num_scenes", 1)
num_iterations = config.inference.get("degrade_num_iterations", 100)
save_images = config.inference.get("degrade_save_images", num_scenes == 1)

single_scene_idx = config.inference.get("degrade_test_scene_idx", None)

if ddp_info.is_main_process:
    print(f"\n=== Token-Space Degradation Test ===")
    print(f"Scenes: {num_scenes}  |  Iterations: {num_iterations}  |  Save images: {save_images}")
    print(f"Output dir : {config.inference_out_dir}")
    os.makedirs(config.inference_out_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Build list of scene indices to process
# ---------------------------------------------------------------------------

if num_scenes == 1 and single_scene_idx is not None:
    candidate_indices = [single_scene_idx]
else:
    candidate_indices = list(range(len(dataset)))

# ---------------------------------------------------------------------------
# Multi-scene loop
# ---------------------------------------------------------------------------

step_psnrs = [[] for _ in range(num_iterations)]
step_ssims = [[] for _ in range(num_iterations)]
step_lpipss = [[] for _ in range(num_iterations)]
all_records = []

processed = 0
skipped = 0

for dataset_idx in candidate_indices:
    if processed >= num_scenes:
        break

    try:
        sample = dataset[dataset_idx]
    except Exception as e:
        if ddp_info.is_main_process:
            print(f"  Skip idx {dataset_idx}: {e}")
        skipped += 1
        continue

    scene_name = sample["scene_name"]
    scene_path = dataset.all_scene_paths[dataset_idx].strip()

    batch = {}
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.unsqueeze(0).to(ddp_info.device)
        else:
            batch[k] = [v]

    with torch.no_grad():
        input_data, target_data = model.module.process_data(
            batch, has_target_image=True, target_has_input=True, compute_rays=True
        )

    num_input_views = input_data.image.shape[1]
    image_indices = input_data.index[0, :, 0].cpu().tolist()

    envmaps = collect_envmaps_and_gt(
        dataset, scene_path, scene_name, image_indices, num_input_views, ddp_info.device
    )

    if len(envmaps) == 0:
        if ddp_info.is_main_process:
            print(f"  Skip '{scene_name}': no envmap variations")
        skipped += 1
        del batch, input_data, target_data
        torch.cuda.empty_cache()
        continue

    processed += 1

    safe_name = "".join(c for c in scene_name if c.isalnum() or c in ("_", "-"))[:100]
    scene_dir = os.path.join(config.inference_out_dir, safe_name)

    if ddp_info.is_main_process:
        print(f"\n[{processed}/{num_scenes}] {scene_name} "
              f"(idx={dataset_idx}, {len(envmaps)} envmaps, views={image_indices})")
        os.makedirs(scene_dir, exist_ok=True)

        if save_images:
            orig = input_data.image[0].detach().cpu()
            for vi in range(num_input_views):
                img = (orig[vi].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(img).save(os.path.join(scene_dir, f"original_input_v{vi}.png"))

        seq = [{"step": i, "envmap_name": envmaps[i % len(envmaps)][3]} for i in range(num_iterations)]
        with open(os.path.join(scene_dir, "envmap_sequence.json"), "w") as f:
            json.dump(seq, f, indent=2)

    dist.barrier()

    # --- Iterative loop for this scene ---
    scene_metrics = []

    with torch.no_grad(), torch.autocast(
        enabled=config.training.use_amp,
        device_type="cuda",
        dtype=amp_dtype_mapping[config.training.amp_dtype],
    ):
        latent_tokens, n_patches, d = model.module.reconstructor(input_data)
        current_tokens = latent_tokens.detach().clone()

        for step in range(num_iterations):
            env_ldr, env_hdr, gt_images, env_name = envmaps[step % len(envmaps)]
            input_data.env_ldr = env_ldr
            input_data.env_hdr = env_hdr

            current_tokens = model.module.edit_scene_with_env(current_tokens, input_data)
            rendered = model.module.renderer(current_tokens, target_data, n_patches, d)

            if np.random.randint(0, 10) < 8 and step < 17:
                current_tokens = latent_tokens.detach().clone()
            else:
                latent_tokens = current_tokens

            if ddp_info.is_main_process:
                metrics = save_step(
                    step, rendered, gt_images, env_name,
                    scene_dir, num_input_views, save_images=save_images,
                )
                scene_metrics.append(metrics)
                step_psnrs[step].append(metrics["psnr"])
                step_ssims[step].append(metrics["ssim"])
                step_lpipss[step].append(metrics["lpips"])
                all_records.append(
                    f"{scene_name},{step},{metrics['psnr']:.4f},"
                    f"{metrics['ssim']:.6f},{metrics['lpips']:.6f},{env_name}"
                )
                print(
                    f"  Step {step:03d}: PSNR={metrics['psnr']:.2f}  "
                    f"SSIM={metrics['ssim']:.4f}  LPIPS={metrics['lpips']:.4f}  "
                    f"(env: {env_name})"
                )

    # Per-scene summary
    if ddp_info.is_main_process:
        with open(os.path.join(scene_dir, "summary.json"), "w") as f:
            json.dump(scene_metrics, f, indent=2)
        if save_images:
            create_flattened_views(scene_dir, num_iterations, num_input_views)

    del batch, input_data, target_data
    torch.cuda.empty_cache()
    dist.barrier()

# ---------------------------------------------------------------------------
# Aggregated results
# ---------------------------------------------------------------------------

if ddp_info.is_main_process:
    print(f"\n=== Done: {processed} scenes processed, {skipped} skipped ===")

    avg_path = os.path.join(config.inference_out_dir, "all_scenes_avg.csv")
    with open(avg_path, "w") as f:
        f.write("step,avg_psnr,avg_ssim,avg_lpips,num_scenes\n")
        for s in range(num_iterations):
            if step_psnrs[s]:
                f.write(
                    f"{s},{np.mean(step_psnrs[s]):.4f},"
                    f"{np.mean(step_ssims[s]):.6f},"
                    f"{np.mean(step_lpipss[s]):.6f},"
                    f"{len(step_psnrs[s])}\n"
                )

    detail_path = os.path.join(config.inference_out_dir, "per_scene_step_metrics.csv")
    with open(detail_path, "w") as f:
        f.write("scene_name,step,psnr,ssim,lpips,envmap_name\n")
        for rec in all_records:
            f.write(rec + "\n")

    print(f"Average metrics: {avg_path}")
    print(f"Per-scene metrics: {detail_path}")

dist.barrier()
dist.destroy_process_group()
exit(0)
