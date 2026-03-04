"""
Token-space iterative editing degradation test.

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

# Force same_pose so target views == context views
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
    import lpips  # noqa: F401  — pre-download weights on main process
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

dist.barrier()
model.eval()

# ---------------------------------------------------------------------------
# Degradation test parameters
# ---------------------------------------------------------------------------

scene_idx = config.inference.get("degrade_test_scene_idx", 0)
num_iterations = config.inference.get("degrade_num_iterations", 100)

if ddp_info.is_main_process:
    print(f"\n=== Token-Space Degradation Test ===")
    print(f"Scene index: {scene_idx}  |  Iterations: {num_iterations}")
    print(f"Output dir : {config.inference_out_dir}")

# ---------------------------------------------------------------------------
# Load one scene and build batch
# ---------------------------------------------------------------------------

sample = dataset[scene_idx]
batch = {}
for k, v in sample.items():
    if isinstance(v, torch.Tensor):
        batch[k] = v.unsqueeze(0).to(ddp_info.device)
    else:
        batch[k] = [v]

scene_name = sample["scene_name"]
scene_path = dataset.all_scene_paths[scene_idx].strip()

# ---------------------------------------------------------------------------
# Process data → input_data / target_data with rays and env_dir
# ---------------------------------------------------------------------------

with torch.no_grad():
    input_data, target_data = model.module.process_data(
        batch, has_target_image=True, target_has_input=True, compute_rays=True
    )

num_input_views = input_data.image.shape[1]
image_indices = input_data.index[0, :, 0].cpu().tolist()

if ddp_info.is_main_process:
    print(f"Scene       : {scene_name}")
    print(f"Context views: {num_input_views}  frame indices: {image_indices}")

# ---------------------------------------------------------------------------
# Collect envmaps + GT images (deterministic order — same as image-space)
# ---------------------------------------------------------------------------

envmaps = collect_envmaps_and_gt(
    dataset, scene_path, scene_name, image_indices, num_input_views, ddp_info.device
)

if ddp_info.is_main_process:
    print(f"Collected {len(envmaps)} envmap variations:")
    for i, (_, _, _, name) in enumerate(envmaps):
        print(f"  [{i}] {name}")

assert len(envmaps) > 0, f"No envmap variations found for object in scene '{scene_name}'"

# ---------------------------------------------------------------------------
# Prepare output directory
# ---------------------------------------------------------------------------

safe_name = "".join(c for c in scene_name if c.isalnum() or c in ("_", "-"))[:100]
scene_dir = os.path.join(config.inference_out_dir, safe_name)

if ddp_info.is_main_process:
    os.makedirs(scene_dir, exist_ok=True)

    # Save original input images
    orig = input_data.image[0].detach().cpu()
    for vi in range(num_input_views):
        img = (orig[vi].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(scene_dir, f"original_input_v{vi}.png"))

    # Save envmap sequence (identical ordering to image-space script)
    seq = [{"step": i, "envmap_name": envmaps[i % len(envmaps)][3]} for i in range(num_iterations)]
    with open(os.path.join(scene_dir, "envmap_sequence.json"), "w") as f:
        json.dump(seq, f, indent=2)

dist.barrier()

# ---------------------------------------------------------------------------
# Iterative loop
# ---------------------------------------------------------------------------

all_metrics = []

with torch.no_grad(), torch.autocast(
    enabled=config.training.use_amp,
    device_type="cuda",
    dtype=amp_dtype_mapping[config.training.amp_dtype],
):
    # Reconstruct ONCE from original input images
    latent_tokens, n_patches, d = model.module.reconstructor(input_data)
    current_tokens = latent_tokens.detach().clone()

    for step in range(num_iterations):
        # 1. Pick envmap for this step (same cycle as image-space)
        env_ldr, env_hdr, gt_images, env_name = envmaps[step % len(envmaps)]
        input_data.env_ldr = env_ldr
        input_data.env_hdr = env_hdr

        # 2. Edit current tokens (NOT re-encoded from pixels)
        current_tokens = model.module.edit_scene_with_env(current_tokens, input_data)

        # 3. Render for visualisation and metrics
        rendered = model.module.renderer(current_tokens, target_data, n_patches, d)

        if np.random.randint(0, 10) <8 and step <17:
            current_tokens = latent_tokens.detach().clone()
        else:
            latent_tokens = current_tokens
            

        # 4. Save results + compute metrics vs GT relit images
        if ddp_info.is_main_process:
            metrics = save_step(step, rendered, gt_images, env_name, scene_dir, num_input_views)
            all_metrics.append(metrics)
            if step % 10 == 0 or step == num_iterations - 1:
                print(
                    f"  Step {step:03d}: PSNR={metrics['psnr']:.2f}  "
                    f"SSIM={metrics['ssim']:.4f}  LPIPS={metrics['lpips']:.4f}  "
                    f"(env: {env_name})"
                )

# ---------------------------------------------------------------------------
# Summary + flattened views
# ---------------------------------------------------------------------------

if ddp_info.is_main_process:
    with open(os.path.join(scene_dir, "summary.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    create_flattened_views(scene_dir, num_iterations, num_input_views)
    print(f"\nFlattened view folders created in {os.path.join(scene_dir, 'flattened')}")
    print(f"=== Done. Results saved to {scene_dir} ===")

torch.cuda.empty_cache()
dist.barrier()
dist.destroy_process_group()
exit(0)
