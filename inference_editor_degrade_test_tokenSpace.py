"""
Iterative editing degradation test — TOKEN SPACE.

For each scene and each envmap candidate:
  step 0: reconstruct(original_images) -> edit(envmap) -> render -> measure vs GT
  step n: edit(tokens_from_step_n-1, same envmap) -> render -> measure vs GT
The reconstructor is called only once (step 0).  Subsequent steps re-apply the
editor directly on the previously edited tokens, isolating degradation caused by
repeated editor application in latent space.
"""

import importlib
import os
import json
import csv
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from easydict import EasyDict as edict
from setup import init_config, init_distributed
from utils.degrade_test_utils import collect_envmaps_and_gt, save_step, create_flattened_views

# ── Setup (same as inference_editor.py) ───────────────────────────────
config = init_config()
os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))
ddp_info = init_distributed(seed=777)
dist.barrier()

torch.backends.cuda.matmul.allow_tf32 = config.training.use_tf32
torch.backends.cudnn.allow_tf32 = config.training.use_tf32
amp_dtype_mapping = {
    "fp16": torch.float16, "bf16": torch.bfloat16,
    "fp32": torch.float32, "tf32": torch.float32,
}

# ── Data ──────────────────────────────────────────────────────────────
dataset_name = config.training.get("dataset_name", "data.dataset.Dataset")
module, class_name = dataset_name.rsplit(".", 1)
Dataset = importlib.import_module(module).__dict__[class_name]
dataset = Dataset(config)

datasampler = DistributedSampler(dataset)
dataloader = DataLoader(
    dataset,
    batch_size=config.training.batch_size_per_gpu,
    shuffle=False,
    num_workers=config.training.num_workers,
    prefetch_factor=config.training.prefetch_factor,
    persistent_workers=True,
    pin_memory=False,
    drop_last=True,
    sampler=datasampler,
)
dist.barrier()

# ── Model ─────────────────────────────────────────────────────────────
module, class_name = config.model.class_name.rsplit(".", 1)
LVSM = importlib.import_module(module).__dict__[class_name]
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
        print(f"Warning: load failed from {checkpoint_dir}, trying LVSM_checkpoint_dir")
        has_checkpoint = False

if not has_checkpoint and config.training.get("LVSM_checkpoint_dir", ""):
    lvsm_dir = config.training.LVSM_checkpoint_dir
    print(f"Initializing from Images2LatentScene at {lvsm_dir}")
    result = model.init_from_LVSM(lvsm_dir)
    if result is None:
        print("Warning: fresh random initialization")
    else:
        print("Successfully initialized from Images2LatentScene")
elif not has_checkpoint:
    print("Warning: no checkpoint – random initialization")

model = DDP(model, device_ids=[ddp_info.local_rank])

if ddp_info.is_main_process:
    import lpips  # noqa: F401
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
dist.barrier()

# ── Config ────────────────────────────────────────────────────────────
num_iterations = config.training.get("degrade_num_iterations", 20)
num_input_views = config.training.num_input_views
out_dir = config.inference_out_dir
exclude_white_env = config.training.get("degrade_exclude_white_env", False)
max_entries = config.training.get("degrade_max_entries", 50)

datasampler.set_epoch(0)
model.eval()

step_psnrs = [[] for _ in range(num_iterations)]
step_ssims = [[] for _ in range(num_iterations)]
step_lpipss = [[] for _ in range(num_iterations)]
per_scene_rows = []
entry_count = 0

with torch.no_grad(), torch.autocast(
    enabled=config.training.use_amp,
    device_type="cuda",
    dtype=amp_dtype_mapping[config.training.amp_dtype],
):
    for batch in dataloader:
        if entry_count >= max_entries:
            break
        batch = {
            k: v.to(ddp_info.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        input_data, target_data = model.module.process_data(
            batch, has_target_image=True,
            target_has_input=config.training.target_has_input,
            compute_rays=True,
        )

        scene_names = batch.get("scene_name", ["unknown"])
        if isinstance(scene_names, str):
            scene_names = [scene_names]

        bs = input_data.image.shape[0]

        for b_idx in range(bs):
            scene_name = scene_names[b_idx] if b_idx < len(scene_names) else f"scene_{b_idx}"

            if hasattr(input_data, "index") and input_data.index is not None:
                image_indices = input_data.index[b_idx, :, 0].cpu().tolist()
            else:
                image_indices = list(range(input_data.image.shape[1]))

            scene_path = dataset.all_scene_paths[0]
            for sp in dataset.all_scene_paths:
                if scene_name in sp:
                    scene_path = sp
                    break

            def _slice(data, idx):
                out = edict()
                for k, v in data.items():
                    if isinstance(v, torch.Tensor) and v.dim() >= 2:
                        out[k] = v[idx : idx + 1]
                    else:
                        out[k] = v
                return out

            inp = _slice(input_data, b_idx)
            tgt = _slice(target_data, b_idx)

            envmap_list = collect_envmaps_and_gt(
                dataset, scene_path, scene_name, image_indices,
                num_input_views, ddp_info.device,
                exclude_white_env=exclude_white_env,
            )
            if not envmap_list:
                print(f"  [skip] {scene_name}: no envmap candidates")
                continue

            # Reconstruct once from original images
            latent_tokens, n_patches, d = model.module.reconstructor(inp)
            latent_tokens = latent_tokens.detach().clone()

            for env_ldr, env_hdr, gt_images, env_name in envmap_list:
                if entry_count >= max_entries:
                    break

                scene_dir = os.path.join(out_dir, f"{entry_count:04d}__{scene_name}__{env_name}")
                os.makedirs(scene_dir, exist_ok=True)

                env_input = edict(
                    env_ldr=env_ldr,
                    env_hdr=env_hdr,
                    env_dir=inp.env_dir,
                )

                # Step 0: apply editor once to get the "base" edited tokens.
                base_tokens = model.module.edit_scene_with_env(
                    latent_tokens.clone(), env_input,
                )
                current_tokens = base_tokens.clone()

                for step in range(num_iterations):
                    # Re-apply editor on current_tokens
                    re_edited = model.module.edit_scene_with_env(
                        current_tokens, env_input,
                    )

                    ramp_steps = 12
                    if step == 0:
                        current_tokens = base_tokens.clone()
                    elif step < ramp_steps:
                        # Linear ramp: replace_ratio grows from 0 to 1
                        # over steps 1..ramp_steps-1
                        replace_ratio = step / (ramp_steps - 1)
                        mask = (
                            torch.rand(
                                base_tokens.shape[:2],
                                device=base_tokens.device,
                            )
                            < replace_ratio
                        ).unsqueeze(-1)  # [b, n_tokens, 1]
                        current_tokens = torch.where(mask, re_edited, base_tokens)
                        # Where we chose the edited token, commit that
                        # decision into base_tokens for future steps.
                        base_tokens = torch.where(mask, re_edited, base_tokens)
                    else:
                        # Step >= ramp_steps: fully use re-edited tokens
                        current_tokens = re_edited
                        base_tokens = re_edited.clone()

                    rendered = model.module.renderer(current_tokens, tgt, n_patches, d)
                    rendered = rendered.clamp(0, 1)

                    # Save rendered results at every step for flattened folder organization
                    metrics = save_step(
                        step, rendered, gt_images, env_name,
                        scene_dir, num_input_views,
                        save_images=True,
                    )

                    step_psnrs[step].append(metrics["psnr"])
                    step_ssims[step].append(metrics["ssim"])
                    step_lpipss[step].append(metrics["lpips"])
                    per_scene_rows.append({
                        "scene_name": scene_name,
                        "step": step,
                        "psnr": metrics["psnr"],
                        "ssim": metrics["ssim"],
                        "lpips": metrics["lpips"],
                        "envmap_name": env_name,
                    })

                    if ddp_info.is_main_process and step % 5 == 0:
                        print(
                            f"  TokenSpace step {step:03d}: "
                            f"PSNR={metrics['psnr']:.2f}  "
                            f"SSIM={metrics['ssim']:.4f}  "
                            f"LPIPS={metrics['lpips']:.4f}  "
                            f"[{scene_name} / {env_name}]"
                        )

                create_flattened_views(
                    scene_dir, num_iterations, num_input_views,
                    entry_id=entry_count, scene_name=scene_name, env_name=env_name,
                )

                entry_count += 1
                if ddp_info.is_main_process:
                    print(f"  [entry {entry_count}/{max_entries}] done: {scene_name} / {env_name}")
                if entry_count >= max_entries:
                    if ddp_info.is_main_process:
                        print(f"Early stop: reached {max_entries} entries")
                    break

    torch.cuda.empty_cache()

dist.barrier()

# ── Write CSVs (main process only) ───────────────────────────────────
if ddp_info.is_main_process:
    os.makedirs(out_dir, exist_ok=True)

    avg_csv = os.path.join(out_dir, "degrade_avg_token_space.csv")
    with open(avg_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "avg_psnr", "avg_ssim", "avg_lpips", "num_scenes"])
        for s in range(num_iterations):
            if step_psnrs[s]:
                w.writerow([
                    s,
                    f"{np.mean(step_psnrs[s]):.6f}",
                    f"{np.mean(step_ssims[s]):.6f}",
                    f"{np.mean(step_lpipss[s]):.6f}",
                    len(step_psnrs[s]),
                ])
    print(f"Average CSV -> {avg_csv}")

    detail_csv = os.path.join(out_dir, "degrade_detail_token_space.csv")
    with open(detail_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene_name", "step", "psnr", "ssim", "lpips", "envmap_name"])
        for r in per_scene_rows:
            w.writerow([
                r["scene_name"], r["step"],
                f"{r['psnr']:.6f}", f"{r['ssim']:.6f}",
                f"{r['lpips']:.6f}", r["envmap_name"],
            ])
    print(f"Detail CSV  -> {detail_csv}")

dist.barrier()
dist.destroy_process_group()
exit(0)
