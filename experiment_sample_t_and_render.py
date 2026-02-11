# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).

import importlib
import os
import torch
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from easydict import EasyDict as edict
from PIL import Image
from setup import init_config, init_distributed
from model.LVSM_x_prediction_editor_noise2relit_overfit_chamfer import (
    chamfer_distance, 
    hungarian_matching_loss
)

# Load config and read(override) arguments from CLI
config = init_config()

os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))

# Set up DDP and Fix random seed
ddp_info = init_distributed(seed=777)
dist.barrier()

# Set up tf32
torch.backends.cuda.matmul.allow_tf32 = config.training.use_tf32
torch.backends.cudnn.allow_tf32 = config.training.use_tf32

# Load dataset
dataset_name = config.training.get("dataset_name", "data.dataset.Dataset")
module, class_name = dataset_name.rsplit(".", 1)
Dataset = importlib.import_module(module).__dict__[class_name]
dataset = Dataset(config)

datasampler = DistributedSampler(dataset)
dataloader = DataLoader(
    dataset,
    batch_size=1,  # Process one sample at a time for this experiment
    shuffle=False,
    num_workers=config.training.num_workers,
    prefetch_factor=config.training.prefetch_factor,
    persistent_workers=True,
    pin_memory=False,
    drop_last=False,
    sampler=datasampler
)

# Import model and load checkpoint
module, class_name = config.model.class_name.rsplit(".", 1)
LVSM = importlib.import_module(module).__dict__[class_name]
model = LVSM(config).to(ddp_info.device)

# Load checkpoint
checkpoint_dir = config.training.get("checkpoint_dir", "")
has_checkpoint = False
if checkpoint_dir:
    if os.path.isdir(checkpoint_dir):
        ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
        has_checkpoint = len(ckpt_files) > 0
    elif os.path.isfile(checkpoint_dir) and checkpoint_dir.endswith(".pt"):
        has_checkpoint = True

if has_checkpoint:
    result = model.load_ckpt(checkpoint_dir)
    if result is not None:
        print(f"Loaded checkpoint from {checkpoint_dir}")
    else:
        print(f"Warning: Failed to load checkpoint from {checkpoint_dir}")
        has_checkpoint = False

if not has_checkpoint and config.training.get("LVSM_checkpoint_dir", ""):
    lvsm_checkpoint_dir = config.training.LVSM_checkpoint_dir
    print(f"Initializing from LVSM checkpoint at {lvsm_checkpoint_dir}")
    result = model.init_from_LVSM(lvsm_checkpoint_dir)
    if result is None:
        print(f"Warning: Failed to initialize from LVSM checkpoint")
    else:
        print(f"Successfully initialized from LVSM checkpoint")

model = DDP(model, device_ids=[ddp_info.local_rank])
model.eval()

# Create output directory
output_dir = config.training.get("experiment_output_dir", "experiment_results/sample_t_and_render")
if ddp_info.is_main_process:
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

dist.barrier()

# Sample 10 different t values uniformly in [0, 1]
num_samples = 10
t_values = np.linspace(0.0, 1.0, num_samples).tolist()

# Latent scale (same as in training)
latent_scale = 0.136
noise_seed = config.training.get("noise_seed", 42)

datasampler.set_epoch(0)

with torch.no_grad(), torch.autocast(
    enabled=config.training.use_amp,
    device_type="cuda",
    dtype=torch.bfloat16 if config.training.amp_dtype == "bf16" else torch.float16,
):
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= config.training.get("num_samples_to_process", 1):
            break
            
        batch = {k: v.to(ddp_info.device) if type(v) == torch.Tensor else v for k, v in batch.items()}
        
        # Process data to get input and target
        input, target = model.module.process_data(
            batch, 
            has_target_image=True, 
            target_has_input=config.training.target_has_input, 
            compute_rays=True
        )
        
        # Get z_A and z_B
        z_A, n_patches, d = model.module.reconstructor(input)
        
        if not (hasattr(input, 'relit_images') and input.relit_images is not None):
            print(f"Warning: Sample {batch_idx} has no relit_images, skipping...")
            continue
            
        target_input = edict(
            image=input.relit_images,
            ray_o=input.ray_o,
            ray_d=input.ray_d
        )
        z_B, _, _ = model.module.reconstructor(target_input)
        
        # Scale latents
        z_A = z_A * latent_scale
        z_B = z_B * latent_scale
        
        # Generate z_0 (noise) with fixed seed
        generator = torch.Generator(device=z_A.device).manual_seed(noise_seed + batch_idx)
        z_0 = torch.randn(z_A.shape, device=z_A.device, dtype=z_A.dtype, generator=generator)
        
        # Prepare output file for distances
        if ddp_info.is_main_process:
            sample_output_dir = os.path.join(output_dir, f"sample_{batch_idx:04d}")
            os.makedirs(sample_output_dir, exist_ok=True)
            distances_file = os.path.join(sample_output_dir, "distances.txt")
            with open(distances_file, 'w') as f:
                f.write("t_value\tMSE_distance\tChamfer_distance\tHungarian_distance\n")
        
        # Process each t value
        for t_val in t_values:
            t = torch.tensor([t_val], device=z_A.device, dtype=z_A.dtype)
            t_expand = t.view(-1, 1, 1)
            
            # Compute z_t = (1-t) * z_0 + t * z_B
            z_t = (1 - t_expand) * z_0 + t_expand * z_B
            
            # Render directly from z_t (scale back first)
            z_t_unscaled = z_t / latent_scale
            rendered_image = model.module.renderer(z_t_unscaled, target, n_patches, d)
            
            # Compute distances between z_t and z_B
            # z_t and z_B are already in shape [batch, n_patches, d]
            # MSE distance
            mse_dist = torch.nn.functional.mse_loss(z_t, z_B).item()
            
            # Chamfer distance (expects [batch, n_patches, d])
            chamfer_dist = chamfer_distance(z_B, z_t).item()
            
            # Hungarian distance (only if enabled)
            compute_hungarian = config.training.get("compute_hungarian_loss", False)
            if compute_hungarian:
                hungarian_dist = hungarian_matching_loss(z_B, z_t).item()
            else:
                hungarian_dist = 0.0  # Not computed
            
            # Save rendered image
            if ddp_info.is_main_process:
                # rendered_image shape: [batch, num_views, channels, height, width]
                # Take first batch and first view
                img_tensor = rendered_image[0, 0].cpu().clamp(0, 1)  # [3, h, w]
                
                # Convert to numpy: [3, h, w] -> [h, w, 3]
                img_np = img_tensor.permute(1, 2, 0).numpy()
                
                # Convert to uint8
                img_np = (img_np * 255).astype(np.uint8)
                
                # Convert to PIL Image
                img_pil = Image.fromarray(img_np, 'RGB')
                
                # Save with t value in filename
                t_str = f"{t_val:.4f}".replace(".", "_")
                img_path = os.path.join(sample_output_dir, f"t_{t_str}.png")
                img_pil.save(img_path)
                
                # Write distances to file
                with open(distances_file, 'a') as f:
                    f.write(f"{t_val:.6f}\t{mse_dist:.6f}\t{chamfer_dist:.6f}\t{hungarian_dist:.6f}\n")
                
                print(f"Sample {batch_idx}, t={t_val:.4f}: MSE={mse_dist:.6f}, Chamfer={chamfer_dist:.6f}, Hungarian={hungarian_dist:.6f}")
        
        if ddp_info.is_main_process:
            print(f"Completed sample {batch_idx}")
        
        torch.cuda.empty_cache()

dist.barrier()

if ddp_info.is_main_process:
    print(f"Experiment completed! Results saved to: {output_dir}")

dist.destroy_process_group()
exit(0)

