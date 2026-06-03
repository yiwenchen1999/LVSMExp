# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).

import importlib
import os
import time
import random
import traceback
import wandb
import torch
from easydict import EasyDict as edict
from rich import print
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from setup import init_config, init_distributed, init_wandb_and_backup
from utils.metric_utils import visualize_intermediate_results, save_interpolated_vis_frames
from utils.training_utils import create_optimizer, create_lr_scheduler, auto_resume_job, print_rank0


def select_local_pose_path(c2w, num_keyframes=6, pos_w=1.0, rot_w=0.5, anchor_idx=None):
    """Select a set of spatially-close camera views and order them into a smooth path.

    Selection is based on camera-pose proximity (NOT frame index), since views with
    adjacent indices are not necessarily spatially close.

    Args:
        c2w (torch.Tensor): [v, 4, 4] camera-to-world matrices for the available views.
        num_keyframes (int): number of views to pick for the path.
        pos_w (float): weight on translation distance.
        rot_w (float): weight on rotation-angle distance (radians).
        anchor_idx (int | None): optional fixed anchor; if None a random view is used.

    Returns:
        list[int]: ordered view indices forming a spatially smooth path.
    """
    v = c2w.shape[0]
    if v <= 1:
        return list(range(v))

    c2w = c2w.detach().float()
    t = c2w[:, :3, 3]                    # [v, 3]
    R = c2w[:, :3, :3].reshape(v, 9)     # [v, 9]

    # Translation distance (euclidean).
    pos_d = torch.cdist(t, t)            # [v, v]
    # Rotation angle distance: trace(R_i^T R_j) == <R_i, R_j>_F for rotation matrices.
    trace = R @ R.t()                    # [v, v]
    cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    rot_d = torch.arccos(cos_theta)      # [v, v], radians

    dist = pos_w * pos_d + rot_w * rot_d  # [v, v]

    k = min(num_keyframes, v)
    if anchor_idx is None:
        anchor_idx = int(torch.randint(0, v, (1,)).item())

    # Pick the (k-1) views closest to the anchor (plus the anchor itself).
    anchor_dists = dist[anchor_idx].clone()
    anchor_dists[anchor_idx] = float("inf")
    nearest = torch.topk(anchor_dists, k - 1, largest=False).indices.tolist()
    selected = [anchor_idx] + nearest

    # Greedy nearest-neighbour ordering over the selected set to form a smooth path.
    remaining = set(selected)
    ordered = [anchor_idx]
    remaining.discard(anchor_idx)
    while remaining:
        last = ordered[-1]
        nxt = min(remaining, key=lambda j: float(dist[last, j]))
        ordered.append(nxt)
        remaining.discard(nxt)
    return ordered


# Load config and read(override) arguments from CLI
config = init_config()

os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))

# Set up DDP for training/inference and Fix random seed
ddp_info = init_distributed(seed=777)
dist.barrier()

# Set up wandb and backup source code
if ddp_info.is_main_process:
    init_wandb_and_backup(config)
dist.barrier()


# Set up tf32
torch.backends.cuda.matmul.allow_tf32 = config.training.use_tf32
torch.backends.cudnn.allow_tf32 = config.training.use_tf32
amp_dtype_mapping = {
    "fp16": torch.float16, 
    "bf16": torch.bfloat16, 
    "fp32": torch.float32, 
    'tf32': torch.float32
}
print(f"------0.0 training script started, basic config set up------")
# Load dataset
dataset_name = config.training.get("dataset_name", "data.dataset.Dataset")
module, class_name = dataset_name.rsplit(".", 1)
Dataset = importlib.import_module(module).__dict__[class_name]
dataset = Dataset(config)
batch_size_per_gpu = config.training.batch_size_per_gpu

datasampler = DistributedSampler(dataset)
dataloader = DataLoader(
    dataset,
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=config.training.num_workers,
    persistent_workers=True,
    pin_memory=False,
    drop_last=True,
    prefetch_factor=config.training.prefetch_factor,
    sampler=datasampler,
)
dataloader_iter = iter(dataloader)

print(f"------1.0 dataloader set up------")
total_train_steps = config.training.train_steps
grad_accum_steps = config.training.grad_accum_steps
total_param_update_steps = total_train_steps
total_train_steps = total_train_steps * grad_accum_steps # real train steps when using gradient accumulation
total_batch_size = batch_size_per_gpu * ddp_info.world_size * grad_accum_steps
total_num_epochs = int(total_param_update_steps * total_batch_size / len(dataset))


module, class_name = config.model.class_name.rsplit(".", 1)
LVSM = importlib.import_module(module).__dict__[class_name]
model = LVSM(config).to(ddp_info.device)
model = DDP(model, device_ids=[ddp_info.local_rank])
print(f"------2.0 model initialized and wrapped with DDP------")

optimizer, optimized_param_dict, all_param_dict = create_optimizer(
    model,
    config.training.weight_decay,
    config.training.lr,
    (config.training.beta1, config.training.beta2),
)
optim_param_list = list(optimized_param_dict.values())


scheduler_type = config.training.get("scheduler_type", "cosine")
lr_scheduler = create_lr_scheduler(
    optimizer,
    total_param_update_steps,
    config.training.warmup,
    scheduler_type=scheduler_type,
)
print(f"------3.0 optimizer set up------")


if config.training.get("resume_ckpt", "") != "":
    ckpt_load_path = config.training.resume_ckpt
else:
    ckpt_load_path = config.training.checkpoint_dir
reset_training_state = config.training.get("reset_training_state", False)
optimizer, lr_scheduler, cur_train_step, cur_param_update_step = auto_resume_job(
    ckpt_load_path,
    model,
    optimizer,
    lr_scheduler,
    reset_training_state,
)
print(f"------4.0 checkpoint loaded and training state resumed------")

enable_grad_scaler = config.training.use_amp and config.training.amp_dtype == "fp16"
scaler = torch.amp.GradScaler('cuda', enabled=enable_grad_scaler)
print_rank0(f"Grad scaler enabled: {enable_grad_scaler}")
dist.barrier()

start_train_step = cur_train_step
model.train()
print(f"------5.0 model set to train mode------")
while cur_train_step <= total_train_steps:
    # print(f"------5.1 training step {cur_train_step} started!------")
    tic = time.time()
    cur_epoch = int(cur_train_step * (total_batch_size / grad_accum_steps) // len(dataset) )
    try:
        # if start_train_step == cur_train_step:
        #     print(f"Current Rank {ddp_info.local_rank} Restarting training from step {cur_train_step}. Resetting dataloader epoch to {cur_epoch}; might take a while...")
        #     datasampler.set_epoch(cur_epoch)
        #     dataloader_iter = iter(dataloader)
        data = next(dataloader_iter)
    except StopIteration:
        print(f"Current Rank {ddp_info.local_rank} Ran out of data. Resetting dataloader epoch to {cur_epoch}; might take a while...")
        datasampler.set_epoch(cur_epoch)
        dataloader_iter = iter(dataloader)
        data = next(dataloader_iter)

    batch = {k: v.to(ddp_info.device) if type(v) == torch.Tensor else v for k, v in data.items()}


    with torch.autocast(
        enabled=config.training.use_amp,
        device_type="cuda",
        dtype=amp_dtype_mapping[config.training.amp_dtype],
    ):
        ret_dict = model(batch)

    update_grads = (cur_train_step + 1) % grad_accum_steps == 0 or cur_train_step == total_train_steps
    if update_grads:
        with model.no_sync(): # no sync grads for efficiency
            scaler.scale(ret_dict.loss_metrics.loss / grad_accum_steps).backward()
    else:
        scaler.scale(ret_dict.loss_metrics.loss / grad_accum_steps).backward()
    cur_train_step += 1

    export_inter_results = ((cur_train_step-1) == start_train_step) or (cur_train_step % config.training.vis_every == 0)

    if update_grads:
        skip_optimizer_step = False
        # Skip optimizer step if loss is NaN or Inf
        if torch.isnan(ret_dict.loss_metrics.loss) or torch.isinf(ret_dict.loss_metrics.loss):
            print(f"NaN or Inf loss detected, skip this iteration")
            skip_optimizer_step = True
            ret_dict.loss_metrics.loss.data = torch.zeros_like(ret_dict.loss_metrics.loss)

        total_grad_norm = None
        # Check gradient norm and update optimizer if everything is fine
        if not skip_optimizer_step:
            # Unscales the gradients
            scaler.unscale_(optimizer) 
            # For all gradients, we safely change the NaN -> 0., inf -> 1e-6, -inf -> 1e-6.
            with torch.no_grad():
                for n, p in optimized_param_dict.items():
                    if p.requires_grad and (p.grad is not None):
                        p.grad.nan_to_num_(nan=0.0, posinf=1e-6, neginf=-1e-6)
        
            # visualize the grad norm of each layer of our transformer (FOR DEBUG)
            if ddp_info.is_main_process and config.training.get("log_grad_norm_details", False):
                grad_norms = {}  # Dictionary to store norms per layer
                for name, param in model.named_parameters():
                    if param.grad is not None:  # Some parameters might not have gradients
                        grad_norms[name] = param.grad.detach().norm().item()  # Detach for safety
                for layer_name, grad_norm in grad_norms.items():
                    wandb.log({"grad_norm_details/" + layer_name: grad_norm}, step=cur_train_step)

            total_grad_norm = 0.0
            if config.training.grad_clip_norm > 0:
                total_grad_norm = torch.nn.utils.clip_grad_norm_(optim_param_list, max_norm=config.training.grad_clip_norm).item()

                if total_grad_norm > config.training.grad_clip_norm * 2.0:
                    print(f"WARNING: step {cur_train_step} grad norm too large {total_grad_norm} > {config.training.grad_clip_norm * 2.0}")

                allowed_gradnorm = config.training.grad_clip_norm * config.training.get("allowed_gradnorm_factor", 5)
                if total_grad_norm > allowed_gradnorm:
                    skip_optimizer_step = True
                    print(f"WARNING: step {cur_train_step} grad norm too large {total_grad_norm} > {allowed_gradnorm}, skipping optimizer step")

                # show grad norm in wandb if it's too large
                display_grad_norm = total_grad_norm > config.training.grad_clip_norm * 2.0 or total_grad_norm > allowed_gradnorm
                if display_grad_norm and ddp_info.is_main_process:
                    wandb.log({"grad_norm": total_grad_norm}, step=cur_train_step)

            # since skip flag may be updated because of grad norm, we check it again
            if not skip_optimizer_step:
                scaler.step(optimizer)
                cur_param_update_step += 1

        scaler.update()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    # log and save checkpoint
    if ddp_info.is_main_process:
        loss_dict = {k: float(f"{v.item():.6f}") for k, v in ret_dict.loss_metrics.items()}
        # print in console
        if (cur_train_step % config.training.print_every == 0) or (cur_train_step < 100 + start_train_step):
            print_str = f"[Epoch {int(cur_epoch):>3d}] | Forwad step: {int(cur_train_step):>6d} (Param update step: {int(cur_param_update_step):>6d})"
            print_str += f" | Iter Time: {time.time() - tic:.2f}s | LR: {optimizer.param_groups[0]['lr']:.6f}\n"
            # Add loss values
            for k, v in loss_dict.items():
                print_str += f"{k}: {v:.6f} | "
            print(print_str)

        # log in wandb
        if (cur_train_step % config.training.wandb_log_every == 0) or (
            cur_train_step < 200 + start_train_step
        ):
            log_dict = {
                "iter": cur_train_step, 
                "forward_pass_step": cur_train_step,
                "param_update_step": cur_param_update_step,
                "lr": optimizer.param_groups[0]["lr"],
                "iter_time": time.time() - tic,
                "grad_norm": total_grad_norm,
                "epoch": cur_epoch,
            }
            log_dict.update({"train/" + k: v for k, v in loss_dict.items()})
            wandb.log(
                log_dict,
                step=cur_train_step,
            )

        # save checkpoint
        if (cur_train_step % config.training.checkpoint_every == 0) or (cur_train_step == total_train_steps):
            if isinstance(model, DDP):
                model_weights = model.module.state_dict()
            else:
                model_weights = model.state_dict()
            checkpoint = {
                "model": model_weights,
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "fwdbwd_pass_step": cur_train_step,
                "param_update_step": cur_param_update_step,
            }
            os.makedirs(config.training.checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(config.training.checkpoint_dir, f"ckpt_{cur_train_step:016}.pt")
            torch.save(checkpoint, ckpt_path)
            print(f"Saved checkpoint at step {cur_train_step} to {os.path.abspath(ckpt_path)}")
        
        # export intermediate visualization results
        if export_inter_results:
            vis_path = os.path.join(config.training.checkpoint_dir, f"iter_{cur_train_step:08d}")
            os.makedirs(vis_path, exist_ok=True)
            visualize_intermediate_results(vis_path, ret_dict)

            # Extra visualization-only pass: pick a set of spatially-close input views
            # (by camera-pose proximity, not frame index), order them into a smooth
            # short trajectory, and interpolate `num_frames` views between every
            # adjacent pair. This does NOT affect loss, optimizer steps or checkpoints
            # (pure no_grad inference + save).
            if config.training.get("vis_interpolate", True):
                try:
                    base_model = model.module if isinstance(model, DDP) else model
                    base_model.eval()
                    vis_input = ret_dict.input
                    v_input = vis_input.image.size(1)
                    if v_input >= 2:
                        num_frames = int(config.training.get("vis_interpolate_frames", 8))
                        num_keyframes = int(config.training.get("vis_interpolate_num_input_keyframes", 6))
                        pos_w = float(config.training.get("vis_interpolate_pose_dist_pos_w", 1.0))
                        rot_w = float(config.training.get("vis_interpolate_pose_dist_rot_w", 0.5))
                        select_mode = config.training.get("vis_interpolate_select", "local_six_pose")
                        # When enabled, `num_frames` counts the views rendered strictly
                        # BETWEEN two context keyframes: we render num_frames+2 poses then
                        # drop the two endpoints (which coincide with the context poses).
                        drop_endpoints = bool(config.training.get("vis_interpolate_drop_endpoints", False))
                        render_frames = num_frames + 2 if drop_endpoints else num_frames

                        c2w_b0 = vis_input.c2w[0]  # [v, 4, 4]
                        if select_mode == "random_two":
                            ordered = random.sample(range(v_input), 2)
                        elif (not torch.isfinite(c2w_b0).all()):
                            print("[interpolate vis] non-finite c2w, falling back to random sampling")
                            k = min(num_keyframes, v_input)
                            ordered = random.sample(range(v_input), k)
                        else:
                            ordered = select_local_pose_path(
                                c2w_b0,
                                num_keyframes=num_keyframes,
                                pos_w=pos_w,
                                rot_w=rot_w,
                            )

                        # Render each adjacent segment and concatenate frames in order.
                        segment_frames = []
                        for a, b in zip(ordered[:-1], ordered[1:]):
                            sel = [a, b]
                            vis_data_batch = edict(
                                input=edict(
                                    image=vis_input.image[:, sel],
                                    ray_o=vis_input.ray_o[:, sel],
                                    ray_d=vis_input.ray_d[:, sel],
                                    c2w=vis_input.c2w[:, sel],
                                    fxfycxcy=vis_input.fxfycxcy[:, sel],
                                ),
                                target=edict(image_h_w=ret_dict.target.image_h_w),
                            )
                            try:
                                with torch.no_grad():
                                    with torch.autocast(
                                        device_type="cuda",
                                        dtype=amp_dtype_mapping[config.training.amp_dtype],
                                    ):
                                        vis_out = base_model.render_video(
                                            vis_data_batch,
                                            traj_type="interpolate",
                                            num_frames=render_frames,
                                            loop_video=False,
                                            order_poses=False,
                                        )
                                seg_render = vis_out.video_rendering  # [b, render_frames, 3, h, w]
                                # Drop the endpoint frames (context-pose renders) so only
                                # the views strictly between the two context frames remain.
                                if drop_endpoints and seg_render.shape[1] >= 3:
                                    seg_render = seg_render[:, 1:-1]
                                segment_frames.append(seg_render)
                            except Exception as seg_e:
                                print(f"[interpolate vis] segment ({a},{b}) skipped: {seg_e}")

                        if len(segment_frames) > 0:
                            all_frames = torch.cat(segment_frames, dim=1)  # [b, total_frames, 3, h, w]
                            scene_name = (
                                ret_dict.input.scene_name[0]
                                if hasattr(ret_dict.input, "scene_name") and ret_dict.input.scene_name is not None
                                else "unknown"
                            )
                            save_interpolated_vis_frames(
                                vis_path,
                                all_frames,
                                input_endpoints=vis_input.image[0, ordered],
                                scene_name=scene_name,
                            )
                except Exception as e:
                    print(f"[interpolate vis] skipped due to error: {e}")
                    traceback.print_exc()

            torch.cuda.empty_cache()
            model.train()

            
    if export_inter_results:
        torch.cuda.empty_cache()
        dist.barrier()
        
print(f"------6.0 training finished!------")
dist.barrier()
dist.destroy_process_group()
