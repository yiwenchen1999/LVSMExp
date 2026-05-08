# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).

import importlib
import os
import time
import json
import wandb
import torch
from rich import print
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from setup import init_config, init_distributed, init_wandb_and_backup
from utils.metric_utils import visualize_intermediate_results
from utils.training_utils import create_optimizer, create_lr_scheduler, auto_resume_job, print_rank0

DEBUG_LOG_PATH = "/Users/yiwenchen/Desktop/ResearchProjects/LightingDiffusion/3dgs/LVSMExp/.cursor/debug-3a0a87.log"
DEBUG_SESSION_ID = "3a0a87"


def _debug_log(run_id, hypothesis_id, location, message, data):
    payload = {
        "sessionId": DEBUG_SESSION_ID,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    log_path = os.path.expanduser(DEBUG_LOG_PATH)
    log_dir = os.path.dirname(log_path)
    try:
        os.makedirs(log_dir, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        # Never let debug logging break training runtime.
        pass


def _collect_optimizer_layout_snapshot(optimizer, optimized_param_dict):
    summary = []
    for name, param in optimized_param_dict.items():
        if not param.requires_grad:
            continue
        grad = param.grad
        state = optimizer.state.get(param, {})
        exp_avg = state.get("exp_avg", None)
        exp_avg_sq = state.get("exp_avg_sq", None)
        summary.append(
            {
                "name": name,
                "param_dtype": str(param.dtype),
                "param_device": str(param.device),
                "param_layout": str(param.layout),
                "param_stride": list(param.stride()),
                "grad_is_none": grad is None,
                "grad_dtype": str(grad.dtype) if grad is not None else None,
                "grad_device": str(grad.device) if grad is not None else None,
                "grad_layout": str(grad.layout) if grad is not None else None,
                "grad_stride": list(grad.stride()) if grad is not None else None,
                "exp_avg_dtype": str(exp_avg.dtype) if exp_avg is not None else None,
                "exp_avg_device": str(exp_avg.device) if exp_avg is not None else None,
                "exp_avg_layout": str(exp_avg.layout) if exp_avg is not None else None,
                "exp_avg_stride": list(exp_avg.stride()) if exp_avg is not None else None,
                "exp_avg_sq_dtype": str(exp_avg_sq.dtype) if exp_avg_sq is not None else None,
                "exp_avg_sq_device": str(exp_avg_sq.device) if exp_avg_sq is not None else None,
                "exp_avg_sq_layout": str(exp_avg_sq.layout) if exp_avg_sq is not None else None,
                "exp_avg_sq_stride": list(exp_avg_sq.stride()) if exp_avg_sq is not None else None,
            }
        )
    return summary


def _group_params_with_decay(named_params, weight_decay, lr):
    decay_params, nodecay_params = [], []
    for _, param in named_params:
        if param.dim() == 1 or getattr(param, "_no_weight_decay", False):
            nodecay_params.append(param)
        else:
            decay_params.append(param)
    groups = []
    if len(decay_params) > 0:
        groups.append({"params": decay_params, "weight_decay": weight_decay, "lr": lr})
    if len(nodecay_params) > 0:
        groups.append({"params": nodecay_params, "weight_decay": 0.0, "lr": lr})
    return groups


def create_optimizer_for_stage(model, config, stage_name, dpt_transfer_cfg):
    weight_decay = config.training.weight_decay
    learning_rate = config.training.lr
    betas = (config.training.beta1, config.training.beta2)

    all_param_dict = {name: param for name, param in model.named_parameters()}
    optimized_param_dict = {name: param for name, param in all_param_dict.items() if param.requires_grad}

    use_scaled_groups = (
        bool(dpt_transfer_cfg.get("enabled", False))
        and str(stage_name).lower() == "stage2"
        and float(dpt_transfer_cfg.get("backbone_lr_scale", 1.0)) != 1.0
    )

    if not use_scaled_groups:
        return create_optimizer(model, weight_decay, learning_rate, betas)

    backbone_lr_scale = float(dpt_transfer_cfg.get("backbone_lr_scale", 0.1))
    head_named = [(n, p) for n, p in optimized_param_dict.items() if "image_token_decoder" in n]
    backbone_named = [(n, p) for n, p in optimized_param_dict.items() if "image_token_decoder" not in n]

    optim_groups = []
    optim_groups.extend(_group_params_with_decay(head_named, weight_decay, learning_rate))
    optim_groups.extend(_group_params_with_decay(backbone_named, weight_decay, learning_rate * backbone_lr_scale))

    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=True)
    print_rank0(
        f"Stage2 optimizer groups enabled: head_lr={learning_rate}, "
        f"backbone_lr={learning_rate * backbone_lr_scale}"
    )
    return optimizer, optimized_param_dict, all_param_dict


# Load config and read(override) arguments from CLI
config = init_config()
relight_signals = config.training.get("relight_signals", ["envmap"])
if isinstance(relight_signals, str):
    relight_signals = [relight_signals]
config.training.relight_signals = list(relight_signals)
print(f"------0.0.1 relight_signals: {config.training.relight_signals}------")
# #region agent log
_debug_log(
    run_id=f"rank-{os.environ.get('RANK', 'na')}",
    hypothesis_id="H1",
    location="train_editor.py:config_init",
    message="Torch/CUDA runtime snapshot at startup",
    data={
        "torch_version": torch.__version__,
        "torch_compiled_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_arch_list": torch.cuda.get_arch_list() if hasattr(torch.cuda, "get_arch_list") else [],
    },
)
# #endregion

os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))

# Set up DDP for training/inference and Fix random seed
ddp_info = init_distributed(seed=777)
# #region agent log
_debug_log(
    run_id=f"rank-{os.environ.get('RANK', 'na')}",
    hypothesis_id="H2",
    location="train_editor.py:post_init_distributed",
    message="Distributed/device assignment snapshot",
    data={
        "local_rank": int(os.environ.get("LOCAL_RANK", -1)),
        "rank": int(os.environ.get("RANK", -1)) if os.environ.get("RANK") is not None else -1,
        "device_repr": str(ddp_info.device),
    },
)
# #endregion
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

# Check if dataloader is empty (can happen with DistributedSampler + drop_last)
if len(dataloader) == 0:
    error_msg = f"Dataloader is empty! This can happen when:\n"
    error_msg += f"  - Dataset size ({len(dataset)}) is smaller than total batch size ({batch_size_per_gpu * ddp_info.world_size})\n"
    error_msg += f"  - Or when using DistributedSampler, some ranks get no data\n"
    error_msg += f"  - Consider reducing batch_size_per_gpu or setting drop_last=False"
    print_rank0(error_msg)
    raise ValueError(error_msg)

print_rank0(f"Dataloader size: {len(dataloader)} batches per epoch")
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
# #region agent log
_debug_log(
    run_id=f"rank-{os.environ.get('RANK', 'na')}",
    hypothesis_id="H3",
    location="train_editor.py:model_to_device",
    message="Model moved to target device",
    data={
        "target_device": str(ddp_info.device),
        "param_device": str(next(model.parameters()).device),
    },
)
# #endregion

dpt_transfer_cfg = config.training.get("dpt_transfer", {})
dpt_transfer_enabled = bool(dpt_transfer_cfg.get("enabled", False))
dpt_train_stage = str(dpt_transfer_cfg.get("train_stage", "auto")).lower()
stage1_steps = int(dpt_transfer_cfg.get("stage1_steps", 0))
if dpt_transfer_enabled and dpt_transfer_cfg.get("use_teacher_distill", False):
    print_rank0("Warning: use_teacher_distill is enabled, but teacher distillation loss is not implemented in this training loop yet.")

if dpt_transfer_enabled and hasattr(model, "set_dpt_transfer_stage"):
    if dpt_train_stage == "stage1":
        active_stage = "stage1"
    elif dpt_train_stage == "stage2":
        active_stage = "stage2"
    else:
        active_stage = "stage1" if stage1_steps > 0 else "stage2"
    model.set_dpt_transfer_stage(active_stage, dpt_transfer_cfg)
else:
    active_stage = "stage2"

# Initialize from Images2LatentScene if LVSM_checkpoint_dir is provided
if config.training.get("LVSM_checkpoint_dir", ""):
    lvsm_checkpoint_dir = config.training.LVSM_checkpoint_dir
    print(f"------2.1 Initializing LatentSceneEditor from Images2LatentScene at {lvsm_checkpoint_dir}------")
    result = model.init_from_LVSM(lvsm_checkpoint_dir)
    if result is None:
        print(f"Warning: Failed to initialize from LVSM checkpoint, using random initialization")
    else:
        print(f"------2.2 Successfully initialized from Images2LatentScene------")

# Freeze reconstructor and renderer layers, only train editor layers
# Do this before DDP wrapping to ensure freeze state is preserved
if config.training.get("freeze_reconstructor_renderer", False) and not dpt_transfer_enabled:
    model.freeze_reconstructor_and_renderer()
    print(f"------2.0.1 frozen reconstructor and renderer, only editor layers trainable------")

model = DDP(model, device_ids=[ddp_info.local_rank], find_unused_parameters=True)
print(f"------2.0 model initialized and wrapped with DDP------")

optimizer, optimized_param_dict, all_param_dict = create_optimizer_for_stage(
    model=model,
    config=config,
    stage_name=active_stage,
    dpt_transfer_cfg=dpt_transfer_cfg,
)
optim_param_list = list(optimized_param_dict.values())
#region agent log
_debug_log(
    run_id="pre-fix",
    hypothesis_id="H1",
    location="train_editor.py:optimizer_init",
    message="optimizer created",
    data={
        "active_stage": active_stage,
        "optimizer_type": type(optimizer).__name__,
        "optimizer_fused": optimizer.defaults.get("fused", None),
        "num_param_groups": len(optimizer.param_groups),
        "trainable_param_count": len(optimized_param_dict),
    },
)
#endregion


scheduler_type = config.training.get("scheduler_type", "cosine")
lr_scheduler = create_lr_scheduler(
    optimizer,
    total_param_update_steps,
    config.training.warmup,
    scheduler_type=scheduler_type,
)
print(f"------3.0 optimizer set up------")


# Load checkpoint: if checkpoint_dir exists, load directly; otherwise use LVSM_checkpoint_dir for initialization
if config.training.get("checkpoint_dir", "") and os.path.exists(config.training.checkpoint_dir):
    # Check if checkpoint exists in checkpoint_dir
    ckpt_files = [f for f in os.listdir(config.training.checkpoint_dir) if f.endswith(".pt")] if os.path.isdir(config.training.checkpoint_dir) else []
    if ckpt_files or os.path.isfile(config.training.checkpoint_dir):
        # Directly load from checkpoint_dir
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
        print(f"------4.0 checkpoint loaded from {ckpt_load_path} and training state resumed------")
    else:
        # No checkpoint in checkpoint_dir, but we already initialized from LVSM if LVSM_checkpoint_dir was provided
        # Start from scratch with initialized weights
        cur_train_step = 0
        cur_param_update_step = 0
        print(f"------4.0 no checkpoint found in checkpoint_dir, starting from initialized weights------")
else:
    # No checkpoint_dir, start fresh (may have been initialized from LVSM)
    cur_train_step = 0
    cur_param_update_step = 0
    print(f"------4.0 no checkpoint_dir specified, starting fresh------")

enable_grad_scaler = config.training.use_amp and config.training.amp_dtype == "fp16"
scaler = torch.amp.GradScaler('cuda', enabled=enable_grad_scaler)
print_rank0(f"Grad scaler enabled: {enable_grad_scaler}")
dist.barrier()

start_train_step = cur_train_step
model.train()
print(f"------5.0 model set to train mode------")
while cur_train_step <= total_train_steps:
    if (
        dpt_transfer_enabled
        and dpt_train_stage == "auto"
        and active_stage == "stage1"
        and cur_param_update_step >= stage1_steps
    ):
        active_stage = "stage2"
        if hasattr(model.module, "set_dpt_transfer_stage"):
            model.module.set_dpt_transfer_stage(active_stage, dpt_transfer_cfg)
        optimizer, optimized_param_dict, all_param_dict = create_optimizer_for_stage(
            model=model,
            config=config,
            stage_name=active_stage,
            dpt_transfer_cfg=dpt_transfer_cfg,
        )
        optim_param_list = list(optimized_param_dict.values())
        lr_scheduler = create_lr_scheduler(
            optimizer,
            total_param_update_steps,
            config.training.warmup,
            scheduler_type=scheduler_type,
        )
        for _ in range(cur_param_update_step):
            lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        print_rank0(f"Switched DPT transfer stage to {active_stage} at param update step {cur_param_update_step}")

    if dpt_transfer_enabled and hasattr(model.module, "set_decoder_head_gate_schedule"):
        model.module.set_decoder_head_gate_schedule(cur_param_update_step, dpt_transfer_cfg)

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
                non_contiguous_grad_params = []
                stride_mismatch_grad_params = []
                for n, p in optimized_param_dict.items():
                    if p.requires_grad and (p.grad is not None):
                        p.grad.nan_to_num_(nan=0.0, posinf=1e-6, neginf=-1e-6)
                        if not p.grad.is_contiguous():
                            non_contiguous_grad_params.append(
                                {
                                    "name": n,
                                    "grad_stride_before": list(p.grad.stride()),
                                    "param_stride": list(p.stride()),
                                }
                            )
                            p.grad = p.grad.contiguous()
                            non_contiguous_grad_params[-1]["grad_stride_after"] = list(p.grad.stride())
                        if p.grad.layout == torch.strided and p.grad.stride() != p.stride():
                            stride_mismatch_grad_params.append(
                                {
                                    "name": n,
                                    "param_stride": list(p.stride()),
                                    "grad_stride_before": list(p.grad.stride()),
                                }
                            )
                            fixed_grad = torch.empty_like(
                                p.grad,
                                device=p.grad.device,
                                dtype=p.grad.dtype,
                                memory_format=torch.contiguous_format,
                            )
                            fixed_grad.copy_(p.grad)
                            p.grad = fixed_grad
                            stride_mismatch_grad_params[-1]["grad_stride_after"] = list(p.grad.stride())

                if len(non_contiguous_grad_params) > 0 and cur_param_update_step < 3:
                    #region agent log
                    _debug_log(
                        run_id="post-fix",
                        hypothesis_id="H4",
                        location="train_editor.py:grad_contiguous_fix",
                        message="made non-contiguous grads contiguous before fused AdamW",
                        data={
                            "cur_train_step": cur_train_step,
                            "cur_param_update_step": cur_param_update_step,
                            "num_fixed_params": len(non_contiguous_grad_params),
                            "fixed_params": non_contiguous_grad_params[:10],
                        },
                    )
                    #endregion
                if len(stride_mismatch_grad_params) > 0 and cur_param_update_step < 3:
                    #region agent log
                    _debug_log(
                        run_id="post-fix",
                        hypothesis_id="H5",
                        location="train_editor.py:grad_stride_alignment_fix",
                        message="aligned grad stride with param stride before fused AdamW",
                        data={
                            "cur_train_step": cur_train_step,
                            "cur_param_update_step": cur_param_update_step,
                            "num_fixed_params": len(stride_mismatch_grad_params),
                            "fixed_params": stride_mismatch_grad_params[:10],
                        },
                    )
                    #endregion
        
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
                if cur_param_update_step < 2:
                    #region agent log
                    _debug_log(
                        run_id="pre-fix",
                        hypothesis_id="H2",
                        location="train_editor.py:before_scaler_step",
                        message="before scaler.step snapshot",
                        data={
                            "cur_train_step": cur_train_step,
                            "cur_param_update_step": cur_param_update_step,
                            "optimizer_fused": optimizer.defaults.get("fused", None),
                            "sampled_params": _collect_optimizer_layout_snapshot(
                                optimizer, dict(list(optimized_param_dict.items())[:8])
                            ),
                        },
                    )
                    #endregion
                try:
                    scaler.step(optimizer)
                except RuntimeError as e:
                    #region agent log
                    _debug_log(
                        run_id="pre-fix",
                        hypothesis_id="H3",
                        location="train_editor.py:scaler_step_exception",
                        message="scaler.step runtime error",
                        data={
                            "error": str(e),
                            "optimizer_fused": optimizer.defaults.get("fused", None),
                            "full_snapshot": _collect_optimizer_layout_snapshot(
                                optimizer, optimized_param_dict
                            )[:50],
                        },
                    )
                    #endregion
                    raise
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
                "train_stage": 1 if active_stage == "stage1" else 2,
            }
            if dpt_transfer_enabled and hasattr(model.module.image_token_decoder, "get_gate_values"):
                gate_vals = model.module.image_token_decoder.get_gate_values()
                for gate_key, gate_val in gate_vals.items():
                    log_dict[f"gates/{gate_key}"] = gate_val
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
            torch.cuda.empty_cache()
            model.train()

            
    if export_inter_results:
        torch.cuda.empty_cache()
        dist.barrier()
        
print(f"------6.0 training finished!------")
dist.barrier()
dist.destroy_process_group()
