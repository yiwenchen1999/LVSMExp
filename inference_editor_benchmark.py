"""
Benchmark script for single-scene relight inference.

Measures per-stage timing and peak memory for:
  1. Data preprocessing
  2. Scene reconstruction (encoder)
  3. Relighting (editor)
  4. Rendering N novel views (decoder, chunked)

Outputs a JSON report and prints a summary table.
"""

import importlib
import json
import os
import time

import psutil
import torch
import torch.distributed as dist
from easydict import EasyDict as edict
from einops import rearrange, repeat
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from setup import init_config, init_distributed
from utils import camera_utils

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

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
dataset_name = config.training.get("dataset_name", "data.dataset.Dataset")
module, class_name = dataset_name.rsplit(".", 1)
Dataset = importlib.import_module(module).__dict__[class_name]
dataset = Dataset(config)

datasampler = DistributedSampler(dataset)
dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=config.training.num_workers,
    prefetch_factor=config.training.prefetch_factor,
    persistent_workers=True,
    pin_memory=False,
    drop_last=True,
    sampler=datasampler,
)

dist.barrier()

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
module, class_name = config.model.class_name.rsplit(".", 1)
LVSM = importlib.import_module(module).__dict__[class_name]
model = LVSM(config).to(ddp_info.device)

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
        has_checkpoint = False

if not has_checkpoint and config.training.get("LVSM_checkpoint_dir", ""):
    lvsm_dir = config.training.LVSM_checkpoint_dir
    print(f"Initializing from Images2LatentScene at {lvsm_dir}")
    model.init_from_LVSM(lvsm_dir)

model = DDP(model, device_ids=[ddp_info.local_rank])

dist.barrier()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
NUM_NOVEL_VIEWS = config.inference.get("benchmark_num_views", 100)
VIEW_CHUNK_SIZE = 4


def render_novel_views(m, latent_tokens, input_dict, target_dict, num_frames):
    """Render *num_frames* novel views from edited latent_tokens.

    Replicates the camera-interpolation + chunked-decoding logic of
    ``LatentSceneEditor.render_video`` but uses the **already-edited**
    latent tokens so that relighting is included.
    """
    device = latent_tokens.device
    bs = latent_tokens.shape[0]
    c2ws = input_dict.c2w
    fxfycxcy = input_dict.fxfycxcy
    h, w = target_dict.image_h_w

    intrinsics = torch.zeros(
        (c2ws.shape[0], c2ws.shape[1], 3, 3), device=device
    )
    intrinsics[:, :, 0, 0] = fxfycxcy[:, :, 0]
    intrinsics[:, :, 1, 1] = fxfycxcy[:, :, 1]
    intrinsics[:, :, 0, 2] = fxfycxcy[:, :, 2]
    intrinsics[:, :, 1, 2] = fxfycxcy[:, :, 3]

    c2ws_loop = torch.cat([c2ws, c2ws[:, [0], :]], dim=1)
    intrinsics_loop = torch.cat([intrinsics, intrinsics[:, [0], :]], dim=1)

    all_c2ws, all_intrinsics = [], []
    for b_idx in range(bs):
        cur_c2ws, cur_intr = camera_utils.get_interpolated_poses_many(
            c2ws_loop[b_idx, :, :3, :4],
            intrinsics_loop[b_idx],
            num_frames,
            order_poses=False,
        )
        all_c2ws.append(cur_c2ws.to(device))
        all_intrinsics.append(cur_intr.to(device))

    all_c2ws = torch.stack(all_c2ws, dim=0)
    all_intrinsics = torch.stack(all_intrinsics, dim=0)

    homo = torch.tensor([[[0, 0, 0, 1]]], device=device).expand(
        all_c2ws.shape[0], all_c2ws.shape[1], -1, -1
    )
    all_c2ws = torch.cat([all_c2ws, homo], dim=2)

    all_fxfycxcy = torch.zeros(
        (all_intrinsics.shape[0], all_intrinsics.shape[1], 4), device=device
    )
    all_fxfycxcy[:, :, 0] = all_intrinsics[:, :, 0, 0]
    all_fxfycxcy[:, :, 1] = all_intrinsics[:, :, 1, 1]
    all_fxfycxcy[:, :, 2] = all_intrinsics[:, :, 0, 2]
    all_fxfycxcy[:, :, 3] = all_intrinsics[:, :, 1, 2]

    ray_o, ray_d = m.process_data.compute_rays(
        fxfycxcy=all_fxfycxcy, c2w=all_c2ws, h=h, w=w, device=device
    )

    target_pose_cond = m.get_posed_input(
        ray_o=ray_o.to(device), ray_d=ray_d.to(device)
    )
    _, num_views, c_dim, ph, pw = target_pose_cond.size()
    target_pose_tokens = m.target_pose_tokenizer(target_pose_cond)
    _, n_patches, d = target_pose_tokens.size()
    target_pose_tokens = target_pose_tokens.reshape(bs, num_views * n_patches, d)

    n_latent_vectors = m.config.model.transformer.n_latent_vectors
    patch_size = m.config.model.target_pose_tokenizer.patch_size
    video_list = []

    for cur_chunk in range(0, num_views, VIEW_CHUNK_SIZE):
        cur_sz = min(VIEW_CHUNK_SIZE, num_views - cur_chunk)
        s_idx = cur_chunk * n_patches
        e_idx = (cur_chunk + cur_sz) * n_patches
        cur_pose = rearrange(
            target_pose_tokens[:, s_idx:e_idx, :],
            "b (v p) d -> (b v) p d",
            v=cur_sz,
            p=n_patches,
        )
        cur_latent = repeat(
            latent_tokens, "b nl d -> (b v) nl d", v=cur_sz
        )
        dec_in = torch.cat((cur_pose, cur_latent), dim=1)
        dec_in = m.transformer_input_layernorm_decoder(dec_in)
        dec_out = m.pass_layers(
            m.transformer_decoder, dec_in, gradient_checkpoint=False
        )
        img_tokens, _ = dec_out.split([n_patches, n_latent_vectors], dim=1)
        rendered = m.image_token_decoder(img_tokens)
        rendered = rearrange(
            rendered,
            "(b v) (h w) (p1 p2 c) -> b v c (h p1) (w p2)",
            v=cur_sz,
            h=h // patch_size,
            w=w // patch_size,
            p1=patch_size,
            p2=patch_size,
            c=3,
        ).cpu()
        video_list.append(rendered)

    return torch.cat(video_list, dim=1)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
datasampler.set_epoch(0)
model.eval()

batch = next(iter(dataloader))
batch = {
    k: v.to(ddp_info.device) if isinstance(v, torch.Tensor) else v
    for k, v in batch.items()
}

m = model.module
n_latent_vectors = config.model.transformer.n_latent_vectors
amp_enabled = config.training.use_amp
amp_dtype = amp_dtype_mapping[config.training.amp_dtype]

process = psutil.Process(os.getpid())

# ---- Warmup (excludes CUDA lazy-init / kernel compilation) ----
print("Warmup run ...")
with torch.no_grad(), torch.autocast(
    enabled=amp_enabled, device_type="cuda", dtype=amp_dtype
):
    _ = model(batch)
torch.cuda.synchronize()
print("Warmup done.\n")

# ---- Reset memory stats ----
torch.cuda.reset_peak_memory_stats()
cpu_mem_before = process.memory_info().rss

# ---- Timed run ----
timings = {}

with torch.no_grad(), torch.autocast(
    enabled=amp_enabled, device_type="cuda", dtype=amp_dtype
):
    # Stage 1: Data preprocessing
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    input_dict, target_dict = m.process_data(
        batch,
        has_target_image=True,
        target_has_input=config.training.target_has_input,
        compute_rays=True,
    )
    torch.cuda.synchronize()
    timings["data_loading_s"] = time.perf_counter() - t0

    # Stage 2: Scene reconstruction
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    latent_tokens, n_patches, d = m.reconstructor(input_dict)
    torch.cuda.synchronize()
    timings["reconstruction_s"] = time.perf_counter() - t1

    # Stage 3: Relighting (editor)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    condition_tokens = m._build_editor_condition_tokens(input_dict, d)
    if condition_tokens is not None:
        editor_input = torch.cat([latent_tokens, condition_tokens], dim=1)
        editor_output = m.pass_layers(
            m.transformer_editor, editor_input, gradient_checkpoint=False
        )
        latent_tokens = editor_output[:, :n_latent_vectors, :]
    torch.cuda.synchronize()
    timings["relight_s"] = time.perf_counter() - t2

    # Stage 4: Render N novel views
    torch.cuda.synchronize()
    t3 = time.perf_counter()
    video = render_novel_views(
        m, latent_tokens, input_dict, target_dict, num_frames=NUM_NOVEL_VIEWS
    )
    torch.cuda.synchronize()
    timings["rendering_novel_views_s"] = time.perf_counter() - t3

timings["end_to_end_s"] = (
    timings["data_loading_s"]
    + timings["reconstruction_s"]
    + timings["relight_s"]
    + timings["rendering_novel_views_s"]
)
timings["per_view_render_ms"] = (
    timings["rendering_novel_views_s"] / NUM_NOVEL_VIEWS * 1000
)

# ---- Memory ----
peak_gpu_bytes = torch.cuda.max_memory_allocated()
peak_gpu_reserved_bytes = torch.cuda.max_memory_reserved()
cpu_mem_after = process.memory_info().rss
peak_cpu_bytes = max(cpu_mem_before, cpu_mem_after)

mem = {
    "peak_gpu_allocated_MB": round(peak_gpu_bytes / 1024**2, 1),
    "peak_gpu_reserved_MB": round(peak_gpu_reserved_bytes / 1024**2, 1),
    "cpu_rss_MB": round(peak_cpu_bytes / 1024**2, 1),
}

# ---- Report ----
report = {
    "num_novel_views": NUM_NOVEL_VIEWS,
    "num_input_views": config.training.num_input_views,
    "image_resolution": list(target_dict.image_h_w),
    **{k: round(v, 4) for k, v in timings.items()},
    **mem,
}

if ddp_info.is_main_process:
    print("\n" + "=" * 60)
    print("  Benchmark Results – Single-Scene Relight Inference")
    print("=" * 60)
    print(f"  Novel views rendered     : {NUM_NOVEL_VIEWS}")
    print(f"  Input views              : {config.training.num_input_views}")
    print(f"  Image resolution         : {target_dict.image_h_w}")
    print("-" * 60)
    print(f"  Data loading             : {timings['data_loading_s']:.4f} s")
    print(f"  Scene reconstruction     : {timings['reconstruction_s']:.4f} s")
    print(f"  Relighting (editor)      : {timings['relight_s']:.4f} s")
    print(
        f"  Rendering {NUM_NOVEL_VIEWS} views"
        f"          : {timings['rendering_novel_views_s']:.4f} s"
    )
    print(f"  Per-view render time     : {timings['per_view_render_ms']:.2f} ms")
    print(f"  End-to-end               : {timings['end_to_end_s']:.4f} s")
    print("-" * 60)
    print(f"  Peak GPU allocated       : {mem['peak_gpu_allocated_MB']:.1f} MB")
    print(f"  Peak GPU reserved        : {mem['peak_gpu_reserved_MB']:.1f} MB")
    print(f"  CPU RSS                  : {mem['cpu_rss_MB']:.1f} MB")
    print("=" * 60)

    out_dir = config.get("inference_out_dir", "experiments/benchmark_relight")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nResults saved to {out_path}")

dist.barrier()
dist.destroy_process_group()
