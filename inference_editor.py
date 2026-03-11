# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).

import importlib
import os
import json
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from easydict import EasyDict as edict
from einops import rearrange, repeat
from setup import init_config, init_distributed
from utils.metric_utils import export_results, export_all_views_results, summarize_evaluation

# Load config and read(override) arguments from CLI
config = init_config()

os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))

# Set up DDP training/inference and Fix random seed
ddp_info = init_distributed(seed=777)
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


# Load data
dataset_name = config.training.get("dataset_name", "data.dataset.Dataset")
module, class_name = dataset_name.rsplit(".", 1)
Dataset = importlib.import_module(module).__dict__[class_name]
dataset = Dataset(config)

datasampler = DistributedSampler(dataset)
# render_all_views requires batch_size=1 because frame counts vary per scene
_batch_size = 1 if config.inference.get("render_all_views", False) else config.training.batch_size_per_gpu
dataloader = DataLoader(
    dataset,
    batch_size=_batch_size,
    shuffle=False,
    num_workers=config.training.num_workers,
    prefetch_factor=config.training.prefetch_factor,
    persistent_workers=True,
    pin_memory=False,
    drop_last=False if config.inference.get("render_all_views", False) else True,
    sampler=datasampler
)
dataloader_iter = iter(dataloader)

dist.barrier()



# Import model and load checkpoint
module, class_name = config.model.class_name.rsplit(".", 1)
LVSM = importlib.import_module(module).__dict__[class_name]
model = LVSM(config).to(ddp_info.device)

# Check if checkpoint_dir has .pt files
checkpoint_dir = config.training.get("checkpoint_dir", "")
has_checkpoint = False
if checkpoint_dir:
    if os.path.isdir(checkpoint_dir):
        ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
        has_checkpoint = len(ckpt_files) > 0
    elif os.path.isfile(checkpoint_dir) and checkpoint_dir.endswith(".pt"):
        has_checkpoint = True

# Try to load from checkpoint_dir first
if has_checkpoint:
    result = model.load_ckpt(checkpoint_dir)
    if result is not None:
        print(f"Loaded checkpoint from {checkpoint_dir}")
    else:
        print(f"Warning: Failed to load checkpoint from {checkpoint_dir}, trying LVSM_checkpoint_dir...")
        has_checkpoint = False  # Mark as failed, will try LVSM_checkpoint_dir

# If no checkpoint in checkpoint_dir, try to initialize from LVSM_checkpoint_dir
if not has_checkpoint and config.training.get("LVSM_checkpoint_dir", ""):
    lvsm_checkpoint_dir = config.training.LVSM_checkpoint_dir
    print(f"Initializing LatentSceneEditor from Images2LatentScene at {lvsm_checkpoint_dir}")
    result = model.init_from_LVSM(lvsm_checkpoint_dir)
    if result is None:
        print(f"Warning: Failed to initialize from LVSM checkpoint at {lvsm_checkpoint_dir}")
        print(f"Warning: Starting completely from fresh (random initialization)")
    else:
        print(f"Successfully initialized from Images2LatentScene")
elif not has_checkpoint:
    print(f"Warning: No checkpoint found in checkpoint_dir and no LVSM_checkpoint_dir specified")
    print(f"Warning: Starting completely from fresh (random initialization)")

model = DDP(model, device_ids=[ddp_info.local_rank])


if ddp_info.is_main_process:  
    condition_reverse = config.training.get("condition_reverse", False)
    print(f"Running inference; save results to: {config.inference_out_dir}")
    if condition_reverse:
        print(f"  condition_reverse=True: input images from relit scene, "
              f"relit/lighting from current scene")
    # avoid multiple processes downloading LPIPS at the same time
    import lpips
    # Suppress the warning by setting weights_only=True
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)

dist.barrier()


def render_all_views_chunked(m, batch, device, view_chunk_size=4):
    """Render all target views using chunked decoding to avoid OOM.

    Steps: process_data -> reconstructor -> editor -> chunked renderer.
    """
    input_dict, target_dict = m.process_data(
        batch,
        has_target_image=True,
        target_has_input=config.training.target_has_input,
        compute_rays=True,
    )

    latent_tokens, n_patches, d = m.reconstructor(input_dict)

    n_latent_vectors = config.model.transformer.n_latent_vectors
    condition_tokens = m._build_editor_condition_tokens(input_dict, d)
    if condition_tokens is not None:
        editor_input = torch.cat([latent_tokens, condition_tokens], dim=1)
        editor_output = m.pass_layers(
            m.transformer_editor, editor_input, gradient_checkpoint=False
        )
        latent_tokens = editor_output[:, :n_latent_vectors, :]

    # Chunked rendering of all target views
    target_pose_cond = m.get_posed_input(
        ray_o=target_dict.ray_o, ray_d=target_dict.ray_d
    )
    bs, num_views, c_dim, ph, pw = target_pose_cond.size()
    h, w = target_dict.image_h_w
    patch_size = config.model.target_pose_tokenizer.patch_size

    target_pose_tokens = m.target_pose_tokenizer(target_pose_cond)
    _, n_per_view, _ = target_pose_tokens.size()
    target_pose_tokens = target_pose_tokens.reshape(bs, num_views * n_per_view, d)

    rendered_chunks = []
    for start in range(0, num_views, view_chunk_size):
        cur_sz = min(view_chunk_size, num_views - start)
        s_idx = start * n_per_view
        e_idx = (start + cur_sz) * n_per_view
        cur_pose = rearrange(
            target_pose_tokens[:, s_idx:e_idx, :],
            "b (v p) d -> (b v) p d", v=cur_sz, p=n_per_view,
        )
        cur_latent = repeat(latent_tokens, "b nl d -> (b v) nl d", v=cur_sz)
        dec_in = torch.cat((cur_pose, cur_latent), dim=1)
        dec_in = m.transformer_input_layernorm_decoder(dec_in)
        dec_out = m.pass_layers(
            m.transformer_decoder, dec_in, gradient_checkpoint=False
        )
        img_tokens, _ = dec_out.split([n_per_view, n_latent_vectors], dim=1)
        rendered = m.image_token_decoder(img_tokens)
        rendered = rearrange(
            rendered,
            "(b v) (hh ww) (p1 p2 c) -> b v c (hh p1) (ww p2)",
            v=cur_sz,
            hh=h // patch_size, ww=w // patch_size,
            p1=patch_size, p2=patch_size, c=3,
        ).cpu()
        rendered_chunks.append(rendered)

    rendered_all = torch.cat(rendered_chunks, dim=1)
    return edict(input=input_dict, target=target_dict, render=rendered_all)


datasampler.set_epoch(0)
model.eval()

_render_all_views = config.inference.get("render_all_views", False)

with torch.no_grad(), torch.autocast(
    enabled=config.training.use_amp,
    device_type="cuda",
    dtype=amp_dtype_mapping[config.training.amp_dtype],
):
    for batch in dataloader:
        batch = {k: v.to(ddp_info.device) if type(v) == torch.Tensor else v for k, v in batch.items()}

        if _render_all_views:
            view_chunk_size = config.inference.get("view_chunk_size", 4)
            result = render_all_views_chunked(
                model.module, batch, ddp_info.device,
                view_chunk_size=view_chunk_size,
            )
            export_all_views_results(
                result, config.inference_out_dir,
                compute_metrics=config.inference.get("compute_metrics", False),
            )
        else:
            result = model(batch)
            if config.inference.get("render_video", False):
                result = model.module.render_video(result, **config.inference.render_video_config)
            export_results(result, config.inference_out_dir, compute_metrics=config.inference.get("compute_metrics"))
    torch.cuda.empty_cache()


dist.barrier()

if ddp_info.is_main_process and config.inference.get("compute_metrics", False):
    summarize_evaluation(config.inference_out_dir)
    if config.inference.get("generate_website", True):
        os.system(f"python generate_html.py {config.inference_out_dir}")
    
    # Generate consolidated metadata summary
    print("Generating consolidated metadata summary...")
    metadata_files = []
    for root, dirs, files in os.walk(config.inference_out_dir):
        for file in files:
            if file == "metadata.json":
                metadata_files.append(os.path.join(root, file))
    
    if metadata_files:
        consolidated_metadata = {}
        for metadata_file in metadata_files:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                scene_name = metadata.get("scene_name")
                if scene_name:
                    consolidated_metadata[scene_name] = metadata
        
        summary_path = os.path.join(config.inference_out_dir, "inference_metadata_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(consolidated_metadata, f, indent=2)
        print(f"Consolidated metadata saved to: {summary_path}")
        print(f"Total scenes: {len(consolidated_metadata)}")
    else:
        print("No metadata files found to consolidate.")
        
dist.barrier()
dist.destroy_process_group()
exit(0)