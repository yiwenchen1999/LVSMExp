
import importlib
import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from setup import init_config, init_distributed
from utils.metric_utils import export_results, summarize_evaluation
from utils.data_utils import create_video_from_frames
from einops import rearrange
import numpy as np
from PIL import Image
import re
import json

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


def parse_scene_name(scene_name):
    """
    Parse scene naming patterns used by env-variation datasets.

    Returns:
        dict or None:
          - object_id
          - input_prefix: dedup key for "same input" scenes
          - lighting_prefix: key for grouping different lighting envs
          - sort_key: stable sorting tuple
    """
    match = re.match(r"^(.+)_env_(\d+)_(\d+)$", scene_name)
    if match:
        object_id = match.group(1)
        env_num = int(match.group(2))
        variation_num = int(match.group(3))
        return {
            "object_id": object_id,
            "input_prefix": f"{object_id}_env_{env_num}",
            "lighting_prefix": f"env_{env_num}",
            "sort_key": (0, env_num, variation_num),
        }

    match = re.match(r"^(.+)_env_(\d+)$", scene_name)
    if match:
        object_id = match.group(1)
        env_num = int(match.group(2))
        return {
            "object_id": object_id,
            "input_prefix": f"{object_id}_env_{env_num}",
            "lighting_prefix": f"env_{env_num}",
            "sort_key": (0, env_num, 0),
        }

    match = re.match(r"^(.+)_white_env_(\d+)$", scene_name)
    if match:
        object_id = match.group(1)
        white_idx = int(match.group(2))
        return {
            "object_id": object_id,
            "input_prefix": f"{object_id}_white_env_{white_idx}",
            "lighting_prefix": f"white_env_{white_idx}",
            "sort_key": (1, white_idx, 0),
        }

    return None


def collect_all_lighting_scenes(scene_name, full_list_path):
    """
    Collect ALL scenes that share the same input prefix as `scene_name`.

    Scenes that share an input prefix (e.g. ``..._env_0_25`` and ``..._env_0_26``)
    correspond to the same reconstructed input but different rotated lighting envs.
    These are the relit frames we iterate over, so we keep every member of
    full_list.txt (no representative deduplication).

    Returns:
        tuple: (input_prefix, lighting_prefix, variation_scenes)
          - input_prefix: dedup key for the shared input / output folder
          - lighting_prefix: key used in output file names (e.g. ``env_0``)
          - variation_scenes: all scene names sharing the input prefix, sorted
            by rotation/variation order.
    """
    parsed_scene = parse_scene_name(scene_name)
    if parsed_scene is None:
        return None, None, []

    input_prefix = parsed_scene["input_prefix"]
    lighting_prefix = parsed_scene["lighting_prefix"]

    if not os.path.exists(full_list_path):
        if ddp_info.is_main_process:
            print(f"Warning: full_list.txt not found at {full_list_path}")
        return input_prefix, lighting_prefix, [scene_name]

    entries = []
    with open(full_list_path, "r") as f:
        for line in f:
            json_path = line.strip()
            if not json_path:
                continue
            json_file = os.path.basename(json_path)
            candidate_scene_name = json_file[:-5] if json_file.endswith(".json") else json_file

            parsed_candidate = parse_scene_name(candidate_scene_name)
            if parsed_candidate is None:
                continue
            # Same shared input -> this is one of the rotated lighting variations.
            if parsed_candidate["input_prefix"] != input_prefix:
                continue

            entries.append((parsed_candidate["sort_key"], candidate_scene_name))

    entries.sort(key=lambda x: (x[0], x[1]))
    variation_scenes = [name for _, name in entries]
    if not variation_scenes:
        variation_scenes = [scene_name]

    return input_prefix, lighting_prefix, variation_scenes


def load_env_variation_data(scene_name, base_dir, image_indices, dataset_class):
    """
    Load data for a specific environment variation scene.
    
    Args:
        scene_name: Scene name to load (e.g., "object_id_env_1_5")
        base_dir: Base directory containing metadata, images, envmaps folders
        image_indices: List of frame indices to load
        dataset_class: Dataset class to use for loading
        
    Returns:
        dict: Data batch for the variation scene
    """
    metadata_dir = os.path.join(base_dir, 'metadata')
    scene_json_path = os.path.join(metadata_dir, f"{scene_name}.json")
    
    if not os.path.exists(scene_json_path):
        return None
    
    # Load scene JSON
    with open(scene_json_path, 'r') as f:
        scene_data = json.load(f)
    
    frames = scene_data.get("frames", [])
    
    # Check if scene has enough frames
    if len(frames) <= max(image_indices):
        return None
    
    # Create a temporary dataset instance to use its loading methods
    # We'll create a minimal config for this
    temp_config = type('Config', (), {
        'training': type('Training', (), {
            'num_views': len(image_indices),
            'num_input_views': dataset_class.config.training.num_input_views,
            'num_target_views': dataset_class.config.training.num_target_views,
            'view_selector': dataset_class.config.training.view_selector,
        })(),
        'inference': type('Inference', (), {
            'if_inference': True,
        })(),
    })()
    
    # Create a dataset instance with the scene
    # We need to modify the dataset to load from a specific scene
    # For now, let's use the dataset's __getitem__ method directly
    # by creating a mock index that points to this scene
    
    # Actually, we need to load the data manually
    # Load images
    image_paths = [frames[ic]["image_path"] for ic in image_indices]
    frames_chosen = [frames[ic] for ic in image_indices]
    
    # Use dataset's preprocess_frames method
    # preprocess_frames returns: (images, intrinsics, c2ws) where all are already stacked tensors
    # images: [v, 3, h, w], intrinsics: [v, 4], c2ws: [v, 4, 4]
    images, fxfycxcy, c2w = dataset_class.preprocess_frames(frames_chosen, image_paths)
    
    # All returns are already tensors, no need to stack
    
    # Load environment maps
    envmaps_dir = os.path.join(base_dir, 'envmaps', scene_name)
    env_ldr_list = []
    env_hdr_list = []
    
    for frame_idx in image_indices:
        env_ldr_path = os.path.join(envmaps_dir, f"{frame_idx:05d}_ldr.png")
        env_hdr_path = os.path.join(envmaps_dir, f"{frame_idx:05d}_hdr.png")
        
        if os.path.exists(env_ldr_path) and os.path.exists(env_hdr_path):
            env_ldr = Image.open(env_ldr_path).convert('RGB')
            env_hdr = Image.open(env_hdr_path).convert('RGB')
            env_ldr = np.array(env_ldr) / 255.0
            env_hdr = np.array(env_hdr) / 255.0
            env_ldr_list.append(torch.from_numpy(env_ldr).float().permute(2, 0, 1))
            env_hdr_list.append(torch.from_numpy(env_hdr).float().permute(2, 0, 1))
        else:
            return None
    
    env_ldr = torch.stack(env_ldr_list, dim=0)  # [v, 3, h, w]
    env_hdr = torch.stack(env_hdr_list, dim=0)  # [v, 3, h, w]
    
    # Create data batch
    # images: [v, 3, h, w] -> [1, v, 3, h, w]
    # fxfycxcy: [v, 4] -> [1, v, 4]
    # c2w: [v, 4, 4] -> [1, v, 4, 4]
    data_batch = {
        "image": images.unsqueeze(0),  # [1, v, 3, h, w]
        "fxfycxcy": fxfycxcy.unsqueeze(0),  # [1, v, 4]
        "c2w": c2w.unsqueeze(0),  # [1, v, 4, 4]
        "scene_name": [scene_name],
        "env_ldr": env_ldr.unsqueeze(0),  # [1, v, 3, h, w]
        "env_hdr": env_hdr.unsqueeze(0),  # [1, v, 3, h, w]
    }
    
    return data_batch


# Load data
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
    drop_last=False,  # Don't drop last batch for inference
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
    print(f"Running exp_rotate_env inference; save results to: {config.inference_out_dir}")
    # avoid multiple processes downloading LPIPS at the same time
    import lpips
    # Suppress the warning by setting weights_only=True
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)

dist.barrier()


datasampler.set_epoch(0)
model.eval()
processed_input_prefixes = set()

with torch.no_grad(), torch.autocast(
    enabled=config.training.use_amp,
    device_type="cuda",
    dtype=amp_dtype_mapping[config.training.amp_dtype],
):
    for batch in dataloader:
        batch = {k: v.to(ddp_info.device) if type(v) == torch.Tensor else v for k, v in batch.items()}
        
        # Get scene name
        scene_name = batch["scene_name"][0] if isinstance(batch["scene_name"], list) else batch["scene_name"][0]
        
        # Extract base directory from scene path in dataset
        # Find the scene path from dataset's all_scene_paths
        scene_path = None
        for path in dataset.all_scene_paths:
            if scene_name in path:
                scene_path = path
                break
        
        if scene_path is None:
            print(f"Warning: Could not find scene path for {scene_name}, processing normally")
            result = model(batch)
            if config.inference.get("render_video", False):
                result = model.module.render_video(result, **config.inference.render_video_config)
            export_results(result, config.inference_out_dir, compute_metrics=config.inference.get("compute_metrics"))
            continue
        
        scene_path_dir = os.path.dirname(scene_path)
        base_dir = os.path.dirname(scene_path_dir)  # Go up from metadata to train/test
        metadata_dir = os.path.join(base_dir, 'metadata')
        
        # Find full_list.txt path
        full_list_path = os.path.join(base_dir, 'full_list.txt')
        
        # Dedup the INPUT: scenes sharing an input prefix (e.g. *_env_0_25, *_env_0_26)
        # use the same reconstructed input once, but we still relight over ALL of them.
        input_prefix, lighting_prefix, variation_scenes = collect_all_lighting_scenes(scene_name, full_list_path)

        if input_prefix is None:
            print(f"Warning: Could not parse scene name {scene_name}, processing normally")
            result = model(batch)
            if config.inference.get("render_video", False):
                result = model.module.render_video(result, **config.inference.render_video_config)
            export_results(result, config.inference_out_dir, compute_metrics=config.inference.get("compute_metrics"))
            continue

        if input_prefix in processed_input_prefixes:
            if ddp_info.is_main_process:
                print(f"Skipping duplicate input prefix '{input_prefix}' from scene '{scene_name}'")
            continue
        processed_input_prefixes.add(input_prefix)

        if not variation_scenes:
            print(f"No lighting scenes found for {scene_name}, processing normally")
            result = model(batch)
            if config.inference.get("render_video", False):
                result = model.module.render_video(result, **config.inference.render_video_config)
            export_results(result, config.inference_out_dir, compute_metrics=config.inference.get("compute_metrics"))
            continue

        if ddp_info.is_main_process:
            print(
                f"Using input prefix '{input_prefix}' (lighting '{lighting_prefix}') "
                f"with {len(variation_scenes)} relit rotations"
            )
        
        # Process original scene first to get input data and reconstruct scene
        input, target = model.module.process_data(
            batch, 
            has_target_image=True, 
            target_has_input=config.training.target_has_input, 
            compute_rays=True
        )
        
        # Reconstruct scene once (using input images) - this is shared across all variations
        latent_tokens, n_patches, d = model.module.reconstructor(input)
        
        # Get target information (will be reused for all variations)
        from easydict import EasyDict as edict
        target_template = edict({
            'ray_o': target.ray_o,
            'ray_d': target.ray_d,
            'image_h_w': target.image_h_w,
        })
        
        # Get input view indices from the original batch
        # We need to know which frame indices were used for input views
        # Get this from the dataset's view_idx_list or infer from input shape
        v_input = input.image.shape[1]
        
        # Prefer the actual sampled context frame ids from this batch.
        # This is required for random_chunk_sampling (and is correct for any sampler),
        # since the context frames are not necessarily 0..v_input-1.
        input_indices = None
        if hasattr(input, "index") and input.index is not None:
            try:
                input_indices = input.index[0, :, 0].detach().cpu().long().tolist()
            except Exception:
                input_indices = None
        
        # Fallback: actual indices from view_idx_list json (legacy eval-index mode)
        if input_indices is None and hasattr(dataset, 'view_idx_list') and scene_name in dataset.view_idx_list:
            view_idx_info = dataset.view_idx_list[scene_name]
            if 'context' in view_idx_info:
                input_indices = view_idx_info['context'][:v_input]
        
        # Fallback: assume sequential indices starting from 0
        if input_indices is None:
            input_indices = list(range(v_input))
        
        # Prepare output dirs once (one folder per shared input).
        light_key = "".join(c for c in lighting_prefix if c.isalnum() or c in ('_', '-'))
        if ddp_info.is_main_process:
            safe_group_name = "".join(c for c in input_prefix if c.isalnum() or c in ('_', '-'))[:100]
            sample_dir = os.path.join(config.inference_out_dir, safe_group_name)
            image_dir = os.path.join(sample_dir, "images")
            envmap_dir = os.path.join(sample_dir, "envmaps")
            before_lit_dir = os.path.join(sample_dir, "before_lit")
            os.makedirs(sample_dir, exist_ok=True)
            os.makedirs(image_dir, exist_ok=True)
            os.makedirs(envmap_dir, exist_ok=True)
            os.makedirs(before_lit_dir, exist_ok=True)

            # Save the GT target views (before relighting) alongside predictions.
            if hasattr(target, "image") and target.image is not None:
                gt_target = target.image  # [1, v_target, 3, h, w]
                for view_idx in range(gt_target.shape[1]):
                    gt_img = gt_target[0, view_idx].detach().cpu().float()  # [3, h, w]
                    gt_img = rearrange(gt_img, "c h w -> h w c")
                    gt_img = (gt_img.numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
                    Image.fromarray(gt_img).save(
                        os.path.join(before_lit_dir, f"light_{light_key}_view_{view_idx:03d}.png")
                    )

        # Iterate over ALL relit rotations sharing this input; render and save
        # incrementally to keep GPU memory bounded. Each target view gets its own
        # rotating sequence/video so all target views are preserved.
        video_frames_by_view = {}  # view_idx -> list of CPU uint8 [h, w, 3]
        target_counter = 0
        for rot_idx, var_scene_name in enumerate(variation_scenes):
            if ddp_info.is_main_process:
                print(
                    f"Processing relit rotation {rot_idx + 1}/{len(variation_scenes)}: {var_scene_name}"
                )

            # Load environment variation data
            var_data = load_env_variation_data(var_scene_name, base_dir, input_indices, dataset)

            if var_data is None:
                if ddp_info.is_main_process:
                    print(f"Warning: Could not load data for variation {var_scene_name}, skipping")
                continue

            # Move to device
            var_data = {k: v.to(ddp_info.device) if type(v) == torch.Tensor else v for k, v in var_data.items()}

            # Get env maps from variation data
            var_env_ldr = var_data["env_ldr"]  # [1, v_input, 3, h, w]
            var_env_hdr = var_data["env_hdr"]  # [1, v_input, 3, h, w]

            # Compute env_dir for the variation (using input c2w)
            var_env_dir = model.module.process_data.compute_env_dir(
                input.c2w,
                envmap_h=256,
                envmap_w=512,
                device=ddp_info.device
            )  # [1, v_input, 3, env_h, env_w]

            # Create input with variation env maps (copy input and update env maps)
            var_input = edict(input)
            var_input.env_ldr = var_env_ldr
            var_input.env_hdr = var_env_hdr
            var_input.env_dir = var_env_dir

            # Edit scene with variation env maps
            edited_latent_tokens = model.module.edit_scene_with_env(latent_tokens, var_input)

            # Render with same target rays
            rendered_images = model.module.renderer(edited_latent_tokens, target_template, n_patches, d)

            if ddp_info.is_main_process:
                v_target = rendered_images.shape[1]

                # Save one relit image per target view for this rotation.
                for view_idx in range(v_target):
                    img = rendered_images[0, view_idx].detach().cpu().float()  # [3, h, w]
                    img = rearrange(img, "c h w -> h w c")
                    img = (img.numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
                    Image.fromarray(img).save(
                        os.path.join(
                            image_dir,
                            f"light_{light_key}_view_{view_idx:03d}_target_{target_counter:03d}.png",
                        )
                    )
                    video_frames_by_view.setdefault(view_idx, []).append(img)

                # Save the corresponding input envmap (view 0) once per rotation.
                env_ldr_img = rearrange(var_env_ldr[0, 0].detach().cpu().float(), "c h w -> h w c")
                env_hdr_img = rearrange(var_env_hdr[0, 0].detach().cpu().float(), "c h w -> h w c")
                env_ldr_img = (env_ldr_img.numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
                env_hdr_img = (env_hdr_img.numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
                Image.fromarray(env_ldr_img).save(
                    os.path.join(envmap_dir, f"light_{light_key}_target_{target_counter:03d}_ldr.png")
                )
                Image.fromarray(env_hdr_img).save(
                    os.path.join(envmap_dir, f"light_{light_key}_target_{target_counter:03d}_hdr.png")
                )

                target_counter += 1

        # Save one grouped video per target view for this lighting prefix.
        if ddp_info.is_main_process and video_frames_by_view:
            for view_idx in sorted(video_frames_by_view.keys()):
                frames = np.stack(video_frames_by_view[view_idx], axis=0)  # [n_rot, h, w, 3]
                # Keep all rotation frames, extend duration by repeating each frame 3x at fps 24.
                frames = np.repeat(frames, repeats=3, axis=0)
                video_path = os.path.join(sample_dir, f"group_{light_key}_view_{view_idx:03d}.mp4")
                create_video_from_frames(frames, video_path, framerate=24)

            num_views = len(video_frames_by_view)
            print(
                f"Saved {target_counter * num_views} relit images, {target_counter} envmaps, "
                f"and {num_views} grouped videos (one per target view) to {sample_dir}"
            )

        torch.cuda.empty_cache()


dist.barrier()

if ddp_info.is_main_process and config.inference.get("compute_metrics", False):
    summarize_evaluation(config.inference_out_dir)
    if config.inference.get("generate_website", True):
        os.system(f"python generate_html.py {config.inference_out_dir}")
dist.barrier()
dist.destroy_process_group()
exit(0)

