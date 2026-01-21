
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


def find_all_env_variations(scene_name, full_list_path):
    """
    Find all environment variation scenes for a given base scene from full_list.txt.
    
    For example:
    - If scene_name is "01c9013483b6427fbc2f478e5e328810_env_0_1",
      find all "01c9013483b6427fbc2f478e5e328810_env_*" scenes
    - If scene_name is "e561fa2f48d64a9fb62ca03daeea41be_white_env_0",
      find all "e561fa2f48d64a9fb62ca03daeea41be*" scenes (env_* and white_env_*)
    
    Args:
        scene_name: Current scene name (e.g., "object_id_env_0_1" or "object_id_white_env_0")
        full_list_path: Path to full_list.txt file containing scene JSON paths
        
    Returns:
        list: Sorted list of variation scene names (sorted by env_num and variation number)
    """
    # Parse scene name to extract object_id
    # Examples:
    #   "01c9013483b6427fbc2f478e5e328810_env_0_1" -> object_id: "01c9013483b6427fbc2f478e5e328810"
    #   "e561fa2f48d64a9fb62ca03daeea41be_white_env_0" -> object_id: "e561fa2f48d64a9fb62ca03daeea41be"
    
    object_id = None
    
    # Try to match pattern: {object_id}_white_env_{env_num} first (more specific)
    # This must come before the generic _env_ pattern to avoid greedy matching
    match = re.match(r'^(.+)_white_env_(\d+)$', scene_name)
    if match:
        object_id = match.group(1)
    else:
        # Try pattern: {object_id}_env_{env_num}_{variation}
        match = re.match(r'^(.+)_env_(\d+)_(\d+)$', scene_name)
        if match:
            object_id = match.group(1)
        else:
            # Try pattern: {object_id}_env_{env_num} (without variation)
            match = re.match(r'^(.+)_env_(\d+)$', scene_name)
            if match:
                object_id = match.group(1)
            else:
                print(f"Warning: Could not parse scene name '{scene_name}', skipping env variations")
                return []
    
    if object_id is None:
        print(f"Warning: Could not extract object_id from scene name '{scene_name}', skipping env variations")
        return []
    
    # Read scene list from full_list.txt
    if not os.path.exists(full_list_path):
        print(f"Warning: full_list.txt not found at {full_list_path}, skipping env variations")
        return []
    
    variation_scenes = []
    
    # Pattern to match: {object_id}_env_{any_env_num}_{variation}
    # This will match all env variations (env_0_*, env_1_*, env_2_*, etc.)
    pattern_env = f"^{re.escape(object_id)}_env_(\\d+)_(\\d+)$"
    
    # Pattern to match: {object_id}_white_env_{env_num} (white_env usually doesn't have variations)
    pattern_white_env = f"^{re.escape(object_id)}_white_env_(\\d+)$"
    
    # Debug: print patterns
    if ddp_info.is_main_process:
        print(f"Looking for variations of scene: {scene_name}")
        print(f"Extracted object_id: {object_id}")
        print(f"Pattern env: {pattern_env}")
        print(f"Pattern white_env: {pattern_white_env}")
    
    with open(full_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Extract scene name from JSON path
            # Path format: /path/to/metadata/scene_name.json
            json_path = line
            json_file = os.path.basename(json_path)
            candidate_scene_name = json_file[:-5] if json_file.endswith('.json') else json_file
            
            # Skip the current scene itself
            if candidate_scene_name == scene_name:
                continue
            
            # Check if candidate starts with object_id (quick check before regex)
            if not candidate_scene_name.startswith(object_id + '_'):
                continue
            
            # Check if it matches env pattern: {object_id}_env_{env_num}_{variation}
            match_env = re.match(pattern_env, candidate_scene_name)
            if match_env:
                env_num = int(match_env.group(1))
                variation_num = int(match_env.group(2))
                # Store as (0, env_num, variation_num, scene_name) for sorting
                # Use 0 prefix to distinguish from white_env
                variation_scenes.append((0, env_num, variation_num, candidate_scene_name))
                if ddp_info.is_main_process:
                    print(f"  Matched env scene: {candidate_scene_name} (env_{env_num}, var_{variation_num})")
            else:
                # Check if it matches white_env pattern: {object_id}_white_env_{env_num}
                match_white = re.match(pattern_white_env, candidate_scene_name)
                if match_white:
                    env_num = int(match_white.group(1))
                    # Store as (1, env_num, 0, scene_name) for sorting
                    # Use 1 prefix to put white_env after regular env
                    variation_scenes.append((1, env_num, 0, candidate_scene_name))
                    if ddp_info.is_main_process:
                        print(f"  Matched white_env scene: {candidate_scene_name} (env_{env_num})")
    
    # Sort by: type (0=env, 1=white_env), env_num, variation_num
    # This ensures order: env_0_1, env_0_2, ..., env_1_1, env_1_2, ..., white_env_0, white_env_1, ...
    variation_scenes.sort(key=lambda x: (x[0], x[1], x[2]))
    
    return [scene_name for _, _, _, scene_name in variation_scenes]


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
    images, fxfycxcy_list, c2w_list = dataset_class.preprocess_frames(frames_chosen, image_paths)
    
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
    data_batch = {
        "image": images.unsqueeze(0),  # [1, v, 3, h, w]
        "fxfycxcy": torch.stack(fxfycxcy_list, dim=0).unsqueeze(0),  # [1, v, 4]
        "c2w": torch.stack(c2w_list, dim=0).unsqueeze(0),  # [1, v, 4, 4]
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
        
        # Find all environment variation scenes from full_list.txt
        variation_scenes = find_all_env_variations(scene_name, full_list_path)
        
        if not variation_scenes:
            print(f"No environment variations found for {scene_name}, processing normally")
            result = model(batch)
            if config.inference.get("render_video", False):
                result = model.module.render_video(result, **config.inference.render_video_config)
            export_results(result, config.inference_out_dir, compute_metrics=config.inference.get("compute_metrics"))
            continue
        
        print(f"Found {len(variation_scenes)} environment variations for {scene_name}")
        
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
        
        # Try to get actual indices from view_idx_list
        input_indices = None
        if hasattr(dataset, 'view_idx_list') and scene_name in dataset.view_idx_list:
            view_idx_info = dataset.view_idx_list[scene_name]
            if 'context' in view_idx_info:
                input_indices = view_idx_info['context'][:v_input]
        
        # Fallback: assume sequential indices starting from 0
        if input_indices is None:
            input_indices = list(range(v_input))
        
        # Store results for all variations
        all_variation_results = []
        
        # Process each variation in order
        for var_idx, var_scene_name in enumerate(variation_scenes):
            if ddp_info.is_main_process:
                print(f"Processing variation {var_idx + 1}/{len(variation_scenes)}: {var_scene_name}")
            
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
            
            # Store result
            all_variation_results.append(rendered_images)  # [1, v_target, 3, h, w]
        
        # Stitch all variation results together
        if all_variation_results:
            # Concatenate all renders horizontally: [1, v_target, 3, h, k*w]
            all_renders = torch.cat(all_variation_results, dim=4)  # [1, v_target, 3, h, k*w]
            
            # Save stitched images and video
            if ddp_info.is_main_process:
                safe_scene_name = "".join(c for c in scene_name if c.isalnum() or c in ('_', '-'))[:100]
                sample_dir = os.path.join(config.inference_out_dir, safe_scene_name)
                os.makedirs(sample_dir, exist_ok=True)
                
                v_target, c, h, total_w = all_renders.shape[1], all_renders.shape[2], all_renders.shape[3], all_renders.shape[4]
                
                # Save stitched image for each target view
                for view_idx in range(v_target):
                    stitched_img = all_renders[0, view_idx].detach().cpu()  # [3, h, k*w]
                    stitched_img = rearrange(stitched_img, "c h w -> h w c")
                    stitched_img = (stitched_img.numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
                    Image.fromarray(stitched_img).save(
                        os.path.join(sample_dir, f"stitched_view_{view_idx:03d}.png")
                    )
                
                # Save as video (each frame is one target view, showing all variations horizontally)
                # Reshape: [1, v_target, 3, h, k*w] -> [v_target, h, k*w, 3]
                video_frames = rearrange(all_renders.squeeze(0), "v c h w -> v h w c")
                video_frames = (video_frames.detach().cpu().numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
                video_path = os.path.join(sample_dir, "env_variations_video.mp4")
                create_video_from_frames(video_frames, video_path, framerate=15)
                print(f"Saved stitched results ({len(variation_scenes)} variations) and video to {sample_dir}")
        
        torch.cuda.empty_cache()


dist.barrier()

if ddp_info.is_main_process and config.inference.get("compute_metrics", False):
    summarize_evaluation(config.inference_out_dir)
    if config.inference.get("generate_website", True):
        os.system(f"python generate_html.py {config.inference_out_dir}")
dist.barrier()
dist.destroy_process_group()
exit(0)

