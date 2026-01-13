# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).

import importlib
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import PIL.Image
from setup import init_config
from utils import data_utils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
import argparse


def denormalize_image(image_tensor):
    """
    Convert normalized image tensor back to [0, 255] range for visualization.
    Handles both [0, 1] and [-1, 1] normalization.
    
    Args:
        image_tensor: torch.Tensor of shape [C, H, W] or [H, W, C], values in [0, 1] or [-1, 1]
    
    Returns:
        numpy array in [0, 255] range, shape [H, W, C]
    """
    if isinstance(image_tensor, torch.Tensor):
        image = image_tensor.cpu().numpy()
    else:
        image = image_tensor
    
    # Handle different tensor shapes
    if len(image.shape) == 3:
        if image.shape[0] == 3 or image.shape[0] == 1:  # [C, H, W]
            image = image.transpose(1, 2, 0)  # [H, W, C]
    
    # Check if normalized to [-1, 1] or [0, 1]
    if image.min() < 0:
        # Assume [-1, 1] range, convert to [0, 1]
        image = (image + 1) / 2.0
    
    # Clamp to [0, 1] and convert to [0, 255]
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)
    
    # Ensure 3 channels
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.shape[2] > 3:
        image = image[:, :, :3]
    
    return image


def save_data(data_dict, input_dict, target_dict, output_dir):
    """
    Save ray_o, ray_d, env_dir as .np files, and images as .png files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save input data
    input_dir = os.path.join(output_dir, "input")
    os.makedirs(input_dir, exist_ok=True)
    
    b, v = input_dict["image"].shape[:2]
    
    for view_idx in range(v):
        view_dir = os.path.join(input_dir, f"view_{view_idx:02d}")
        os.makedirs(view_dir, exist_ok=True)
        
        # Save ray_o and ray_d
        ray_o = input_dict["ray_o"][0, view_idx].cpu().numpy()  # [3, H, W]
        ray_d = input_dict["ray_d"][0, view_idx].cpu().numpy()  # [3, H, W]
        np.save(os.path.join(view_dir, "ray_o.npy"), ray_o)
        np.save(os.path.join(view_dir, "ray_d.npy"), ray_d)
        
        # Save env_dir
        env_dir = input_dict["env_dir"][0, view_idx].cpu().numpy()  # [3, envH, envW]
        np.save(os.path.join(view_dir, "env_dir.npy"), env_dir)
        
        # Save RGB image
        image = input_dict["image"][0, view_idx]  # [3, H, W]
        image_denorm = denormalize_image(image)
        PIL.Image.fromarray(image_denorm).save(os.path.join(view_dir, "image.png"))
        
        # Save relit_image if available
        if "relit_images" in input_dict and input_dict["relit_images"] is not None:
            relit_image = input_dict["relit_images"][0, view_idx]  # [3, H, W]
            relit_image_denorm = denormalize_image(relit_image)
            PIL.Image.fromarray(relit_image_denorm).save(os.path.join(view_dir, "relit_image.png"))
        
        # Save env_hdr if available
        if "env_hdr" in input_dict and input_dict["env_hdr"] is not None:
            env_hdr = input_dict["env_hdr"][0, view_idx]  # [3, envH, envW]
            env_hdr_denorm = denormalize_image(env_hdr)
            PIL.Image.fromarray(env_hdr_denorm).save(os.path.join(view_dir, "env_hdr.png"))
    
    # Save target data
    target_dir = os.path.join(output_dir, "target")
    os.makedirs(target_dir, exist_ok=True)
    
    b, v = target_dict["image"].shape[:2]
    
    for view_idx in range(v):
        view_dir = os.path.join(target_dir, f"view_{view_idx:02d}")
        os.makedirs(view_dir, exist_ok=True)
        
        # Save ray_o and ray_d
        ray_o = target_dict["ray_o"][0, view_idx].cpu().numpy()  # [3, H, W]
        ray_d = target_dict["ray_d"][0, view_idx].cpu().numpy()  # [3, H, W]
        np.save(os.path.join(view_dir, "ray_o.npy"), ray_o)
        np.save(os.path.join(view_dir, "ray_d.npy"), ray_d)
        
        # Save env_dir
        env_dir = target_dict["env_dir"][0, view_idx].cpu().numpy()  # [3, envH, envW]
        np.save(os.path.join(view_dir, "env_dir.npy"), env_dir)
        
        # Save RGB image
        image = target_dict["image"][0, view_idx]  # [3, H, W]
        image_denorm = denormalize_image(image)
        PIL.Image.fromarray(image_denorm).save(os.path.join(view_dir, "image.png"))
        
        # Save relit_image if available
        if "relit_images" in target_dict and target_dict["relit_images"] is not None:
            relit_image = target_dict["relit_images"][0, view_idx]  # [3, H, W]
            relit_image_denorm = denormalize_image(relit_image)
            PIL.Image.fromarray(relit_image_denorm).save(os.path.join(view_dir, "relit_image.png"))
        
        # Save env_hdr if available
        if "env_hdr" in target_dict and target_dict["env_hdr"] is not None:
            env_hdr = target_dict["env_hdr"][0, view_idx]  # [3, envH, envW]
            env_hdr_denorm = denormalize_image(env_hdr)
            PIL.Image.fromarray(env_hdr_denorm).save(os.path.join(view_dir, "env_hdr.png"))
    
    print(f"Data saved to {output_dir}")


def visualize_rays_and_envdir(input_dict, target_dict, output_dir, subsample=10):
    """
    Visualize ray_o, ray_d, and env_dir in 3D space.
    
    Args:
        input_dict: Input data dictionary
        target_dict: Target data dictionary
        output_dir: Output directory for visualizations
        subsample: Subsample factor for rays (to reduce number of rays for visualization)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize input views
    input_viz_dir = os.path.join(output_dir, "visualizations", "input")
    os.makedirs(input_viz_dir, exist_ok=True)
    
    b, v = input_dict["image"].shape[:2]
    
    for view_idx in range(v):
        visualize_single_view(
            input_dict, view_idx, input_viz_dir, subsample, prefix="input"
        )
    
    # Visualize target views
    target_viz_dir = os.path.join(output_dir, "visualizations", "target")
    os.makedirs(target_viz_dir, exist_ok=True)
    
    b, v = target_dict["image"].shape[:2]
    
    for view_idx in range(v):
        visualize_single_view(
            target_dict, view_idx, target_viz_dir, subsample, prefix="target"
        )


def visualize_single_view(data_dict, view_idx, output_dir, subsample=10, prefix=""):
    """
    Visualize rays and env_dir for a single view.
    """
    # Get data for this view
    ray_o = data_dict["ray_o"][0, view_idx].cpu().numpy()  # [3, H, W]
    ray_d = data_dict["ray_d"][0, view_idx].cpu().numpy()  # [3, H, W]
    
    # Get relit_image for colors
    has_relit = "relit_images" in data_dict and data_dict["relit_images"] is not None
    if has_relit:
        relit_image = data_dict["relit_images"][0, view_idx].cpu().numpy()  # [3, H, W]
        relit_image = relit_image.transpose(1, 2, 0)  # [H, W, 3]
        # Normalize to [0, 1] if needed
        if relit_image.min() < 0:
            relit_image = (relit_image + 1) / 2.0
        relit_image = np.clip(relit_image, 0, 1)
    else:
        # Use regular image if relit_image not available
        image = data_dict["image"][0, view_idx].cpu().numpy()  # [3, H, W]
        relit_image = image.transpose(1, 2, 0)  # [H, W, 3]
        if relit_image.min() < 0:
            relit_image = (relit_image + 1) / 2.0
        relit_image = np.clip(relit_image, 0, 1)
    
    # Get env_dir and env_hdr
    env_dir = data_dict["env_dir"][0, view_idx].cpu().numpy()  # [3, envH, envW]
    has_env_hdr = "env_hdr" in data_dict and data_dict["env_hdr"] is not None
    if has_env_hdr:
        env_hdr = data_dict["env_hdr"][0, view_idx].cpu().numpy()  # [3, envH, envW]
        env_hdr = env_hdr.transpose(1, 2, 0)  # [envH, envW, 3]
        if env_hdr.min() < 0:
            env_hdr = (env_hdr + 1) / 2.0
        env_hdr = np.clip(env_hdr, 0, 1)
    else:
        env_hdr = None
    
    H, W = ray_o.shape[1], ray_o.shape[2]
    envH, envW = env_dir.shape[1], env_dir.shape[2]
    
    # Reshape ray data
    ray_o_flat = ray_o.transpose(1, 2, 0).reshape(-1, 3)  # [H*W, 3]
    ray_d_flat = ray_d.transpose(1, 2, 0).reshape(-1, 3)  # [H*W, 3]
    relit_image_flat = relit_image.reshape(-1, 3)  # [H*W, 3]
    
    # Normalize ray_d to unit length
    ray_d_norm = ray_d_flat / (np.linalg.norm(ray_d_flat, axis=1, keepdims=True) + 1e-8)
    
    # Subsample rays
    indices = np.arange(H * W)
    subsampled_indices = indices[::subsample]
    ray_o_sub = ray_o_flat[subsampled_indices]  # [N, 3]
    ray_d_norm_sub = ray_d_norm[subsampled_indices]  # [N, 3]
    colors_sub = relit_image_flat[subsampled_indices]  # [N, 3]
    
    # Compute ray endpoints (ray_o + ray_d * 1.0)
    ray_endpoints = ray_o_sub + ray_d_norm_sub * 1.0  # [N, 3]
    
    # Create 3D plot
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot rays as lines
    for i in range(len(ray_o_sub)):
        ax.plot(
            [ray_o_sub[i, 0], ray_endpoints[i, 0]],
            [ray_o_sub[i, 1], ray_endpoints[i, 1]],
            [ray_o_sub[i, 2], ray_endpoints[i, 2]],
            color=colors_sub[i],
            linewidth=0.5,
            alpha=0.6
        )
    
    # Plot env_dir points
    if env_dir is not None:
        env_dir_flat = env_dir.transpose(1, 2, 0).reshape(-1, 3)  # [envH*envW, 3]
        env_points = 3.0 * env_dir_flat  # [envH*envW, 3]
        
        if env_hdr is not None:
            env_hdr_flat = env_hdr.reshape(-1, 3)  # [envH*envW, 3]
            # Subsample env points
            env_subsample = max(1, envH * envW // 1000)  # Limit to ~1000 points
            env_indices = np.arange(envH * envW)[::env_subsample]
            env_points_sub = env_points[env_indices]
            env_hdr_sub = env_hdr_flat[env_indices]
            
            ax.scatter(
                env_points_sub[:, 0],
                env_points_sub[:, 1],
                env_points_sub[:, 2],
                c=env_hdr_sub,
                s=1,
                alpha=0.8
            )
        else:
            # Use default color if env_hdr not available
            env_subsample = max(1, envH * envW // 1000)
            env_indices = np.arange(envH * envW)[::env_subsample]
            env_points_sub = env_points[env_indices]
            
            ax.scatter(
                env_points_sub[:, 0],
                env_points_sub[:, 1],
                env_points_sub[:, 2],
                c='red',
                s=1,
                alpha=0.5
            )
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'{prefix} View {view_idx:02d} - Rays and Environment Directions')
    
    # Set equal aspect ratio
    max_range = np.array([
        ray_o_sub.max() - ray_o_sub.min(),
        ray_endpoints.max() - ray_endpoints.min()
    ]).max()
    mid_x = (ray_o_sub[:, 0].max() + ray_o_sub[:, 0].min()) * 0.5
    mid_y = (ray_o_sub[:, 1].max() + ray_o_sub[:, 1].min()) * 0.5
    mid_z = (ray_o_sub[:, 2].max() + ray_o_sub[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    # Save figure
    output_path = os.path.join(output_dir, f"{prefix}_view_{view_idx:02d}_rays_envdir.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Test data preprocessing visualization")
    parser.add_argument("--config", "-c", required=True, help="Path to config file")
    parser.add_argument("--output-dir", "-o", default="./test_data_output", help="Output directory")
    parser.add_argument("--subsample", type=int, default=10, help="Subsample factor for rays (default: 10)")
    parser.add_argument("--batch-idx", type=int, default=0, help="Batch index to visualize (default: 0)")
    args, unknown = parser.parse_known_args()
    
    # Set up sys.argv for init_config (it expects --config in sys.argv)
    import sys
    original_argv = sys.argv.copy()
    sys.argv = ['test_data_visualization.py', '--config', args.config] + unknown
    
    # Load config
    config = init_config()
    
    # Restore original argv
    sys.argv = original_argv
    
    # Override batch size to 1
    config.training.batch_size_per_gpu = 1
    
    # Load dataset
    dataset_name = config.training.get("dataset_name", "data.dataset.Dataset")
    module, class_name = dataset_name.rsplit(".", 1)
    Dataset = importlib.import_module(module).__dict__[class_name]
    dataset = Dataset(config)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,  # Use 0 workers for debugging
        pin_memory=False,
        drop_last=False,
    )
    
    print(f"Dataset loaded with {len(dataset)} scenes")
    print(f"Loading batch {args.batch_idx}...")
    
    # Get one batch
    dataloader_iter = iter(dataloader)
    for i in range(args.batch_idx + 1):
        try:
            data = next(dataloader_iter)
        except StopIteration:
            print(f"Error: Batch index {args.batch_idx} is out of range")
            return
    
    # Move to device (CPU for simplicity)
    device = "cpu"
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    
    print("Batch loaded. Processing data...")
    
    # Initialize ProcessData
    process_data = data_utils.ProcessData(config)
    
    # Process data (like in train.py)
    input_dict, target_dict = process_data(
        batch,
        has_target_image=True,
        target_has_input=config.training.target_has_input,
        compute_rays=True
    )
    
    print("Data processed. Saving data...")
    
    # Save data
    save_data(batch, input_dict, target_dict, args.output_dir)
    
    print("Creating visualizations...")
    
    # Create visualizations
    visualize_rays_and_envdir(input_dict, target_dict, args.output_dir, subsample=args.subsample)
    
    print(f"All done! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()

