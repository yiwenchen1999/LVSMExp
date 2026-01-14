#!/usr/bin/env python3
"""
Preview script to visualize processed scenes.
Samples one image from each scene (default: 65th image) and displays them in a grid.
"""

import os
import json
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def check_image_black_or_white(img, threshold=0.9):
    """
    Check if an image is mostly black or mostly white.
    
    Args:
        img: PIL Image (RGB)
        threshold: Threshold ratio (default: 0.9, meaning 90%)
        
    Returns:
        tuple: (is_mostly_black, is_mostly_white, black_ratio, white_ratio)
    """
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Calculate total pixels
    total_pixels = img_array.shape[0] * img_array.shape[1]
    
    # Check for fully black pixels (RGB = [0, 0, 0])
    black_pixels = np.all(img_array == [0, 0, 0], axis=2)
    black_count = np.sum(black_pixels)
    black_ratio = black_count / total_pixels
    
    # Check for fully white pixels (RGB = [255, 255, 255])
    white_pixels = np.all(img_array == [255, 255, 255], axis=2)
    white_count = np.sum(white_pixels)
    white_ratio = white_count / total_pixels
    
    is_mostly_black = black_ratio > threshold
    is_mostly_white = white_ratio > threshold
    
    return is_mostly_black, is_mostly_white, black_ratio, white_ratio


def load_scene_image(scene_json_path, image_idx=64):
    """
    Load an image from a scene JSON file.
    
    Args:
        scene_json_path: Path to the scene JSON file
        image_idx: Index of the image to load (0-based, default: 64 for 65th image)
        
    Returns:
        tuple: (PIL Image or None if failed, scene_name)
    """
    try:
        with open(scene_json_path, 'r') as f:
            scene_data = json.load(f)
        
        frames = scene_data.get('frames', [])
        if image_idx >= len(frames):
            # If requested index is out of range, use the last image
            image_idx = len(frames) - 1
        
        if image_idx < 0 or len(frames) == 0:
            return None
        
        image_path = frames[image_idx]['image_path']
        
        # The image_path in JSON is already an absolute path (from preprocess_objaverse.py line 524)
        # But check if it exists, if not, try to construct it from the JSON path
        if not os.path.exists(image_path):
            # Try to construct path from JSON location
            # Structure: output_root/split/metadata/scene.json
            # Images: output_root/split/images/scene_name/00000.png
            json_dir = os.path.dirname(scene_json_path)
            metadata_dir = os.path.dirname(json_dir) if os.path.basename(json_dir) == 'metadata' else json_dir
            split_dir = os.path.dirname(metadata_dir) if os.path.basename(metadata_dir) in ['test', 'train'] else metadata_dir
            
            scene_name = os.path.basename(scene_json_path).replace('.json', '')
            image_filename = os.path.basename(image_path)
            image_path = os.path.join(split_dir, 'images', scene_name, image_filename)
        
        scene_name = os.path.basename(scene_json_path).replace('.json', '')
        
        if os.path.exists(image_path):
            img = Image.open(image_path).convert('RGB')
            return img, scene_name
        else:
            print(f"Warning: Image not found: {image_path}")
            return None, scene_name
    except Exception as e:
        print(f"Error loading scene {scene_json_path}: {e}")
        scene_name = os.path.basename(scene_json_path).replace('.json', '')
        return None, scene_name


def add_text_to_image(img, text, font_size=20):
    """
    Add text label to an image.
    
    Args:
        img: PIL Image
        text: Text to add
        font_size: Font size
        
    Returns:
        PIL Image with text
    """
    # Create a copy to avoid modifying original
    img_with_text = img.copy()
    draw = ImageDraw.Draw(img_with_text)
    
    # Try to load a font, fallback to default if not available
    try:
        # Try to use a default font
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except:
        try:
            # Try another common font path
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except:
            # Fallback to default font
            font = ImageFont.load_default()
    
    # Get text bounding box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Calculate position (bottom center)
    img_width, img_height = img.size
    x = (img_width - text_width) // 2
    y = img_height - text_height - 10
    
    # Draw background rectangle for better text visibility
    padding = 5
    draw.rectangle(
        [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
        fill=(0, 0, 0, 180)  # Semi-transparent black
    )
    
    # Draw text
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    
    return img_with_text


def create_preview_grid(full_list_path, output_path_template, image_idx=64, grid_cols=8, grid_rows=4, images_per_grid=32):
    """
    Create preview grids of images from all processed scenes.
    Saves multiple preview images (preview1.png, preview2.png, ...) if there are many scenes.
    
    Args:
        full_list_path: Path to full_list.txt file
        output_path_template: Template for output paths (e.g., "preview.png" -> "preview1.png", "preview2.png", ...)
        image_idx: Index of image to sample from each scene (0-based, default: 64 for 65th image)
        grid_cols: Number of columns in the grid (default: 8)
        grid_rows: Number of rows in the grid (default: 4)
        images_per_grid: Number of images per grid (default: 32)
    """
    # Read scene list
    if not os.path.exists(full_list_path):
        print(f"Error: full_list.txt not found at {full_list_path}")
        return
    
    with open(full_list_path, 'r') as f:
        scene_paths = [line.strip() for line in f.readlines() if line.strip()]
    
    if not scene_paths:
        print(f"Error: No scenes found in {full_list_path}")
        return
    
    total_scenes = len(scene_paths)
    print(f"Found {total_scenes} scenes in {full_list_path}")
    print(f"Sampling image index {image_idx} (the {image_idx + 1}th image) from each scene")
    print(f"Creating {grid_rows}x{grid_cols} grids with {images_per_grid} images per grid")
    print(f"Checking for broken scenes (images with >90% black or white pixels)")
    
    # Calculate number of preview files needed
    num_previews = (total_scenes + images_per_grid - 1) // images_per_grid
    print(f"Will create {num_previews} preview files")
    
    # List to store all broken scene names
    broken_scenes_list = []
    
    # Determine output directory and base name
    output_dir = os.path.dirname(output_path_template) if os.path.dirname(output_path_template) else '.'
    output_base = os.path.basename(output_path_template)
    # Remove extension if present
    if '.' in output_base:
        base_name, ext = os.path.splitext(output_base)
    else:
        base_name = output_base
        ext = '.png'
    
    # Process scenes in batches
    for preview_idx in range(num_previews):
        start_idx = preview_idx * images_per_grid
        end_idx = min(start_idx + images_per_grid, total_scenes)
        batch_scene_paths = scene_paths[start_idx:end_idx]
        
        print(f"\nProcessing batch {preview_idx + 1}/{num_previews} (scenes {start_idx + 1}-{end_idx})...")
        
        # Load images for this batch
        images = []
        scene_names = []
        broken_scenes_batch = []  # Track broken scenes in this batch
        
        for scene_path in batch_scene_paths:
            img, scene_name = load_scene_image(scene_path, image_idx)
            scene_names.append(scene_name)
            
            if img is not None:
                # Check if image is mostly black or white
                is_mostly_black, is_mostly_white, black_ratio, white_ratio = check_image_black_or_white(img, threshold=0.9)
                
                if is_mostly_black or is_mostly_white:
                    broken_scenes_batch.append(scene_name)
                    print(f"  Broken scene detected: {scene_name} (black: {black_ratio:.2%}, white: {white_ratio:.2%})")
                
                images.append(img)
            else:
                # Create a placeholder image if loading failed
                placeholder = Image.new('RGB', (256, 256), color=(128, 128, 128))
                images.append(placeholder)
        
        # Append broken scenes from this batch to the main list
        if broken_scenes_batch:
            broken_scenes_list.extend(broken_scenes_batch)
        
        if not images:
            print(f"Warning: No images could be loaded for batch {preview_idx + 1}")
            continue
        
        # Resize all images to the same size (use the first image's size or a default)
        target_size = (256, 256)  # Default size
        if images[0] is not None:
            # Use a reasonable size based on first image
            first_img_size = images[0].size
            # Keep aspect ratio but limit to reasonable size
            max_dim = 256
            if first_img_size[0] > first_img_size[1]:
                target_size = (max_dim, int(first_img_size[1] * max_dim / first_img_size[0]))
            else:
                target_size = (int(first_img_size[0] * max_dim / first_img_size[1]), max_dim)
        
        resized_images = []
        for i, img in enumerate(images):
            # Resize image
            img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
            # Add scene name text
            scene_name = scene_names[i]
            # Truncate scene name if too long (max 15 characters)
            max_name_length = 15
            if len(scene_name) > max_name_length:
                scene_name = scene_name[:max_name_length] + "..."
            img_with_text = add_text_to_image(img_resized, scene_name, font_size=12)
            resized_images.append(img_with_text)
        
        # Create grid
        img_width, img_height = target_size
        grid_width = grid_cols * img_width
        grid_height = grid_rows * img_height
        
        # Create blank canvas
        grid_image = Image.new('RGB', (grid_width, grid_height), color=(255, 255, 255))
        
        # Paste images into grid
        for idx, img in enumerate(resized_images):
            if idx >= images_per_grid:
                break
            
            row = idx // grid_cols
            col = idx % grid_cols
            
            if row >= grid_rows:
                break
            
            x = col * img_width
            y = row * img_height
            
            grid_image.paste(img, (x, y))
        
        # Generate output path
        output_path = os.path.join(output_dir, f"{base_name}{preview_idx + 1}{ext}")
        
        # Save the grid
        grid_image.save(output_path)
        print(f"Preview {preview_idx + 1} saved to {output_path} ({len(resized_images)} images)")
    
    # Save broken scenes list to file
    if broken_scenes_list:
        # Determine output directory for broken_scene.txt
        output_dir = os.path.dirname(output_path_template) if os.path.dirname(output_path_template) else '.'
        broken_scene_file = os.path.join(output_dir, 'broken_scene.txt')
        
        with open(broken_scene_file, 'w') as f:
            for scene_name in sorted(set(broken_scenes_list)):  # Remove duplicates and sort
                f.write(f"{scene_name}\n")
        
        print(f"\nFound {len(set(broken_scenes_list))} broken scenes (images with >90% black or white pixels)")
        print(f"Broken scenes list saved to {broken_scene_file}")
    else:
        print(f"\nNo broken scenes detected (all images have <90% black/white pixels)")
    
    print(f"\nAll previews complete! Created {num_previews} preview files.")


def main():
    parser = argparse.ArgumentParser(description='Preview processed scenes')
    parser.add_argument('--full-list', '-f', required=True,
                       help='Path to full_list.txt file (e.g., data_samples/objaverse_processed/test/full_list.txt)')
    parser.add_argument('--output', '-o', required=True,
                       help='Output path template for preview images (e.g., preview.png -> preview1.png, preview2.png, ...)')
    parser.add_argument('--image-idx', '-i', type=int, default=64,
                       help='Index of image to sample from each scene (0-based, default: 64 for 65th image)')
    parser.add_argument('--grid-cols', type=int, default=8,
                       help='Number of columns in the grid (default: 8)')
    parser.add_argument('--grid-rows', type=int, default=4,
                       help='Number of rows in the grid (default: 4)')
    parser.add_argument('--images-per-grid', type=int, default=32,
                       help='Number of images per grid (default: 32)')
    
    args = parser.parse_args()
    
    create_preview_grid(
        args.full_list,
        args.output,
        image_idx=args.image_idx,
        grid_cols=args.grid_cols,
        grid_rows=args.grid_rows,
        images_per_grid=args.images_per_grid
    )


if __name__ == '__main__':
    main()

