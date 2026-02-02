# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).

import random
import traceback
import os
import numpy as np
import PIL
import torch
from torch.utils.data import Dataset
import json
import torch.nn.functional as F



class Dataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config

        try:
            with open(self.config.training.dataset_path, 'r') as f:
                self.all_scene_paths = f.read().splitlines()
            self.all_scene_paths = [path for path in self.all_scene_paths if path.strip()]
        
        except Exception as e:
            print(f"Error reading dataset paths from '{self.config.training.dataset_path}'")
            raise e
        
        # Filter scenes if whiteEnvInput is enabled
        # Only keep scenes ending with "white_env_0" as input images
        self.whiteEnvInput = self.config.training.get("whiteEnvInput", False)
        if self.whiteEnvInput:
            total_scenes_before = len(self.all_scene_paths)
            filtered_scene_paths = []
            for scene_path in self.all_scene_paths:
                # Extract scene name from path (e.g., "/path/to/metadata/scene_name.json" -> "scene_name")
                file_name = os.path.basename(scene_path)
                scene_name = file_name.replace('.json', '')
                # Only keep scenes ending with "white_env_0"
                if scene_name.endswith('_white_env_0'):
                    filtered_scene_paths.append(scene_path)
            self.all_scene_paths = filtered_scene_paths
            print(f"whiteEnvInput enabled: Filtered to {len(self.all_scene_paths)} scenes ending with 'white_env_0' (from {total_scenes_before} total)")

        self.inference = self.config.inference.get("if_inference", False)
        # Load file that specifies the input and target view indices to use for inference
        if self.inference:
            self.view_idx_list = dict()
            if self.config.inference.get("view_idx_file_path", None) is not None:
                if os.path.exists(self.config.inference.view_idx_file_path):
                    with open(self.config.inference.view_idx_file_path, 'r') as f:
                        self.view_idx_list = json.load(f)
                        # filter out None values, i.e. scenes that don't have specified input and targetviews
                        self.view_idx_list_filtered = [k for k, v in self.view_idx_list.items() if v is not None]
                    filtered_scene_paths = []
                    for scene in self.all_scene_paths:
                        file_name = scene.split("/")[-1]
                        scene_name = file_name.split(".")[0]
                        if scene_name in self.view_idx_list_filtered:
                            filtered_scene_paths.append(scene)

                    self.all_scene_paths = filtered_scene_paths

        # Check if we should load relit images
        self.use_relit_images = self.config.training.get("use_relit_images", True)


    def __len__(self):
        return len(self.all_scene_paths)


    def preprocess_frames(self, frames_chosen, image_paths_chosen):
        resize_h = self.config.model.image_tokenizer.image_size
        patch_size = self.config.model.image_tokenizer.patch_size
        square_crop = self.config.training.get("square_crop", False)

        images = []
        intrinsics = []
        for cur_frame, cur_image_path in zip(frames_chosen, image_paths_chosen):
            try:
                # Try to open the image file
                image = PIL.Image.open(cur_image_path)
                # Verify the image is valid by attempting to load it
                image.load()
            except (PIL.UnidentifiedImageError, OSError, IOError, Exception) as e:
                # If image file is corrupted or cannot be identified, raise an error
                error_msg = f"Error loading image file '{cur_image_path}': {type(e).__name__}: {str(e)}"
                print(error_msg)
                raise RuntimeError(error_msg) from e
            
            original_image_w, original_image_h = image.size
            
            resize_w = int(resize_h / original_image_h * original_image_w)
            resize_w = int(round(resize_w / patch_size) * patch_size)
            # if torch.distributed.get_rank() == 0:
            #     import ipdb; ipdb.set_trace()

            image = image.resize((resize_w, resize_h), resample=PIL.Image.LANCZOS)
            if square_crop:
                min_size = min(resize_h, resize_w)
                start_h = (resize_h - min_size) // 2
                start_w = (resize_w - min_size) // 2
                image = image.crop((start_w, start_h, start_w + min_size, start_h + min_size))

            image = np.array(image) / 255.0
            
            # Handle RGBA images: alpha blend with white background
            if image.shape[2] == 4:  # RGBA image
                rgb = image[:, :, :3]  # Extract RGB channels
                alpha = image[:, :, 3:4]  # Extract alpha channel [h, w, 1]
                # Alpha blend with white background: RGB * alpha + (1 - alpha) * white
                # white = 1.0 (normalized)
                image = rgb * alpha + (1.0 - alpha) * 1.0
            elif image.shape[2] == 3:  # RGB image
                image = image
            else:
                # Convert grayscale or other formats to RGB
                if len(image.shape) == 2:  # Grayscale
                    image = np.stack([image, image, image], axis=2)
                else:
                    # Take first 3 channels if more than 3
                    image = image[:, :, :3]
            
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            fxfycxcy = np.array(cur_frame["fxfycxcy"])
            resize_ratio_x = resize_w / original_image_w
            resize_ratio_y = resize_h / original_image_h
            fxfycxcy *= (resize_ratio_x, resize_ratio_y, resize_ratio_x, resize_ratio_y)
            if square_crop:
                fxfycxcy[2] -= start_w
                fxfycxcy[3] -= start_h
            fxfycxcy = torch.from_numpy(fxfycxcy).float()
            images.append(image)
            intrinsics.append(fxfycxcy)

        images = torch.stack(images, dim=0)
        intrinsics = torch.stack(intrinsics, dim=0)
        w2cs = np.stack([np.array(frame["w2c"]) for frame in frames_chosen])
        c2ws = np.linalg.inv(w2cs) # (num_frames, 4, 4)
        c2ws = torch.from_numpy(c2ws).float()
        return images, intrinsics, c2ws

    def preprocess_poses(
        self,
        in_c2ws: torch.Tensor,
        scene_scale_factor=1.35,
    ):
        """
        Preprocess the poses to:
        1. translate and rotate the scene to align the average camera direction and position
        2. rescale the whole scene to a fixed scale
        """

        # Translation and Rotation
        # align coordinate system (OpenCV coordinate) to the mean camera
        # center is the average of all camera centers
        # average direction vectors are computed from all camera direction vectors (average down and forward)
        center = in_c2ws[:, :3, 3].mean(0)
        avg_forward = F.normalize(in_c2ws[:, :3, 2].mean(0), dim=-1) # average forward direction (z of opencv camera)
        avg_down = in_c2ws[:, :3, 1].mean(0) # average down direction (y of opencv camera)
        avg_right = F.normalize(torch.cross(avg_down, avg_forward, dim=-1), dim=-1) # (x of opencv camera)
        avg_down = F.normalize(torch.cross(avg_forward, avg_right, dim=-1), dim=-1) # (y of opencv camera)

        avg_pose = torch.eye(4, device=in_c2ws.device) # average c2w matrix
        avg_pose[:3, :3] = torch.stack([avg_right, avg_down, avg_forward], dim=-1)
        avg_pose[:3, 3] = center 
        avg_pose = torch.linalg.inv(avg_pose) # average w2c matrix
        in_c2ws = avg_pose @ in_c2ws 


        # Rescale the whole scene to a fixed scale
        scene_scale = torch.max(torch.abs(in_c2ws[:, :3, 3]))
        scene_scale = scene_scale_factor * scene_scale

        in_c2ws[:, :3, 3] /= scene_scale

        return in_c2ws

    def view_selector(self, frames):
        # print(f"frames: {len(frames)}")
        # print(f"num_views: {self.config.training.num_views}")
        if len(frames) < self.config.training.num_views:
            print(f"Not enough frames to sample")
            return None
        # sample view candidates
        view_selector_config = self.config.training.view_selector
        min_frame_dist = view_selector_config.get("min_frame_dist", 25)
        max_frame_dist = min(len(frames) - 1, view_selector_config.get("max_frame_dist", 100))
        if max_frame_dist <= min_frame_dist:
            print(f"max_frame_dist: {max_frame_dist}")
            print(f"min_frame_dist: {min_frame_dist}")
            print(f"max_frame_dist <= min_frame_dist")
            return None
        frame_dist = random.randint(min_frame_dist, max_frame_dist)
        if len(frames) <= frame_dist:
            print(f"len(frames): {len(frames)}")
            print(f"frame_dist: {frame_dist}")
            print(f"len(frames) <= frame_dist")
            return None
        start_frame = random.randint(0, len(frames) - frame_dist - 1)
        end_frame = start_frame + frame_dist
        # Check if we have enough frames in the range to sample
        # We need num_views-2 samples from range(start_frame+1, end_frame)
        # which has size: end_frame - start_frame - 1 = frame_dist - 1
        num_samples_needed = self.config.training.num_views - 2
        available_range_size = end_frame - start_frame - 1
        if available_range_size < num_samples_needed:
            print(f"available_range_size: {available_range_size}")
            print(f"num_samples_needed: {num_samples_needed}")
            print(f"available_range_size < num_samples_needed")
            return None
        sampled_frames = random.sample(range(start_frame + 1, end_frame), num_samples_needed)
        image_indices = [start_frame, end_frame] + sampled_frames
        return image_indices

    def __getitem__(self, idx, max_retries=10, _retry_count=0):
        """
        Get item from dataset with error handling and retry logic.
        
        Args:
            idx: Index of the item to retrieve
            max_retries: Maximum number of retries before giving up
            _retry_count: Internal counter for retry attempts
        """
        if _retry_count >= max_retries:
            # If we've exhausted retries, raise an error with helpful message
            error_msg = f"Failed to load data after {max_retries} retries. Last attempted index: {idx}"
            print(error_msg)
            raise RuntimeError(error_msg)
        
        try:
            scene_path = self.all_scene_paths[idx].strip()
            
            # Check if file exists
            if not os.path.exists(scene_path):
                raise FileNotFoundError(f"Scene JSON file not found: {scene_path}")
            
            # Load scene JSON
            try:
                with open(scene_path, 'r') as f:
                    data_json = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                raise RuntimeError(f"Error reading/parsing scene JSON '{scene_path}': {type(e).__name__}: {str(e)}") from e
            
            frames = data_json.get("frames", [])
            if not frames:
                raise ValueError(f"No frames found in scene JSON: {scene_path}")
            
            scene_name = data_json.get("scene_name", f"scene_{idx}")
            # print(f"scene_name: {scene_name}")
            # print(f"frames: {len(frames)}")

            if self.inference and scene_name in self.view_idx_list:
                current_view_idx = self.view_idx_list[scene_name]
                image_indices = current_view_idx["context"] + current_view_idx["target"]
            else:
                # sample input and target views
                image_indices = self.view_selector(frames)
                if image_indices is None:
                    # Fallback: try another random index
                    print(f"view_selector returned None for scene {scene_name} at index {idx}, trying another random index")
                    return self.__getitem__(random.randint(0, len(self) - 1), max_retries=max_retries, _retry_count=_retry_count + 1)
            
            # Validate image indices
            if not image_indices or len(image_indices) == 0:
                raise ValueError(f"No valid image indices selected for scene {scene_name}")
            
            # Check if all indices are valid
            for ic in image_indices:
                if ic < 0 or ic >= len(frames):
                    raise IndexError(f"Image index {ic} out of range for scene {scene_name} (has {len(frames)} frames)")
            
            image_paths_chosen = [frames[ic]["image_path"] for ic in image_indices]
            frames_chosen = [frames[ic] for ic in image_indices]
            
            # Check if all image paths exist
            for img_path in image_paths_chosen:
                if not os.path.exists(img_path):
                    raise FileNotFoundError(f"Image file not found: {img_path}")
            
            # Preprocess frames (this may raise PIL.UnidentifiedImageError or other errors)
            input_images, input_intrinsics, input_c2ws = self.preprocess_frames(frames_chosen, image_paths_chosen)

            # centerize and scale the poses (for unbounded scenes)
            scene_scale_factor = self.config.training.get("scene_scale_factor", 1.35)
            input_c2ws = self.preprocess_poses(input_c2ws, scene_scale_factor)
            

            # Load relit images and environment maps from a different env scene with the same object_id
            relit_images = None
            env_ldr = None
            env_hdr = None
            albedos = None
            
            # Extract object_id and env_name from scene_name
            # Examples:
            #   "0007a7c8fcb44074b20fa4e14b8730a6_env_0" -> object_id="0007a7c8fcb44074b20fa4e14b8730a6", env_name="env_0"
            #   "00b100ac52b34afaa95ed4000cd9a4bb_white_env_0" -> object_id="00b100ac52b34afaa95ed4000cd9a4bb", env_name="white_env_0"
            
            # Check for white_env_ prefix first
            # todo: make this part slimmer
            if scene_name.endswith('_white_env_0') or '_white_env_' in scene_name:
                # Find the last occurrence of '_white_env_'
                idx = scene_name.rfind('_white_env_')
                if idx != -1:
                    object_id = scene_name[:idx]
                    current_env_name = scene_name[idx+1:]  # Includes 'white_env_0'
                else:
                    error_msg = f"Failed to parse scene_name '{scene_name}': expected '_white_env_' pattern"
                    print(f"Error: {error_msg}")
                    raise ValueError(error_msg)
            elif scene_name.endswith('_env_0') or '_env_' in scene_name:
                # Find the last occurrence of '_env_'
                idx = scene_name.rfind('_env_')
                if idx != -1:
                    object_id = scene_name[:idx]
                    current_env_name = scene_name[idx+1:]  # Includes 'env_0'
                else:
                    error_msg = f"Failed to parse scene_name '{scene_name}': expected '_env_' pattern"
                    print(f"Error: {error_msg}")
                    raise ValueError(error_msg)
            else:
                # Fallback to simple rsplit
                scene_name_parts = scene_name.rsplit('_', 1)
                if len(scene_name_parts) != 2:
                    error_msg = f"Failed to parse scene_name '{scene_name}' into object_id and env_name"
                    print(f"Error: {error_msg}")
                    raise ValueError(error_msg)
                object_id = scene_name_parts[0]
                current_env_name = scene_name_parts[1]
            
            # Extract base directory from scene_path (e.g., ".../train/metadata/...")
            # This is needed for both relit images and albedo loading
            scene_path_dir = os.path.dirname(scene_path)
            base_dir = os.path.dirname(scene_path_dir)  # Go up from metadata to train/test
            
            # Load relit images and environment maps only if configured
            if self.use_relit_images:
                
                # Find all scenes with the same object_id but different env_name
                metadata_dir = os.path.join(base_dir, 'metadata')
                if not os.path.exists(metadata_dir):
                    error_msg = f"Metadata directory not found: {metadata_dir}"
                    print(f"Error: {error_msg}")
                    raise FileNotFoundError(error_msg)
                
                all_scene_json_files = [f for f in os.listdir(metadata_dir) if f.endswith('.json')]
                candidate_scenes = []
                for json_file in all_scene_json_files:
                    candidate_scene_name = json_file[:-5]  # Remove .json extension
                    # Check if scene has the same object_id and different env_name
                    if candidate_scene_name.startswith(object_id + '_') and candidate_scene_name != scene_name:
                        # Filter: only include scenes ending with _env_x (not white_env_x)
                        # This ensures relit_images come from *_env_x scenes, not white_env_x
                        if '_env_' in candidate_scene_name and not candidate_scene_name.endswith('_white_env_0') and not '_white_env_' in candidate_scene_name:
                            candidate_scenes.append(candidate_scene_name)
                
                # Check if we have candidate scenes
                if not candidate_scenes:
                    error_msg = f"No candidate relit scenes found for object_id '{object_id}' (current scene: {scene_name})"
                    print(f"Error: {error_msg}")
                    raise ValueError(error_msg)
                
                # Randomly select one candidate scene
                relit_scene_name = random.choice(candidate_scenes)
                relit_scene_path = os.path.join(metadata_dir, relit_scene_name + '.json')
                
                # Load relit scene JSON
                if not os.path.exists(relit_scene_path):
                    error_msg = f"Relit scene JSON not found: {relit_scene_path}"
                    print(f"Error: {error_msg}")
                    raise FileNotFoundError(error_msg)
                
                with open(relit_scene_path, 'r') as f:
                    relit_data_json = json.load(f)
                relit_frames = relit_data_json.get("frames", [])
                
                # Check if relit scene has enough frames
                if len(relit_frames) <= max(image_indices):
                    error_msg = f"Relit scene '{relit_scene_name}' has only {len(relit_frames)} frames, but need frame index {max(image_indices)}"
                    print(f"Error: {error_msg}")
                    raise IndexError(error_msg)
                
                # Load relit images with same indices
                relit_image_paths = [relit_frames[ic]["image_path"] for ic in image_indices]
                relit_frames_chosen = [relit_frames[ic] for ic in image_indices]
                
                # Check if all relit image paths exist
                missing_images = [img_path for img_path in relit_image_paths if not os.path.exists(img_path)]
                if missing_images:
                    error_msg = f"Missing relit image files for scene '{relit_scene_name}': {missing_images[:3]}..." if len(missing_images) > 3 else f"Missing relit image files: {missing_images}"
                    print(f"Error: {error_msg}")
                    raise FileNotFoundError(error_msg)
                
                # Load relit images
                relit_images, _, _ = self.preprocess_frames(relit_frames_chosen, relit_image_paths)
                
                # Load environment maps from envmaps folder
                envmaps_dir = os.path.join(base_dir, 'envmaps', relit_scene_name)
                if not os.path.exists(envmaps_dir):
                    error_msg = f"Environment maps directory not found: {envmaps_dir}"
                    print(f"Error: {error_msg}")
                    raise FileNotFoundError(error_msg)
                
                env_ldr_list = []
                env_hdr_list = []
                
                for ic in image_indices:
                    env_ldr_path = os.path.join(envmaps_dir, f"{ic:05d}_ldr.png")
                    env_hdr_path = os.path.join(envmaps_dir, f"{ic:05d}_hdr.png")
                    
                    if not os.path.exists(env_ldr_path):
                        error_msg = f"Environment LDR file not found: {env_ldr_path}"
                        print(f"Error: {error_msg}")
                        raise FileNotFoundError(error_msg)
                    
                    if not os.path.exists(env_hdr_path):
                        error_msg = f"Environment HDR file not found: {env_hdr_path}"
                        print(f"Error: {error_msg}")
                        raise FileNotFoundError(error_msg)
                    
                    try:
                        env_ldr_img = PIL.Image.open(env_ldr_path)
                        env_ldr_img.load()
                        env_ldr_array = np.array(env_ldr_img) / 255.0
                        if len(env_ldr_array.shape) == 2:
                            env_ldr_array = np.stack([env_ldr_array, env_ldr_array, env_ldr_array], axis=2)
                        elif env_ldr_array.shape[2] == 4:
                            rgb = env_ldr_array[:, :, :3]
                            alpha = env_ldr_array[:, :, 3:4]
                            env_ldr_array = rgb * alpha + (1.0 - alpha) * 1.0
                        elif env_ldr_array.shape[2] == 3:
                            pass
                        else:
                            env_ldr_array = env_ldr_array[:, :, :3]
                        env_ldr_tensor = torch.from_numpy(env_ldr_array).permute(2, 0, 1).float()
                        
                        env_hdr_img = PIL.Image.open(env_hdr_path)
                        env_hdr_img.load()
                        env_hdr_array = np.array(env_hdr_img) / 255.0
                        if len(env_hdr_array.shape) == 2:
                            env_hdr_array = np.stack([env_hdr_array, env_hdr_array, env_hdr_array], axis=2)
                        elif env_hdr_array.shape[2] == 4:
                            rgb = env_hdr_array[:, :, :3]
                            alpha = env_hdr_array[:, :, 3:4]
                            env_hdr_array = rgb * alpha + (1.0 - alpha) * 1.0
                        elif env_hdr_array.shape[2] == 3:
                            pass
                        else:
                            env_hdr_array = env_hdr_array[:, :, :3]
                        env_hdr_tensor = torch.from_numpy(env_hdr_array).permute(2, 0, 1).float()
                        
                        env_ldr_list.append(env_ldr_tensor)
                        env_hdr_list.append(env_hdr_tensor)
                    except Exception as e:
                        error_msg = f"Failed to load environment map for frame {ic} in scene '{relit_scene_name}': {type(e).__name__}: {str(e)}"
                        print(f"Error: {error_msg}")
                        traceback.print_exc()
                        raise RuntimeError(error_msg) from e
                
                if len(env_ldr_list) != len(image_indices) or len(env_hdr_list) != len(image_indices):
                    error_msg = f"Mismatch in environment map count: expected {len(image_indices)}, got {len(env_ldr_list)} LDR and {len(env_hdr_list)} HDR"
                    print(f"Error: {error_msg}")
                    raise ValueError(error_msg)
                
                env_ldr = torch.stack(env_ldr_list, dim=0)  # [v, 3, h, w]
                env_hdr = torch.stack(env_hdr_list, dim=0)  # [v, 3, h, w]
            else:
                # Skip loading relit images and envmaps if not configured
                relit_images = None
                env_ldr = None
                env_hdr = None

            # Load albedo images from albedos folder (shared across all scenes with same object_id)
            albedos_dir = os.path.join(base_dir, 'albedos', object_id)
            if os.path.exists(albedos_dir):
                albedo_list = []
                for ic in image_indices:
                    albedo_path = os.path.join(albedos_dir, f"{ic:05d}.png")
                    
                    if not os.path.exists(albedo_path):
                        error_msg = f"Albedo file not found: {albedo_path}"
                        print(f"Error: {error_msg}")
                        raise FileNotFoundError(error_msg)
                    
                    try:
                        albedo_img = PIL.Image.open(albedo_path)
                        albedo_img.load()
                        
                        # Apply same preprocessing as input images (resize, crop, etc.)
                        # Get original dimensions
                        original_albedo_w, original_albedo_h = albedo_img.size
                        resize_h = self.config.model.image_tokenizer.image_size
                        patch_size = self.config.model.image_tokenizer.patch_size
                        square_crop = self.config.training.get("square_crop", False)
                        
                        resize_w = int(resize_h / original_albedo_h * original_albedo_w)
                        resize_w = int(round(resize_w / patch_size) * patch_size)
                        
                        # Resize albedo to match input image size
                        albedo_img = albedo_img.resize((resize_w, resize_h), resample=PIL.Image.LANCZOS)
                        if square_crop:
                            min_size = min(resize_h, resize_w)
                            start_h = (resize_h - min_size) // 2
                            start_w = (resize_w - min_size) // 2
                            albedo_img = albedo_img.crop((start_w, start_h, start_w + min_size, start_h + min_size))
                        
                        # Convert to numpy array
                        albedo_array = np.array(albedo_img) / 255.0
                        
                        # Handle different image formats
                        if len(albedo_array.shape) == 2:
                            # Grayscale: convert to RGB
                            albedo_array = np.stack([albedo_array, albedo_array, albedo_array], axis=2)
                        elif albedo_array.shape[2] == 4:
                            # RGBA: alpha blend with white background
                            rgb = albedo_array[:, :, :3]
                            alpha = albedo_array[:, :, 3:4]
                            albedo_array = rgb * alpha + (1.0 - alpha) * 1.0
                        elif albedo_array.shape[2] == 3:
                            # RGB: use as is
                            pass
                        else:
                            # More than 4 channels: take first 3
                            albedo_array = albedo_array[:, :, :3]
                        
                        albedo_tensor = torch.from_numpy(albedo_array).permute(2, 0, 1).float()
                        albedo_list.append(albedo_tensor)
                    except Exception as e:
                        error_msg = f"Failed to load albedo for frame {ic} in object_id '{object_id}': {type(e).__name__}: {str(e)}"
                        print(f"Error: {error_msg}")
                        traceback.print_exc()
                        raise RuntimeError(error_msg) from e
                
                if len(albedo_list) != len(image_indices):
                    error_msg = f"Mismatch in albedo count: expected {len(image_indices)}, got {len(albedo_list)}"
                    print(f"Error: {error_msg}")
                    raise ValueError(error_msg)
                
                albedos = torch.stack(albedo_list, dim=0)  # [v, 3, h, w]
            else:
                # Albedo directory doesn't exist, set to None
                raise ValueError(f"Albedo directory not found: {albedos_dir}")

            image_indices = torch.tensor(image_indices).long().unsqueeze(-1)  # [v, 1]
            scene_indices = torch.full_like(image_indices, idx)  # [v, 1]
            indices = torch.cat([image_indices, scene_indices], dim=-1)  # [v, 2]

            result_dict = {
                "image": input_images,
                "c2w": input_c2ws,
                "fxfycxcy": input_intrinsics,
                "index": indices,
                "scene_name": scene_name
            }
            
            # Add optional relit images and environment maps
            # Always include these keys (even if None) to ensure consistent batch structure
            # This prevents KeyError during DataLoader collation when some samples have these fields and others don't
            if self.use_relit_images:
                result_dict["relit_images"] = relit_images
                result_dict["env_ldr"] = env_ldr
                result_dict["env_hdr"] = env_hdr
            result_dict["albedos"] = albedos
            
            return result_dict
        except (ValueError, IndexError, KeyError, FileNotFoundError, RuntimeError, 
                PIL.UnidentifiedImageError, OSError, IOError, json.JSONDecodeError,
                torch.cuda.OutOfMemoryError) as e:
            # Fallback for any data loading errors
            # Try another random index instead
            error_type = type(e).__name__
            error_msg = f"Error loading scene at index {idx} (attempt {_retry_count + 1}/{max_retries}): {error_type}: {str(e)}"
            print(error_msg)
            if _retry_count < 2:  # Only print traceback for first few attempts
                traceback.print_exc()
            
            # Try another random index
            new_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(new_idx, max_retries=max_retries, _retry_count=_retry_count + 1)

