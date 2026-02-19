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
        self.relight_signals = self.config.training.get("relight_signals", ["envmap"])
        if isinstance(self.relight_signals, str):
            self.relight_signals = [self.relight_signals]
        self.relight_signals = list(self.relight_signals)
        self.use_relight_envmap = "envmap" in self.relight_signals
        self.use_relight_point_light = "point_light" in self.relight_signals
        self.point_light_num_rays = int(self.config.training.get("point_light_num_rays", 1024))

        # Check if we should load albedo images
        self.use_albedos = self.config.training.get("use_albedos", False)
        
        # Check if we should use white_env_0 scene images as albedo instead of loading from albedos folder
        self.white_env_as_albedo = self.config.training.get("white_env_as_albedo", False)


    def __len__(self):
        return len(self.all_scene_paths)

    def _scene_lighting_type(self, scene_name: str) -> str:
        if "_white_env_" in scene_name or "_env_" in scene_name:
            return "envmap"
        if "_white_pl_" in scene_name or "_rgb_pl_" in scene_name:
            return "point_light"
        return "other"

    def _extract_object_id(self, scene_name: str) -> str:
        split_tags = ["_white_env_", "_env_", "_white_pl_", "_rgb_pl_"]
        for tag in split_tags:
            idx = scene_name.rfind(tag)
            if idx != -1:
                return scene_name[:idx]
        # Fallback for unknown naming
        return scene_name.rsplit("_", 1)[0]

    def _load_point_light_rays(self, base_dir: str, scene_name: str, num_views: int) -> torch.Tensor:
        rays_path = os.path.join(base_dir, "point_light_rays", f"{scene_name}.npy")
        if not os.path.exists(rays_path):
            raise FileNotFoundError(f"Point light rays file not found: {rays_path}")
        rays = np.load(rays_path)  # [N, 10]
        if rays.ndim != 2 or rays.shape[1] != 10:
            raise ValueError(f"Point light rays should have shape [N,10], got {rays.shape} at {rays_path}")

        sampled = []
        n_total = rays.shape[0]
        for _ in range(num_views):
            replace = n_total < self.point_light_num_rays
            sampled_idx = np.random.choice(n_total, size=self.point_light_num_rays, replace=replace)
            sampled.append(torch.from_numpy(rays[sampled_idx]).float())  # [num_rays, 10]
        return torch.stack(sampled, dim=0)  # [v, num_rays, 10]


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
            
            object_id = self._extract_object_id(scene_name)
            
            # Extract base directory from scene_path (e.g., ".../train/metadata/...")
            # This is needed for both relit images and albedo loading
            scene_path_dir = os.path.dirname(scene_path)
            base_dir = os.path.dirname(scene_path_dir)  # Go up from metadata to train/test
            
            point_light_rays = None

            # Load relit images and lighting conditions only if configured
            if self.use_relit_images:
                
                # Find all scenes with the same object_id but different env_name
                metadata_dir = os.path.join(base_dir, 'metadata')
                if not os.path.exists(metadata_dir):
                    error_msg = f"Metadata directory not found: {metadata_dir}"
                    print(f"Error: {error_msg}")
                    raise FileNotFoundError(error_msg)
                
                all_scene_json_files = [f for f in os.listdir(metadata_dir) if f.endswith('.json')]
                candidate_scenes_by_type = {"envmap": [], "point_light": []}
                for json_file in all_scene_json_files:
                    candidate_scene_name = json_file[:-5]  # Remove .json extension
                    if not (candidate_scene_name.startswith(object_id + '_') and candidate_scene_name != scene_name):
                        continue
                    scene_type = self._scene_lighting_type(candidate_scene_name)
                    if scene_type in candidate_scenes_by_type:
                        candidate_scenes_by_type[scene_type].append(candidate_scene_name)

                enabled_signal_types = []
                if self.use_relight_envmap:
                    enabled_signal_types.append("envmap")
                if self.use_relight_point_light:
                    enabled_signal_types.append("point_light")
                if len(enabled_signal_types) == 0:
                    enabled_signal_types = ["envmap"]

                candidate_scenes = []
                for sig in enabled_signal_types:
                    candidate_scenes.extend(candidate_scenes_by_type[sig])

                if not candidate_scenes:
                    error_msg = (
                        f"No candidate relit scenes found for object_id '{object_id}' "
                        f"with relight_signals={enabled_signal_types} (current scene: {scene_name})"
                    )
                    print(f"Error: {error_msg}")
                    raise ValueError(error_msg)

                # Sample one relit scene for relit image supervision
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
                
                # Load envmaps if enabled by relight_signals
                if self.use_relight_envmap:
                    if self._scene_lighting_type(relit_scene_name) == "envmap":
                        env_scene_name = relit_scene_name
                    else:
                        if len(candidate_scenes_by_type["envmap"]) == 0:
                            error_msg = (
                                f"relight_signals includes envmap but no envmap-lit scenes found for object '{object_id}'"
                            )
                            print(f"Error: {error_msg}")
                            raise ValueError(error_msg)
                        env_scene_name = random.choice(candidate_scenes_by_type["envmap"])

                    envmaps_dir = os.path.join(base_dir, 'envmaps', env_scene_name)
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
                            elif env_ldr_array.shape[2] != 3:
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
                            elif env_hdr_array.shape[2] != 3:
                                env_hdr_array = env_hdr_array[:, :, :3]
                            env_hdr_tensor = torch.from_numpy(env_hdr_array).permute(2, 0, 1).float()

                            env_ldr_list.append(env_ldr_tensor)
                            env_hdr_list.append(env_hdr_tensor)
                        except Exception as e:
                            error_msg = f"Failed to load environment map for frame {ic} in scene '{env_scene_name}': {type(e).__name__}: {str(e)}"
                            print(f"Error: {error_msg}")
                            traceback.print_exc()
                            raise RuntimeError(error_msg) from e

                    if len(env_ldr_list) != len(image_indices) or len(env_hdr_list) != len(image_indices):
                        error_msg = f"Mismatch in environment map count: expected {len(image_indices)}, got {len(env_ldr_list)} LDR and {len(env_hdr_list)} HDR"
                        print(f"Error: {error_msg}")
                        raise ValueError(error_msg)
                    env_ldr = torch.stack(env_ldr_list, dim=0)  # [v, 3, h, w]
                    env_hdr = torch.stack(env_hdr_list, dim=0)  # [v, 3, h, w]

                # Load point light rays if enabled by relight_signals
                if self.use_relight_point_light:
                    if self._scene_lighting_type(relit_scene_name) == "point_light":
                        point_light_scene_name = relit_scene_name
                    else:
                        if len(candidate_scenes_by_type["point_light"]) == 0:
                            error_msg = (
                                f"relight_signals includes point_light but no point-light scenes found for object '{object_id}'"
                            )
                            print(f"Error: {error_msg}")
                            raise ValueError(error_msg)
                        point_light_scene_name = random.choice(candidate_scenes_by_type["point_light"])
                    point_light_rays = self._load_point_light_rays(
                        base_dir=base_dir,
                        scene_name=point_light_scene_name,
                        num_views=len(image_indices),
                    )  # [v, num_rays, 10]
            else:
                # Skip loading relit signals if not configured
                relit_images = None
                env_ldr = None
                env_hdr = None
                point_light_rays = None

            # Load albedo images
            # Only load if configured to do so
            if self.use_albedos:
                if self.white_env_as_albedo:
                    # Load albedo from white_env_0 scene instead of albedos folder
                    white_env_scene_name = object_id + '_white_env_0'
                    metadata_dir = os.path.join(base_dir, 'metadata')
                    if not os.path.exists(metadata_dir):
                        error_msg = f"Metadata directory not found: {metadata_dir}"
                        print(f"Error: {error_msg}")
                        raise FileNotFoundError(error_msg)
                    
                    white_env_scene_path = os.path.join(metadata_dir, white_env_scene_name + '.json')
                    if not os.path.exists(white_env_scene_path):
                        error_msg = f"White env scene JSON not found: {white_env_scene_path}"
                        print(f"Error: {error_msg}")
                        raise FileNotFoundError(error_msg)
                    
                    # Load white env scene JSON
                    with open(white_env_scene_path, 'r') as f:
                        white_env_data_json = json.load(f)
                    white_env_frames = white_env_data_json.get("frames", [])
                    
                    # Check if white env scene has enough frames
                    if len(white_env_frames) <= max(image_indices):
                        error_msg = f"White env scene '{white_env_scene_name}' has only {len(white_env_frames)} frames, but need frame index {max(image_indices)}"
                        print(f"Error: {error_msg}")
                        raise IndexError(error_msg)
                    
                    # Load white env images with same indices
                    white_env_image_paths = [white_env_frames[ic]["image_path"] for ic in image_indices]
                    white_env_frames_chosen = [white_env_frames[ic] for ic in image_indices]
                    
                    # Check if all white env image paths exist
                    missing_images = [img_path for img_path in white_env_image_paths if not os.path.exists(img_path)]
                    if missing_images:
                        error_msg = f"Missing white env image files for scene '{white_env_scene_name}': {missing_images[:3]}..." if len(missing_images) > 3 else f"Missing white env image files: {missing_images}"
                        print(f"Error: {error_msg}")
                        raise FileNotFoundError(error_msg)
                    
                    # Load and preprocess white env images as albedo (using same preprocessing as input images)
                    albedos, _, _ = self.preprocess_frames(white_env_frames_chosen, white_env_image_paths)
                else:
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
                        error_msg = f"Albedo directory not found: {albedos_dir}"
                        print(f"Error: {error_msg}")
                        raise FileNotFoundError(error_msg)
            else:
                # Skip loading albedo if not configured
                albedos = None

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
                if self.use_relight_envmap:
                    result_dict["env_ldr"] = env_ldr
                    result_dict["env_hdr"] = env_hdr
                if self.use_relight_point_light:
                    result_dict["point_light_rays"] = point_light_rays
            if self.use_albedos:
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

