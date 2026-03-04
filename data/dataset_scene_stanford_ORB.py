# Stanford ORB dataset loader for LVSM relighting training.
# Scenes with the same prefix before "_scene" are lighting-variation pairs.
# Each scene has a set of envmaps (not all frames); one is randomly chosen per sample.

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
            self.all_scene_paths = [p for p in self.all_scene_paths if p.strip()]
        except Exception as e:
            print(f"Error reading dataset paths from '{self.config.training.dataset_path}'")
            raise e

        self.inference = self.config.inference.get("if_inference", False)
        if self.inference:
            self.view_idx_list = dict()
            view_idx_path = self.config.inference.get("view_idx_file_path", None)
            if view_idx_path is not None and os.path.exists(view_idx_path):
                with open(view_idx_path, 'r') as f:
                    self.view_idx_list = json.load(f)
                    self.view_idx_list_filtered = [
                        k for k, v in self.view_idx_list.items() if v is not None
                    ]
                filtered = []
                for scene in self.all_scene_paths:
                    scene_name = os.path.basename(scene).replace('.json', '')
                    if scene_name in self.view_idx_list_filtered:
                        filtered.append(scene)
                self.all_scene_paths = filtered

        # Build per-object scene index for fast relit-scene lookup
        self._object_scenes = {}  # object_id -> [scene_name, ...]
        for scene_path in self.all_scene_paths:
            sn = os.path.basename(scene_path).replace('.json', '')
            oid = self._extract_object_id(sn)
            self._object_scenes.setdefault(oid, []).append(sn)

        print(f"[StanfordORB] Loaded {len(self.all_scene_paths)} scenes, "
              f"{len(self._object_scenes)} objects")

    def __len__(self):
        return len(self.all_scene_paths)

    @staticmethod
    def _extract_object_id(scene_name: str) -> str:
        """'teapot_scene006' -> 'teapot', 'baking_scene001' -> 'baking'."""
        idx = scene_name.find("_scene")
        if idx != -1:
            return scene_name[:idx]
        return scene_name.rsplit("_", 1)[0]

    # ------------------------------------------------------------------
    # Frame preprocessing (image loading, intrinsics, c2w)
    # ------------------------------------------------------------------

    def preprocess_frames(self, frames_chosen, image_paths_chosen):
        resize_h = self.config.model.image_tokenizer.image_size
        patch_size = self.config.model.image_tokenizer.patch_size
        square_crop = self.config.training.get("square_crop", False)

        images = []
        intrinsics = []
        for cur_frame, cur_image_path in zip(frames_chosen, image_paths_chosen):
            try:
                image = PIL.Image.open(cur_image_path)
                image.load()
            except Exception as e:
                raise RuntimeError(
                    f"Error loading image '{cur_image_path}': {e}"
                ) from e

            original_w, original_h = image.size
            resize_w = int(resize_h / original_h * original_w)
            resize_w = int(round(resize_w / patch_size) * patch_size)

            image = image.resize((resize_w, resize_h), resample=PIL.Image.LANCZOS)
            if square_crop:
                min_size = min(resize_h, resize_w)
                start_h = (resize_h - min_size) // 2
                start_w = (resize_w - min_size) // 2
                image = image.crop(
                    (start_w, start_h, start_w + min_size, start_h + min_size)
                )

            image = np.array(image) / 255.0
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=2)
            elif image.shape[2] == 4:
                rgb, alpha = image[:, :, :3], image[:, :, 3:4]
                image = rgb * alpha + (1.0 - alpha)
            else:
                image = image[:, :, :3]

            image = torch.from_numpy(image).permute(2, 0, 1).float()
            fxfycxcy = np.array(cur_frame["fxfycxcy"])
            rx = resize_w / original_w
            ry = resize_h / original_h
            fxfycxcy *= (rx, ry, rx, ry)
            if square_crop:
                fxfycxcy[2] -= start_w
                fxfycxcy[3] -= start_h
            fxfycxcy = torch.from_numpy(fxfycxcy).float()
            images.append(image)
            intrinsics.append(fxfycxcy)

        images = torch.stack(images, dim=0)
        intrinsics = torch.stack(intrinsics, dim=0)
        w2cs = np.stack([np.array(f["w2c"]) for f in frames_chosen])
        c2ws = np.linalg.inv(w2cs)
        c2ws = torch.from_numpy(c2ws).float()
        return images, intrinsics, c2ws

    # ------------------------------------------------------------------
    # Pose normalisation – returns the transform so we can apply it to envmap poses
    # ------------------------------------------------------------------

    def preprocess_poses(self, in_c2ws, scene_scale_factor=1.35):
        """Normalise poses; also returns avg_pose and scene_scale for envmap c2w."""
        center = in_c2ws[:, :3, 3].mean(0)
        avg_forward = F.normalize(in_c2ws[:, :3, 2].mean(0), dim=-1)
        avg_down = in_c2ws[:, :3, 1].mean(0)
        avg_right = F.normalize(torch.cross(avg_down, avg_forward, dim=-1), dim=-1)
        avg_down = F.normalize(torch.cross(avg_forward, avg_right, dim=-1), dim=-1)

        avg_c2w = torch.eye(4, device=in_c2ws.device)
        avg_c2w[:3, :3] = torch.stack([avg_right, avg_down, avg_forward], dim=-1)
        avg_c2w[:3, 3] = center
        avg_pose = torch.linalg.inv(avg_c2w)  # w2c of average camera
        in_c2ws = avg_pose @ in_c2ws

        scene_scale = torch.max(torch.abs(in_c2ws[:, :3, 3]))
        scene_scale = scene_scale_factor * scene_scale
        in_c2ws[:, :3, 3] /= scene_scale

        return in_c2ws, avg_pose, scene_scale

    # ------------------------------------------------------------------
    # View selection
    # ------------------------------------------------------------------

    def view_selector(self, frames):
        num_views = self.config.training.num_views
        if len(frames) < num_views:
            return None
        vs = self.config.training.view_selector
        min_fd = vs.get("min_frame_dist", 25)
        max_fd = min(len(frames) - 1, vs.get("max_frame_dist", 100))
        if max_fd <= min_fd:
            return None
        fd = random.randint(min_fd, max_fd)
        if len(frames) <= fd:
            return None
        start = random.randint(0, len(frames) - fd - 1)
        end = start + fd
        need = num_views - 2
        avail = end - start - 1
        if avail < need:
            return None
        mid = random.sample(range(start + 1, end), need)
        return [start, end] + mid

    # ------------------------------------------------------------------
    # Envmap helpers
    # ------------------------------------------------------------------

    def _list_envmap_indices(self, envmaps_dir):
        """Return sorted list of available envmap indices in the directory."""
        if not os.path.exists(envmaps_dir):
            return []
        files = [f for f in os.listdir(envmaps_dir) if f.endswith('_ldr.png')]
        indices = sorted(int(f.split('_')[0]) for f in files)
        return indices

    def _load_envmap_pair(self, envmaps_dir, idx):
        """Load LDR and HDR envmap PNGs at *idx* and return as [3, H, W] tensors."""
        ldr_path = os.path.join(envmaps_dir, f"{idx:05d}_ldr.png")
        hdr_path = os.path.join(envmaps_dir, f"{idx:05d}_hdr.png")
        tensors = []
        for path in (ldr_path, hdr_path):
            img = PIL.Image.open(path)
            img.load()
            arr = np.array(img) / 255.0
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=2)
            elif arr.shape[2] == 4:
                rgb, a = arr[:, :, :3], arr[:, :, 3:4]
                arr = rgb * a + (1.0 - a)
            else:
                arr = arr[:, :, :3]
            tensors.append(torch.from_numpy(arr).permute(2, 0, 1).float())
        return tensors[0], tensors[1]  # env_ldr, env_hdr

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, idx, max_retries=10, _retry_count=0):
        if _retry_count >= max_retries:
            raise RuntimeError(
                f"Failed to load data after {max_retries} retries (last idx={idx})"
            )

        try:
            scene_path = self.all_scene_paths[idx].strip()
            if not os.path.exists(scene_path):
                raise FileNotFoundError(f"Scene JSON not found: {scene_path}")

            with open(scene_path, 'r') as f:
                data_json = json.load(f)

            frames = data_json.get("frames", [])
            if not frames:
                raise ValueError(f"No frames in {scene_path}")

            scene_name = data_json.get("scene_name", f"scene_{idx}")

            # ---- view selection ----
            if self.inference and scene_name in self.view_idx_list:
                vi = self.view_idx_list[scene_name]
                image_indices = vi["context"] + vi["target"]
            else:
                image_indices = self.view_selector(frames)
                if image_indices is None:
                    print(f"[StanfordORB] view_selector returned None for "
                          f"{scene_name}, retrying")
                    return self.__getitem__(
                        random.randint(0, len(self) - 1),
                        max_retries=max_retries,
                        _retry_count=_retry_count + 1,
                    )

            for ic in image_indices:
                if ic < 0 or ic >= len(frames):
                    raise IndexError(
                        f"Index {ic} out of range for {scene_name} "
                        f"({len(frames)} frames)"
                    )

            frames_chosen = [frames[ic] for ic in image_indices]
            image_paths_chosen = [f["image_path"] for f in frames_chosen]
            for p in image_paths_chosen:
                if not os.path.exists(p):
                    raise FileNotFoundError(f"Image not found: {p}")

            input_images, input_intrinsics, input_c2ws = self.preprocess_frames(
                frames_chosen, image_paths_chosen
            )

            scene_scale_factor = self.config.training.get("scene_scale_factor", 1.35)
            input_c2ws, avg_pose, scene_scale = self.preprocess_poses(
                input_c2ws, scene_scale_factor
            )

            # ---- relit scene selection (same object, different lighting) ----
            object_id = self._extract_object_id(scene_name)
            base_dir = os.path.dirname(os.path.dirname(scene_path))
            metadata_dir = os.path.join(base_dir, "metadata")

            candidates = [
                s for s in self._object_scenes.get(object_id, [])
                if s != scene_name
            ]
            if not candidates:
                candidates = [scene_name]

            relit_scene_name = random.choice(candidates)
            relit_scene_path = os.path.join(
                metadata_dir, relit_scene_name + ".json"
            )
            if not os.path.exists(relit_scene_path):
                raise FileNotFoundError(
                    f"Relit scene JSON not found: {relit_scene_path}"
                )

            with open(relit_scene_path, 'r') as f:
                relit_data_json = json.load(f)
            relit_frames = relit_data_json.get("frames", [])

            if len(relit_frames) <= max(image_indices):
                raise IndexError(
                    f"Relit scene '{relit_scene_name}' has {len(relit_frames)} "
                    f"frames, need index {max(image_indices)}"
                )

            relit_image_paths = [relit_frames[ic]["image_path"] for ic in image_indices]
            relit_frames_chosen = [relit_frames[ic] for ic in image_indices]
            for p in relit_image_paths:
                if not os.path.exists(p):
                    raise FileNotFoundError(f"Relit image not found: {p}")

            relit_images, relit_intrinsics, relit_c2ws = self.preprocess_frames(
                relit_frames_chosen, relit_image_paths
            )
            # normalise relit poses with the SAME transform as input poses
            relit_c2ws = avg_pose @ relit_c2ws
            relit_c2ws[:, :3, 3] /= scene_scale

            # ---- envmap loading (randomly choose one available envmap) ----
            envmaps_dir = os.path.join(base_dir, "envmaps", relit_scene_name)
            env_indices = self._list_envmap_indices(envmaps_dir)
            # keep only indices that are valid frame indices in the relit scene
            env_indices = [i for i in env_indices if i < len(relit_frames)]

            num_views = len(image_indices)

            if env_indices:
                chosen_env_idx = random.choice(env_indices)
                env_ldr, env_hdr = self._load_envmap_pair(envmaps_dir, chosen_env_idx)

                # envmap camera pose from the relit scene metadata
                envmap_frame = relit_frames[chosen_env_idx]
                w2c = np.array(envmap_frame["w2c"])
                envmap_c2w_raw = torch.from_numpy(np.linalg.inv(w2c)).float()

                # apply the SAME normalisation as the input scene
                envmap_c2w_single = (avg_pose @ envmap_c2w_raw)
                envmap_c2w_single[:3, 3] /= scene_scale

                # broadcast to all views so fetch_views can slice normally
                env_ldr = env_ldr.unsqueeze(0).expand(num_views, -1, -1, -1).clone()
                env_hdr = env_hdr.unsqueeze(0).expand(num_views, -1, -1, -1).clone()
                envmap_c2w = envmap_c2w_single.unsqueeze(0).expand(
                    num_views, -1, -1
                ).clone()
            else:
                env_ldr = torch.zeros(num_views, 3, 256, 512, dtype=torch.float32)
                env_hdr = torch.zeros(num_views, 3, 256, 512, dtype=torch.float32)
                envmap_c2w = torch.eye(4, dtype=torch.float32).unsqueeze(0).expand(
                    num_views, -1, -1
                ).clone()

            # ---- pack result ----
            idx_tensor = torch.tensor(image_indices).long().unsqueeze(-1)
            scene_idx_tensor = torch.full_like(idx_tensor, idx)
            indices = torch.cat([idx_tensor, scene_idx_tensor], dim=-1)

            return {
                "image": input_images,
                "c2w": input_c2ws,
                "fxfycxcy": input_intrinsics,
                "index": indices,
                "scene_name": scene_name,
                "relit_images": relit_images,
                "relit_scene_name": relit_scene_name,
                "relit_c2w": relit_c2ws,
                "relit_fxfycxcy": relit_intrinsics,
                "env_ldr": env_ldr,
                "env_hdr": env_hdr,
                "envmap_c2w": envmap_c2w,
            }

        except Exception as e:
            etype = type(e).__name__
            print(
                f"[StanfordORB] Error idx={idx} "
                f"(attempt {_retry_count+1}/{max_retries}): {etype}: {e}"
            )
            if _retry_count < 2:
                traceback.print_exc()
            new_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(
                new_idx, max_retries=max_retries, _retry_count=_retry_count + 1
            )
