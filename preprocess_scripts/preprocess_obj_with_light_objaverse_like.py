#!/usr/bin/env python3
"""
Convert data_samples/obj_with_light to an Objaverse-like layout:

  <output_root>/train/images/<scene>/<frame:05d>.png
  <output_root>/train/metadata/<scene>.json
  <output_root>/train/full_list.txt
  <output_root>/test/images/<scene>/<frame:05d>.png
  <output_root>/test/metadata/<scene>.json
  <output_root>/test/envmaps/<scene>/<frame:05d>_ldr.png, _hdr.png
  <output_root>/test/full_list.txt

Per scene convention:
  - source <scene>/test/inputs     -> output split=train
  - source <scene>/test (gt_*)     -> output split=test

Envmap handling (test split only):
  - reads gt_env_<id>.hdr/.exr and gt_world_to_env_<id>.txt
  - rotates envmap to the corresponding gt_camera_<id>.txt frame
  - writes Objaverse-style LDR/HDR PNG pairs
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import pyexr

    HAS_PYEXR = True
except ImportError:
    HAS_PYEXR = False

try:
    import imageio

    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False


def read_camera_txt(path: Path):
    rows = [list(map(float, line.split())) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) < 7:
        raise ValueError(f"Invalid camera file (expected >=7 rows): {path}")
    K = np.array(rows[0:3], dtype=np.float64)
    R = np.array(rows[3:6], dtype=np.float64)
    t = np.array(rows[6], dtype=np.float64)
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = R
    w2c[:3, 3] = t
    return K, w2c


def build_frame_entry(image_abs_path: Path, K: np.ndarray, w2c: np.ndarray):
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    return {
        "image_path": str(image_abs_path.resolve()),
        "fxfycxcy": [fx, fy, cx, cy],
        "w2c": w2c.tolist(),
    }


def process_image_with_mask(
    image_path: Path, mask_path: Path | None, target_size: int, apply_mask: bool
) -> tuple[np.ndarray, tuple[int, int, int], float]:
    with Image.open(image_path) as image:
        rgb = np.array(image.convert("RGB"), dtype=np.float32)
    if apply_mask:
        if mask_path is None:
            raise ValueError(f"Mask is required when apply_mask=True: {image_path}")
        with Image.open(mask_path) as mask_image:
            alpha = np.array(mask_image.convert("L"), dtype=np.float32) / 255.0

        if rgb.shape[:2] != alpha.shape[:2]:
            raise ValueError(
                f"Mask shape mismatch for {image_path.name}: image={rgb.shape[:2]}, mask={alpha.shape[:2]}"
            )

        # Alpha blend foreground with white background (supports soft mask edges).
        alpha = alpha[..., None]
        rgb = rgb * alpha + 255.0 * (1.0 - alpha)
    rgb = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)

    h, w = rgb.shape[:2]
    crop_size = min(h, w)
    x0 = (w - crop_size) // 2
    y0 = (h - crop_size) // 2
    cropped = rgb[y0 : y0 + crop_size, x0 : x0 + crop_size]

    if crop_size != target_size:
        resized = np.array(
            Image.fromarray(cropped).resize((target_size, target_size), Image.Resampling.BICUBIC),
            dtype=np.uint8,
        )
    else:
        resized = cropped

    scale = target_size / float(crop_size)
    return resized, (x0, y0, crop_size), scale


def update_intrinsics_for_crop_resize(
    K: np.ndarray, crop_xy_size: tuple[int, int, int], scale: float
) -> np.ndarray:
    x0, y0, _ = crop_xy_size
    K_new = K.copy()
    K_new[0, 0] = K[0, 0] * scale
    K_new[1, 1] = K[1, 1] * scale
    K_new[0, 2] = (K[0, 2] - x0) * scale
    K_new[1, 2] = (K[1, 2] - y0) * scale
    return K_new


def extract_frame_id(name: str, pattern: str):
    m = re.match(pattern, name)
    if not m:
        return None
    return int(m.group(1))


def read_env_hdr(path: Path) -> np.ndarray | None:
    """Read HDR envmap from EXR/HDR file and return float RGB."""
    suffix = path.suffix.lower()
    if suffix == ".exr" and HAS_PYEXR:
        try:
            rgb = pyexr.read(str(path))
            if rgb is not None and rgb.ndim >= 3 and rgb.shape[-1] >= 3:
                return rgb[:, :, :3].astype(np.float64)
            return None
        except Exception:
            return None

    if suffix in (".hdr", ".exr") and HAS_IMAGEIO:
        try:
            rgb = imageio.imread(str(path))
            if rgb is None:
                return None
            if rgb.ndim == 2:
                rgb = np.stack([rgb] * 3, axis=-1)
            if rgb.ndim >= 3 and rgb.shape[-1] >= 3:
                return rgb[:, :, :3].astype(np.float64)
            return None
        except Exception:
            return None
    return None


def split_envmap_ldr_hdr(env_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert linear HDR envmap into Objaverse-style LDR/HDR uint8 PNG arrays."""
    env_ldr = np.clip(env_raw, 0, 1) ** (1 / 2.2)
    env_hdr = np.log1p(10 * np.clip(env_raw, 0, None))
    max_val = float(np.max(env_hdr))
    if max_val > 1e-8:
        env_hdr = np.clip(env_hdr / max_val, 0, 1)
    else:
        env_hdr = np.clip(env_hdr, 0, 1)
    return (np.uint8(env_ldr * 255), np.uint8(env_hdr * 255))


def read_world_to_env_rotation(path: Path) -> np.ndarray:
    """Read world_to_env transform and return its 3x3 rotation block."""
    rows = [list(map(float, line.split())) for line in path.read_text().splitlines() if line.strip()]
    mat = np.array(rows, dtype=np.float64)
    if mat.shape == (4, 4):
        return mat[:3, :3]
    if mat.shape == (3, 4):
        return mat[:, :3]
    if mat.shape == (3, 3):
        return mat
    raise ValueError(f"Invalid world_to_env format: {path}, got shape={mat.shape}")


def _dirs_from_equirect_uv(height: int, width: int) -> np.ndarray:
    """
    Build direction grid for dataset's equirect convention:
      +Z -> top, -Z -> bottom, +X -> center, +Y -> u=0.25.
    """
    u = (np.arange(width, dtype=np.float64) + 0.5) / float(width)
    v = (np.arange(height, dtype=np.float64) + 0.5) / float(height)
    uu, vv = np.meshgrid(u, v, indexing="xy")

    phi = (0.5 - uu) * (2.0 * np.pi)
    theta = vv * np.pi
    sin_theta = np.sin(theta)

    x = sin_theta * np.cos(phi)
    y = sin_theta * np.sin(phi)
    z = np.cos(theta)
    return np.stack([x, y, z], axis=-1)  # [H, W, 3]


def _equirect_uv_from_dirs(dirs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert xyz directions to normalized equirect UV in dataset convention."""
    x = dirs[..., 0]
    y = dirs[..., 1]
    z = np.clip(dirs[..., 2], -1.0, 1.0)
    phi = np.arctan2(y, x)
    theta = np.arccos(z)
    u = np.mod(0.5 - phi / (2.0 * np.pi), 1.0)
    v = np.clip(theta / np.pi, 0.0, 1.0)
    return u, v


def _sample_equirect_bilinear(env: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinear sample equirect image; wraps horizontally and clamps vertically."""
    h, w = env.shape[:2]
    x = u * w - 0.5
    y = v * h - 0.5

    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    wx = (x - x0)[..., None]
    wy = (y - y0)[..., None]

    x0 = np.mod(x0, w)
    x1 = np.mod(x1, w)
    y0 = np.clip(y0, 0, h - 1)
    y1 = np.clip(y1, 0, h - 1)

    Ia = env[y0, x0]
    Ib = env[y0, x1]
    Ic = env[y1, x0]
    Id = env[y1, x1]

    top = Ia * (1.0 - wx) + Ib * wx
    bottom = Ic * (1.0 - wx) + Id * wx
    return top * (1.0 - wy) + bottom * wy


def rotate_envmap_to_camera(env_raw: np.ndarray, R_w2env: np.ndarray, R_w2c: np.ndarray) -> np.ndarray:
    """
    Rotate per-frame envmap into the corresponding camera coordinate frame.
    Uses world_to_env and world_to_camera rotations.
    """
    h, w = env_raw.shape[:2]
    dirs_cam = _dirs_from_equirect_uv(h, w)

    # d_env = R_w2env * d_world, d_world = R_c2w * d_cam
    # => d_env = (R_w2env * R_c2w) * d_cam
    R_c2w = R_w2c.T
    R_cam2env = R_w2env @ R_c2w

    dirs_env = dirs_cam @ R_cam2env.T

    # Align camera-frame convention with envmap convention used by obj_with_light.
    # Keep upside-down behavior here (z flip). Left-right flip is applied later on
    # fully processed outputs (LDR/HDR), per user request.
    axis_fix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    dirs_env = dirs_env @ axis_fix.T

    u_src, v_src = _equirect_uv_from_dirs(dirs_env)
    return _sample_equirect_bilinear(env_raw, u_src, v_src)


def process_split(
    scene_name: str,
    src_img_dir: Path,
    img_prefix: str,
    cam_dir: Path,
    cam_prefix: str,
    mask_dir: Path,
    mask_prefix: str,
    output_root: Path,
    split: str,
    target_size: int,
    apply_mask: bool,
    enable_test_envmaps: bool,
):
    out_images_dir = output_root / split / "images" / scene_name
    out_meta_dir = output_root / split / "metadata"
    process_test_envmaps = split == "test" and enable_test_envmaps
    out_envmaps_dir = output_root / split / "envmaps" / scene_name
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_meta_dir.mkdir(parents=True, exist_ok=True)
    if process_test_envmaps:
        out_envmaps_dir.mkdir(parents=True, exist_ok=True)

    img_pattern = rf"{re.escape(img_prefix)}(\d+)\.png$"
    cam_pattern = rf"{re.escape(cam_prefix)}(\d+)\.txt$"

    img_map = {}
    for p in src_img_dir.glob(f"{img_prefix}*.png"):
        fid = extract_frame_id(p.name, img_pattern)
        if fid is not None:
            img_map[fid] = p

    cam_map = {}
    for p in cam_dir.glob(f"{cam_prefix}*.txt"):
        fid = extract_frame_id(p.name, cam_pattern)
        if fid is not None:
            cam_map[fid] = p

    mask_map = {}
    if apply_mask:
        mask_pattern = rf"{re.escape(mask_prefix)}(\d+)\.png$"
        for p in mask_dir.glob(f"{mask_prefix}*.png"):
            fid = extract_frame_id(p.name, mask_pattern)
            if fid is not None:
                mask_map[fid] = p

    env_map = {}
    env_w2e_map = {}
    if process_test_envmaps:
        for p in src_img_dir.glob("gt_env_*.hdr"):
            fid = extract_frame_id(p.name, r"gt_env_(\d+)\.hdr$")
            if fid is not None:
                env_map[fid] = p
        for p in src_img_dir.glob("gt_env_*.exr"):
            fid = extract_frame_id(p.name, r"gt_env_(\d+)\.exr$")
            if fid is not None:
                env_map[fid] = p
        for p in src_img_dir.glob("gt_world_to_env_*.txt"):
            fid = extract_frame_id(p.name, r"gt_world_to_env_(\d+)\.txt$")
            if fid is not None:
                env_w2e_map[fid] = p

    common_ids = sorted(set(img_map.keys()) & set(cam_map.keys()))
    if apply_mask:
        common_ids = sorted(set(common_ids) & set(mask_map.keys()))
    if not common_ids:
        return None

    frames = []
    n_env_written = 0
    n_env_skipped = 0
    for out_idx, fid in enumerate(common_ids):
        src_img = img_map[fid]
        src_cam = cam_map[fid]
        src_mask = mask_map.get(fid) if apply_mask else None
        out_img = out_images_dir / f"{out_idx:05d}.png"

        K, w2c = read_camera_txt(src_cam)
        rgb_out, crop_xy_size, scale = process_image_with_mask(
            src_img, src_mask, target_size, apply_mask
        )
        K_out = update_intrinsics_for_crop_resize(K, crop_xy_size, scale)
        Image.fromarray(rgb_out).save(out_img)
        frames.append(build_frame_entry(out_img, K_out, w2c))

        if process_test_envmaps:
            env_path = env_map.get(fid)
            w2e_path = env_w2e_map.get(fid)
            if env_path is None or w2e_path is None:
                n_env_skipped += 1
            else:
                env_raw = read_env_hdr(env_path)
                if env_raw is None:
                    n_env_skipped += 1
                else:
                    try:
                        R_w2env = read_world_to_env_rotation(w2e_path)
                        R_w2c = w2c[:3, :3]
                        env_rot = rotate_envmap_to_camera(env_raw, R_w2env, R_w2c)
                        env_ldr, env_hdr = split_envmap_ldr_hdr(env_rot)
                        # Apply left-right flip on fully processed envmaps (not on source envmap).
                        env_ldr = np.flip(env_ldr, axis=1).copy()
                        env_hdr = np.flip(env_hdr, axis=1).copy()
                        Image.fromarray(env_ldr).save(out_envmaps_dir / f"{out_idx:05d}_ldr.png")
                        Image.fromarray(env_hdr).save(out_envmaps_dir / f"{out_idx:05d}_hdr.png")
                        n_env_written += 1
                    except Exception:
                        n_env_skipped += 1

    scene_meta = {"scene_name": scene_name, "frames": frames}
    out_json = out_meta_dir / f"{scene_name}.json"
    out_json.write_text(json.dumps(scene_meta, indent=2))
    if process_test_envmaps:
        print(f"  [Env] {scene_name}/{split}: written={n_env_written}, skipped={n_env_skipped}")
    return out_json


def write_full_list(output_root: Path, split: str):
    meta_dir = output_root / split / "metadata"
    if not meta_dir.exists():
        return 0
    json_paths = sorted(meta_dir.glob("*.json"))
    full_list_path = output_root / split / "full_list.txt"
    full_list_path.parent.mkdir(parents=True, exist_ok=True)
    with full_list_path.open("w") as f:
        for p in json_paths:
            f.write(f"{p.resolve()}\n")
    return len(json_paths)


def parse_args():
    parser = argparse.ArgumentParser("Preprocess obj_with_light to Objaverse-like train/test format")
    parser.add_argument(
        "--input-root",
        type=str,
        default="data_samples/obj_with_light",
        help="Root containing obj_with_light scene folders.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data_samples/processed_obj_with_light_objaverse_like",
        help="Output root in Objaverse-like format.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=512,
        help="Output square image size after center crop and resize.",
    )
    parser.add_argument(
        "--apply-mask",
        dest="apply_mask",
        action="store_true",
        help="Apply mask blending onto a white background before crop/resize.",
    )
    parser.add_argument(
        "--no-apply-mask",
        dest="apply_mask",
        action="store_false",
        help="Disable mask blending and keep original RGB before crop/resize.",
    )
    parser.set_defaults(apply_mask=True)
    return parser.parse_args()


def main():
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    envmap_enabled = HAS_PYEXR or HAS_IMAGEIO
    if not envmap_enabled:
        print("[Warn] Neither pyexr nor imageio found. test envmap export will be skipped.")

    scene_dirs = [d for d in sorted(input_root.iterdir()) if d.is_dir()]
    if not scene_dirs:
        print(f"[Warn] No scene folders found under {input_root}")
        return

    n_train = 0
    n_test = 0
    for scene_dir in scene_dirs:
        scene_name = scene_dir.name
        scene_test_root = scene_dir / "test"
        scene_inputs_root = scene_test_root / "inputs"
        if not scene_test_root.exists() or not scene_inputs_root.exists():
            print(f"[Skip] Missing required folders for {scene_name}: expected test/ and test/inputs/")
            continue

        train_json = process_split(
            scene_name=scene_name,
            src_img_dir=scene_inputs_root,
            img_prefix="image_",
            cam_dir=scene_inputs_root,
            cam_prefix="camera_",
            mask_dir=scene_inputs_root,
            mask_prefix="mask_",
            output_root=output_root,
            split="train",
            target_size=args.target_size,
            apply_mask=args.apply_mask,
            enable_test_envmaps=envmap_enabled,
        )
        if train_json is not None:
            n_train += 1

        test_json = process_split(
            scene_name=scene_name,
            src_img_dir=scene_test_root,
            img_prefix="gt_image_",
            cam_dir=scene_test_root,
            cam_prefix="gt_camera_",
            mask_dir=scene_test_root,
            mask_prefix="gt_mask_",
            output_root=output_root,
            split="test",
            target_size=args.target_size,
            apply_mask=args.apply_mask,
            enable_test_envmaps=envmap_enabled,
        )
        if test_json is not None:
            n_test += 1

        print(f"[Done] {scene_name}: train={'yes' if train_json else 'no'}, test={'yes' if test_json else 'no'}")

    n_train_list = write_full_list(output_root, "train")
    n_test_list = write_full_list(output_root, "test")
    print(f"[Done] train full_list scenes: {n_train_list}")
    print(f"[Done] test full_list scenes: {n_test_list}")
    print(f"[Done] Processed scenes -> train: {n_train}, test: {n_test}")


if __name__ == "__main__":
    main()
