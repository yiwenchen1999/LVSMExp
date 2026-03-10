#!/usr/bin/env python3
"""
Preprocess TensoIR benchmark data to match objaverse_processed_with_envmaps format.

Input structure (tenIR_bench):
  {object}/test_{idx}/
    metadata.json          - cam_angle_x (rad), cam_transform_mat (4x4 flat), envmap, imh, imw
    rgba_{env}.png         - RGBA rendered under env lighting
    albedo.png             - object albedo

Output structure (matching objaverse_processed_with_envmaps/test/):
  metadata/{object}_{env}.json    - scene_name, frames[{image_path, fxfycxcy, w2c}]
  images/{object}_{env}/00000.png - composited RGB images (white bg)
  envmaps/{object}_{env}/00000_hdr.png, 00000_ldr.png
  albedos/{object}/00000.png
  full_list.txt

Usage:
    python preprocess_scripts/preprocess_tenIR.py \\
        --input data_samples/tenIR_bench \\
        --output data_samples/tenIR_processed \\
        --hdri-dir data_samples/high_res_envmaps_1k
"""

import os
import sys
import json
import re
import math
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Camera utilities (from preprocess_objaverse.py)
# ---------------------------------------------------------------------------

def blender_to_opencv_c2w(c2w_blender):
    """Convert OpenGL/Blender c2w to OpenCV c2w.
    OpenGL: +X right, +Y up, -Z forward
    OpenCV: +X right, +Y down, +Z forward
    """
    c2w_opencv = c2w_blender.copy()
    transform = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    c2w_opencv[:3, :3] = c2w_blender[:3, :3] @ transform
    return c2w_opencv


def fov_rad_to_fxfycxcy(fov_rad, width, height):
    """Convert horizontal FOV (radians) to [fx, fy, cx, cy]."""
    fx = fy = (width / 2.0) / math.tan(fov_rad / 2.0)
    cx = width / 2.0
    cy = height / 2.0
    return [fx, fy, cx, cy]


# ---------------------------------------------------------------------------
# Environment-map utilities (pure numpy, no torch dependency)
# ---------------------------------------------------------------------------

def read_hdr(path):
    """Read Radiance .hdr (RGBE) file -> float32 RGB numpy array (H, W, 3)."""
    with open(path, 'rb') as f:
        line = b''
        while True:
            c = f.read(1)
            if c == b'\n':
                if line == b'':
                    break
                line = b''
            else:
                line += c
        # Resolution line: -Y <H> +X <W>
        res_line = f.readline().decode().strip()
        parts = res_line.split()
        height = int(parts[1])
        width = int(parts[3])
        rgb = np.zeros((height, width, 3), dtype=np.float32)
        for y in range(height):
            scanline = _read_hdr_scanline(f, width)
            rgb[y] = scanline
    return rgb


def _read_hdr_scanline(f, width):
    """Read one scanline from an HDR file (new-style RLE or flat)."""
    header = f.read(4)
    if len(header) < 4:
        raise ValueError("Unexpected EOF in HDR scanline")
    r, g, b, e = header
    if r == 2 and g == 2 and b < 128:
        # New-style RLE scanline
        scan_w = (b << 8) | e
        if scan_w != width:
            raise ValueError(f"Scanline width mismatch: {scan_w} != {width}")
        components = np.zeros((4, width), dtype=np.uint8)
        for ch in range(4):
            ptr = 0
            while ptr < width:
                code = f.read(1)
                if len(code) == 0:
                    raise ValueError("Unexpected EOF in RLE data")
                code = code[0]
                if code > 128:
                    count = code - 128
                    val = f.read(1)[0]
                    components[ch, ptr:ptr + count] = val
                    ptr += count
                else:
                    count = code
                    data = f.read(count)
                    components[ch, ptr:ptr + count] = np.frombuffer(data, dtype=np.uint8)
                    ptr += count
        rgbe = components.T  # (width, 4)
    else:
        # Flat (uncompressed) — first pixel already read
        rest = f.read((width - 1) * 4)
        flat = np.frombuffer(header + rest, dtype=np.uint8).reshape(width, 4)
        rgbe = flat

    # RGBE -> float RGB
    exp = rgbe[:, 3].astype(np.int32)
    mask = exp > 0
    result = np.zeros((width, 3), dtype=np.float32)
    if np.any(mask):
        scale = np.ldexp(1.0, exp[mask] - (128 + 8))
        result[mask, 0] = rgbe[mask, 0] * scale
        result[mask, 1] = rgbe[mask, 1] * scale
        result[mask, 2] = rgbe[mask, 2] * scale
    return result


def generate_envir_map_dir(envmap_h, envmap_w):
    """Generate (H, W, 3) direction grid for equirectangular envmap."""
    lat_step = np.pi / envmap_h
    lng_step = 2 * np.pi / envmap_w
    theta = np.linspace(np.pi / 2 - 0.5 * lat_step,
                        -np.pi / 2 + 0.5 * lat_step, envmap_h)
    phi = np.linspace(np.pi - 0.5 * lng_step,
                      -np.pi + 0.5 * lng_step, envmap_w)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
    view_dirs = np.stack([
        np.cos(phi_grid) * np.cos(theta_grid),
        np.sin(phi_grid) * np.cos(theta_grid),
        np.sin(theta_grid),
    ], axis=-1).reshape(-1, 3)
    return view_dirs


def sample_envmap(envir_map, directions):
    """Sample equirectangular envmap at given 3D directions (N,3) -> (N,3) RGB.
    Uses bilinear interpolation.
    """
    env_h, env_w = envir_map.shape[:2]
    dirs = np.clip(directions, -1, 1).astype(np.float64)
    theta = np.arccos(np.clip(dirs[:, 2], -1, 1))  # [0, pi]
    phi = np.arctan2(dirs[:, 1], dirs[:, 0])        # [-pi, pi]

    # Map to pixel coordinates (continuous)
    v = theta / np.pi * (env_h - 1)                      # [0, H-1]
    u = (-phi / np.pi * 0.5 + 0.5) * (env_w - 1)        # [0, W-1]

    u0 = np.floor(u).astype(int)
    v0 = np.floor(v).astype(int)
    u1 = u0 + 1
    v1 = v0 + 1

    wu = (u - u0).astype(np.float32)
    wv = (v - v0).astype(np.float32)

    u0 = np.clip(u0, 0, env_w - 1)
    u1 = np.clip(u1, 0, env_w - 1)
    v0 = np.clip(v0, 0, env_h - 1)
    v1 = np.clip(v1, 0, env_h - 1)

    c00 = envir_map[v0, u0]
    c01 = envir_map[v0, u1]
    c10 = envir_map[v1, u0]
    c11 = envir_map[v1, u1]

    wu = wu[:, None]
    wv = wv[:, None]
    result = (c00 * (1 - wu) * (1 - wv) +
              c01 * wu * (1 - wv) +
              c10 * (1 - wu) * wv +
              c11 * wu * wv)
    return result


def rotate_and_preprocess_envir_map(envir_map_np, c2w_opengl, euler_rotation_rad=None,
                                     view_dirs=None):
    """Rotate envmap by euler Z-rotation + camera pose; return (hdr_rescaled, ldr)."""
    env_h, env_w = envir_map_np.shape[:2]
    if view_dirs is None:
        view_dirs = generate_envir_map_dir(env_h, env_w)

    processed = envir_map_np.copy()

    if euler_rotation_rad is not None and euler_rotation_rad != 0.0:
        rotation_deg = np.degrees(euler_rotation_rad)
        shift = int((rotation_deg / 360.0) * env_w)
        processed = np.roll(processed, shift, axis=1)

    c2w_rotation = c2w_opengl[:3, :3]
    w2c_rotation = c2w_rotation.T
    axis_aligned_transform = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    axis_aligned_R = (axis_aligned_transform @ w2c_rotation).astype(np.float64)
    view_dirs_world = view_dirs @ axis_aligned_R

    rotated_hdr_rgb = sample_envmap(processed, view_dirs_world)
    rotated_hdr_rgb = rotated_hdr_rgb.reshape(env_h, env_w, 3)

    ldr = np.clip(rotated_hdr_rgb, 0, 1) ** (1 / 2.2)
    hdr_log = np.log1p(10 * rotated_hdr_rgb)
    max_val = np.max(hdr_log)
    hdr_rescaled = np.clip(hdr_log / max_val if max_val > 0 else hdr_log, 0, 1)
    return hdr_rescaled, ldr


# ---------------------------------------------------------------------------
# TensoIR helpers
# ---------------------------------------------------------------------------

def parse_cam_transform_mat(mat_str):
    """Parse comma-separated 4x4 matrix string to numpy array (row-major)."""
    vals = [float(v) for v in mat_str.split(',')]
    assert len(vals) == 16, f"Expected 16 values, got {len(vals)}"
    return np.array(vals).reshape(4, 4)


def parse_env_name(filename):
    """Parse env name from rgba_{env}.png.
    Returns (env_label, hdr_basename, rotation_degrees).
    e.g. 'rgba_sunset_120.png' -> ('sunset_120', 'sunset', 120)
         'rgba_bridge.png' -> ('bridge', 'bridge', 0)
    """
    name = filename.replace('rgba_', '').replace('.png', '')
    match = re.match(r'^(.+?)_(\d{3})$', name)
    if match:
        base = match.group(1)
        rotation = int(match.group(2))
        return name, base, rotation
    return name, name, 0


def rgba_to_rgb_white_bg(rgba_array):
    """Composite RGBA onto white background -> uint8 RGB."""
    rgba = rgba_array.astype(np.float32) / 255.0
    alpha = rgba[:, :, 3:4]
    rgb = rgba[:, :, :3] * alpha + (1.0 - alpha)
    return (rgb * 255).clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_object(object_name, object_dir, output_root, hdri_dir, split='test'):
    """Process one TensoIR object, creating one scene per env."""

    view_dirs_list = sorted([
        d for d in os.listdir(object_dir)
        if os.path.isdir(os.path.join(object_dir, d)) and d.startswith(f'{split}_')
    ])
    if not view_dirs_list:
        print(f"  No {split}_* views found in {object_dir}, skipping")
        return []

    # Collect per-view metadata and available envs
    view_metas = []
    all_envs = set()
    for vdir in view_dirs_list:
        vpath = os.path.join(object_dir, vdir)
        meta_path = os.path.join(vpath, 'metadata.json')
        if not os.path.exists(meta_path):
            print(f"  Warning: {meta_path} missing, skipping view")
            continue
        with open(meta_path) as f:
            meta = json.load(f)

        envs_in_view = []
        for fn in sorted(os.listdir(vpath)):
            if fn.startswith('rgba_') and fn.endswith('.png'):
                envs_in_view.append(fn)
                env_label, _, _ = parse_env_name(fn)
                all_envs.add(env_label)

        view_metas.append({
            'view_name': vdir,
            'view_path': vpath,
            'meta': meta,
            'env_files': {parse_env_name(fn)[0]: fn for fn in envs_in_view},
        })

    if not view_metas:
        print(f"  No valid views for {object_name}")
        return []

    all_envs = sorted(all_envs)
    print(f"  Views: {len(view_metas)}, Envs: {len(all_envs)}")

    # Precompute per-view camera data
    for vm in view_metas:
        m = vm['meta']
        c2w_opengl = parse_cam_transform_mat(m['cam_transform_mat'])
        c2w_opencv = blender_to_opencv_c2w(c2w_opengl)
        w2c = np.linalg.inv(c2w_opencv)
        fov_rad = m['cam_angle_x']
        imw, imh = m['imw'], m['imh']
        fxfycxcy = fov_rad_to_fxfycxcy(fov_rad, imw, imh)
        vm['c2w_opengl'] = c2w_opengl
        vm['w2c'] = w2c
        vm['fxfycxcy'] = fxfycxcy
        vm['imw'] = imw
        vm['imh'] = imh

    # Process albedos (shared across envs)
    albedo_out_dir = os.path.join(output_root, split, 'albedos', object_name)
    os.makedirs(albedo_out_dir, exist_ok=True)
    for view_idx, vm in enumerate(view_metas):
        src_albedo = os.path.join(vm['view_path'], 'albedo.png')
        dst_albedo = os.path.join(albedo_out_dir, f"{view_idx:05d}.png")
        if not os.path.exists(dst_albedo) and os.path.exists(src_albedo):
            img = Image.open(src_albedo)
            arr = np.array(img)
            if arr.shape[2] == 4:
                arr = rgba_to_rgb_white_bg(arr)
            Image.fromarray(arr).save(dst_albedo)
    print(f"  Albedos saved to {albedo_out_dir}")

    # Load & cache HDR envmaps
    hdr_cache = {}

    def get_hdr(base_name):
        if base_name in hdr_cache:
            return hdr_cache[base_name]
        hdr_path = os.path.join(hdri_dir, f"{base_name}.hdr")
        if not os.path.exists(hdr_path):
            print(f"  Warning: {hdr_path} not found")
            hdr_cache[base_name] = None
            return None
        hdr = read_hdr(hdr_path)
        hdr_cache[base_name] = hdr
        return hdr

    # Precompute envmap direction grid (assumes all envmaps same resolution)
    envmap_view_dirs = None

    scene_names = []

    for env_label in all_envs:
        _, hdr_base, rotation_deg = parse_env_name(f"rgba_{env_label}.png")
        scene_name = f"{object_name}_{env_label}"

        out_meta_dir = os.path.join(output_root, split, 'metadata')
        out_images_dir = os.path.join(output_root, split, 'images', scene_name)
        out_envmaps_dir = os.path.join(output_root, split, 'envmaps', scene_name)
        os.makedirs(out_meta_dir, exist_ok=True)
        os.makedirs(out_images_dir, exist_ok=True)
        os.makedirs(out_envmaps_dir, exist_ok=True)

        hdr_data = get_hdr(hdr_base) if hdri_dir else None
        if hdr_data is not None and envmap_view_dirs is None:
            eh, ew = hdr_data.shape[:2]
            envmap_view_dirs = generate_envir_map_dir(eh, ew)

        euler_rot_rad = math.radians(rotation_deg) if rotation_deg != 0 else None

        frames = []
        for view_idx, vm in enumerate(view_metas):
            env_file = vm['env_files'].get(env_label)
            if env_file is None:
                continue

            # Image: RGBA -> RGB white bg
            out_img_path = os.path.join(out_images_dir, f"{view_idx:05d}.png")
            if not os.path.exists(out_img_path):
                rgba = np.array(Image.open(os.path.join(vm['view_path'], env_file)))
                rgb = rgba_to_rgb_white_bg(rgba)
                Image.fromarray(rgb).save(out_img_path)

            # Envmap per view
            out_hdr_path = os.path.join(out_envmaps_dir, f"{view_idx:05d}_hdr.png")
            out_ldr_path = os.path.join(out_envmaps_dir, f"{view_idx:05d}_ldr.png")
            if hdr_data is not None and not os.path.exists(out_hdr_path):
                hdr_rescaled, ldr = rotate_and_preprocess_envir_map(
                    hdr_data, vm['c2w_opengl'],
                    euler_rotation_rad=euler_rot_rad,
                    view_dirs=envmap_view_dirs,
                )
                Image.fromarray((hdr_rescaled * 255).astype(np.uint8)).save(out_hdr_path)
                Image.fromarray((ldr * 255).astype(np.uint8)).save(out_ldr_path)

            frames.append({
                "image_path": os.path.abspath(out_img_path),
                "fxfycxcy": vm['fxfycxcy'],
                "w2c": vm['w2c'].tolist(),
            })

        scene_data = {"scene_name": scene_name, "frames": frames}
        json_path = os.path.join(out_meta_dir, f"{scene_name}.json")
        with open(json_path, 'w') as f:
            json.dump(scene_data, f, indent=2)

        scene_names.append(scene_name)
        print(f"  Scene {scene_name}: {len(frames)} frames")

    return scene_names


def main():
    parser = argparse.ArgumentParser(description='Preprocess TensoIR bench to objaverse format')
    parser.add_argument('--input', '-i', required=True,
                        help='Input tenIR_bench directory')
    parser.add_argument('--output', '-o', required=True,
                        help='Output directory (e.g. data_samples/tenIR_processed)')
    parser.add_argument('--hdri-dir', required=True,
                        help='Directory with .hdr envmaps (e.g. data_samples/high_res_envmaps_1k)')
    parser.add_argument('--split', default='test', choices=['train', 'test', 'val'],
                        help='Which view split to process (default: test)')
    args = parser.parse_args()

    input_root = args.input
    output_root = args.output
    hdri_dir = args.hdri_dir
    split = args.split

    objects = sorted([
        d for d in os.listdir(input_root)
        if os.path.isdir(os.path.join(input_root, d))
    ])
    print(f"Found {len(objects)} objects: {objects}")

    all_scenes = []
    for obj in objects:
        print(f"\nProcessing object: {obj}")
        scenes = process_object(
            obj,
            os.path.join(input_root, obj),
            output_root,
            hdri_dir,
            split=split,
        )
        all_scenes.extend(scenes)

    # Write full_list.txt
    meta_dir = os.path.join(output_root, split, 'metadata')
    full_list_path = os.path.join(output_root, split, 'full_list.txt')
    with open(full_list_path, 'w') as f:
        for scene_name in sorted(all_scenes):
            json_path = os.path.join(meta_dir, f"{scene_name}.json")
            f.write(f"{os.path.abspath(json_path)}\n")
    print(f"\nWrote full_list.txt with {len(all_scenes)} scenes")
    print("Done.")


if __name__ == '__main__':
    main()
