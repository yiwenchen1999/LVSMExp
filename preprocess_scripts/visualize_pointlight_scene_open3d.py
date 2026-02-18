#!/usr/bin/env python3
"""
Visualize one processed point-light scene with Open3D LineSet.

What it draws:
- Point-light rays from point_light_rays/{scene_name}.npy as black lines.
  Each ray starts at ray_o and extends along normalized ray_d with length 1.
- Camera frustums from metadata/{scene_name}.json in world space.
  First camera is red, last camera is green, intermediate cameras linearly interpolate.
"""

import argparse
import json
import os
from typing import List, Tuple

import numpy as np

try:
    import open3d as o3d
except ImportError as exc:
    raise ImportError(
        "open3d is required for visualization. Please install open3d first."
    ) from exc


def load_scene_data(processed_root: str, split: str, scene_name: str):
    metadata_path = os.path.join(processed_root, split, "metadata", f"{scene_name}.json")
    rays_path = os.path.join(processed_root, split, "point_light_rays", f"{scene_name}.npy")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    if not os.path.exists(rays_path):
        raise FileNotFoundError(f"Point light rays not found: {rays_path}")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    rays = np.load(rays_path)

    if rays.ndim != 2 or rays.shape[1] != 10:
        raise ValueError(f"Expected point_light_rays shape [N,10], got {rays.shape}")

    return metadata, rays


def build_rays_lineset(rays: np.ndarray, ray_length: float = 1.0) -> o3d.geometry.LineSet:
    """
    rays: [N, 10] = intensity(1), color(3), ray_o(3), ray_d(3)
    """
    ray_o = rays[:, 4:7].astype(np.float64)
    ray_d = rays[:, 7:10].astype(np.float64)
    ray_d = ray_d / (np.linalg.norm(ray_d, axis=1, keepdims=True) + 1e-8)
    ray_end = ray_o + ray_d * float(ray_length)

    n = ray_o.shape[0]
    points = np.vstack([ray_o, ray_end])  # [2N, 3]
    lines = np.stack([np.arange(n), np.arange(n) + n], axis=1).astype(np.int32)
    colors = np.zeros((n, 3), dtype=np.float64)  # black rays

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def camera_color(i: int, total: int) -> np.ndarray:
    """Interpolate camera color from red (first) to green (last)."""
    if total <= 1:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    t = float(i) / float(total - 1)
    return np.array([1.0 - t, t, 0.0], dtype=np.float64)


def get_intrinsics_from_frame(frame: dict) -> Tuple[float, float, float, float, int, int]:
    fxfycxcy = frame["fxfycxcy"]
    fx, fy, cx, cy = [float(x) for x in fxfycxcy]
    # In this pipeline cx=w/2, cy=h/2; recover integer image size from principal point.
    width = int(round(cx * 2.0))
    height = int(round(cy * 2.0))
    return fx, fy, cx, cy, width, height


def make_camera_frustum_world_points(
    w2c: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    frustum_depth: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      points_world: [5,3] (camera center + 4 image-plane corners)
      lines: [8,2] line indices for frustum
    """
    c2w = np.linalg.inv(w2c)

    # Pixel corners on image plane (z=1 in camera coordinates).
    corners_uv = np.array(
        [
            [0.0, 0.0],
            [width - 1.0, 0.0],
            [width - 1.0, height - 1.0],
            [0.0, height - 1.0],
        ],
        dtype=np.float64,
    )

    corners_cam = []
    for u, v in corners_uv:
        x = (u - cx) / fx
        y = (v - cy) / fy
        d = np.array([x, y, 1.0], dtype=np.float64)
        d = d / (np.linalg.norm(d) + 1e-8)
        corners_cam.append(d * frustum_depth)
    corners_cam = np.stack(corners_cam, axis=0)  # [4,3]

    center_cam = np.zeros((1, 3), dtype=np.float64)
    points_cam = np.vstack([center_cam, corners_cam])  # [5,3]

    points_cam_h = np.concatenate([points_cam, np.ones((5, 1), dtype=np.float64)], axis=1)
    points_world_h = (c2w @ points_cam_h.T).T
    points_world = points_world_h[:, :3] / points_world_h[:, 3:4]

    lines = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],  # rays from camera center to corners
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 1],  # image-plane rectangle
        ],
        dtype=np.int32,
    )
    return points_world, lines


def build_camera_frustums_linesets(metadata: dict, frustum_depth: float) -> List[o3d.geometry.LineSet]:
    frames = metadata.get("frames", [])
    if not frames:
        raise ValueError("No frames found in metadata.")

    line_sets: List[o3d.geometry.LineSet] = []
    total = len(frames)
    for i, frame in enumerate(frames):
        fx, fy, cx, cy, width, height = get_intrinsics_from_frame(frame)
        w2c = np.array(frame["w2c"], dtype=np.float64)
        points_world, lines = make_camera_frustum_world_points(
            w2c=w2c,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=width,
            height=height,
            frustum_depth=frustum_depth,
        )

        color = camera_color(i, total)
        colors = np.tile(color[None, :], (lines.shape[0], 1))

        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(points_world)
        ls.lines = o3d.utility.Vector2iVector(lines)
        ls.colors = o3d.utility.Vector3dVector(colors)
        line_sets.append(ls)
    return line_sets


def main():
    parser = argparse.ArgumentParser(description="Visualize processed point-light scene with Open3D LineSet")
    parser.add_argument(
        "--processed-root",
        type=str,
        default="data_samples/processed_objaverse_with_pointLights",
        help="Root of processed dataset",
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument(
        "--scene-name",
        type=str,
        default="d1854791c6234fc2bfd78433ecd01776_white_pl_0",
        help="Scene name (without extension)",
    )
    parser.add_argument(
        "--ray-length",
        type=float,
        default=1.0,
        help="Length of each visualized light ray",
    )
    parser.add_argument(
        "--frustum-depth",
        type=float,
        default=0.2,
        help="Depth/scale of camera frustums in world units",
    )
    args = parser.parse_args()

    metadata, rays = load_scene_data(args.processed_root, args.split, args.scene_name)

    rays_ls = build_rays_lineset(rays, ray_length=args.ray_length)
    cam_line_sets = build_camera_frustums_linesets(metadata, frustum_depth=args.frustum_depth)

    geometries: List[o3d.geometry.Geometry] = [rays_ls] + cam_line_sets

    print(f"Scene: {args.scene_name}")
    print(f"Rays: {rays.shape[0]}")
    print(f"Cameras: {len(metadata.get('frames', []))}")
    print("Opening Open3D viewer...")
    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"Point Light + Cameras: {args.scene_name}",
    )


if __name__ == "__main__":
    main()
