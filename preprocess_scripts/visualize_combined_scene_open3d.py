#!/usr/bin/env python3
"""
Visualize a processed combined lighting scene with Open3D.

Draws:
- Point/area light rays from point_light_rays/{scene_name}.npy (colored by intensity).
  Each ray starts at ray_o and extends along ray_d with configurable length.
- Camera frustums from metadata/{scene_name}.json in world space.
  First camera is red, last camera is green, intermediate cameras interpolate.
- A wireframe sphere at the origin showing the scene bounding sphere.
- Light source positions as colored spheres (point lights = yellow, area lights = cyan).
- Optional: envmap preview as a textured sphere (if envmaps are present).
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
        "open3d is required for visualization. Install: pip install open3d"
    ) from exc


def load_scene_data(processed_root: str, split: str, scene_name: str):
    metadata_path = os.path.join(processed_root, split, "metadata", f"{scene_name}.json")
    rays_path = os.path.join(processed_root, split, "point_light_rays", f"{scene_name}.npy")
    envmaps_dir = os.path.join(processed_root, split, "envmaps", scene_name)

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    rays = None
    if os.path.exists(rays_path):
        rays = np.load(rays_path)
        if rays.ndim != 2 or rays.shape[1] != 10:
            raise ValueError(f"Expected rays shape [N,10], got {rays.shape}")

    has_envmaps = os.path.exists(envmaps_dir) and len(os.listdir(envmaps_dir)) > 0

    return metadata, rays, has_envmaps, envmaps_dir


def load_combined_json(source_root: str, object_id: str, split: str, folder_name: str):
    """Try to load the original combined.json for light source positions."""
    path = os.path.join(source_root, object_id, split, folder_name, "combined.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def build_rays_lineset(rays: np.ndarray, ray_length: float = 1.0, max_rays: int = 2000) -> o3d.geometry.LineSet:
    """
    rays: [N, 10] = intensity(1), color(3), ray_o(3), ray_d(3)
    Subsample to max_rays for performance. Color by ray RGB.
    """
    if rays.shape[0] > max_rays:
        idx = np.random.choice(rays.shape[0], max_rays, replace=False)
        rays = rays[idx]

    intensity = rays[:, 0:1].astype(np.float64)
    ray_color = rays[:, 1:4].astype(np.float64)
    ray_o = rays[:, 4:7].astype(np.float64)
    ray_d = rays[:, 7:10].astype(np.float64)
    ray_d = ray_d / (np.linalg.norm(ray_d, axis=1, keepdims=True) + 1e-8)
    ray_end = ray_o + ray_d * float(ray_length)

    n = ray_o.shape[0]
    points = np.vstack([ray_o, ray_end])
    lines = np.stack([np.arange(n), np.arange(n) + n], axis=1).astype(np.int32)

    # Color lines: use ray color scaled by normalized intensity
    max_int = intensity.max() if intensity.max() > 0 else 1.0
    alpha = (intensity / max_int).clip(0.1, 1.0)
    colors = (ray_color * alpha).clip(0, 1)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def camera_color(i: int, total: int) -> np.ndarray:
    if total <= 1:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    t = float(i) / float(total - 1)
    return np.array([1.0 - t, t, 0.0], dtype=np.float64)


def get_intrinsics_from_frame(frame: dict) -> Tuple[float, float, float, float, int, int]:
    fxfycxcy = frame["fxfycxcy"]
    fx, fy, cx, cy = [float(x) for x in fxfycxcy]
    width = int(round(cx * 2.0))
    height = int(round(cy * 2.0))
    return fx, fy, cx, cy, width, height


def make_camera_frustum_world_points(
    w2c: np.ndarray, fx: float, fy: float, cx: float, cy: float,
    width: int, height: int, frustum_depth: float,
) -> Tuple[np.ndarray, np.ndarray]:
    c2w = np.linalg.inv(w2c)
    corners_uv = np.array([
        [0.0, 0.0], [width - 1.0, 0.0],
        [width - 1.0, height - 1.0], [0.0, height - 1.0],
    ], dtype=np.float64)

    corners_cam = []
    for u, v in corners_uv:
        x = (u - cx) / fx
        y = (v - cy) / fy
        d = np.array([x, y, 1.0], dtype=np.float64)
        d = d / (np.linalg.norm(d) + 1e-8)
        corners_cam.append(d * frustum_depth)
    corners_cam = np.stack(corners_cam, axis=0)

    center_cam = np.zeros((1, 3), dtype=np.float64)
    points_cam = np.vstack([center_cam, corners_cam])
    points_cam_h = np.concatenate([points_cam, np.ones((5, 1), dtype=np.float64)], axis=1)
    points_world_h = (c2w @ points_cam_h.T).T
    points_world = points_world_h[:, :3] / points_world_h[:, 3:4]

    lines = np.array([
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1],
    ], dtype=np.int32)
    return points_world, lines


def build_camera_frustums_linesets(metadata: dict, frustum_depth: float) -> List[o3d.geometry.LineSet]:
    frames = metadata.get("frames", [])
    if not frames:
        return []

    line_sets = []
    total = len(frames)
    for i, frame in enumerate(frames):
        fx, fy, cx, cy, width, height = get_intrinsics_from_frame(frame)
        w2c = np.array(frame["w2c"], dtype=np.float64)
        points_world, lines = make_camera_frustum_world_points(
            w2c=w2c, fx=fx, fy=fy, cx=cx, cy=cy,
            width=width, height=height, frustum_depth=frustum_depth,
        )
        color = camera_color(i, total)
        colors = np.tile(color[None, :], (lines.shape[0], 1))

        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(points_world)
        ls.lines = o3d.utility.Vector2iVector(lines)
        ls.colors = o3d.utility.Vector3dVector(colors)
        line_sets.append(ls)
    return line_sets


def make_wireframe_sphere(center=(0, 0, 0), radius=3.0, color=(0.5, 0.5, 0.5)):
    """Create a wireframe sphere as LineSet."""
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=20)
    mesh.translate(np.array(center, dtype=np.float64))
    wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    wireframe.paint_uniform_color(color)
    return wireframe


def make_light_source_marker(pos, color=(1.0, 1.0, 0.0), radius=0.08):
    """Create a small sphere at a light source position."""
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=10)
    sphere.translate(np.array(pos, dtype=np.float64))
    sphere.paint_uniform_color(color)
    return sphere


def extract_light_positions_from_rays(rays: np.ndarray):
    """Extract unique light source origins from ray array."""
    ray_o = rays[:, 4:7]
    # Cluster by rounding to find unique origins
    rounded = np.round(ray_o, decimals=2)
    unique_origins = np.unique(rounded, axis=0)
    return unique_origins


def main():
    parser = argparse.ArgumentParser(
        description="Visualize combined lighting scene (rays + envmap + cameras) with Open3D"
    )
    parser.add_argument(
        "--processed-root", type=str,
        default="data_samples/scene_light_combined_processed",
        help="Root of processed dataset",
    )
    parser.add_argument(
        "--source-root", type=str,
        default="data_samples/scene_light_combined",
        help="Root of original source data (for reading combined.json light positions)",
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument(
        "--scene-name", type=str,
        default="6c70f11081e4438b878f4c007a48ab65_combined_3",
        help="Scene name (without extension)",
    )
    parser.add_argument("--ray-length", type=float, default=1.5, help="Visualized ray length")
    parser.add_argument("--frustum-depth", type=float, default=0.15, help="Camera frustum depth")
    parser.add_argument("--max-rays", type=int, default=2000, help="Max rays to display")
    parser.add_argument("--sphere-radius", type=float, default=3.0, help="Scene sphere radius")
    parser.add_argument("--no-sphere", action="store_true", help="Hide scene sphere")
    parser.add_argument("--no-rays", action="store_true", help="Hide light rays")
    parser.add_argument("--no-cameras", action="store_true", help="Hide camera frustums")
    args = parser.parse_args()

    metadata, rays, has_envmaps, envmaps_dir = load_scene_data(
        args.processed_root, args.split, args.scene_name
    )

    geometries: List[o3d.geometry.Geometry] = []

    # Origin coordinate frame
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    geometries.append(coord)

    # Scene bounding sphere
    if not args.no_sphere:
        sphere_wire = make_wireframe_sphere(radius=args.sphere_radius, color=(0.4, 0.4, 0.4))
        geometries.append(sphere_wire)

    # Light rays
    ray_count = 0
    if rays is not None and not args.no_rays:
        ray_count = rays.shape[0]
        rays_ls = build_rays_lineset(rays, ray_length=args.ray_length, max_rays=args.max_rays)
        geometries.append(rays_ls)

        # Extract and mark light source positions from rays
        unique_origins = extract_light_positions_from_rays(rays)
        for origin in unique_origins:
            marker = make_light_source_marker(origin, color=(1.0, 0.8, 0.0), radius=0.1)
            geometries.append(marker)

    # Also try to load original combined.json for explicit light positions
    # Parse object_id and folder_name from scene_name
    parts = args.scene_name.split("_combined_")
    if len(parts) == 2:
        object_id = parts[0]
        folder_name = f"combined_{parts[1]}"
        comb_data = load_combined_json(args.source_root, object_id, args.split, folder_name)
        if comb_data:
            # Mark point light positions
            if "point_lights" in comb_data:
                for pos in comb_data["point_lights"]["pos"]:
                    marker = make_light_source_marker(pos, color=(1.0, 1.0, 0.0), radius=0.12)
                    geometries.append(marker)
            elif "pos" in comb_data and "power" in comb_data:
                marker = make_light_source_marker(comb_data["pos"], color=(1.0, 1.0, 0.0), radius=0.12)
                geometries.append(marker)

            # Mark area light positions
            if "area_light" in comb_data:
                al = comb_data["area_light"]
                marker = make_light_source_marker(al["pos"], color=(0.0, 1.0, 1.0), radius=0.15)
                geometries.append(marker)

    # Camera frustums
    if not args.no_cameras:
        cam_line_sets = build_camera_frustums_linesets(metadata, frustum_depth=args.frustum_depth)
        geometries.extend(cam_line_sets)

    num_cameras = len(metadata.get("frames", []))
    print(f"Scene: {args.scene_name}")
    print(f"Cameras: {num_cameras}")
    if rays is not None:
        print(f"Total rays: {ray_count} (showing {min(ray_count, args.max_rays)})")
    else:
        print("No light rays found")
    print(f"Has envmaps: {has_envmaps}")
    print(f"Scene sphere radius: {args.sphere_radius}")
    print("Light markers: yellow = point light, cyan = area light")
    print("Camera colors: red (first) -> green (last)")
    print("Opening Open3D viewer...")

    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"Combined Scene: {args.scene_name}",
    )


if __name__ == "__main__":
    main()
