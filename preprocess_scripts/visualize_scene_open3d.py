#!/usr/bin/env python3
"""
Visualize a single scene's cameras and light sources with Open3D.

Draws:
- A wireframe sphere (grid-ball) at the scene center.
- Light ray source positions as colored point cloud.
- Selected camera frustums in world space.
- Saves a preview PNG with transparent background via offscreen rendering.
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


def get_intrinsics_from_frame(frame: dict) -> Tuple[float, float, float, float, int, int]:
    fxfycxcy = frame["fxfycxcy"]
    fx, fy, cx, cy = [float(x) for x in fxfycxcy]
    width = int(round(cx * 2.0))
    height = int(round(cy * 2.0))
    return fx, fy, cx, cy, width, height


def make_tube(p0: np.ndarray, p1: np.ndarray, radius: float, color, resolution: int = 6):
    """Create a cylinder (tube) mesh between two 3D points."""
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    diff = p1 - p0
    length = np.linalg.norm(diff)
    if length < 1e-10:
        return None
    cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length, resolution=resolution, split=1)
    mid = (p0 + p1) / 2.0
    direction = diff / length
    # Default cylinder axis is along Z
    z_axis = np.array([0.0, 0.0, 1.0])
    v = np.cross(z_axis, direction)
    s = np.linalg.norm(v)
    c = np.dot(z_axis, direction)
    if s < 1e-8:
        if c > 0:
            R = np.eye(3)
        else:
            R = np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = mid
    cyl.transform(T)
    cyl.paint_uniform_color(color)
    cyl.compute_vertex_normals()
    return cyl


def make_camera_frustum(
    w2c: np.ndarray, fx: float, fy: float, cx: float, cy: float,
    width: int, height: int, frustum_depth: float, color: np.ndarray,
    tube_radius: float = 0.04,
) -> List[o3d.geometry.TriangleMesh]:
    """Build camera frustum as thick tube meshes."""
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

    edges = [
        (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 2), (2, 3), (3, 4), (4, 1),
    ]
    tubes = []
    for i, j in edges:
        t = make_tube(points_world[i], points_world[j], tube_radius, color)
        if t is not None:
            tubes.append(t)
    return tubes


def make_shaded_sphere(center=(0, 0, 0), radius=3.0, color=(0.72, 0.75, 0.80), grid_color=(0.38, 0.38, 0.42)):
    """Create a shaded sphere with a wireframe grid overlay for a 3D look."""
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=30)
    mesh.translate(np.array(center, dtype=np.float64))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)

    grid_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius * 1.001, resolution=16)
    grid_mesh.translate(np.array(center, dtype=np.float64))
    wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(grid_mesh)
    wireframe.paint_uniform_color(grid_color)

    return mesh, wireframe


def is_area_light(rays: np.ndarray) -> bool:
    """Detect whether a ray array represents an area light (many unique origins)."""
    rounded = np.round(rays[:, 4:7], decimals=1)
    return len(np.unique(rounded, axis=0)) > 10


def extract_light_positions(rays: np.ndarray) -> np.ndarray:
    """Extract light source positions. For point lights returns exact origins;
    for area lights (many unique origins) returns the centroid."""
    ray_o = rays[:, 4:7]
    rounded = np.round(ray_o, decimals=1)
    unique = np.unique(rounded, axis=0)
    if len(unique) <= 10:
        return unique
    return ray_o.mean(axis=0, keepdims=True)


def make_area_light_plane(rays: np.ndarray, color=(0.6, 0.6, 0.6), plane_size=2.0):
    """Create a flat plane mesh representing an area light source.
    The plane is centered at the median of ray origins, oriented via PCA."""
    ray_o = rays[:, 4:7].astype(np.float64)
    centroid = np.median(ray_o, axis=0)

    # Normal faces toward scene center (0,0,0)
    normal = -centroid / (np.linalg.norm(centroid) + 1e-8)

    up_candidate = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(normal, up_candidate)) > 0.99:
        up_candidate = np.array([0.0, 1.0, 0.0])
    tangent = np.cross(normal, up_candidate)
    tangent /= np.linalg.norm(tangent) + 1e-8
    bitangent = np.cross(normal, tangent)
    bitangent /= np.linalg.norm(bitangent) + 1e-8

    half = plane_size / 2.0
    corners = np.array([
        centroid - half * tangent - half * bitangent,
        centroid + half * tangent - half * bitangent,
        centroid + half * tangent + half * bitangent,
        centroid - half * tangent + half * bitangent,
    ])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(corners)
    mesh.triangles = o3d.utility.Vector3iVector(np.array([[0, 1, 2], [0, 2, 3]]))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return mesh, centroid


def make_light_points_cloud(rays: np.ndarray, max_points: int = 2000) -> o3d.geometry.PointCloud:
    """Subsample ray origins and show them as a point cloud."""
    ray_o = rays[:, 4:7].astype(np.float64)
    ray_color = rays[:, 1:4].astype(np.float64)
    if ray_o.shape[0] > max_points:
        idx = np.random.choice(ray_o.shape[0], max_points, replace=False)
        ray_o = ray_o[idx]
        ray_color = ray_color[idx]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ray_o)
    pcd.colors = o3d.utility.Vector3dVector(ray_color.clip(0, 1))
    return pcd


def make_camera_label_sphere(position: np.ndarray, color, radius=0.08):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=10)
    sphere.translate(position)
    sphere.paint_uniform_color(color)
    return sphere


def save_transparent_preview(geometries, output_path, img_w=1920, img_h=1080):
    """Render the scene offscreen and save as PNG with transparent background."""
    from PIL import Image

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=img_w, height=img_h, visible=False)
    for g in geometries:
        vis.add_geometry(g)

    opt = vis.get_render_option()
    opt.background_color = np.array([0.0, 0.0, 0.0])
    opt.point_size = 5.0
    opt.line_width = 2.0

    ctr = vis.get_view_control()
    ctr.set_zoom(0.5)
    ctr.set_front([0.5, -0.5, 0.7])
    ctr.set_lookat([0.0, 0.0, 0.0])
    ctr.set_up([0.0, 0.0, 1.0])

    vis.poll_events()
    vis.update_renderer()

    img_o3d = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()

    img_np = (np.asarray(img_o3d) * 255).astype(np.uint8)

    # Build alpha: black background pixels become transparent
    gray = img_np.astype(np.float32).mean(axis=2)
    alpha = np.where(gray < 2.0, 0, 255).astype(np.uint8)
    rgba = np.dstack([img_np, alpha])

    Image.fromarray(rgba, "RGBA").save(output_path)
    print(f"Saved transparent preview: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize scene cameras and light sources with Open3D")
    parser.add_argument("--scene-name", type=str,
                        default="6daa24e4320d403780b6790ea46c8ee3_white_pl_0")
    parser.add_argument("--visdata-root", type=str, default="visdata")
    parser.add_argument("--camera-indices", type=int, nargs="+", default=[1, 6],
                        help="0-based camera indices to visualize (default: 1 6, i.e. 2nd and 7th)")
    parser.add_argument("--frustum-depth", type=float, default=1.2)
    parser.add_argument("--sphere-radius", type=float, default=3.2)
    parser.add_argument("--max-light-points", type=int, default=2000)
    parser.add_argument("--extra-rays", type=str, nargs="*", default=[],
                        help="Additional .npy ray files to show light sources from (different color)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output PNG path for transparent preview (default: visdata/<scene_name>_preview.png)")
    parser.add_argument("--no-interactive", action="store_true",
                        help="Skip interactive viewer, only save preview image")
    args = parser.parse_args()

    rays_path = os.path.join(args.visdata_root, f"{args.scene_name}.npy")
    metadata_path = os.path.join(args.visdata_root, "metadata", f"{args.scene_name}.json")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    rays = None
    if os.path.exists(rays_path):
        rays = np.load(rays_path)
        assert rays.ndim == 2 and rays.shape[1] == 10, f"Expected [N,10], got {rays.shape}"

    frames = metadata["frames"]
    geometries: List[o3d.geometry.Geometry] = []

    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    geometries.append(coord)

    sphere_mesh, sphere_grid = make_shaded_sphere(radius=args.sphere_radius)
    geometries.append(sphere_mesh)
    geometries.append(sphere_grid)

    if rays is not None:
        if is_area_light(rays):
            plane_mesh, centroid = make_area_light_plane(rays, color=(0.55, 0.55, 0.58), plane_size=2.0)
            geometries.append(plane_mesh)
            np.random.seed(42)
            for target in np.random.randn(8, 3) * 0.5:
                ray_tube = make_tube(centroid, target, radius=0.015, color=(1.0, 0.9, 0.3))
                if ray_tube is not None:
                    geometries.append(ray_tube)
            print(f"Light: area light, centroid=({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f})")
        else:
            light_pcd = make_light_points_cloud(rays, max_points=args.max_light_points)
            geometries.append(light_pcd)

            unique_origins = extract_light_positions(rays)
            for origin in unique_origins:
                marker = make_camera_label_sphere(origin, color=(1.0, 0.8, 0.0), radius=0.4)
                geometries.append(marker)
                np.random.seed(42)
                for target in np.random.randn(8, 3) * 0.5:
                    ray_tube = make_tube(origin, target, radius=0.015, color=(1.0, 0.9, 0.3))
                    if ray_tube is not None:
                        geometries.append(ray_tube)

            print(f"Light sources: {len(unique_origins)} point source(s)")
            for i, o in enumerate(unique_origins):
                print(f"  Light {i}: pos=({o[0]:.3f}, {o[1]:.3f}, {o[2]:.3f})")

    extra_light_colors = [
        (0.0, 1.0, 0.5),   # green
        (0.0, 0.8, 1.0),   # cyan
        (1.0, 0.4, 1.0),   # magenta
        (1.0, 0.5, 0.0),   # orange
    ]
    for ei, extra_path in enumerate(args.extra_rays):
        if not os.path.exists(extra_path):
            print(f"Warning: extra rays file not found: {extra_path}, skipping")
            continue
        extra_rays = np.load(extra_path)
        extra_color = extra_light_colors[ei % len(extra_light_colors)]
        extra_ray_color = (extra_color[0] * 0.8, extra_color[1] * 0.8, extra_color[2] * 0.8)
        extra_name = os.path.basename(extra_path)

        if is_area_light(extra_rays):
            plane_mesh, centroid = make_area_light_plane(extra_rays, color=(0.55, 0.55, 0.58), plane_size=2.0)
            geometries.append(plane_mesh)
            np.random.seed(99 + ei)
            for target in np.random.randn(8, 3) * 0.5:
                ray_tube = make_tube(centroid, target, radius=0.015, color=extra_ray_color)
                if ray_tube is not None:
                    geometries.append(ray_tube)
            print(f"Extra light ({extra_name}): area light, centroid=({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f})")
        else:
            extra_pcd = make_light_points_cloud(extra_rays, max_points=args.max_light_points)
            geometries.append(extra_pcd)
            extra_origins = extract_light_positions(extra_rays)
            for origin in extra_origins:
                marker = make_camera_label_sphere(origin, color=extra_color, radius=0.4)
                geometries.append(marker)
                np.random.seed(99 + ei)
                for target in np.random.randn(8, 3) * 0.5:
                    ray_tube = make_tube(origin, target, radius=0.015, color=extra_ray_color)
                    if ray_tube is not None:
                        geometries.append(ray_tube)
            print(f"Extra light ({extra_name}): {len(extra_origins)} point source(s)")
            for i, o in enumerate(extra_origins):
                print(f"  Extra Light {i}: pos=({o[0]:.3f}, {o[1]:.3f}, {o[2]:.3f})")

    cam_colors = [
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
        np.array([0.0, 0.8, 0.0], dtype=np.float64),
        np.array([1.0, 0.5, 0.0], dtype=np.float64),
        np.array([0.8, 0.0, 0.8], dtype=np.float64),
    ]

    for ci, cam_idx in enumerate(args.camera_indices):
        if cam_idx < 0 or cam_idx >= len(frames):
            print(f"Warning: camera index {cam_idx} out of range [0, {len(frames)-1}], skipping")
            continue
        frame = frames[cam_idx]
        fx, fy, cx, cy, width, height = get_intrinsics_from_frame(frame)
        w2c = np.array(frame["w2c"], dtype=np.float64)
        color = cam_colors[ci % len(cam_colors)]

        frustum_tubes = make_camera_frustum(
            w2c=w2c, fx=fx, fy=fy, cx=cx, cy=cy,
            width=width, height=height,
            frustum_depth=args.frustum_depth, color=color,
            tube_radius=0.04,
        )
        geometries.extend(frustum_tubes)

        c2w = np.linalg.inv(w2c)
        cam_pos = c2w[:3, 3]
        label_sphere = make_camera_label_sphere(cam_pos, color=color, radius=0.35)
        geometries.append(label_sphere)

        color_names = ["red", "blue", "green", "orange", "purple"]
        print(f"Camera {cam_idx} (#{cam_idx+1}): pos=({cam_pos[0]:.3f}, {cam_pos[1]:.3f}, {cam_pos[2]:.3f})  color={color_names[ci % len(color_names)]}")

    print(f"\nScene: {args.scene_name}")
    print(f"Total cameras in metadata: {len(frames)}")
    print(f"Visualizing cameras: {args.camera_indices}")

    output_path = args.output or os.path.join(args.visdata_root, f"{args.scene_name}_preview.png")
    save_transparent_preview(geometries, output_path)

    if not args.no_interactive:
        print("Opening Open3D viewer...")
        o3d.visualization.draw_geometries(
            geometries,
            window_name=f"Scene: {args.scene_name}",
        )


if __name__ == "__main__":
    main()
