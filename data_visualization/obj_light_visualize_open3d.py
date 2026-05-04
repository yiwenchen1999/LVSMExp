from pathlib import Path

import numpy as np
import open3d as o3d


def read_camera(path: Path):
    rows = [list(map(float, line.split())) for line in path.read_text().splitlines() if line.strip()]
    R = np.array(rows[3:6], dtype=np.float64)
    t = np.array(rows[6], dtype=np.float64)
    C = -R.T @ t
    return R, t, C


def main():
    repo_root = Path(__file__).resolve().parent.parent
    base = repo_root / "data_samples" / "obj_with_light" / "apple" / "test"
    inputs_dir = base / "inputs"

    # Scene center from axis-aligned bbox.
    bbox = np.array(
        [float(x.strip()) for x in (inputs_dir / "object_bounding_box.txt").read_text().splitlines() if x.strip()],
        dtype=np.float64,
    )
    scene_center = (bbox[0::2] + bbox[1::2]) / 2.0

    input_files = sorted(inputs_dir.glob("camera_*.txt"))
    cams = [read_camera(p) for p in input_files]
    gt0 = read_camera(base / "gt_camera_0000.txt")

    centers = np.stack([c[2] for c in cams], axis=0)
    radius = np.linalg.norm(centers - scene_center[None, :], axis=1)

    # Auto-detect whether camera forward is +Z or -Z in camera frame.
    angles_plus, angles_minus = [], []
    for R, _, C in cams:
        to_center = scene_center - C
        to_center /= np.linalg.norm(to_center) + 1e-12
        d_plus = R.T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        d_minus = R.T @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
        d_plus /= np.linalg.norm(d_plus) + 1e-12
        d_minus /= np.linalg.norm(d_minus) + 1e-12
        angles_plus.append(np.degrees(np.arccos(np.clip(np.dot(d_plus, to_center), -1.0, 1.0))))
        angles_minus.append(np.degrees(np.arccos(np.clip(np.dot(d_minus, to_center), -1.0, 1.0))))

    forward_axis = "+Z" if np.mean(angles_plus) < np.mean(angles_minus) else "-Z"
    print(f"[info] inferred camera forward axis: {forward_axis}")

    geoms = []

    mesh_path = base / "neus_mesh.ply"
    if mesh_path.exists():
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        if len(mesh.vertices) > 0:
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color([0.75, 0.75, 0.75])
            geoms.append(mesh)

    center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    center_sphere.paint_uniform_color([1.0, 0.0, 0.0])  # red scene center
    center_sphere.translate(scene_center)
    geoms.append(center_sphere)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(centers)
    pcd.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0.1, 0.6, 1.0]]), (centers.shape[0], 1)))
    geoms.append(pcd)

    gt0_center = gt0[2]
    gt0_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
    gt0_marker.paint_uniform_color([1.0, 0.6, 0.0])  # orange gt0
    gt0_marker.translate(gt0_center)
    geoms.append(gt0_marker)

    # Look direction rays.
    scale = float(np.median(radius) * 0.4)
    line_points = []
    line_indices = []
    line_colors = []
    axis_vec = np.array([0.0, 0.0, 1.0]) if forward_axis == "+Z" else np.array([0.0, 0.0, -1.0])

    for i, (R, _, C) in enumerate(cams):
        d = R.T @ axis_vec
        d /= np.linalg.norm(d) + 1e-12
        line_points.extend([C, C + d * scale])
        line_indices.append([2 * i, 2 * i + 1])
        line_colors.append([0.1, 0.8, 0.1])  # green input rays

    start = len(line_points)
    d_gt = gt0[0].T @ axis_vec
    d_gt /= np.linalg.norm(d_gt) + 1e-12
    line_points.extend([gt0_center, gt0_center + d_gt * scale])
    line_indices.append([start, start + 1])
    line_colors.append([1.0, 0.6, 0.0])  # orange gt0 ray

    lines = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.array(line_points)),
        lines=o3d.utility.Vector2iVector(np.array(line_indices, dtype=np.int32)),
    )
    lines.colors = o3d.utility.Vector3dVector(np.array(line_colors))
    geoms.append(lines)

    print("[info] blue points: input camera centers")
    print("[info] green lines: input look directions")
    print("[info] orange sphere+line: gt_camera_0000")
    print("[info] red sphere: scene center from object_bounding_box")
    o3d.visualization.draw_geometries(geoms, window_name="apple camera poses")


if __name__ == "__main__":
    main()
