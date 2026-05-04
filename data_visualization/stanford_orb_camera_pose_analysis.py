from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from PIL import Image


def load_transforms(path: Path):
    data = json.loads(path.read_text())
    cam_angle_x = float(data["camera_angle_x"])
    frames = data["frames"]
    c2ws = []
    files = []
    for f in frames:
        c2ws.append(np.array(f["transform_matrix"], dtype=np.float64))
        files.append(f["file_path"])
    return cam_angle_x, np.stack(c2ws, axis=0), files


def closest_point_to_rays(origins: np.ndarray, dirs: np.ndarray):
    # Solve sum_i (I - d_i d_i^T) x = sum_i (I - d_i d_i^T) o_i
    A = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros((3,), dtype=np.float64)
    for o, d in zip(origins, dirs):
        d = d / (np.linalg.norm(d) + 1e-12)
        M = np.eye(3) - np.outer(d, d)
        A += M
        b += M @ o
    return np.linalg.solve(A + 1e-9 * np.eye(3), b)


def angle_stats(forward_dirs: np.ndarray, origins: np.ndarray, target: np.ndarray):
    v = target[None, :] - origins
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    c = np.sum(forward_dirs * v, axis=1)
    c = np.clip(c, -1.0, 1.0)
    a = np.degrees(np.arccos(c))
    return a


def describe(vals: np.ndarray):
    return {
        "min": float(vals.min()),
        "max": float(vals.max()),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "median": float(np.median(vals)),
    }


def main():
    scene_root = Path("/Users/yiwenchen/Desktop/ResearchProjects/LightingDiffusion/3dgs/LVSMExp/data_samples/stanford_ORB/gnome_scene005")
    out_dir = scene_root / "camera_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_json = scene_root / "transforms_train.json"
    test_json = scene_root / "transforms_test.json"

    fovx_train, c2w_train, _ = load_transforms(train_json)
    fovx_test, c2w_test, _ = load_transforms(test_json)
    c2w_all = np.concatenate([c2w_train, c2w_test], axis=0)

    # Camera centers from c2w translation
    centers_train = c2w_train[:, :3, 3]
    centers_test = c2w_test[:, :3, 3]
    centers_all = c2w_all[:, :3, 3]

    # Blender/NeRF usually uses -Z as forward in camera coords.
    dirs_plus = c2w_all[:, :3, :3] @ np.array([0.0, 0.0, 1.0])
    dirs_minus = c2w_all[:, :3, :3] @ np.array([0.0, 0.0, -1.0])
    dirs_plus = dirs_plus / (np.linalg.norm(dirs_plus, axis=1, keepdims=True) + 1e-12)
    dirs_minus = dirs_minus / (np.linalg.norm(dirs_minus, axis=1, keepdims=True) + 1e-12)

    world_center = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    # Choose forward axis by smaller angle to world center.
    ang_w_plus = angle_stats(dirs_plus, centers_all, world_center)
    ang_w_minus = angle_stats(dirs_minus, centers_all, world_center)
    if ang_w_minus.mean() <= ang_w_plus.mean():
        forward_axis = "-Z"
        dirs = dirs_minus
        ang_world = ang_w_minus
    else:
        forward_axis = "+Z"
        dirs = dirs_plus
        ang_world = ang_w_plus

    scene_center = closest_point_to_rays(centers_all, dirs)
    ang_scene = angle_stats(dirs, centers_all, scene_center)

    dist_world = np.linalg.norm(centers_all - world_center[None, :], axis=1)
    dist_scene = np.linalg.norm(centers_all - scene_center[None, :], axis=1)

    # FOV
    sample_image = scene_root / "train" / "0000.png"
    if sample_image.exists():
        w, h = Image.open(sample_image).size
    else:
        w, h = 800, 800
    fovy_train = float(2.0 * np.arctan(np.tan(fovx_train / 2.0) * (h / w)))
    fovy_test = float(2.0 * np.arctan(np.tan(fovx_test / 2.0) * (h / w)))

    summary = {
        "scene_root": str(scene_root),
        "num_cameras": {
            "train": int(len(centers_train)),
            "test": int(len(centers_test)),
            "all": int(len(centers_all)),
        },
        "image_size": {"width": int(w), "height": int(h)},
        "fov": {
            "camera_angle_x_rad_train": float(fovx_train),
            "camera_angle_x_deg_train": float(np.degrees(fovx_train)),
            "camera_angle_y_rad_train": float(fovy_train),
            "camera_angle_y_deg_train": float(np.degrees(fovy_train)),
            "camera_angle_x_rad_test": float(fovx_test),
            "camera_angle_x_deg_test": float(np.degrees(fovx_test)),
            "camera_angle_y_rad_test": float(fovy_test),
            "camera_angle_y_deg_test": float(np.degrees(fovy_test)),
        },
        "camera_centers": {
            "min_xyz": centers_all.min(axis=0).tolist(),
            "max_xyz": centers_all.max(axis=0).tolist(),
            "mean_xyz": centers_all.mean(axis=0).tolist(),
            "std_xyz": centers_all.std(axis=0).tolist(),
        },
        "forward_axis_inferred": forward_axis,
        "world_center": world_center.tolist(),
        "estimated_scene_center_from_rays": scene_center.tolist(),
        "distance_to_world_center": describe(dist_world),
        "distance_to_estimated_scene_center": describe(dist_scene),
        "look_at_angle_to_world_center_deg": describe(ang_world),
        "look_at_angle_to_estimated_scene_center_deg": describe(ang_scene),
    }
    (out_dir / "stanford_orb_camera_summary.json").write_text(json.dumps(summary, indent=2))

    # PLY: camera centers (train blue, test orange)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(centers_all)
    colors = np.vstack(
        [
            np.tile(np.array([[0.1, 0.5, 1.0]]), (len(centers_train), 1)),
            np.tile(np.array([[1.0, 0.55, 0.1]]), (len(centers_test), 1)),
        ]
    )
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(out_dir / "stanford_orb_camera_centers.ply"), pcd)

    # PLY: look direction rays
    ray_scale = float(np.median(dist_world) * 0.25)
    line_points = []
    line_idx = []
    line_colors = []
    for i, (c, d) in enumerate(zip(centers_all, dirs)):
        p2 = c + d * ray_scale
        line_points.extend([c, p2])
        line_idx.append([2 * i, 2 * i + 1])
        line_colors.append([0.2, 0.8, 0.2])
    lines = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.array(line_points)),
        lines=o3d.utility.Vector2iVector(np.array(line_idx, dtype=np.int32)),
    )
    lines.colors = o3d.utility.Vector3dVector(np.array(line_colors))
    o3d.io.write_line_set(str(out_dir / "stanford_orb_camera_look_dirs.ply"), lines)

    # PLY: center markers
    center_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.08)
    center_mesh.paint_uniform_color([1.0, 0.0, 0.0])  # world center red
    center_mesh.translate(world_center)
    o3d.io.write_triangle_mesh(str(out_dir / "stanford_orb_world_center.ply"), center_mesh)

    scene_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.08)
    scene_mesh.paint_uniform_color([0.8, 0.0, 0.8])  # estimated scene center purple
    scene_mesh.translate(scene_center)
    o3d.io.write_triangle_mesh(str(out_dir / "stanford_orb_estimated_scene_center.ply"), scene_mesh)

    # PNG visualization
    fig = plt.figure(figsize=(15, 10), dpi=160)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0], height_ratios=[1.0, 1.0])

    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax3d.scatter(centers_train[:, 0], centers_train[:, 1], centers_train[:, 2], s=20, c="tab:blue", label="train")
    ax3d.scatter(centers_test[:, 0], centers_test[:, 1], centers_test[:, 2], s=24, c="tab:orange", label="test")
    ax3d.scatter([world_center[0]], [world_center[1]], [world_center[2]], s=90, c="red", marker="o", label="world center")
    ax3d.scatter([scene_center[0]], [scene_center[1]], [scene_center[2]], s=90, c="purple", marker="^", label="estimated scene center")
    for c, d in zip(centers_all, dirs):
        p2 = c + d * ray_scale
        ax3d.plot([c[0], p2[0]], [c[1], p2[1]], [c[2], p2[2]], c="tab:green", alpha=0.35, linewidth=0.8)
    ax3d.set_title(f"Camera Positions + Look Directions (forward={forward_axis})")
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.legend(loc="upper right", fontsize=8)
    ax3d.view_init(elev=18, azim=-42)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.hist(ang_world, bins=14, alpha=0.75, color="tab:red", edgecolor="black")
    ax1.set_title("Look-Angle to World Center")
    ax1.set_xlabel("angle (deg)")
    ax1.set_ylabel("count")

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.hist(dist_world, bins=14, alpha=0.75, color="tab:blue", edgecolor="black", label="dist to world center")
    ax2.hist(dist_scene, bins=14, alpha=0.55, color="tab:purple", edgecolor="black", label="dist to estimated scene center")
    ax2.set_title("Camera Distance Distribution")
    ax2.set_xlabel("distance")
    ax2.set_ylabel("count")
    ax2.legend(fontsize=8)

    fov_txt = (
        f"FOVx train/test: {np.degrees(fovx_train):.2f}/{np.degrees(fovx_test):.2f} deg\n"
        f"FOVy train/test: {np.degrees(fovy_train):.2f}/{np.degrees(fovy_test):.2f} deg\n"
        f"look-to-world mean: {ang_world.mean():.2f} deg\n"
        f"look-to-scene(mean): {ang_scene.mean():.2f} deg\n"
        f"dist-to-world mean: {dist_world.mean():.3f}"
    )
    fig.text(0.57, 0.02, fov_txt, fontsize=9, family="monospace")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_dir / "stanford_orb_camera_pose_analysis.png")
    plt.close(fig)

    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
