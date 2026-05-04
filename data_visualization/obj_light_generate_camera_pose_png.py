from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def read_camera(path: Path):
    rows = [list(map(float, l.split())) for l in path.read_text().splitlines() if l.strip()]
    R = np.array(rows[3:6], dtype=np.float64)
    t = np.array(rows[6], dtype=np.float64)
    C = -R.T @ t
    return R, t, C


def main():
    scripts_dir = Path(__file__).resolve().parent
    repo_root = scripts_dir.parent
    test_dir = repo_root / "data_samples" / "obj_with_light" / "apple" / "test"
    out_dir = test_dir / "camera_analysis"
    inputs_dir = test_dir / "inputs"

    input_files = sorted(inputs_dir.glob("camera_*.txt"))
    cams = [read_camera(p) for p in input_files]
    R_gt, _, C_gt = read_camera(test_dir / "gt_camera_0000.txt")

    bbox = np.array(
        [float(x.strip()) for x in (inputs_dir / "object_bounding_box.txt").read_text().splitlines() if x.strip()],
        dtype=np.float64,
    )
    scene_center = (bbox[0::2] + bbox[1::2]) / 2.0

    centers = np.stack([c[2] for c in cams], axis=0)
    vecs = centers - scene_center[None, :]
    radius = np.linalg.norm(vecs, axis=1)
    xy = np.linalg.norm(vecs[:, :2], axis=1)
    az = np.degrees(np.arctan2(vecs[:, 1], vecs[:, 0]))
    el = np.degrees(np.arctan2(vecs[:, 2], xy))

    angles_plus, angles_minus = [], []
    dirs_plus, dirs_minus = [], []
    for R, _, C in cams:
        to_center = scene_center - C
        to_center /= np.linalg.norm(to_center) + 1e-12

        d_plus = R.T @ np.array([0.0, 0.0, 1.0])
        d_minus = R.T @ np.array([0.0, 0.0, -1.0])
        d_plus /= np.linalg.norm(d_plus) + 1e-12
        d_minus /= np.linalg.norm(d_minus) + 1e-12

        dirs_plus.append(d_plus)
        dirs_minus.append(d_minus)
        angles_plus.append(np.degrees(np.arccos(np.clip(np.dot(d_plus, to_center), -1.0, 1.0))))
        angles_minus.append(np.degrees(np.arccos(np.clip(np.dot(d_minus, to_center), -1.0, 1.0))))

    angles_plus = np.array(angles_plus)
    angles_minus = np.array(angles_minus)
    if angles_plus.mean() <= angles_minus.mean():
        forward_axis = "+Z"
        look_angles = angles_plus
        look_dirs = np.stack(dirs_plus, axis=0)
    else:
        forward_axis = "-Z"
        look_angles = angles_minus
        look_dirs = np.stack(dirs_minus, axis=0)

    to_center_gt = scene_center - C_gt
    to_center_gt /= np.linalg.norm(to_center_gt) + 1e-12
    d_gt = R_gt.T @ (np.array([0.0, 0.0, 1.0]) if forward_axis == "+Z" else np.array([0.0, 0.0, -1.0]))
    d_gt /= np.linalg.norm(d_gt) + 1e-12
    look_angle_gt = float(np.degrees(np.arccos(np.clip(np.dot(d_gt, to_center_gt), -1.0, 1.0))))

    mesh = o3d.io.read_triangle_mesh(str(test_dir / "neus_mesh.ply"))
    mesh_pts = None
    if len(mesh.vertices) > 0:
        mesh_pts = np.asarray(mesh.vertices)
        if mesh_pts.shape[0] > 15000:
            idx = np.random.RandomState(0).choice(mesh_pts.shape[0], 15000, replace=False)
            mesh_pts = mesh_pts[idx]

    fig = plt.figure(figsize=(16, 11), dpi=150)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1.0], height_ratios=[1.0, 1.0])

    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    if mesh_pts is not None:
        ax3d.scatter(mesh_pts[:, 0], mesh_pts[:, 1], mesh_pts[:, 2], s=0.3, c="lightgray", alpha=0.5)
    ax3d.scatter(centers[:, 0], centers[:, 1], centers[:, 2], s=22, c="tab:blue", label="input cameras")
    ax3d.scatter([C_gt[0]], [C_gt[1]], [C_gt[2]], s=90, c="orange", marker="*", label="gt_camera_0000")
    ax3d.scatter([scene_center[0]], [scene_center[1]], [scene_center[2]], s=70, c="red", marker="o", label="scene center")

    ray_scale = float(np.median(radius) * 0.35)
    for C, d in zip(centers, look_dirs):
        p2 = C + d * ray_scale
        ax3d.plot([C[0], p2[0]], [C[1], p2[1]], [C[2], p2[2]], c="tab:green", linewidth=0.7, alpha=0.7)
    p2_gt = C_gt + d_gt * ray_scale
    ax3d.plot([C_gt[0], p2_gt[0]], [C_gt[1], p2_gt[1]], [C_gt[2], p2_gt[2]], c="orange", linewidth=2.0)

    ax3d.set_title(f"Camera Poses + Scene Center (forward={forward_axis})")
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.legend(loc="upper right", fontsize=8)
    ax3d.view_init(elev=24, azim=-45)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.hist(look_angles, bins=12, color="tab:green", alpha=0.8, edgecolor="black")
    ax1.axvline(look_angle_gt, color="orange", linestyle="--", linewidth=2, label=f"gt0={look_angle_gt:.2f} deg")
    ax1.set_title("Look-at Angle to Scene Center")
    ax1.set_xlabel("angle (deg)")
    ax1.set_ylabel("count")
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.scatter(az, el, c=radius, cmap="viridis", s=28)
    cb = plt.colorbar(ax2.collections[0], ax=ax2)
    cb.set_label("radius to center")
    ax2.set_title("Azimuth/Elevation Distribution")
    ax2.set_xlabel("azimuth (deg)")
    ax2.set_ylabel("elevation (deg)")
    ax2.grid(True, alpha=0.3)

    summary = (
        f"N={len(cams)}\n"
        f"center={np.round(scene_center, 4)}\n"
        f"radius mean+-std={radius.mean():.4f}+-{radius.std():.4f}\n"
        f"look mean+-std={look_angles.mean():.2f}+-{look_angles.std():.2f} deg\n"
        f"look range=[{look_angles.min():.2f}, {look_angles.max():.2f}] deg"
    )
    fig.text(0.57, 0.02, summary, fontsize=9, family="monospace")

    png_path = out_dir / "camera_pose_analysis.png"
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(png_path)
    plt.close(fig)

    stats = {
        "forward_axis": forward_axis,
        "num_cameras": len(cams),
        "scene_center": scene_center.tolist(),
        "radius": {
            "min": float(radius.min()),
            "max": float(radius.max()),
            "mean": float(radius.mean()),
            "std": float(radius.std()),
        },
        "look_angle_deg": {
            "min": float(look_angles.min()),
            "max": float(look_angles.max()),
            "mean": float(look_angles.mean()),
            "std": float(look_angles.std()),
        },
        "gt_camera_0000_look_angle_deg": look_angle_gt,
    }
    (out_dir / "camera_pose_analysis_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
