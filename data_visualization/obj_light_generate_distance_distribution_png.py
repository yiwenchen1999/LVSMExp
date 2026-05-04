from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_camera_center(path: Path):
    rows = [list(map(float, l.split())) for l in path.read_text().splitlines() if l.strip()]
    R = np.array(rows[3:6], dtype=np.float64)
    t = np.array(rows[6], dtype=np.float64)
    return -R.T @ t


def main():
    scripts_dir = Path(__file__).resolve().parent
    repo_root = scripts_dir.parent
    test_dir = repo_root / "data_samples" / "obj_with_light" / "apple" / "test"
    out_dir = test_dir / "camera_analysis"
    inputs_dir = test_dir / "inputs"

    bbox = np.array(
        [float(x.strip()) for x in (inputs_dir / "object_bounding_box.txt").read_text().splitlines() if x.strip()],
        dtype=np.float64,
    )
    scene_center = (bbox[0::2] + bbox[1::2]) / 2.0

    input_files = sorted(inputs_dir.glob("camera_*.txt"))
    gt_files = sorted(test_dir.glob("gt_camera_*.txt"))

    input_centers = np.stack([read_camera_center(p) for p in input_files], axis=0)
    gt_centers = np.stack([read_camera_center(p) for p in gt_files], axis=0)

    d_input = np.linalg.norm(input_centers - scene_center[None, :], axis=1)
    d_gt = np.linalg.norm(gt_centers - scene_center[None, :], axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=160)

    bins = np.linspace(min(d_input.min(), d_gt.min()), max(d_input.max(), d_gt.max()), 14)
    axes[0].hist(d_input, bins=bins, alpha=0.75, color="tab:blue", edgecolor="black", label="inputs (N=45)")
    axes[0].hist(d_gt, bins=bins, alpha=0.65, color="tab:orange", edgecolor="black", label="gt (N=9)")
    axes[0].axvline(d_input.mean(), color="tab:blue", linestyle="--", linewidth=1.8)
    axes[0].axvline(d_gt.mean(), color="tab:orange", linestyle="--", linewidth=1.8)
    axes[0].set_title("Distance to Scene Center")
    axes[0].set_xlabel("distance (world units)")
    axes[0].set_ylabel("count")
    axes[0].legend(fontsize=8)

    axes[1].boxplot([d_input, d_gt], tick_labels=["inputs", "gt"], showmeans=True)
    axes[1].set_title("Distance Boxplot")
    axes[1].set_ylabel("distance (world units)")
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle("Object-Light Camera Distance Distribution", y=1.02, fontsize=12)
    fig.tight_layout()

    png_path = out_dir / "obj_light_camera_distance_distribution.png"
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    stats = {
        "scene_center_bbox_world": scene_center.tolist(),
        "inputs_distance": {
            "min": float(d_input.min()),
            "max": float(d_input.max()),
            "mean": float(d_input.mean()),
            "std": float(d_input.std()),
            "median": float(np.median(d_input)),
        },
        "gt_distance": {
            "min": float(d_gt.min()),
            "max": float(d_gt.max()),
            "mean": float(d_gt.mean()),
            "std": float(d_gt.std()),
            "median": float(np.median(d_gt)),
        },
    }
    (out_dir / "obj_light_camera_distance_distribution_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
