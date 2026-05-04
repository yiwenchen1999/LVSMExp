from pathlib import Path
import json

import numpy as np
import open3d as o3d


def read_camera(path: Path):
    rows = [list(map(float, l.split())) for l in path.read_text().splitlines() if l.strip()]
    R = np.array(rows[3:6], dtype=np.float64)
    t = np.array(rows[6], dtype=np.float64)
    C = -R.T @ t
    return R, t, C


def summarize_distances(centers: np.ndarray, center: np.ndarray):
    d = np.linalg.norm(centers - center[None, :], axis=1)
    return {
        "min": float(d.min()),
        "max": float(d.max()),
        "mean": float(d.mean()),
        "std": float(d.std()),
        "median": float(np.median(d)),
    }


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
    bbox_min = bbox[0::2]
    bbox_max = bbox[1::2]
    scene_center_bbox = (bbox_min + bbox_max) / 2.0

    input_files = sorted(inputs_dir.glob("camera_*.txt"))
    gt_files = sorted(test_dir.glob("gt_camera_*.txt"))
    input_centers = np.stack([read_camera(p)[2] for p in input_files], axis=0)
    gt_centers = np.stack([read_camera(p)[2] for p in gt_files], axis=0)

    # Optional geometric center of mesh vertices (not the same as bbox center in general).
    mesh = o3d.io.read_triangle_mesh(str(test_dir / "neus_mesh.ply"))
    mesh_vertex_center = None
    if len(mesh.vertices) > 0:
        mesh_vertex_center = np.asarray(mesh.vertices).mean(axis=0)

    report = {
        "scene_center_definition": "AABB center from inputs/object_bounding_box.txt in world coordinates",
        "scene_center_bbox_world": scene_center_bbox.tolist(),
        "bbox_min_world": bbox_min.tolist(),
        "bbox_max_world": bbox_max.tolist(),
        "input_cameras_count": int(input_centers.shape[0]),
        "gt_cameras_count": int(gt_centers.shape[0]),
        "distance_to_bbox_center": {
            "inputs": summarize_distances(input_centers, scene_center_bbox),
            "gt": summarize_distances(gt_centers, scene_center_bbox),
        },
    }

    if mesh_vertex_center is not None:
        report["mesh_vertex_mean_world"] = mesh_vertex_center.tolist()
        report["distance_between_bbox_center_and_mesh_vertex_mean"] = float(
            np.linalg.norm(scene_center_bbox - mesh_vertex_center)
        )

    out_path = out_dir / "camera_distance_to_center_report.json"
    out_path.write_text(json.dumps(report, indent=2))

    print(f"Wrote {out_path}")
    print("inputs distance mean:", report["distance_to_bbox_center"]["inputs"]["mean"])
    print("gt distance mean:", report["distance_to_bbox_center"]["gt"]["mean"])


if __name__ == "__main__":
    main()
