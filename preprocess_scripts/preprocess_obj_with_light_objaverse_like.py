#!/usr/bin/env python3
"""
Convert data_samples/obj_with_light to an Objaverse-like layout:

  <output_root>/train/images/<scene>/<frame:05d>.png
  <output_root>/train/metadata/<scene>.json
  <output_root>/train/full_list.txt
  <output_root>/test/images/<scene>/<frame:05d>.png
  <output_root>/test/metadata/<scene>.json
  <output_root>/test/full_list.txt

Per scene convention:
  - source <scene>/test/inputs     -> output split=train
  - source <scene>/test (gt_*)     -> output split=test
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np


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


def extract_frame_id(name: str, pattern: str):
    m = re.match(pattern, name)
    if not m:
        return None
    return int(m.group(1))


def process_split(
    scene_name: str,
    src_img_dir: Path,
    img_prefix: str,
    cam_dir: Path,
    cam_prefix: str,
    output_root: Path,
    split: str,
):
    out_images_dir = output_root / split / "images" / scene_name
    out_meta_dir = output_root / split / "metadata"
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_meta_dir.mkdir(parents=True, exist_ok=True)

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

    common_ids = sorted(set(img_map.keys()) & set(cam_map.keys()))
    if not common_ids:
        return None

    frames = []
    for out_idx, fid in enumerate(common_ids):
        src_img = img_map[fid]
        src_cam = cam_map[fid]
        out_img = out_images_dir / f"{out_idx:05d}.png"
        shutil.copy2(src_img, out_img)

        K, w2c = read_camera_txt(src_cam)
        frames.append(build_frame_entry(out_img, K, w2c))

    scene_meta = {"scene_name": scene_name, "frames": frames}
    out_json = out_meta_dir / f"{scene_name}.json"
    out_json.write_text(json.dumps(scene_meta, indent=2))
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
    return parser.parse_args()


def main():
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

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
            output_root=output_root,
            split="train",
        )
        if train_json is not None:
            n_train += 1

        test_json = process_split(
            scene_name=scene_name,
            src_img_dir=scene_test_root,
            img_prefix="gt_image_",
            cam_dir=scene_test_root,
            cam_prefix="gt_camera_",
            output_root=output_root,
            split="test",
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
