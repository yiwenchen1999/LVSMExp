#!/usr/bin/env python3
"""
Convert Stanford-ORB scenes to re10k-like metadata/image format.

Output layout:
  <output_root>/<split>/images/<scene_name>/<frame:05d>.png
  <output_root>/<split>/metadata/<scene_name>.json
  <output_root>/<split>/full_list.txt

Each metadata JSON contains:
  {
    "scene_name": "...",
    "frames": [
      {
        "image_path": "/abs/path/to/image.png",
        "fxfycxcy": [fx, fy, cx, cy],
        "w2c": [[...], [...], [...], [...]]
      }
    ]
  }
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def blender_to_opencv_c2w(c2w_blender: np.ndarray) -> np.ndarray:
    """Convert Blender camera convention to OpenCV camera convention."""
    c2w_opencv = c2w_blender.copy()
    transform = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    c2w_opencv[:3, :3] = c2w_blender[:3, :3] @ transform
    return c2w_opencv


def fov_to_fxfycxcy(fov_degrees: float, image_width: int, image_height: int):
    """Convert horizontal FOV to [fx, fy, cx, cy]."""
    fov_rad = math.radians(fov_degrees)
    fx = fy = (image_width / 2.0) / math.tan(fov_rad / 2.0)
    cx = image_width / 2.0
    cy = image_height / 2.0
    return [fx, fy, cx, cy]


def find_image_path(scene_dir: Path, frame_rel: str) -> Path:
    rel = frame_rel.replace("./", "")
    base = scene_dir / rel
    candidates = []
    if base.suffix:
        candidates.append(base)
    for ext in [".png", ".jpg", ".jpeg", ".webp", ".PNG", ".JPG", ".JPEG", ".WEBP"]:
        candidates.append(Path(f"{base}{ext}"))

    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Cannot find image for frame path '{frame_rel}' under {scene_dir}")


def find_mask_path(scene_dir: Path, frame_rel: str, image_path: Path, split: str) -> Path | None:
    rel = frame_rel.replace("./", "")
    base = scene_dir / rel
    split_name = Path(rel).parts[0] if len(Path(rel).parts) > 1 else ""
    stem = base.name

    # Stanford-ORB canonical structure:
    #   <scene>/<split>/<id>.png
    #   <scene>/<split>_mask/<id>.png
    canonical_mask = scene_dir / f"{split}_mask" / f"{stem}.png"
    if canonical_mask.exists():
        return canonical_mask

    candidates = [
        image_path.with_name(f"{image_path.stem}_mask.png"),
        image_path.with_name(f"{image_path.stem}.mask.png"),
        image_path.with_name(f"{image_path.stem}_alpha.png"),
        base.parent / f"{stem}_mask.png",
        base.parent / f"{stem}.mask.png",
        base.parent / f"mask_{stem}.png",
        scene_dir / "mask" / f"{stem}.png",
        scene_dir / "masks" / f"{stem}.png",
    ]

    if split_name:
        rel_tail = Path(*Path(rel).parts[1:]) if len(Path(rel).parts) > 1 else Path(stem)
        candidates.extend(
            [
                scene_dir / "mask" / split_name / f"{rel_tail}.png",
                scene_dir / "masks" / split_name / f"{rel_tail}.png",
                scene_dir / rel.replace(f"{split_name}/", "mask/"),
                scene_dir / f"{rel.replace(f'{split_name}/', 'mask/')}.png",
                scene_dir / rel.replace(f"{split_name}/", "masks/"),
                scene_dir / f"{rel.replace(f'{split_name}/', 'masks/')}.png",
            ]
        )

    for c in candidates:
        if c.exists():
            return c
    return None


def load_mask(image_path: Path, mask_path: Path | None) -> np.ndarray:
    with Image.open(image_path) as img:
        if "A" in img.getbands():
            alpha = np.array(img.getchannel("A"), dtype=np.uint8)
            return alpha > 0

    if mask_path is None:
        with Image.open(image_path) as img:
            h, w = img.height, img.width
        return np.ones((h, w), dtype=bool)

    with Image.open(mask_path) as m:
        m_arr = np.array(m.convert("L"), dtype=np.uint8)
    return m_arr > 127


def apply_mask_white_background(image_path: Path, mask: np.ndarray) -> np.ndarray:
    with Image.open(image_path) as img:
        rgb = np.array(img.convert("RGB"), dtype=np.uint8)
    if mask.shape[:2] != rgb.shape[:2]:
        raise ValueError(
            f"Mask shape {mask.shape[:2]} does not match image shape {rgb.shape[:2]}: {image_path}"
        )
    out = rgb.copy()
    out[~mask] = 255
    return out


def expand_or_crop_to_target_fov(
    rgb: np.ndarray, source_fov_deg: float, target_fov_deg: float, background_value: int = 255
) -> np.ndarray:
    """
    If source FOV is narrower than target, pad with background.
    If source FOV is wider than target, center-crop.
    """
    h, w = rgb.shape[:2]
    ratio = math.tan(math.radians(target_fov_deg / 2.0)) / max(
        math.tan(math.radians(source_fov_deg / 2.0)), 1e-8
    )

    if abs(ratio - 1.0) < 1e-6:
        return rgb

    if ratio > 1.0:
        new_w = max(1, int(round(w * ratio)))
        new_h = max(1, int(round(h * ratio)))
        canvas = np.full((new_h, new_w, 3), background_value, dtype=np.uint8)
        y0 = (new_h - h) // 2
        x0 = (new_w - w) // 2
        canvas[y0 : y0 + h, x0 : x0 + w] = rgb
        return canvas

    crop_w = max(1, int(round(w * ratio)))
    crop_h = max(1, int(round(h * ratio)))
    x0 = max(0, (w - crop_w) // 2)
    y0 = max(0, (h - crop_h) // 2)
    return rgb[y0 : y0 + crop_h, x0 : x0 + crop_w]


def resize_to_square(rgb: np.ndarray, size: int) -> np.ndarray:
    img = Image.fromarray(rgb)
    img = img.resize((size, size), Image.Resampling.BICUBIC)
    return np.array(img, dtype=np.uint8)


def process_scene(
    scene_dir: Path,
    output_root: Path,
    split: str,
    target_fov: float,
    target_size: int,
    adjust_fov: bool,
):
    transforms_path = scene_dir / f"transforms_{split}.json"
    if not transforms_path.exists():
        return None

    with open(transforms_path, "r") as f:
        tf_data = json.load(f)

    if "frames" not in tf_data or len(tf_data["frames"]) == 0:
        print(f"[Skip] No frames in {transforms_path}")
        return None

    source_fov_deg = math.degrees(float(tf_data["camera_angle_x"]))
    scene_name = scene_dir.name
    out_images_dir = output_root / split / "images" / scene_name
    out_meta_dir = output_root / split / "metadata"
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_meta_dir.mkdir(parents=True, exist_ok=True)

    fxfycxcy = fov_to_fxfycxcy(target_fov, target_size, target_size)
    frames_out = []

    for idx, frame in enumerate(tf_data["frames"]):
        frame_rel = frame["file_path"]
        image_path = find_image_path(scene_dir, frame_rel)
        mask_path = find_mask_path(scene_dir, frame_rel, image_path, split)
        if mask_path is None:
            raise FileNotFoundError(
                f"Mask not found for frame '{frame_rel}' in scene {scene_dir}. "
                f"Expected canonical path: {scene_dir / f'{split}_mask' / (Path(frame_rel).name + '.png')}"
            )

        mask = load_mask(image_path, mask_path)
        rgb = apply_mask_white_background(image_path, mask)
        if adjust_fov:
            rgb = expand_or_crop_to_target_fov(rgb, source_fov_deg, target_fov)
        rgb = resize_to_square(rgb, target_size)

        frame_stem = Path(frame_rel).name
        out_name = f"{frame_stem}.png" if frame_stem.isdigit() else f"{idx:05d}.png"
        out_path = out_images_dir / out_name
        Image.fromarray(rgb).save(out_path)

        c2w_blender = np.array(frame["transform_matrix"], dtype=np.float64)
        c2w_opencv = blender_to_opencv_c2w(c2w_blender)
        w2c = np.linalg.inv(c2w_opencv)

        frames_out.append(
            {
                "image_path": str(out_path.resolve()),
                "fxfycxcy": fxfycxcy,
                "w2c": w2c.tolist(),
            }
        )

    scene_meta = {"scene_name": scene_name, "frames": frames_out}
    meta_path = out_meta_dir / f"{scene_name}.json"
    with open(meta_path, "w") as f:
        json.dump(scene_meta, f, indent=2)

    print(
        f"[Done] {scene_name} ({split}): {len(frames_out)} frames, "
        f"source_fov={source_fov_deg:.3f}, target_fov={target_fov}"
    )
    return scene_name


def collect_scene_dirs(input_root: Path, split: str):
    # Single scene directory
    if (input_root / f"transforms_{split}.json").exists():
        return [input_root]

    scene_dirs = []
    for d in sorted(input_root.iterdir()):
        if not d.is_dir():
            continue
        if (d / f"transforms_{split}.json").exists():
            scene_dirs.append(d)
    return scene_dirs


def write_full_list(output_root: Path, split: str):
    meta_dir = output_root / split / "metadata"
    if not meta_dir.exists():
        return
    json_paths = sorted(meta_dir.glob("*.json"))
    full_list_path = output_root / split / "full_list.txt"
    with open(full_list_path, "w") as f:
        for p in json_paths:
            f.write(f"{p.resolve()}\n")
    print(f"[Done] Wrote {full_list_path} ({len(json_paths)} scenes)")


def parse_args():
    parser = argparse.ArgumentParser("Preprocess Stanford-ORB to re10k-like format")
    parser.add_argument(
        "--input-root",
        type=str,
        default="data_samples/stanford_ORB",
        help="Root containing Stanford-ORB scene folders, or a single scene folder.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data_samples/stanford_ORB_processed",
        help="Output root directory in re10k-like format.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test", "both"],
        help="Which split(s) to process.",
    )
    parser.add_argument("--target-size", type=int, default=512, help="Output square image size.")
    parser.add_argument("--target-fov", type=float, default=30.0, help="Target horizontal FOV in degrees.")
    parser.add_argument(
        "--adjust-fov",
        action="store_true",
        help="If set, adjust image framing from source FOV to target FOV before resize. "
        "Default is disabled (resize only).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not input_root.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_root}")

    splits = ["train", "test"] if args.split == "both" else [args.split]
    for split in splits:
        scene_dirs = collect_scene_dirs(input_root, split)
        if not scene_dirs:
            print(f"[Warn] No scenes found for split='{split}' under {input_root}")
            continue
        print(f"[Info] Processing split='{split}' with {len(scene_dirs)} scene(s)")
        for scene_dir in scene_dirs:
            process_scene(
                scene_dir=scene_dir,
                output_root=output_root,
                split=split,
                target_fov=args.target_fov,
                target_size=args.target_size,
                adjust_fov=args.adjust_fov,
            )
        write_full_list(output_root, split)


if __name__ == "__main__":
    main()
