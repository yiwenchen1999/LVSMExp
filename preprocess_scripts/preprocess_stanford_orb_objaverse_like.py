#!/usr/bin/env python3
"""
Convert Stanford-ORB scenes into an Objaverse-like layout:

  <output_root>/train/images/<scene>/<frame:05d>.png
  <output_root>/train/metadata/<scene>.json
  <output_root>/train/envmaps/<scene>/<frame:05d>_ldr.png, _hdr.png (if --env-gt-root)
  <output_root>/train/full_list.txt
  <output_root>/test/images/<scene>/<frame:05d>.png
  <output_root>/test/metadata/<scene>.json
  <output_root>/test/envmaps/<scene>/<frame:05d>_ldr.png, _hdr.png (if --env-gt-root)
  <output_root>/test/full_list.txt

Envmaps follow preprocess_objaverse naming (LDR/HDR split via split_envmap_ldr_hdr).
Stanford ORB horizontal rotation matches preprocess_stanford_orb (roll -width/4).

This intentionally does NOT create point_light_rays/.
"""

import argparse
from pathlib import Path

from preprocess_stanford_orb import (
    HAS_IMAGEIO,
    HAS_PYEXR,
    collect_scene_dirs,
    process_scene,
    write_full_list,
)


def parse_args():
    parser = argparse.ArgumentParser("Preprocess Stanford-ORB to Objaverse-like train/test format")
    parser.add_argument(
        "--input-root",
        type=str,
        default="data_samples/stanford_ORB",
        help="Root containing Stanford-ORB scene folders, or a single scene folder.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data_samples/processed_stanford_ORB_objaverse_like",
        help="Output root in Objaverse-like format (no point_light_rays).",
    )
    parser.add_argument("--target-size", type=int, default=512, help="Output square image size.")
    parser.add_argument("--target-fov", type=float, default=30.0, help="Target horizontal FOV in degrees.")
    parser.add_argument(
        "--adjust-fov",
        action="store_true",
        help="If set, adjust framing from source FOV to target FOV before resize.",
    )
    parser.add_argument(
        "--no-adjust-fov",
        action="store_false",
        dest="adjust_fov",
        help="Explicitly disable FOV adjustment (resize/crop scaling only).",
    )
    parser.set_defaults(adjust_fov=False)
    parser.add_argument(
        "--env-gt-root",
        type=str,
        default="data_samples/stanford_ORB_gt",
        help="GT root with per-scene env EXRs: <root>/<scene>/env_map/<frame_stem>.exr "
        "(same stems as image filenames in transforms). Use empty string to skip envmaps.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    print(f"[Info] input_root={input_root}")
    print(f"[Info] output_root={output_root}")
    print("[Info] point_light_rays folder is intentionally omitted.")

    env_gt_root = None
    if args.env_gt_root and str(args.env_gt_root).strip():
        env_gt_path = Path(args.env_gt_root).expanduser().resolve()
        if not env_gt_path.exists():
            print(f"[Warn] env-gt-root does not exist: {env_gt_path}, skipping envmap processing")
        elif not (HAS_PYEXR or HAS_IMAGEIO):
            print("[Warn] Neither pyexr nor imageio available, skipping envmap processing")
        else:
            env_gt_root = env_gt_path
            print(f"[Info] env_gt_root={env_gt_root}")
    else:
        print("[Info] envmap export disabled (--env-gt-root empty).")

    for split in ["train", "test"]:
        scene_dirs = collect_scene_dirs(input_root, split)
        if not scene_dirs:
            print(f"[Warn] No scenes with transforms_{split}.json found under {input_root}")
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
                env_gt_root=env_gt_root,
            )

        write_full_list(output_root, split)

    print("[Done] Stanford-ORB conversion completed.")


if __name__ == "__main__":
    main()
