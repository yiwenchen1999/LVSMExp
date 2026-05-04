import argparse
import json
import os
from pathlib import Path


def read_source_metadata_paths(input_root: Path, split: str):
    full_list = input_root / split / "full_list.txt"
    if full_list.exists():
        paths = []
        for line in full_list.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            p = Path(line)
            if p.exists():
                paths.append(p)
        if paths:
            return paths

    metadata_dir = input_root / split / "metadata"
    if not metadata_dir.exists():
        return []
    return sorted(metadata_dir.glob("*.json"))


def infer_tag(input_root: Path):
    name = input_root.name
    for tag in ("far", "near", "normal"):
        if name.endswith(f"_{tag}") or f"robustTest_{tag}" in name:
            return tag
    return name.replace("/", "_")


def rewrite_frame_image_path(frame, source_root: Path, split: str, scene_name: str, prefer_image_tar: bool):
    image_name = os.path.basename(frame.get("image_path", ""))
    if not image_name:
        return frame

    frame_new = dict(frame)
    tar_path = source_root / split / "images" / f"{scene_name}.tar"
    image_path = source_root / split / "images" / scene_name / image_name

    if prefer_image_tar and tar_path.exists():
        frame_new["image_path"] = f"{tar_path.resolve()}::{image_name}"
    else:
        frame_new["image_path"] = str(image_path.resolve())
    return frame_new


def main():
    parser = argparse.ArgumentParser(
        description="Merge robustness metadata/full_list from far/near/normal outputs into one merged full_list."
    )
    parser.add_argument(
        "--input-roots",
        nargs="+",
        required=True,
        help="Input roots of robustness outputs (e.g. ..._far ..._near ..._normal).",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Output root for merged metadata/full_list.",
    )
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument(
        "--prefer-image-tar",
        action="store_true",
        help="If set, write image_path as '<images_tar_abs_path>::<filename>' when tar exists.",
    )
    args = parser.parse_args()

    input_roots = [Path(p) for p in args.input_roots]
    output_root = Path(args.output_root)
    output_metadata_dir = output_root / args.split / "metadata"
    output_metadata_dir.mkdir(parents=True, exist_ok=True)

    merged_paths = []
    seen_names = set()

    for input_root in input_roots:
        tag = infer_tag(input_root)
        src_json_paths = read_source_metadata_paths(input_root, args.split)
        print(f"[merge] {input_root} ({tag}): {len(src_json_paths)} metadata files")

        for src_json in src_json_paths:
            with open(src_json, "r") as f:
                data = json.load(f)

            src_scene_name = data.get("scene_name", src_json.stem)
            merged_scene_name = f"{src_scene_name}__{tag}"
            if merged_scene_name in seen_names:
                raise RuntimeError(f"Duplicate merged scene name detected: {merged_scene_name}")
            seen_names.add(merged_scene_name)

            frames = data.get("frames", [])
            frames_new = [
                rewrite_frame_image_path(
                    frame=frame,
                    source_root=input_root,
                    split=args.split,
                    scene_name=src_scene_name,
                    prefer_image_tar=args.prefer_image_tar,
                )
                for frame in frames
            ]

            data_new = dict(data)
            data_new["scene_name"] = merged_scene_name
            data_new["frames"] = frames_new

            out_json = output_metadata_dir / f"{merged_scene_name}.json"
            with open(out_json, "w") as f:
                json.dump(data_new, f, indent=2)
            merged_paths.append(out_json.resolve())

    full_list_path = output_root / args.split / "full_list.txt"
    full_list_path.parent.mkdir(parents=True, exist_ok=True)
    with open(full_list_path, "w") as f:
        for p in sorted(merged_paths):
            f.write(f"{p}\n")

    print(f"[merge] wrote {len(merged_paths)} entries to {full_list_path}")
    print(f"[merge] merged metadata at {output_metadata_dir}")


if __name__ == "__main__":
    main()
