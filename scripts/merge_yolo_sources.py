"""
Merge YOLOv8 datasets into one training source.

Default inputs:
    data/plate-dataset/kaggle-yolo
    data/plate-dataset/datacluster-yolo
    data/plate-dataset/roboflow-yolo

Output:
    data/plate-dataset/datacluster-roboflow-yolo/data.yaml
"""
from __future__ import annotations

import argparse
import pathlib
import shutil

import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_names(data_yaml: pathlib.Path) -> dict[int, str]:
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = payload.get("names", {0: "license_plate"})
    if isinstance(names, list):
        return {index: name for index, name in enumerate(names)}
    return {int(index): str(name) for index, name in names.items()}


def resolve_dataset_root(dataset_dir: pathlib.Path) -> pathlib.Path:
    if (dataset_dir / "data.yaml").exists():
        return dataset_dir
    nested = sorted(dataset_dir.rglob("data.yaml"), key=lambda path: len(path.parts))
    if not nested:
        raise FileNotFoundError(f"No data.yaml found under {dataset_dir}")
    return nested[0].parent


def remap_class_id(old_id: int, names: dict[int, str], target_names: dict[str, int]) -> int:
    name = names.get(old_id, "license_plate")
    lowered = name.lower()
    if "double" in lowered or "two" in lowered:
        return target_names["plate_double_line"] if "plate_double_line" in target_names else target_names["license_plate"]
    if "single" in lowered:
        return target_names["plate_single_line"] if "plate_single_line" in target_names else target_names["license_plate"]
    return target_names.get(name, target_names["license_plate"])


def copy_split(source_root: pathlib.Path, output_dir: pathlib.Path, source_index: int, target_names: dict[str, int]):
    names = load_names(source_root / "data.yaml")
    copied = 0
    for split in ("train", "val", "test"):
        source_images = source_root / "images" / split
        source_labels = source_root / "labels" / split
        if not source_images.exists() or not source_labels.exists():
            continue
        target_split = "val" if split == "test" else split
        (output_dir / "images" / target_split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / target_split).mkdir(parents=True, exist_ok=True)

        for image_path in source_images.iterdir():
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label_path = source_labels / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            target_stem = f"s{source_index}_{image_path.stem}"
            shutil.copy2(image_path, output_dir / "images" / target_split / f"{target_stem}{image_path.suffix.lower()}")
            remapped_lines = []
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                class_id = remap_class_id(int(float(parts[0])), names, target_names)
                remapped_lines.append(" ".join([str(class_id), *parts[1:5]]))
            if remapped_lines:
                (output_dir / "labels" / target_split / f"{target_stem}.txt").write_text("\n".join(remapped_lines) + "\n", encoding="utf-8")
                copied += 1
    return copied


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", default=[
        "data/plate-dataset/kaggle-yolo",
        "data/plate-dataset/datacluster-yolo",
        "data/plate-dataset/roboflow-yolo",
    ], help="YOLO dataset root; repeatable")
    parser.add_argument("--output", default="data/plate-dataset/datacluster-roboflow-yolo")
    parser.add_argument("--two-class", action="store_true", help="Keep single/double-line classes if present")
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output).resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)

    if args.two_class:
        target_names = {"plate_single_line": 0, "plate_double_line": 1, "license_plate": 0}
        yaml_names = ["plate_single_line", "plate_double_line"]
    else:
        target_names = {"license_plate": 0, "plate_single_line": 0, "plate_double_line": 0}
        yaml_names = ["license_plate"]

    total = 0
    used_sources = []
    for index, source in enumerate(args.source, start=1):
        source_path = pathlib.Path(source).resolve()
        if not source_path.exists():
            print(f"Skipping missing source: {source_path}")
            continue
        try:
            root = resolve_dataset_root(source_path)
        except FileNotFoundError as exc:
            print(f"Skipping source without data.yaml: {source_path} ({exc})")
            continue
        count = copy_split(root, output_dir, index, target_names)
        if count:
            total += count
            used_sources.append(str(root))
            print(f"Copied {count} labelled images from {root}")

    if total == 0:
        raise SystemExit("No labelled YOLO images were copied. Download/convert DataCluster and Roboflow first.")

    (output_dir / "data.yaml").write_text(
        "\n".join([
            f"path: {output_dir.as_posix()}",
            "train: images/train",
            "val: images/val",
            "names:",
            *[f"  {index}: {name}" for index, name in enumerate(yaml_names)],
            "",
        ]),
        encoding="utf-8",
    )
    (output_dir / "sources.txt").write_text("\n".join(used_sources) + "\n", encoding="utf-8")
    print(f"Wrote {output_dir / 'data.yaml'}")
    print(f"Total labelled images: {total}")


if __name__ == "__main__":
    main()
