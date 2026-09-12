"""
Prepare Kaggle "kedarsai/indian-license-plates-with-labels" as YOLOv8.

The project already has a downloaded/extracted copy under:
    data/plate-dataset/kaggle/extracted

This script also supports downloading through kagglehub when credentials are
configured, then writes:
    data/plate-dataset/kaggle-yolo/data.yaml
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import zipfile

from sklearn.model_selection import train_test_split


DATASET_SLUG = "kedarsai/indian-license-plates-with-labels"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def maybe_download(raw_dir: pathlib.Path):
    import kagglehub

    downloaded = pathlib.Path(kagglehub.dataset_download(DATASET_SLUG))
    raw_dir.mkdir(parents=True, exist_ok=True)
    for item in downloaded.iterdir():
        target = raw_dir / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)
    return raw_dir


def maybe_extract(raw_dir: pathlib.Path, extracted_dir: pathlib.Path):
    if (extracted_dir / "images").exists() and (extracted_dir / "labels").exists():
        return extracted_dir
    archives = sorted(raw_dir.glob("*.zip"))
    if not archives:
        return raw_dir
    extracted_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archives[0]) as archive:
        archive.extractall(extracted_dir)
    return extracted_dir


def find_source_root(path: pathlib.Path):
    if (path / "images").exists() and (path / "labels").exists():
        return path
    nested = [candidate for candidate in path.rglob("*") if (candidate / "images").exists() and (candidate / "labels").exists()]
    if not nested:
        raise FileNotFoundError(f"No images/labels YOLO folders found under {path}")
    return sorted(nested, key=lambda item: len(item.parts))[0]


def write_dataset(source_root: pathlib.Path, output_dir: pathlib.Path, val_size: float):
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    flat_images = source_root / "images"
    has_split = any((flat_images / split).exists() for split in ("train", "val", "test"))
    copied = 0

    if has_split:
        split_pairs = [("train", "train"), ("val", "val"), ("test", "val")]
        image_groups = []
        for source_split, target_split in split_pairs:
            for image_path in (flat_images / source_split).glob("*"):
                if image_path.suffix.lower() in IMAGE_SUFFIXES:
                    image_groups.append((image_path, target_split))
    else:
        images = sorted(path for path in flat_images.glob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
        train_images, val_images = train_test_split(images, test_size=val_size, random_state=42) if len(images) > 1 else (images, [])
        image_groups = [(path, "train") for path in train_images] + [(path, "val") for path in val_images]

    for image_path, target_split in image_groups:
        label_path = source_root / "labels" / f"{image_path.stem}.txt"
        if not label_path.exists():
            label_path = source_root / "labels" / target_split / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        shutil.copy2(image_path, output_dir / "images" / target_split / image_path.name)
        shutil.copy2(label_path, output_dir / "labels" / target_split / f"{image_path.stem}.txt")
        copied += 1

    if copied == 0:
        raise SystemExit(f"No labelled images copied from {source_root}")

    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(
        "\n".join([
            f"path: {output_dir.as_posix()}",
            "train: images/train",
            "val: images/val",
            "names:",
            "  0: license_plate",
            "",
        ]),
        encoding="utf-8",
    )
    return data_yaml, copied


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/plate-dataset/kaggle")
    parser.add_argument("--extracted-dir", default="data/plate-dataset/kaggle/extracted")
    parser.add_argument("--output", default="data/plate-dataset/kaggle-yolo")
    parser.add_argument("--download", action="store_true", help="Download via kagglehub before converting")
    parser.add_argument("--val-size", type=float, default=0.2)
    args = parser.parse_args()

    raw_dir = pathlib.Path(args.raw_dir).resolve()
    extracted_dir = pathlib.Path(args.extracted_dir).resolve()
    output_dir = pathlib.Path(args.output).resolve()
    if args.download:
        maybe_download(raw_dir)
    source_root = find_source_root(maybe_extract(raw_dir, extracted_dir))
    data_yaml, copied = write_dataset(source_root, output_dir, args.val_size)
    print(f"Wrote {data_yaml}")
    print(f"Copied {copied} labelled images from {source_root}")


if __name__ == "__main__":
    main()
