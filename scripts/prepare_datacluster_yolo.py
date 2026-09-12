"""
Download DataCluster's Indian number plate dataset from Hugging Face and
convert its Pascal VOC XML annotations into YOLOv8 format.

Output:
    data/plate-dataset/datacluster-yolo/data.yaml
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict

from huggingface_hub import snapshot_download
from sklearn.model_selection import train_test_split


REPO_ID = "Dataclusterlabspvtltd/indian-number-plates-dataset"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def yolo_line_from_voc_box(box, image_width, image_height):
    xmin = max(0.0, float(box.findtext("xmin", 0)))
    ymin = max(0.0, float(box.findtext("ymin", 0)))
    xmax = min(float(image_width - 1), float(box.findtext("xmax", 0)))
    ymax = min(float(image_height - 1), float(box.findtext("ymax", 0)))
    if xmax <= xmin or ymax <= ymin:
        return None
    cx = ((xmin + xmax) / 2) / image_width
    cy = ((ymin + ymax) / 2) / image_height
    width = (xmax - xmin) / image_width
    height = (ymax - ymin) / image_height
    return f"0 {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}"


def parse_voc_annotations(source_dir: pathlib.Path):
    image_dir = source_dir / "images"
    annotation_dir = source_dir / "Annotations"
    image_by_name = {path.name: path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES}
    records = defaultdict(list)

    for xml_path in annotation_dir.glob("*.xml"):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        filename = root.findtext("filename") or f"{xml_path.stem}.jpg"
        image_path = image_by_name.get(filename) or image_by_name.get(f"{xml_path.stem}.jpg")
        if image_path is None:
            continue
        width = int(float(root.findtext("size/width", 0)))
        height = int(float(root.findtext("size/height", 0)))
        if width <= 0 or height <= 0:
            continue
        for obj in root.findall("object"):
            line = yolo_line_from_voc_box(obj.find("bndbox"), width, height)
            if line:
                records[image_path].append(line)

    return records


def write_yolo_dataset(records, output_dir: pathlib.Path, val_size: float):
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    images = sorted(records)
    train_images, val_images = train_test_split(images, test_size=val_size, random_state=42) if len(images) > 1 else (images, [])
    split_for = {path: "train" for path in train_images}
    split_for.update({path: "val" for path in val_images})

    for image_path, lines in records.items():
        split = split_for[image_path]
        shutil.copy2(image_path, output_dir / "images" / split / image_path.name)
        (output_dir / "labels" / split / f"{image_path.stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

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
    return data_yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/plate-dataset/datacluster-raw", help="Downloaded Hugging Face snapshot folder")
    parser.add_argument("--output", default="data/plate-dataset/datacluster-yolo", help="YOLOv8 output folder")
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--skip-download", action="store_true", help="Convert an already downloaded --raw-dir")
    args = parser.parse_args()

    raw_dir = pathlib.Path(args.raw_dir).resolve()
    output_dir = pathlib.Path(args.output).resolve()

    if not args.skip_download:
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=str(raw_dir),
        )

    records = parse_voc_annotations(raw_dir)
    if not records:
        raise SystemExit(f"No Pascal VOC plate boxes found under {raw_dir}")
    data_yaml = write_yolo_dataset(records, output_dir, args.val_size)
    print(f"Wrote {data_yaml}")
    print(f"Converted {sum(len(lines) for lines in records.values())} plates from {len(records)} images.")


if __name__ == "__main__":
    main()
