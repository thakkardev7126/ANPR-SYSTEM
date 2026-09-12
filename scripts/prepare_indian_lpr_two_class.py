"""
Prepare an Indian_LPR-style dataset as YOLOv8 two-class plate layout data.

Indian_LPR's public repository does not publish the full road dataset. After
you place an authorized copy locally, run this converter against its root or
annotation file. It preserves the existing one-class dataset and writes a new:

    data/plate-dataset/indian-lpr-two-class/data.yaml

Supported annotations:
- CSV with image path columns plus either four-point columns, bbox columns, or
  one row per character/plate.
- COCO-style JSON with images/annotations and bbox or segmentation.

Class mapping:
0 = plate_single_line
1 = plate_double_line
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import shutil
from collections import defaultdict

import cv2
import pandas as pd
from sklearn.model_selection import train_test_split


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGE_COLUMNS = ("image", "image_path", "img", "img_path", "filename", "file", "file_name", "path")
LAYOUT_COLUMNS = ("layout", "plate_layout", "class", "class_name", "label", "type")
TEXT_COLUMNS = ("plate_text", "text", "number", "registration", "license_plate")


def find_images(root: pathlib.Path) -> dict[str, pathlib.Path]:
    images = {}
    for image_path in root.rglob("*"):
        if image_path.suffix.lower() in IMAGE_SUFFIXES:
            images[image_path.name] = image_path
            images[image_path.stem] = image_path
            try:
                images[str(image_path.relative_to(root)).replace("\\", "/")] = image_path
            except ValueError:
                pass
    return images


def find_annotation(source: pathlib.Path, explicit: str | None) -> pathlib.Path:
    if explicit:
        path = pathlib.Path(explicit)
        if not path.is_absolute():
            path = source / path
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    candidates = []
    for path in source.rglob("*"):
        if path.suffix.lower() in {".csv", ".json"}:
            score = sum(token in path.name.lower() for token in ("annot", "label", "plate", "bbox", "train"))
            candidates.append((score, path))
    if not candidates:
        raise FileNotFoundError("No CSV/JSON annotation file found. Pass --annotations explicitly.")
    return sorted(candidates, key=lambda item: (-item[0], len(str(item[1]))))[0][1]


def image_size(image_path: pathlib.Path) -> tuple[int, int]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    height, width = image.shape[:2]
    return width, height


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def yolo_box_from_xyxy(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> tuple[float, float, float, float]:
    x1, x2 = sorted((clamp(x1, 0, width - 1), clamp(x2, 0, width - 1)))
    y1, y2 = sorted((clamp(y1, 0, height - 1), clamp(y2, 0, height - 1)))
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    return ((x1 + x2) / 2 / width, (y1 + y2) / 2 / height, box_w / width, box_h / height)


def layout_id_from_values(layout_value=None, text_value=None, box_w=None, box_h=None) -> int:
    value = f"{layout_value or ''} {text_value or ''}".lower()
    if any(token in value for token in ("double", "two", "2line", "2-line", "multi", "non-hsrp", "\n")):
        return 1
    if any(token in value for token in ("single", "one", "1line", "1-line", "hsrp")):
        return 0
    if box_w and box_h:
        return 1 if (box_w / max(1.0, box_h)) < 2.9 else 0
    return 0


def resolve_image(row, images: dict[str, pathlib.Path]) -> pathlib.Path | None:
    for column in IMAGE_COLUMNS:
        if column in row and pd.notna(row[column]):
            value = str(row[column]).replace("\\", "/")
            return images.get(value) or images.get(pathlib.Path(value).name) or images.get(pathlib.Path(value).stem)
    return None


def row_text(row):
    for column in TEXT_COLUMNS:
        if column in row and pd.notna(row[column]):
            return str(row[column])
    return None


def row_layout(row):
    for column in LAYOUT_COLUMNS:
        if column in row and pd.notna(row[column]):
            return str(row[column])
    return None


def parse_csv_annotations(annotation_path: pathlib.Path, images: dict[str, pathlib.Path]):
    table = pd.read_csv(annotation_path)
    table.columns = [str(column).strip().lower() for column in table.columns]
    records = []

    point_sets = [
        ("x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"),
        ("tl_x", "tl_y", "tr_x", "tr_y", "br_x", "br_y", "bl_x", "bl_y"),
    ]
    bbox_sets = [
        ("xmin", "ymin", "xmax", "ymax"),
        ("x_min", "y_min", "x_max", "y_max"),
        ("left", "top", "right", "bottom"),
        ("x", "y", "width", "height"),
    ]

    grouped = defaultdict(list)
    for _, row in table.iterrows():
        row_dict = row.to_dict()
        image_path = resolve_image(row_dict, images)
        if image_path is not None:
            grouped[image_path].append(row_dict)

    for image_path, rows in grouped.items():
        width, height = image_size(image_path)
        for row in rows:
            coords = None
            for names in point_sets:
                if all(name in row and pd.notna(row[name]) for name in names):
                    xs = [float(row[name]) for name in names[0::2]]
                    ys = [float(row[name]) for name in names[1::2]]
                    coords = (min(xs), min(ys), max(xs), max(ys))
                    break
            if coords is None:
                for names in bbox_sets:
                    if all(name in row and pd.notna(row[name]) for name in names):
                        values = [float(row[name]) for name in names]
                        if names == ("x", "y", "width", "height"):
                            coords = (values[0], values[1], values[0] + values[2], values[1] + values[3])
                        else:
                            coords = tuple(values)
                        break
            if coords is None:
                continue

            x1, y1, x2, y2 = coords
            class_id = layout_id_from_values(row_layout(row), row_text(row), x2 - x1, y2 - y1)
            records.append((image_path, class_id, yolo_box_from_xyxy(x1, y1, x2, y2, width, height)))

    return records


def parse_coco_annotations(annotation_path: pathlib.Path, images: dict[str, pathlib.Path]):
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    image_by_id = {}
    for item in payload.get("images", []):
        filename = item.get("file_name") or item.get("filename")
        if filename:
            image_by_id[item["id"]] = images.get(filename) or images.get(pathlib.Path(filename).name)

    records = []
    for ann in payload.get("annotations", []):
        image_path = image_by_id.get(ann.get("image_id"))
        if image_path is None:
            continue
        width, height = image_size(image_path)
        if ann.get("bbox"):
            x, y, box_w, box_h = [float(value) for value in ann["bbox"][:4]]
            coords = (x, y, x + box_w, y + box_h)
        elif ann.get("segmentation"):
            points = ann["segmentation"][0] if isinstance(ann["segmentation"], list) else ann["segmentation"]
            xs = [float(value) for value in points[0::2]]
            ys = [float(value) for value in points[1::2]]
            coords = (min(xs), min(ys), max(xs), max(ys))
        else:
            continue
        x1, y1, x2, y2 = coords
        class_id = layout_id_from_values(ann.get("layout") or ann.get("category_name"), ann.get("text"), x2 - x1, y2 - y1)
        records.append((image_path, class_id, yolo_box_from_xyxy(x1, y1, x2, y2, width, height)))
    return records


def write_yolo_dataset(records, output_dir: pathlib.Path, val_size: float):
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    image_paths = sorted({record[0] for record in records})
    train_paths, val_paths = train_test_split(image_paths, test_size=val_size, random_state=42) if len(image_paths) > 1 else (image_paths, [])
    split_by_path = {path: "train" for path in train_paths}
    split_by_path.update({path: "val" for path in val_paths})
    records_by_image = defaultdict(list)
    for image_path, class_id, box in records:
        records_by_image[image_path].append((class_id, box))

    for image_path, labels in records_by_image.items():
        split = split_by_path[image_path]
        target_image = output_dir / "images" / split / image_path.name
        shutil.copy2(image_path, target_image)
        label_lines = [
            f"{class_id} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}"
            for class_id, box in labels
        ]
        (output_dir / "labels" / split / f"{image_path.stem}.txt").write_text("\n".join(label_lines) + "\n", encoding="utf-8")

    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(
        "\n".join([
            f"path: {output_dir.as_posix()}",
            "train: images/train",
            "val: images/val",
            "names:",
            "  0: plate_single_line",
            "  1: plate_double_line",
            "",
        ]),
        encoding="utf-8",
    )
    return data_yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/plate-dataset/Indian_LPR", help="Authorized Indian_LPR dataset root")
    parser.add_argument("--annotations", help="CSV/JSON annotation path, relative to --source or absolute")
    parser.add_argument("--output", default="data/plate-dataset/indian-lpr-two-class", help="YOLOv8 output folder")
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation split size")
    args = parser.parse_args()

    source = pathlib.Path(args.source).resolve()
    output = pathlib.Path(args.output).resolve()
    if not source.exists():
        raise SystemExit(f"Dataset source not found: {source}")

    images = find_images(source)
    if not images:
        raise SystemExit(f"No images found under: {source}")

    annotation_path = find_annotation(source, args.annotations)
    if annotation_path.suffix.lower() == ".json":
        records = parse_coco_annotations(annotation_path, images)
    else:
        records = parse_csv_annotations(annotation_path, images)

    if not records:
        raise SystemExit(f"No plate boxes could be converted from: {annotation_path}")

    data_yaml = write_yolo_dataset(records, output, args.val_size)
    single = sum(1 for _, class_id, _ in records if class_id == 0)
    double = sum(1 for _, class_id, _ in records if class_id == 1)
    print(f"Wrote {data_yaml}")
    print(f"Converted {len(records)} plates from {annotation_path}")
    print(f"plate_single_line={single}, plate_double_line={double}")


if __name__ == "__main__":
    main()
