"""
Generate fog/rain/mud/blur/folded variants for a YOLOv8 plate dataset.

Uses Albumentations bbox-aware transforms so license-plate boxes remain valid.

Default input:
    data/plate-dataset/datacluster-roboflow-yolo

Default output:
    data/plate-dataset/anpr-augmented-yolo/data.yaml
"""
from __future__ import annotations

import argparse
import pathlib
import shutil

import albumentations as A
import cv2
import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_dataset(data_yaml: pathlib.Path):
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    root = pathlib.Path(payload.get("path") or data_yaml.parent).resolve()
    names = payload.get("names", {0: "license_plate"})
    return root, names


def yolo_rows(label_path: pathlib.Path):
    rows = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        rows.append((int(float(parts[0])), [float(value) for value in parts[1:5]]))
    return rows


def write_yolo_rows(label_path: pathlib.Path, classes, boxes):
    lines = []
    for class_id, box in zip(classes, boxes):
        x, y, width, height = box
        if width <= 0 or height <= 0:
            continue
        lines.append(f"{int(class_id)} {x:.6f} {y:.6f} {width:.6f} {height:.6f}")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def transform_for(kind: str):
    bbox_params = A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.25, clip=True)
    common = {
        "fog": A.Compose([
            A.RandomFog(fog_coef_range=(0.15, 0.45), alpha_coef=0.12, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.18, p=0.8),
        ], bbox_params=bbox_params),
        "rain": A.Compose([
            A.RandomRain(drop_length=12, drop_width=1, blur_value=4, brightness_coefficient=0.78, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=(-0.10, 0.05), contrast_limit=0.12, p=0.7),
        ], bbox_params=bbox_params),
        "mud": A.Compose([
            A.Spatter(mean=(0.45, 0.65), std=(0.15, 0.30), gauss_sigma=(2.0, 4.0), intensity=(0.25, 0.55), mode="mud", p=1.0),
            A.CoarseDropout(num_holes_range=(2, 7), hole_height_range=(0.02, 0.08), hole_width_range=(0.02, 0.12), fill=45, p=0.8),
        ], bbox_params=bbox_params),
        "blur": A.Compose([
            A.OneOf([
                A.MotionBlur(blur_limit=(5, 13), p=1.0),
                A.Defocus(radius=(2, 5), alias_blur=(0.1, 0.35), p=1.0),
            ], p=1.0),
            A.ImageCompression(quality_range=(35, 75), p=0.8),
        ], bbox_params=bbox_params),
        "folded": A.Compose([
            A.Perspective(scale=(0.04, 0.12), keep_size=True, fit_output=False, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.18, p=0.6),
            A.ElasticTransform(alpha=35, sigma=5, p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.22, p=0.8),
        ], bbox_params=bbox_params),
    }
    return common[kind]


def copy_original(image_path, label_path, output_root, split, prefix):
    target_image = output_root / "images" / split / f"{prefix}{image_path.name}"
    target_label = output_root / "labels" / split / f"{prefix}{image_path.stem}.txt"
    shutil.copy2(image_path, target_image)
    shutil.copy2(label_path, target_label)


def augment_one(image_path, label_path, output_root, split, kind, index):
    image = cv2.imread(str(image_path))
    if image is None:
        return 0
    rows = yolo_rows(label_path)
    if not rows:
        return 0
    classes = [row[0] for row in rows]
    boxes = [row[1] for row in rows]
    transformed = transform_for(kind)(image=image, bboxes=boxes, class_labels=classes)
    out_boxes = transformed["bboxes"]
    out_classes = transformed["class_labels"]
    if not out_boxes:
        return 0

    target_stem = f"{image_path.stem}_{kind}_{index}"
    target_image = output_root / "images" / split / f"{target_stem}.jpg"
    target_label = output_root / "labels" / split / f"{target_stem}.txt"
    cv2.imwrite(str(target_image), transformed["image"])
    return 1 if write_yolo_rows(target_label, out_classes, out_boxes) else 0


def write_data_yaml(output_root: pathlib.Path, names):
    if isinstance(names, list):
        name_lines = [f"  {index}: {name}" for index, name in enumerate(names)]
    else:
        name_lines = [f"  {int(index)}: {name}" for index, name in names.items()]
    (output_root / "data.yaml").write_text(
        "\n".join([
            f"path: {output_root.as_posix()}",
            "train: images/train",
            "val: images/val",
            "names:",
            *name_lines,
            "",
        ]),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/plate-dataset/datacluster-roboflow-yolo/data.yaml")
    parser.add_argument("--output", default="data/plate-dataset/anpr-augmented-yolo")
    parser.add_argument("--per-kind", type=int, default=1, help="Variants per augmentation kind for each training image")
    parser.add_argument("--kinds", default="fog,rain,mud,blur,folded")
    parser.add_argument("--include-original", action="store_true", default=True)
    args = parser.parse_args()

    source_root, names = load_dataset(pathlib.Path(args.data).resolve())
    output_root = pathlib.Path(args.output).resolve()
    if output_root.exists():
        shutil.rmtree(output_root)
    for split in ("train", "val"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    kinds = [kind.strip() for kind in args.kinds.split(",") if kind.strip()]
    invalid = [kind for kind in kinds if kind not in {"fog", "rain", "mud", "blur", "folded"}]
    if invalid:
        raise SystemExit(f"Unknown augmentation kind(s): {', '.join(invalid)}")

    original_count = 0
    augmented_count = 0
    for split in ("train", "val"):
        image_dir = source_root / "images" / split
        label_dir = source_root / "labels" / split
        if not image_dir.exists() or not label_dir.exists():
            continue
        for image_path in image_dir.iterdir():
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            if args.include_original:
                copy_original(image_path, label_path, output_root, split, "orig_")
                original_count += 1
            if split != "train":
                continue
            for kind in kinds:
                for index in range(args.per_kind):
                    augmented_count += augment_one(image_path, label_path, output_root, split, kind, index + 1)

    write_data_yaml(output_root, names)
    print(f"Wrote {output_root / 'data.yaml'}")
    print(f"Original images copied: {original_count}")
    print(f"Augmented train images generated: {augmented_count}")


if __name__ == "__main__":
    main()
