from pathlib import Path
import argparse
import os
import re

from ultralytics import YOLO

parser = argparse.ArgumentParser(description="Train/resume the YOLOv8 ANPR plate detector.")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of the dataset to use, useful for CPU smoke tests")
parser.add_argument("--batch", type=int, default=16)
parser.add_argument("--imgsz", type=int, default=640)
parser.add_argument("--dry-run", action="store_true", help="Print selected dataset/model without training")
args = parser.parse_args()

project_root = Path(__file__).resolve().parent.parent
latest_weights_path = project_root / "runs" / "detect" / "indian-plates" / "weights" / "last.pt"
legacy_weights_path = project_root / "runs" / "detect" / "runs" / "indian-plates" / "plate-detector-fast" / "weights" / "last.pt"
weights_path = latest_weights_path if latest_weights_path.exists() else legacy_weights_path
augmented_yaml = project_root / "data" / "plate-dataset" / "anpr-augmented-yolo" / "data.yaml"
combined_yaml = project_root / "data" / "plate-dataset" / "datacluster-roboflow-yolo" / "data.yaml"
roboflow_yaml = project_root / "data" / "plate-dataset" / "roboflow-yolo" / "data.yaml"
datacluster_yaml = project_root / "data" / "plate-dataset" / "datacluster-yolo" / "data.yaml"
two_class_yaml = project_root / "data" / "plate-dataset" / "indian-lpr-two-class" / "data.yaml"
if augmented_yaml.exists():
    default_yaml = augmented_yaml
elif combined_yaml.exists():
    default_yaml = combined_yaml
elif roboflow_yaml.exists():
    default_yaml = roboflow_yaml
elif datacluster_yaml.exists():
    default_yaml = datacluster_yaml
elif two_class_yaml.exists():
    default_yaml = two_class_yaml
else:
    default_yaml = project_root / "data" / "plate-dataset" / "indian-yolo" / "data.yaml"
data_yaml = Path(os.getenv("ANPR_DATA_YAML", default_yaml))
run_project = project_root / "runs" / "detect"
run_name = "indian-plates"  # safe local output directory

dataset_text = data_yaml.read_text(encoding="utf-8")
dataset_classes = set(re.findall(r":\s*(plate_single_line|plate_double_line|license_plate)\b", dataset_text))
print(f"Selected dataset: {data_yaml}")
if {"plate_single_line", "plate_double_line"}.issubset(dataset_classes):
    print("Training a two-class plate layout detector: single-line + double-line.")
elif "license_plate" in dataset_classes:
    print("Training the current one-class detector: license_plate.")
    print("To train native layout classes, provide a YOLO data.yaml with plate_single_line and plate_double_line labels via ANPR_DATA_YAML.")
else:
    print(f"Could not identify plate class names in {data_yaml}; Ultralytics will validate the dataset.")

if args.dry_run:
    print("Dry run only; training was not started.")
    raise SystemExit(0)

if weights_path.exists():
    print(f"Loading existing weights from: {weights_path}")
    model = YOLO(str(weights_path))
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        fraction=args.fraction,
        batch=args.batch,
        imgsz=args.imgsz,
        project=str(run_project),
        name=run_name,
        exist_ok=True,
        pretrained=True,
    )
else:
    print(f"Checkpoint not found at {weights_path}. Starting a fresh training run.")
    model = YOLO("yolov8n.pt")
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        fraction=args.fraction,
        batch=args.batch,
        imgsz=args.imgsz,
        project=str(run_project),
        name=run_name,
        exist_ok=True,
    )
