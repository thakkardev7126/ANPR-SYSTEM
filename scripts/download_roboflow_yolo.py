"""
Download a Roboflow Universe/project version in YOLOv8 format.

Required environment variables:
    ROBOFLOW_API_KEY
    ROBOFLOW_WORKSPACE
    ROBOFLOW_PROJECT
    ROBOFLOW_VERSION

Example:
    $env:ROBOFLOW_API_KEY = "..."
    $env:ROBOFLOW_WORKSPACE = "recommendationsystemlivecamerafeed"
    $env:ROBOFLOW_PROJECT = "indian-number-plates-9oobq-bwb4x"
    $env:ROBOFLOW_VERSION = "1"
    py -3.13 scripts\\download_roboflow_yolo.py
"""
from __future__ import annotations

import argparse
import os
import pathlib


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"Set {name} before downloading from Roboflow.")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/plate-dataset/roboflow-yolo", help="YOLOv8 output folder")
    parser.add_argument("--format", default="yolov8", help="Roboflow export format")
    args = parser.parse_args()

    try:
        from roboflow import Roboflow
    except ImportError as exc:
        raise SystemExit("Install the Roboflow SDK first: pip install roboflow") from exc

    api_key = require_env("ROBOFLOW_API_KEY")
    workspace_name = require_env("ROBOFLOW_WORKSPACE")
    project_name = require_env("ROBOFLOW_PROJECT")
    version_number = int(require_env("ROBOFLOW_VERSION"))
    output_dir = pathlib.Path(args.output).resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace_name).project(project_name)
    version = project.version(version_number)
    dataset = version.download(args.format, location=str(output_dir))
    print(f"Wrote Roboflow dataset to {dataset.location}")


if __name__ == "__main__":
    main()
