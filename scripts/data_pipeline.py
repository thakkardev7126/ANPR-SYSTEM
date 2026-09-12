"""Dataset acquisition and preparation helpers for Indian ANPR experiments.

This module deliberately does not silently download gated datasets. Provide a
public URL or a configured Kaggle/Hugging Face source explicitly, and record
source metadata beside the downloaded archive.
"""
import argparse
import hashlib
import json
import os
import pathlib
import urllib.request
from urllib.parse import urlparse
from datetime import datetime, timezone


DEFAULT_MANIFEST = {
    "sources": [
        {
            "name": "indian-lpr-in-the-wild",
            "provider": "github/reference",
            "reference": "https://github.com/sanchit2843/Indian_LPR",
            "notes": "The public repo says the full 16,192-image road dataset is not public because of legal restrictions. Use scripts/prepare_indian_lpr_two_class.py after placing an authorized copy locally.",
        },
        {
            "name": "datacluster-indian-number-plates-huggingface",
            "provider": "huggingface",
            "reference": "Dataclusterlabspvtltd/indian-number-plates-dataset",
            "notes": "Integrated by scripts/prepare_datacluster_yolo.py. Public sample ships Pascal VOC XML and is converted to YOLOv8 license_plate labels.",
        },
        {
            "name": "kaggle-kedarsai-indian-license-plates-with-labels",
            "provider": "kaggle",
            "reference": "kedarsai/indian-license-plates-with-labels",
            "notes": "Integrated by scripts/prepare_kaggle_yolo.py as the baseline labelled YOLO source.",
        },
        {
            "name": "roboflow-universe-indian-license-plate",
            "provider": "roboflow",
            "reference": "https://universe.roboflow.com/search?q=indian%20license%20plate",
            "notes": "Integrated by scripts/download_roboflow_yolo.py after ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE, ROBOFLOW_PROJECT, and ROBOFLOW_VERSION are configured.",
        },
        {
            "name": "datacluster-indian-number-plates-github",
            "provider": "github/reference",
            "reference": "https://github.com/datacluster-labs/Indian-Number-Plates-Dataset",
            "notes": "Reference repo for the 20,000+ image DataCluster dataset. Public GitHub content points to sample/full-dataset access; full YOLO/COCO/Pascal export may require contacting DataCluster.",
        },
        {
            "name": "datacluster-indian-licence-plate-image-github",
            "provider": "github/reference",
            "reference": "https://github.com/datacluster-labs/Indian-Licence-Plate-Image-Dataset",
            "notes": "Reference repo for the 6,000+ image DataCluster dataset. Public GitHub content includes sample images; train-ready annotations require the full export.",
        },
    ],
    "plate_formats": [
        "MH12AB1234", "DL01C5678", "KA01AB1234", "BH12AB1234",
        "two-line plates: concatenate OCR lines before validation",
    ],
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_url(url, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = pathlib.Path(urlparse(url).path).name or "dataset.download"
    destination = output_dir / filename
    urllib.request.urlretrieve(url, destination)
    return {"url": url, "path": str(destination), "sha256": sha256(destination)}


def write_manifest(output_dir, fetched):
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(DEFAULT_MANIFEST)
    payload["fetched"] = fetched
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    destination = output_dir / "manifest.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/indian_anpr", help="Dataset output directory")
    parser.add_argument("--url", action="append", default=[], help="Explicit public archive URL; repeatable")
    parser.add_argument("--manifest-only", action="store_true", help="Write source metadata without downloading")
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output)
    fetched = []
    if not args.manifest_only:
        for url in args.url:
            fetched.append(fetch_url(url, output_dir))
    manifest = write_manifest(output_dir, fetched)
    print(f"Wrote {manifest}")
    if not args.url and not args.manifest_only:
        print("No URLs supplied. Use --url for an explicit dataset export.")


if __name__ == "__main__":
    main()
