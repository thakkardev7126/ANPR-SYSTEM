"""Privacy-safe image derivatives for ANPR evidence.

This layer is intentionally separate from plate detection/OCR. It masks
faces/persons for display while keeping the original frame available only
through RBAC-protected evidence access.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PrivacyResult:
    privacy_image_path: str | None
    status: str
    metadata: dict


def _clamp_box(box, width, height):
    x = max(0, int(box.get("x", 0)))
    y = max(0, int(box.get("y", 0)))
    w = max(0, int(box.get("width", 0)))
    h = max(0, int(box.get("height", 0)))
    x2 = min(width, x + w)
    y2 = min(height, y + h)
    if x >= x2 or y >= y2:
        return None
    return {"x": x, "y": y, "width": x2 - x, "height": y2 - y}


def detect_faces(image_bgr):
    """Detect faces with OpenCV's bundled Haar cascade when available."""
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        return []
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    boxes = detector.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(24, 24))
    return [{"x": int(x), "y": int(y), "width": int(w), "height": int(h)} for x, y, w, h in boxes]


def detect_persons(image_bgr):
    """Detect pedestrian/person regions using OpenCV's pretrained HOG detector."""
    height, width = image_bgr.shape[:2]
    if height < 96 or width < 64:
        return []
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    boxes, weights = hog.detectMultiScale(
        image_bgr,
        winStride=(8, 8),
        padding=(16, 16),
        scale=1.05,
    )
    regions = []
    for index, (x, y, w, h) in enumerate(boxes):
        confidence = float(weights[index]) if index < len(weights) else 0.0
        if confidence < 0.0:
            continue
        regions.append({"x": int(x), "y": int(y), "width": int(w), "height": int(h), "confidence": confidence})
    return regions


def _pixelate_region(image, box, blocks=9):
    x, y, w, h = box["x"], box["y"], box["width"], box["height"]
    roi = image[y:y + h, x:x + w]
    if roi.size == 0:
        return
    small_w = max(1, w // blocks)
    small_h = max(1, h // blocks)
    reduced = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    image[y:y + h, x:x + w] = cv2.resize(reduced, (w, h), interpolation=cv2.INTER_NEAREST)
    blurred = cv2.GaussianBlur(image[y:y + h, x:x + w], (0, 0), sigmaX=8, sigmaY=8)
    image[y:y + h, x:x + w] = blurred


def _restore_plate_regions(masked, original, plate_boxes):
    height, width = original.shape[:2]
    for raw_box in plate_boxes or []:
        box = _clamp_box(raw_box, width, height)
        if not box:
            continue
        pad_x = max(3, int(box["width"] * 0.08))
        pad_y = max(3, int(box["height"] * 0.18))
        x1 = max(0, box["x"] - pad_x)
        y1 = max(0, box["y"] - pad_y)
        x2 = min(width, box["x"] + box["width"] + pad_x)
        y2 = min(height, box["y"] + box["height"] + pad_y)
        masked[y1:y2, x1:x2] = original[y1:y2, x1:x2]


def mask_privacy_regions(image_bgr, plate_boxes=None):
    """Return a masked copy and metadata for an in-memory frame."""
    height, width = image_bgr.shape[:2]
    masked = image_bgr.copy()
    face_boxes = [_clamp_box(box, width, height) for box in detect_faces(image_bgr)]
    person_boxes = [_clamp_box(box, width, height) for box in detect_persons(image_bgr)]
    face_boxes = [box for box in face_boxes if box]
    person_boxes = [box for box in person_boxes if box]
    for box in person_boxes:
        _pixelate_region(masked, box, blocks=7)
    for box in face_boxes:
        _pixelate_region(masked, box, blocks=5)
    _restore_plate_regions(masked, image_bgr, plate_boxes or [])
    return masked, {
        "faces": len(face_boxes),
        "persons": len(person_boxes),
        "plate_regions_preserved": len(plate_boxes or []),
        "method": "opencv-haar-face-and-hog-person",
        "processed_at": dt.datetime.utcnow().isoformat() + "Z",
    }


def create_privacy_safe_derivative(source_path, output_path, plate_boxes=None):
    """Create a privacy-safe display image and never raise into the ANPR path."""
    try:
        original = cv2.imread(source_path)
        if original is None:
            return PrivacyResult(None, "source_unreadable", {"error": "source_unreadable"})
        masked, metadata = mask_privacy_regions(original, plate_boxes or [])

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if not cv2.imwrite(output_path, masked):
            return PrivacyResult(None, "write_failed", {"error": "write_failed"})
        return PrivacyResult(
            output_path,
            "processed",
            metadata,
        )
    except Exception as exc:  # pragma: no cover - exercised through failure-safety tests
        return PrivacyResult(None, "failed", {"error": str(exc)})


def metadata_json(metadata):
    return json.dumps(metadata or {}, sort_keys=True)
