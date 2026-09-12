"""Vehicle appearance extraction for observation-level Re-ID foundations.

This module does not decide whether two observations are the same vehicle. It
only creates compact, normalized visual descriptors and a similarity primitive
that later phases can use.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass

import cv2
import numpy as np

from app.image_quality import assess_quality


APPEARANCE_EMBEDDING_VERSION = "appearance-v1"
OPENCV_MODEL_NAME = "opencv-hsv-hog-v1"
TORCHVISION_MODEL_NAME = "torchvision-mobilenet-v3-small-imagenet"
TORCHVISION_MODEL_VERSION = "MobileNet_V3_Small_Weights.DEFAULT"
_torch_lock = threading.Lock()
_torch_model = None
_torch_weights = None
_torch_device = None


@dataclass
class AppearanceResult:
    vehicle_type: str | None
    vehicle_color: str | None
    crop: np.ndarray | None
    embedding: list[float] | None
    embedding_model: str | None
    embedding_version: str | None
    quality: float
    available: bool
    error: str | None = None


def crop_vehicle_region(image_bgr, plate_bbox=None):
    """Return a normalized vehicle crop using an existing plate bbox when present."""
    if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
        return None
    height, width = image_bgr.shape[:2]
    if plate_bbox:
        try:
            x = float(plate_bbox.get("x", 0))
            y = float(plate_bbox.get("y", 0))
            w = float(plate_bbox.get("width", 0))
            h = float(plate_bbox.get("height", 0))
        except (TypeError, ValueError, AttributeError):
            x = y = w = h = 0
        if w > 0 and h > 0:
            left = max(0, int(x - 2.2 * w))
            top = max(0, int(y - 5.0 * h))
            right = min(width, int(x + 3.2 * w))
            bottom = min(height, int(y + 2.0 * h))
            crop = image_bgr[top:bottom, left:right]
            if crop.size:
                return crop.copy()
    return image_bgr.copy()


def normalize_vehicle_crop(crop_bgr, size=(224, 224)):
    if crop_bgr is None or getattr(crop_bgr, "size", 0) == 0:
        return None
    height, width = crop_bgr.shape[:2]
    if height < 8 or width < 8:
        return None
    return cv2.resize(crop_bgr, size, interpolation=cv2.INTER_AREA if max(height, width) > max(size) else cv2.INTER_CUBIC)


def classify_vehicle_color(crop_bgr):
    if crop_bgr is None or getattr(crop_bgr, "size", 0) == 0:
        return None
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    masks = {
        "white": cv2.inRange(hsv, np.array([0, 0, 170]), np.array([179, 55, 255])),
        "black": cv2.inRange(hsv, np.array([0, 0, 0]), np.array([179, 255, 55])),
        "gray": cv2.inRange(hsv, np.array([0, 0, 56]), np.array([179, 45, 190])),
        "red": cv2.inRange(hsv, np.array([0, 60, 60]), np.array([10, 255, 255])) |
               cv2.inRange(hsv, np.array([170, 60, 60]), np.array([179, 255, 255])),
        "yellow": cv2.inRange(hsv, np.array([15, 70, 70]), np.array([40, 255, 255])),
        "green": cv2.inRange(hsv, np.array([40, 45, 45]), np.array([95, 255, 255])),
        "blue": cv2.inRange(hsv, np.array([95, 50, 45]), np.array([135, 255, 255])),
    }
    scores = {name: cv2.countNonZero(mask) / float(crop_bgr.shape[0] * crop_bgr.shape[1]) for name, mask in masks.items()}
    color, score = max(scores.items(), key=lambda item: item[1])
    return color if score >= 0.12 else None


def _l2_normalize(vector):
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return None
    return (vector / norm).astype(np.float32)


def _opencv_embedding(crop_bgr):
    crop = normalize_vehicle_crop(crop_bgr)
    if crop is None:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).reshape(-1)
    hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).reshape(-1)
    hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256]).reshape(-1)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    hog = np.zeros(16, dtype=np.float32)
    bins = np.floor((angle % 180) / (180 / len(hog))).astype(np.int32)
    for index in range(len(hog)):
        hog[index] = float(magnitude[bins == index].sum())
    return _l2_normalize(np.concatenate([hist_h, hist_s, hist_v, hog]))


def _load_torchvision_model():
    global _torch_model, _torch_weights, _torch_device
    if _torch_model is not None:
        return _torch_model, _torch_weights, _torch_device
    with _torch_lock:
        if _torch_model is not None:
            return _torch_model, _torch_weights, _torch_device
        import torch
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
        weights = MobileNet_V3_Small_Weights.DEFAULT
        model = mobilenet_v3_small(weights=weights)
        model.classifier = torch.nn.Identity()
        device = torch.device("cuda" if torch.cuda.is_available() and os.getenv("ANPR_APPEARANCE_DEVICE", "auto") != "cpu" else "cpu")
        model.to(device).eval()
        _torch_model, _torch_weights, _torch_device = model, weights, device
        return _torch_model, _torch_weights, _torch_device


def _torchvision_embedding(crop_bgr):
    crop = normalize_vehicle_crop(crop_bgr)
    if crop is None:
        return None
    import torch
    model, weights, device = _load_torchvision_model()
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    transforms = weights.transforms()
    from PIL import Image
    tensor = transforms(Image.fromarray(rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        vector = model(tensor).detach().cpu().numpy().reshape(-1)
    return _l2_normalize(vector)


def create_embedding(crop_bgr):
    backend = os.getenv("ANPR_APPEARANCE_BACKEND", "auto").lower()
    if backend in {"torchvision", "mobilenet", "auto"}:
        try:
            embedding = _torchvision_embedding(crop_bgr)
            if embedding is not None:
                return embedding, TORCHVISION_MODEL_NAME, TORCHVISION_MODEL_VERSION
        except Exception:
            if backend != "auto":
                raise
    embedding = _opencv_embedding(crop_bgr)
    if embedding is None:
        return None, None, None
    return embedding, OPENCV_MODEL_NAME, APPEARANCE_EMBEDDING_VERSION


def serialize_embedding(embedding):
    if embedding is None:
        return None
    return json.dumps([round(float(value), 6) for value in np.asarray(embedding, dtype=np.float32).reshape(-1)])


def deserialize_embedding(value):
    if not value:
        return None
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
        return _l2_normalize(parsed)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def appearance_similarity(left, right):
    left_vector = deserialize_embedding(left)
    right_vector = deserialize_embedding(right)
    if left_vector is None or right_vector is None or left_vector.shape != right_vector.shape:
        return None
    return round(float(np.clip(np.dot(left_vector, right_vector), -1.0, 1.0)), 4)


def analyze_vehicle_appearance(image_path, plate_bbox=None):
    image = cv2.imread(image_path)
    if image is None:
        return AppearanceResult(None, None, None, None, None, None, 0.0, False, "image_unreadable")
    crop = crop_vehicle_region(image, plate_bbox)
    normalized = normalize_vehicle_crop(crop)
    if normalized is None:
        return AppearanceResult(None, None, None, None, None, None, 0.0, False, "invalid_crop")
    quality = assess_quality(normalized)
    if quality["unreadable"]:
        return AppearanceResult(None, classify_vehicle_color(normalized), normalized, None, None, None,
                                quality["score"], False, "low_quality_crop")
    embedding, model, version = create_embedding(normalized)
    return AppearanceResult(
        vehicle_type=None,
        vehicle_color=classify_vehicle_color(normalized),
        crop=normalized,
        embedding=embedding.tolist() if embedding is not None else None,
        embedding_model=model,
        embedding_version=version,
        quality=quality["score"],
        available=embedding is not None,
    )
