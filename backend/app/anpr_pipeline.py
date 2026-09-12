"""
ANPR pipeline: plate localisation (pretrained YOLOv8 detector) + text
recognition (PaddleOCR), with basic cleanup for Indian plate formats.

Why this replaces the old Haar-cascade + Tesseract pipeline:
- Haar cascades are hand-crafted edge filters, not learned from data. They
  fail badly on angled, distant, low-light, or blurry real-world CCTV/ANPR
  footage (exactly the conditions this project needs to handle).
- YOLOv8 here uses a real pretrained license-plate detector (trained on an
  actual plate dataset), which is far more robust to angle/scale/lighting.
- Tesseract is built for scanned documents, not embossed/stylised plate
  fonts. PaddleOCR uses deep text-recognition models that handle this much
  better, especially on tilted or low-contrast plates.

Model weights: the app first checks runs/detect/indian-plates/weights/best.pt
from local training, then falls back to backend/license_plate_detector.pt. Set
ANPR_MODEL_PATH to force another YOLO .pt file.
"""
import cv2
import re
import json
import os
import threading
import time
from queue import Queue, Empty, Full

import numpy as np

from app.plate_rules import (
    AMBIGUOUS_PLATE_CHARS,
    INDIAN_STATE_CODES,
    LETTER_POSITION_CORRECTIONS,
    NUMBER_POSITION_CORRECTIONS,
    OCR_ACCEPT_CONFIDENCE,
    PENDING_REVIEW_STATUS,
    classify_plate_color,
    normalize_plate_text,
    status_for_plate,
    sanitize_plate_text,
    strip_hsrp_noise,
)

from app.image_quality import padded_plate_crop, bbox_frame_edges, assess_quality, enhanced_variants

inference_lock = threading.Lock()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TRAINED_MODEL_PATH = os.path.join(PROJECT_ROOT, "runs", "detect", "indian-plates", "weights", "best.pt")
FALLBACK_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "license_plate_detector.pt")
MODEL_PATH = os.getenv(
    "ANPR_MODEL_PATH",
    TRAINED_MODEL_PATH if os.path.exists(TRAINED_MODEL_PATH) else FALLBACK_MODEL_PATH,
)
os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "BOS")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

plate_detector = None
ocr_reader = None
model_lock = threading.Lock()
ocr_lock = threading.Lock()


def get_plate_detector():
    """Load YOLO lazily so camera/status endpoints stay responsive at startup."""
    global plate_detector
    if plate_detector is None:
        with model_lock:
            if plate_detector is None:
                from ultralytics import YOLO
                plate_detector = YOLO(MODEL_PATH)
    return plate_detector


def _create_paddle_ocr():
    """Create a PaddleOCR reader while supporting both v3 and older v2 args."""
    from paddleocr import PaddleOCR

    stable_kwargs = {
        "lang": "en",
        "ocr_version": os.getenv("PADDLEOCR_VERSION", "PP-OCRv5"),
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        "enable_mkldnn": False,
        "device": os.getenv("PADDLEOCR_DEVICE", "cpu"),
    }
    engine = os.getenv("PADDLEOCR_ENGINE")
    if engine:
        stable_kwargs["engine"] = engine

    try:
        return PaddleOCR(**stable_kwargs)
    except ValueError as exc:
        # Some installed PaddleOCR builds do not expose every engine.
        if "engine" not in str(exc).lower() or "engine" not in stable_kwargs:
            raise
        stable_kwargs.pop("engine", None)
        return PaddleOCR(**stable_kwargs)
    except TypeError:
        return PaddleOCR(lang="en", use_angle_cls=False, show_log=False)


def get_ocr_reader():
    """Load PaddleOCR lazily on first OCR request, not while booting FastAPI."""
    global ocr_reader
    if ocr_reader is None:
        with ocr_lock:
            if ocr_reader is None:
                ocr_reader = _create_paddle_ocr()
    return ocr_reader

CONFUSION_PAIRS = {"6": "G", "0": "O", "O": "0", "1": "I", "I": "1", "8": "B", "B": "8"}

DETECTION_CONF_THRESHOLD = 0.20
MIN_PLATE_SHARPNESS = 100.0
STRICT_INDIAN_PLATE_RE = re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$")
HSRP_NOISE_STRINGS = {"IN", "IND", "INDIA"}
DETECT_IMGSZ = int(os.getenv("ANPR_DETECT_IMGSZ", "960"))
MAX_OCR_DETECTIONS = int(os.getenv("ANPR_MAX_OCR_DETECTIONS", "4"))
FAST_OCR_VARIANTS = int(os.getenv("ANPR_FAST_OCR_VARIANTS", "3"))
MAX_FALLBACK_REGIONS = int(os.getenv("ANPR_MAX_FALLBACK_REGIONS", "3"))


class LatestFrameBuffer:
    """A single-slot buffer that always favors the newest frame."""

    def __init__(self):
        self._queue = Queue(maxsize=1)

    def put(self, frame):
        try:
            self._queue.put_nowait(frame)
        except Full:
            try:
                self._queue.get_nowait()
            except Empty:
                pass
            self._queue.put_nowait(frame)

    def get(self, timeout=0.5):
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None


class MobileStreamReceiver:
    """Non-blocking OpenCV receiver for HTTP/MJPEG and RTSP phone streams."""

    def __init__(self, stream_url, reconnect_delay=2.0):
        self.stream_url = stream_url
        self.reconnect_delay = reconnect_delay
        self.frames = LatestFrameBuffer()
        self._stop_event = threading.Event()
        self._connected = threading.Event()
        self._thread = None
        self._capture = None
        self._latest_frame = None
        self._frame_lock = threading.Lock()

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._capture is not None:
            self._capture.release()
        if self._thread:
            self._thread.join(timeout=2)
        self._connected.clear()

    def get_latest_frame(self, timeout=0.5):
        return self.frames.get(timeout)

    def is_connected(self):
        return self._connected.is_set()

    def _open_capture(self):
        if self.stream_url.lower().startswith("rtsp://"):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        capture = cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def _receive_loop(self):
        while not self._stop_event.is_set():
            self._capture = self._open_capture()
            if not self._capture.isOpened():
                self._capture.release()
                self._connected.clear()
                self._stop_event.wait(self.reconnect_delay)
                continue

            self._connected.set()
            while not self._stop_event.is_set():
                ok, frame = self._capture.read()
                if not ok or frame is None:
                    self._connected.clear()
                    break
                self.frames.put(frame)
                with self._frame_lock:
                    self._latest_frame = frame

            self._capture.release()
            self._capture = None
            self._stop_event.wait(self.reconnect_delay)


class StreamProcessor:
    """Consumes only the newest frame and invokes a callback at a fixed interval."""

    def __init__(self, receiver, callback, process_every=3):
        self.receiver = receiver
        self.callback = callback
        self.process_every = max(1, process_every)
        self._stop_event = threading.Event()
        self._thread = None
        self._frame_number = 0

    def start(self):
        self.receiver.start()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self.receiver.stop()
        if self._thread:
            self._thread.join(timeout=2)

    def _process_loop(self):
        while not self._stop_event.is_set():
            frame = self.receiver.get_latest_frame()
            if frame is None:
                continue
            self._frame_number += 1
            if self._frame_number % self.process_every == 0:
                self.callback(frame, self._frame_number)


def infer_plate_layout(crop_bgr, class_name=None):
    """Detect whether a plate crop should be read as one line or two lines."""
    if class_name:
        lowered = str(class_name).lower()
        if "double" in lowered or "two" in lowered:
            return "double_line"
        if "single" in lowered:
            return "single_line"

    if crop_bgr is None or crop_bgr.size == 0:
        return "unknown"
    height, width = crop_bgr.shape[:2]
    if height == 0:
        return "unknown"
    aspect_ratio = width / float(height)
    return "double_line" if aspect_ratio < 2.9 and height >= 45 else "single_line"


def _class_name_for_box(result, class_id):
    names = getattr(result, "names", None) or getattr(get_plate_detector(), "names", {})
    if isinstance(names, dict):
        return names.get(class_id, str(class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return names[class_id]
    return str(class_id)


def locate_plates(image_bgr):
    """Original-resolution detections, with overlapping tiles for large frames."""
    detector = get_plate_detector()
    ih, iw = image_bgr.shape[:2]
    regions = [(0, 0, image_bgr)]
    if max(ih, iw) > 1600:
        tile = 1280
        xs = sorted(set(list(range(0, max(1, iw-tile), 1024)) + [max(0, iw-tile)]))
        ys = sorted(set(list(range(0, max(1, ih-tile), 1024)) + [max(0, ih-tile)]))
        tiles = [(x, y, image_bgr[y:y+tile, x:x+tile]) for y in ys for x in xs]
        regions.extend(tiles)
    detections = []
    for ox, oy, region in regions:
        with inference_lock:
            results = detector.predict(region, conf=DETECTION_CONF_THRESHOLD, imgsz=DETECT_IMGSZ, verbose=False)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x, y = max(0, int(x1)+ox), max(0, int(y1)+oy)
                right, bottom = min(iw, int(x2)+ox), min(ih, int(y2)+oy)
                width, height = right-x, bottom-y
                if width <= 0 or height <= 0:
                    continue
                class_id = int(box.cls[0]) if getattr(box, "cls", None) is not None else 0
                confidence = float(box.conf[0]) if getattr(box, "conf", None) is not None else 1.0
                class_name = _class_name_for_box(result, class_id)
                detections.append(dict(x=x, y=y, width=width, height=height,
                    confidence=round(confidence, 4), class_id=class_id, class_name=class_name,
                    layout=infer_plate_layout(image_bgr[y:bottom, x:right], class_name)))
    if not detections:
        return []
    keep = cv2.dnn.NMSBoxes([[d["x"], d["y"], d["width"], d["height"]] for d in detections],
                           [d["confidence"] for d in detections], DETECTION_CONF_THRESHOLD, .45)
    return [detections[int(i)] for i in np.asarray(keep).reshape(-1)]


def _rank_plate_detections(detections):
    def score(box):
        area = max(0, box.get("width", 0)) * max(0, box.get("height", 0))
        return (box.get("confidence") or 0.0, area)
    return sorted(detections, key=score, reverse=True)

def detection_bbox(detection):
    if isinstance(detection, dict):
        return detection["x"], detection["y"], detection["width"], detection["height"]
    return detection


def is_frame_sharp(image_crop, threshold=MIN_PLATE_SHARPNESS):
    """Reject blurry plate crops before OCR."""
    if image_crop is None or image_crop.size == 0:
        return False
    gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY) if len(image_crop.shape) == 3 else image_crop
    return cv2.Laplacian(gray, cv2.CV_64F).var() > threshold


def plate_status(confidence):
    """Use the production-style review label expected by the dashboard."""
    return status_for_plate(confidence)


def apply_positional_plate_corrections(text):
    rule = normalize_plate_text(text)
    return rule.normalized_text or strip_hsrp_noise(text)


def enforce_strict_indian_plate_regex(text):
    rule = normalize_plate_text(text)
    return rule.normalized_text if rule.valid_format else None


def clean_plate_text(raw_text):
    """Use one full-string parser; never salvage a plausible substring."""
    rule = normalize_plate_text(raw_text)
    return rule.normalized_text, rule.confidence_hint

def _order_quad_points(points):
    points = points.reshape(4, 2).astype("float32")
    ordered = points
    sums = ordered.sum(axis=1)
    diffs = np.diff(ordered, axis=1).reshape(-1)
    return np.array([
        ordered[np.argmin(sums)],
        ordered[np.argmin(diffs)],
        ordered[np.argmax(sums)],
        ordered[np.argmax(diffs)],
    ], dtype="float32")


def perspective_unwarp_plate(crop_bgr):
    """Flatten a slanted plate crop when a four-corner contour is available."""
    if crop_bgr is None or crop_bgr.size == 0:
        return crop_bgr
    height, width = crop_bgr.shape[:2]
    if height < 20 or width < 60:
        return crop_bgr

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 7, 45, 45)
    edges = cv2.Canny(blurred, 60, 180)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:6]

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < width * height * 0.12:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        if len(approx) != 4:
            continue

        src = _order_quad_points(approx)
        top_width = np.linalg.norm(src[1] - src[0])
        bottom_width = np.linalg.norm(src[2] - src[3])
        left_height = np.linalg.norm(src[3] - src[0])
        right_height = np.linalg.norm(src[2] - src[1])
        target_width = int(max(top_width, bottom_width))
        target_height = int(max(left_height, right_height))
        if target_width < 60 or target_height < 20:
            continue

        dst = np.array([
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ], dtype="float32")
        transform = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(crop_bgr, transform, (target_width, target_height))

    return crop_bgr


def _trim_blank_border(image_bgr):
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(thresholded)
    if coords is None:
        return image_bgr
    x, y, width, height = cv2.boundingRect(coords)
    pad_x, pad_y = max(2, width // 25), max(2, height // 8)
    y1, y2 = max(0, y - pad_y), min(image_bgr.shape[0], y + height + pad_y)
    x1, x2 = max(0, x - pad_x), min(image_bgr.shape[1], x + width + pad_x)
    return image_bgr[y1:y2, x1:x2]


def flatten_double_line_plate(crop_bgr):
    """Split a two-line plate and concatenate both rows into one OCR stream."""
    if crop_bgr is None or crop_bgr.size == 0:
        return crop_bgr
    height, width = crop_bgr.shape[:2]
    if height < 35:
        return crop_bgr

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.equalizeHist(gray)
    _, ink = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    row_density = ink.sum(axis=1)
    lower, upper = int(height * 0.35), int(height * 0.65)
    split = int(np.argmin(row_density[lower:upper]) + lower) if upper > lower else height // 2

    top = _trim_blank_border(crop_bgr[:max(1, split), :])
    bottom = _trim_blank_border(crop_bgr[min(height - 1, split):, :])
    if top is None or bottom is None or top.size == 0 or bottom.size == 0:
        return crop_bgr

    target_height = max(top.shape[0], bottom.shape[0], 24)
    top_width = max(1, int(top.shape[1] * target_height / max(1, top.shape[0])))
    bottom_width = max(1, int(bottom.shape[1] * target_height / max(1, bottom.shape[0])))
    top = cv2.resize(top, (top_width, target_height), interpolation=cv2.INTER_CUBIC)
    bottom = cv2.resize(bottom, (bottom_width, target_height), interpolation=cv2.INTER_CUBIC)
    spacer = np.full((target_height, max(6, target_height // 5), 3), 255, dtype=np.uint8)
    return cv2.hconcat([top, spacer, bottom])


def normalize_plate_crop_for_ocr(crop_bgr, layout=None):
    crop = perspective_unwarp_plate(crop_bgr)
    layout = layout or infer_plate_layout(crop)
    if layout == "double_line":
        crop = flatten_double_line_plate(crop)
    return crop


def prepare_ocr_images(crop_bgr, layout=None):
    crop = normalize_plate_crop_for_ocr(crop_bgr, layout)
    return enhanced_variants(crop)[0]

def _ocr_box_origin(box):
    if box is None:
        return 0.0, 0.0
    box = np.asarray(box, dtype="float32")
    if box.ndim == 1 and box.size >= 4:
        return float(box[0]), float(box[1])
    if box.ndim >= 2 and box.shape[-1] >= 2:
        points = box.reshape(-1, box.shape[-1])[:, :2]
        return float(np.min(points[:, 0])), float(np.min(points[:, 1]))
    return 0.0, 0.0


def _extract_paddle_result_blocks(ocr_output):
    """Yield text blocks with coordinates from PaddleOCR v3 or v2 results."""
    def iter_text_scores(payload):
        texts = payload.get("rec_texts")
        scores = payload.get("rec_scores")
        if texts is None:
            return
        if scores is None:
            scores = [0.0] * len(texts)
        boxes = None
        for key in ("rec_polys", "dt_polys", "rec_boxes", "det_polys"):
            value = payload.get(key)
            if value is not None:
                boxes = value
                break
        boxes = boxes if boxes is not None else []
        for index, (raw_text, score) in enumerate(zip(list(texts), list(scores))):
            box = boxes[index] if index < len(boxes) else None
            x, y = _ocr_box_origin(box)
            yield {
                "raw_text": raw_text,
                "confidence": score,
                "box": box.tolist() if hasattr(box, "tolist") else box,
                "x": x,
                "y": y,
            }

    for page in ocr_output or []:
        if isinstance(page, dict):
            payload = page.get("res", page)
            found_items = False
            for block in iter_text_scores(payload):
                found_items = True
                yield block
            if found_items:
                continue

        if hasattr(page, "json"):
            payload = page.json() if callable(page.json) else page.json
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except json.JSONDecodeError:
                    payload = {}
            if isinstance(payload, dict):
                payload = payload.get("res", payload)
                for block in iter_text_scores(payload):
                    yield block
                continue

        # PaddleOCR 2.x commonly returns: [[box, (text, confidence)], ...].
        lines = page if isinstance(page, list) else []
        for line in lines:
            if not isinstance(line, (list, tuple)) or len(line) < 2:
                continue
            text_score = line[1]
            if isinstance(text_score, (list, tuple)) and len(text_score) >= 2:
                x, y = _ocr_box_origin(line[0])
                yield {
                    "raw_text": text_score[0],
                    "confidence": text_score[1],
                    "box": line[0],
                    "x": x,
                    "y": y,
                }


def _is_hsrp_noise(raw_text):
    cleaned = sanitize_plate_text(str(raw_text or ""))
    return cleaned in HSRP_NOISE_STRINGS


def _spatially_join_ocr_blocks(blocks, y_margin=None):
    filtered = [block for block in blocks if block.get("raw_text") and not _is_hsrp_noise(block.get("raw_text"))]
    if not filtered:
        return "", []

    if y_margin is None:
        heights = []
        for block in filtered:
            box = block.get("box")
            if box is not None:
                points = np.asarray(box)
                if points.ndim == 2 and points.shape[1] >= 2:
                    heights.append(float(np.ptp(points[:, 1])))
                elif points.ndim == 1 and points.size == 4:
                    heights.append(float(points[3]-points[1]))
        y_margin = max(3.0, float(np.median(heights)) * .55) if heights else 15.0
    ordered_by_y = sorted(filtered, key=lambda block: (float(block.get("y") or 0.0), float(block.get("x") or 0.0)))
    rows = []
    for block in ordered_by_y:
        y = float(block.get("y") or 0.0)
        row = next((line for line in rows if abs(line["y"] - y) <= y_margin), None)
        if row is None:
            rows.append({"y": y, "blocks": [block]})
        else:
            row["blocks"].append(block)
            row["y"] = sum(float(item.get("y") or 0.0) for item in row["blocks"]) / len(row["blocks"])

    raw_lines = []
    for row in sorted(rows, key=lambda line: line["y"]):
        line_blocks = sorted(row["blocks"], key=lambda block: float(block.get("x") or 0.0))
        raw_lines.append("".join(str(block.get("raw_text") or "") for block in line_blocks))

    joined = strip_hsrp_noise("".join(raw_lines))
    return apply_positional_plate_corrections(joined), raw_lines


def _mean_block_confidence(blocks):
    scores = [float(block.get("confidence") or 0.0) for block in blocks if not _is_hsrp_noise(block.get("raw_text"))]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def read_plate_text(crop_bgr, layout=None, max_variants=None):
    """Run PaddleOCR on a cropped plate region."""
    candidates = []
    quality = assess_quality(crop_bgr)
    if quality["unreadable"]:
        return candidates

    vehicle_category, color_confidence = classify_plate_color(crop_bgr)
    variants = prepare_ocr_images(crop_bgr, layout)
    if max_variants is not None:
        variants = variants[:max_variants]
    elif FAST_OCR_VARIANTS > 0:
        variants = variants[:FAST_OCR_VARIANTS]

    def has_accepted_plate():
        return any(
            candidate.get("valid_format")
            and candidate.get("status") != PENDING_REVIEW_STATUS
            and (candidate.get("confidence") or 0.0) >= OCR_ACCEPT_CONFIDENCE
            for candidate in candidates
        )

    for variant_name, ocr_image in variants:
        reader = get_ocr_reader()
        with ocr_lock:
            try:
                if hasattr(reader, "predict"):
                    ocr_results = reader.predict(ocr_image)
                else:
                    ocr_results = reader.ocr(ocr_image, cls=False)
            except Exception as exc:
                candidates.append({
                    "variant": f"paddleocr_{variant_name}",
                    "raw_text": "",
                    "text": None,
                    "confidence": 0.0,
                    "status": "failed",
                    "needs_review": True,
                    "valid_format": False,
                    "violations": [f"ocr_error:{exc}"],
                    "corrections": [],
                })
                continue

        blocks = list(_extract_paddle_result_blocks(ocr_results))
        joined_text, raw_lines = _spatially_join_ocr_blocks(blocks)
        if joined_text:
            joined_conf = _mean_block_confidence(blocks)
            cleaned, pattern_conf = clean_plate_text(joined_text)
            if cleaned:
                rule_result = normalize_plate_text(
                    cleaned,
                    vehicle_category=vehicle_category,
                    color_confidence=color_confidence,
                )
                combined_conf = round((joined_conf + pattern_conf) / 2, 3)
                regex_passed = status_for_plate(combined_conf, rule_result) != PENDING_REVIEW_STATUS
                candidates.append({
                    "variant": f"paddleocr_{variant_name}_spatial_join",
                    "raw_text": " ".join(raw_lines),
                    "raw_lines": raw_lines,
                    "joined_text": joined_text,
                    "text": cleaned,
                    "confidence": combined_conf,
                    "status": status_for_plate(combined_conf, rule_result),
                    "regex_status": "Passed" if regex_passed else "Needs Review",
                    "needs_review": not regex_passed,
                    "valid_format": rule_result.valid_format,
                    "state_code": rule_result.state_code,
                    "district_code": rule_result.district_code,
                    "vehicle_category": vehicle_category,
                    "color_confidence": color_confidence,
                    "violations": rule_result.violations,
                    "corrections": rule_result.corrections,
                })
            else:
                confidence = round(min(joined_conf, 0.79), 3)
                candidates.append({
                    "variant": f"paddleocr_{variant_name}_spatial_join",
                    "raw_text": " ".join(raw_lines),
                    "raw_lines": raw_lines,
                    "joined_text": joined_text,
                    "text": joined_text,
                    "confidence": confidence,
                    "status": PENDING_REVIEW_STATUS,
                    "regex_status": "Needs Review",
                    "needs_review": True,
                    "valid_format": False,
                    "state_code": None,
                    "district_code": None,
                    "vehicle_category": vehicle_category,
                    "color_confidence": color_confidence,
                    "violations": ["format_mismatch"],
                    "corrections": [],
                })

        for block in blocks:
            raw_text = block.get("raw_text")
            if _is_hsrp_noise(raw_text):
                continue
            ocr_conf = block.get("confidence") or 0.0
            raw_clean = strip_hsrp_noise(str(raw_text or ""))
            if not raw_clean:
                continue
            cleaned, pattern_conf = clean_plate_text(raw_text)
            if cleaned:
                rule_result = normalize_plate_text(
                    cleaned,
                    vehicle_category=vehicle_category,
                    color_confidence=color_confidence,
                )
                # Blend PaddleOCR's own confidence with the pattern-match confidence.
                combined_conf = round((float(ocr_conf) + pattern_conf) / 2, 3)
                candidates.append({
                    "variant": f"paddleocr_{variant_name}",
                    "raw_text": str(raw_text),
                    "raw_lines": [str(raw_text)],
                    "joined_text": raw_clean,
                    "text": cleaned,
                    "confidence": combined_conf,
                    "status": status_for_plate(combined_conf, rule_result),
                    "regex_status": "Passed" if status_for_plate(combined_conf, rule_result) != PENDING_REVIEW_STATUS else "Needs Review",
                    "needs_review": status_for_plate(combined_conf, rule_result) == PENDING_REVIEW_STATUS,
                    "valid_format": rule_result.valid_format,
                    "state_code": rule_result.state_code,
                    "district_code": rule_result.district_code,
                    "vehicle_category": vehicle_category,
                    "color_confidence": color_confidence,
                    "violations": rule_result.violations,
                    "corrections": rule_result.corrections,
                })
            else:
                confidence = round(min(float(ocr_conf or 0.0), 0.79), 3)
                candidates.append({
                    "variant": f"paddleocr_{variant_name}",
                    "raw_text": str(raw_text),
                    "raw_lines": [str(raw_text)],
                    "joined_text": raw_clean,
                    "text": raw_clean,
                    "confidence": confidence,
                    "status": PENDING_REVIEW_STATUS,
                    "regex_status": "Needs Review",
                    "needs_review": True,
                    "valid_format": False,
                    "state_code": None,
                    "district_code": None,
                    "vehicle_category": vehicle_category,
                    "color_confidence": color_confidence,
                    "violations": ["format_mismatch"],
                    "corrections": [],
                })

        if has_accepted_plate():
            break

    for candidate in candidates:
        candidate["quality"] = quality
        if "tiny_plate" in quality["flags"] or "blur" in quality["flags"]:
            candidate["confidence"] = min(candidate["confidence"], .79)
            candidate["status"] = PENDING_REVIEW_STATUS
            candidate["needs_review"] = True
            candidate["regex_status"] = "Needs Review"
        candidate["violations"] = list(dict.fromkeys(candidate.get("violations", []) + quality["flags"]))
    return candidates


def _save_debug_crop(image_path, crop_bgr, suffix):
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    source = os.path.abspath(image_path)
    upload_dir = os.path.dirname(source)
    stem = os.path.splitext(os.path.basename(source))[0]
    crop_name = f"{stem}_{suffix}.jpg"
    crop_path = os.path.join(upload_dir, crop_name)
    cv2.imwrite(crop_path, crop_bgr)
    if os.path.basename(upload_dir) == "uploads":
        return f"/uploads/{crop_name}"
    return crop_path


def _fit_ocr_region(crop_bgr, max_dim=480):
    height, width = crop_bgr.shape[:2]
    scale = max_dim / max(height, width)
    if scale >= 1.0:
        return crop_bgr
    return cv2.resize(crop_bgr, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)


def plate_like_regions(image_bgr, max_regions=6):
    height, width = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.bilateralFilter(gray, 7, 55, 55)
    edges = cv2.Canny(gray, 80, 180)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    image_area = height * width
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h <= 0:
            continue
        aspect = w / h
        area = w * h
        if not (1.2 <= aspect <= 8.5):
            continue
        if not (image_area * 0.002 <= area <= image_area * 0.60):
            continue
        pad_x, pad_y = max(8, int(w * 0.12)), max(6, int(h * 0.25))
        x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
        x2, y2 = min(width, x + w + pad_x), min(height, y + h + pad_y)
        crop = image_bgr[y1:y2, x1:x2]
        vertical_bias = 1.0 + (y / max(1, height))
        score = area * vertical_bias
        bbox = dict(x=x, y=y, width=w, height=h)
        regions.append((score, f"contour_{len(regions) + 1}", crop, bbox))

    regions.sort(key=lambda item: item[0], reverse=True)
    return [(name, crop, bbox) for _, name, crop, bbox in regions[:max_regions] if crop is not None and crop.size]


def fallback_ocr_regions(image_bgr):
    height, width = image_bgr.shape[:2]
    contour_regions = plate_like_regions(image_bgr, max_regions=MAX_FALLBACK_REGIONS)
    regions = [("center_plate_band", .08, .34, .92, .72), ("lower_plate_band", .10, .52, .90, .82)]
    bounded_regions = []
    for name, left, top, right, bottom in regions:
        x, y, x2, y2 = int(left*width), int(top*height), int(right*width), int(bottom*height)
        crop = image_bgr[y:y2, x:x2]
        if crop.size:
            bounded_regions.append((name, crop, dict(x=x,y=y,width=x2-x,height=y2-y)))
    return contour_regions + bounded_regions


def process_image(image_path):
    """
    Full pipeline for one uploaded photo.
    Returns: (plate_text, confidence, status, raw_candidates_json)
    """
    image = cv2.imread(image_path)
    if image is None:
        return None, 0.0, "failed", json.dumps([])

    plates = _rank_plate_detections(locate_plates(image))[:max(1, MAX_OCR_DETECTIONS)]
    all_candidates = []
    debug = {
        "yolo_boxes_found": len(plates),
        "box_confidence_scores": [
            detection.get("confidence") for detection in plates if isinstance(detection, dict)
        ],
        "raw_paddleocr_text": [],
        "detected_ocr_raw_lines": [],
        "joined_normalized_plate": None,
        "regex_corrected_plate": None,
        "regex_status": None,
    }

    for detection_index, detection in enumerate(plates, start=1):
        x, y, w, h = detection_bbox(detection)
        crop, frame_edges = padded_plate_crop(image, (x, y, w, h))
        layout = detection.get("layout") if isinstance(detection, dict) else None
        crop_candidates = read_plate_text(crop, layout=layout)
        crop_image_path = _save_debug_crop(image_path, crop, f"crop_{detection_index}") if crop_candidates else None
        if not crop_candidates:
            crop_candidates = [dict(text=None, confidence=0.0, status=PENDING_REVIEW_STATUS,
                                    needs_review=True, violations=["unreadable_crop"], quality=assess_quality(crop))]
        for candidate in crop_candidates:
            candidate["detection_id"] = detection_index
            candidate["partial"] = bool(frame_edges)
            candidate["frame_edges"] = frame_edges
            if frame_edges:
                candidate["status"] = PENDING_REVIEW_STATUS
                candidate["needs_review"] = True
                candidate["confidence"] = min(candidate.get("confidence", 0), .79)
                candidate["violations"] = candidate.get("violations", []) + ["frame_edge_crop"]
            candidate["bbox"] = {"x": x, "y": y, "width": w, "height": h}
            candidate["detector_confidence"] = detection.get("confidence") if isinstance(detection, dict) else None
            candidate["layout"] = layout
            candidate["crop_image_path"] = crop_image_path
            if candidate.get("raw_text"):
                debug["raw_paddleocr_text"].append(candidate["raw_text"])
            if candidate.get("raw_lines") and not debug["detected_ocr_raw_lines"]:
                debug["detected_ocr_raw_lines"] = candidate["raw_lines"]
            if candidate.get("joined_text") and not debug["joined_normalized_plate"]:
                debug["joined_normalized_plate"] = candidate["joined_text"]
            if candidate.get("regex_status") and not debug["regex_status"]:
                debug["regex_status"] = candidate["regex_status"]
        all_candidates.extend(crop_candidates)

    # Fallback: if the detector misses, try a few bounded plate-like regions so
    # uploads do not stall on full-resolution screenshots.
    if not any(candidate.get("valid_format") for candidate in all_candidates):
        for region_index, (region_name, region, bbox) in enumerate(fallback_ocr_regions(image), 1):
            if "plate_band" in region_name and any(c.get("valid_format") for c in all_candidates):
                break
            fallback_candidates = read_plate_text(region)
            frame_edges = bbox_frame_edges(image, (bbox["x"], bbox["y"], bbox["width"], bbox["height"]))
            fallback_crop_path = _save_debug_crop(image_path, region, region_name) if fallback_candidates else None
            for candidate in fallback_candidates:
                candidate["detection_id"] = -region_index
                candidate["bbox"] = bbox
                candidate["partial"] = bool(frame_edges)
                candidate["frame_edges"] = frame_edges
                if frame_edges:
                    candidate["status"] = PENDING_REVIEW_STATUS
                    candidate["needs_review"] = True
                    candidate["confidence"] = min(candidate.get("confidence", 0), .79)
                    candidate["violations"] = candidate.get("violations", []) + ["frame_edge_crop"]
                candidate["layout"] = candidate.get("layout") or f"fallback_{region_name}"
                candidate["crop_image_path"] = fallback_crop_path
                if candidate.get("raw_text"):
                    debug["raw_paddleocr_text"].append(candidate["raw_text"])
                if candidate.get("raw_lines") and not debug["detected_ocr_raw_lines"]:
                    debug["detected_ocr_raw_lines"] = candidate["raw_lines"]
                if candidate.get("joined_text") and not debug["joined_normalized_plate"]:
                    debug["joined_normalized_plate"] = candidate["joined_text"]
                if candidate.get("regex_status") and not debug["regex_status"]:
                    debug["regex_status"] = candidate["regex_status"]
            all_candidates.extend(fallback_candidates)
            if any(candidate.get("valid_format") and candidate.get("status") != PENDING_REVIEW_STATUS
                   for candidate in all_candidates):
                break

    if not all_candidates:
        return None, 0.0, PENDING_REVIEW_STATUS, json.dumps([{"debug": debug}])

    # Votes stay within a detection; the compatibility result should prefer a
    # valid plate over larger non-plate text that can appear in screenshots.
    grouped = {}
    for candidate in all_candidates:
        grouped.setdefault(candidate.get("detection_id", 0), []).append(candidate)
    winners = [best_plate_candidate(group) for group in grouped.values()]
    winners = [winner for winner in winners if winner]
    if not winners:
        return None, 0.0, PENDING_REVIEW_STATUS, json.dumps([{"debug": debug}, *all_candidates])
    best = max(winners, key=lambda c: (
        bool(c.get("valid_format")),
        not bool(c.get("needs_review") or c.get("status") == PENDING_REVIEW_STATUS),
        c["confidence"],
        (c.get("bbox") or {}).get("width", 0) * (c.get("bbox") or {}).get("height", 0),
    ))
    best_text, best_conf = best.get("text"), best["confidence"]
    status = best.get("status") or PENDING_REVIEW_STATUS
    debug["joined_normalized_plate"] = best_text
    debug["regex_corrected_plate"] = best_text
    debug["regex_status"] = "Passed" if status == "ok" else "Needs Review"
    return best_text, best_conf, status, json.dumps([{"debug": debug}, *all_candidates])


def best_plate_candidate(candidates):
    readable = [c for c in candidates if c.get("text")]
    if not readable:
        return max(candidates, key=lambda c: c.get("confidence", 0), default=None)
    votes = {}
    for candidate in readable:
        votes[candidate["text"]] = votes.get(candidate["text"], 0) + 1
    return max(readable, key=lambda c: (bool(c.get("valid_format")), votes[c["text"]], c["confidence"]))
