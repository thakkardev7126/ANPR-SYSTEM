import os
import shutil
import uuid
import datetime
import json
import re
import asyncio
import logging
import base64
import csv
import threading
import time
import cv2
import numpy as np
from contextlib import asynccontextmanager, suppress
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, Query, Request
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.database import get_db, PlateEvent, Camera, Base, engine, AsyncPlateEventWriter, VehicleMatchCandidate, VehicleAnomaly, PlateSuspicionEvent, RouteAnomalyEvent
from app.database import DATABASE_PATH, DATABASE_URL
from app.database import SessionLocal, backfill_vehicle_ids, ensure_vehicle_for_event, plate_similarity_score, persist_plate_event
from app.database import HotlistAlert, HotlistNotification
from app.hotlist import match_event, alert_payload
from app.hotlist_api import router as hotlist_router
from app.image_quality import padded_plate_crop, bbox_frame_edges, assess_quality, PartialPlateHistory
from app.location import resolve_location, camera_location, valid_coordinates, time_window, iso_utc
from app.anpr_pipeline import (
    MODEL_PATH,
    OCR_ACCEPT_CONFIDENCE,
    process_image,
    get_ocr_reader,
    get_plate_detector,
    MobileStreamReceiver,
    StreamProcessor,
    locate_plates,
    detection_bbox,
    read_plate_text,
    is_frame_sharp,
    plate_status,
    best_plate_candidate,
    infer_plate_layout,
    fallback_ocr_regions,
)
from app.plate_rules import PENDING_REVIEW_STATUS, normalize_plate_text, status_for_plate
from app.trajectory import (
    build_plate_complete_trajectory,
    build_trajectory,
    build_vehicle_trajectory,
    get_vehicle_by_id,
    get_vehicle_by_plate,
    haversine_km,
    list_all_vehicles,
    vehicle_history,
    vehicle_payload,
)
from app.vehicle_appearance import analyze_vehicle_appearance, appearance_similarity, serialize_embedding
from app.vehicle_matching import match_payload, matches_for_observation, matches_for_vehicle, matching_thresholds, review_match
from app.travel_time import speed_summary, vehicle_speed_history
from app.anomalies import anomaly_payload, speed_policy
from app.plate_suspicion import (
    plate_suspicion_summary,
    suspicion_config,
    suspicion_payload,
)
from app.route_anomaly import (
    route_anomaly_config,
    route_anomaly_payload,
    route_anomaly_summary,
)
from app.investigation import build_plate_investigation, build_vehicle_investigation
from app.traffic_analytics import (
    camera_metrics,
    congestion_metrics,
    density_metrics,
    dwell_metrics,
    flow_metrics,
    heatmap_points,
    lane_metrics,
    od_matrix,
    road_metrics,
    timeseries,
    traffic_dashboard,
    traffic_policy,
    traffic_summary,
    validate_bucket,
)
from app.road_network import (
    create_connection,
    disable_connection,
    get_connection,
    list_road_network,
    outgoing_connections,
    road_connection_payload,
    update_connection,
)
from app.seed_cameras import seed as seed_cameras

UPLOAD_DIR = os.getenv("ANPR_UPLOAD_DIR", os.path.join(os.path.dirname(__file__), "..", "uploads"))
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REVIEW_FEEDBACK_DIR = os.getenv("ANPR_REVIEW_DIR", os.path.join(PROJECT_ROOT, "data", "review_feedback"))
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend"))
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(os.path.join(REVIEW_FEEDBACK_DIR, "images"), exist_ok=True)

LIVE_TRACK_OCR_INTERVAL_SECONDS = 8.0
LIVE_TRACK_EVENT_INTERVAL_SECONDS = 12.0

event_writer = AsyncPlateEventWriter()


@asynccontextmanager
async def lifespan(_app):
    await event_writer.start()
    cursor, _ = await asyncio.to_thread(hotlist_notifications_after, None)
    notification_task = asyncio.create_task(publish_hotlist_notifications(cursor))
    model_warmup_task = asyncio.create_task(warm_up_ocr_models())
    yield
    for node in list(stream_nodes.values()):
        await asyncio.to_thread(node.stop)
    stream_nodes.clear()
    await event_writer.stop()
    notification_task.cancel()
    with suppress(asyncio.CancelledError):
        await notification_task
    model_warmup_task.cancel()
    with suppress(asyncio.CancelledError):
        await model_warmup_task


app = FastAPI(title="City-Wide ANPR Trajectory Tracking — Demo API", lifespan=lifespan)
app.include_router(hotlist_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # open for the hackathon demo; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)
seed_cameras()
with SessionLocal() as _startup_db:
    backfill_vehicle_ids(_startup_db)

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


async def warm_up_ocr_models():
    """Load heavy OCR models in the background so the first scan does not time out."""
    try:
        await asyncio.to_thread(get_plate_detector)
        await asyncio.to_thread(get_ocr_reader)
    except Exception:
        logging.exception("OCR model warm-up failed; the first scan will load models lazily")


@app.get("/api/system/status")
def system_status(db: Session = Depends(get_db)):
    """Show the runtime configuration used by the local demo."""
    return {
        "database_path": DATABASE_PATH,
        "database_url": DATABASE_URL,
        "detector_model_path": MODEL_PATH,
        "ocr_engine": "PaddleOCR",
        "ocr_accept_confidence": OCR_ACCEPT_CONFIDENCE,
        "camera_count": db.query(Camera).count(),
        "cameras": [
            {"camera_id": c.camera_id, "label": c.label, "lat": camera_location(c)[0], "lng": camera_location(c)[1], "location_known": bool(c.location_known)}
            for c in db.query(Camera).order_by(Camera.camera_id.asc()).all()
        ],
    }


class StreamNode:
    def __init__(self, camera_id, stream_url):
        self.camera_id = camera_id
        self.receiver = MobileStreamReceiver(stream_url)
        self.latest = None
        self.lock = threading.Lock()
        import supervision as sv
        self.sv = sv
        self.tracker = sv.ByteTrack()
        self.last_live_plate = None
        self.last_live_plate_at = 0.0
        self.track_read_cache = {}
        self.track_last_ocr_at = {}
        self.track_last_event_at = {}
        self.track_max_confidence = {}
        self.track_last_seen = {}
        self.partial_history = PartialPlateHistory()
        self.processor = StreamProcessor(self.receiver, self._process_frame, process_every=3)

    def _read_tracked_plate(self, track_id, crop, frame_confidence=1.0, layout=None, frame_edges=None):
        now = time.monotonic()
        cached = self.track_read_cache.get(track_id)
        last_ocr_at = self.track_last_ocr_at.get(track_id, 0.0)
        max_confidence = self.track_max_confidence.get(track_id, -1.0)
        quality = assess_quality(crop)
        frame_confidence = quality["score"]
        is_better_frame = frame_confidence > max_confidence
        if cached and cached["best"].get("partial") and not frame_edges:
            is_better_frame = True
        if cached and not is_better_frame and now - last_ocr_at < LIVE_TRACK_OCR_INTERVAL_SECONDS:
            return cached["candidates"], cached["best"], True

        if quality["unreadable"]:
            if cached:
                return cached["candidates"], cached["best"], True
            return [], None, True

        candidates = read_plate_text(crop, layout=layout)
        if frame_edges:
            for candidate in candidates:
                candidate["partial"] = True
                candidate["status"] = PENDING_REVIEW_STATUS
                candidate["confidence"] = min(candidate.get("confidence", 0), .79)
                candidate["violations"] = candidate.get("violations", []) + ["frame_edge_crop"]
        best = best_plate_candidate(candidates)
        self.track_last_ocr_at[track_id] = now
        self.track_max_confidence[track_id] = max(max_confidence, frame_confidence)
        if (cached and best and not is_better_frame and not frame_edges
                and cached["best"].get("valid_format") and not cached["best"].get("partial")
                and (not best.get("valid_format") or best["confidence"] < cached["best"]["confidence"])):
            return cached["candidates"], cached["best"], True
        if best:
            self.track_read_cache[track_id] = {"candidates": candidates, "best": best}
        elif cached:
            return cached["candidates"], cached["best"], True
        return candidates, best, False

    def _enqueue_live_event(self, track_id, best, candidates, bbox):
        if not _candidate_is_complete_valid(best):
            return
        now = time.monotonic()
        event_key = f"{track_id}:{best['text']}"
        if now - self.track_last_event_at.get(event_key, 0.0) < LIVE_TRACK_EVENT_INTERVAL_SECONDS:
            return
        x, y, width, height = bbox
        event_writer.enqueue_from_thread({
            "camera_id": self.camera_id,
            "track_id": track_id,
            "plate_text": best["text"],
            "confidence": best["confidence"],
            "status": best.get("status") or plate_status(best["confidence"]),
            "plate_category": best.get("vehicle_category"),
            "layout": best.get("layout"),
            "rule_violations": json.dumps(best.get("violations", [])),
            "bbox_x": x, "bbox_y": y, "bbox_width": width, "bbox_height": height,
            "raw_ocr_candidates": json.dumps(candidates),
        })
        self.track_last_event_at[event_key] = now

    def _process_frame(self, frame, frame_number):
        boxes = locate_plates(frame)
        tracked = self.tracker.update_with_detections(
            self.sv.Detections(
                xyxy=np.array([
                    [
                        detection_bbox(box)[0],
                        detection_bbox(box)[1],
                        detection_bbox(box)[0] + detection_bbox(box)[2],
                        detection_bbox(box)[1] + detection_bbox(box)[3],
                    ]
                    for box in boxes
                ], dtype=float)
                if boxes else np.empty((0, 4), dtype=float),
                confidence=np.array([
                    box.get("confidence", 1.0) if isinstance(box, dict) else 1.0
                    for box in boxes
                ], dtype=float),
                class_id=np.array([
                    box.get("class_id", 0) if isinstance(box, dict) else 0
                    for box in boxes
                ], dtype=int),
            )
        )
        detections = []
        for index, xyxy in enumerate(tracked.xyxy):
            x, y, x2, y2 = map(int, xyxy)
            width, height = x2 - x, y2 - y
            crop, frame_edges = padded_plate_crop(frame, (x, y, width, height))
            track_id = int(tracked.tracker_id[index]) if tracked.tracker_id is not None else f"box-{index}"
            detector_confidence = float(tracked.confidence[index]) if tracked.confidence is not None else 1.0
            layout = infer_plate_layout(crop)
            candidates, best, used_cache = self._read_tracked_plate(
                track_id,
                crop,
                frame_confidence=detector_confidence,
                layout=layout,
                frame_edges=frame_edges,
            )
            if best:
                best = dict(best)
                best["layout"] = layout
                hypotheses = self.partial_history.observe(track_id, best.get("raw_text") or best.get("text"),
                                                          time.monotonic())
                best["partial_hypotheses"] = hypotheses
                if frame_edges:
                    best["status"] = PENDING_REVIEW_STATUS
                    best["confidence"] = min(best["confidence"], .79)
                    best["partial"] = True
                    best["violations"] = list(set(best.get("violations", []) + ["frame_edge_crop"]))
                candidates = [best, *candidates]
            self.track_last_seen[track_id] = time.monotonic()
            visible_best = best if _candidate_is_complete_valid(best) else None
            detection = {
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "track_id": track_id,
                "plate_text": visible_best["text"] if visible_best else None,
                "confidence": visible_best["confidence"] if visible_best else 0.0,
                "detector_confidence": detector_confidence,
                "layout": layout,
                "status": visible_best.get("status") if visible_best else PENDING_REVIEW_STATUS,
                "plate_category": visible_best.get("vehicle_category") if visible_best else None,
                "ocr_cached": used_cache,
                "partial": bool(frame_edges),
                "partial_hypotheses": best.get("partial_hypotheses", []) if best else [],
                "quality": assess_quality(crop),
            }
            detections.append(detection)
            if best and not used_cache:
                self._enqueue_live_event(track_id, best, candidates, (x, y, width, height))

        # Poor detections must not prevent contour/region OCR recovery.
        if not any(normalize_plate_text(d.get("plate_text")).valid_format for d in detections):
            for region_index, (region_name, crop, bbox) in enumerate(fallback_ocr_regions(frame)):
                if "plate_band" in region_name and any(normalize_plate_text(d.get("plate_text")).valid_format for d in detections):
                    break
                fallback_candidates = read_plate_text(crop)
                frame_edges = bbox_frame_edges(frame, (bbox["x"], bbox["y"], bbox["width"], bbox["height"]))
                for candidate in fallback_candidates:
                    candidate["partial"] = bool(frame_edges)
                    candidate["frame_edges"] = frame_edges
                    if frame_edges:
                        candidate["status"] = PENDING_REVIEW_STATUS
                        candidate["needs_review"] = True
                        candidate["confidence"] = min(candidate.get("confidence", 0), .79)
                        candidate["violations"] = candidate.get("violations", []) + ["frame_edge_crop"]
                best = best_plate_candidate(fallback_candidates)
                if not best:
                    continue
                track_id = f"fallback-{region_index}"
                best = dict(best, layout=infer_plate_layout(crop))
                if _candidate_is_complete_valid(best):
                    detections.append({**bbox, "track_id": track_id, "plate_text": best["text"],
                                       "confidence": best["confidence"], "status": best["status"],
                                       "partial": bool(frame_edges), "quality": assess_quality(crop)})
                    self._enqueue_live_event(track_id, best, fallback_candidates,
                                             (bbox["x"], bbox["y"], bbox["width"], bbox["height"]))
        cutoff = time.monotonic() - 30
        stale = [key for key, seen in self.track_last_seen.items() if seen < cutoff]
        for key in stale:
            for cache in (self.track_last_seen, self.track_read_cache, self.track_last_ocr_at, self.track_max_confidence):
                cache.pop(key, None)
        self.track_last_event_at = {key: seen for key, seen in self.track_last_event_at.items() if seen >= cutoff}
        with self.lock:
            self.latest = {"camera_id": self.camera_id, "frame_number": frame_number, "detections": detections}

    def start(self):
        self.processor.start()

    def stop(self):
        self.processor.stop()

    def snapshot(self):
        with self.receiver._frame_lock:
            return self.receiver._latest_frame.copy() if self.receiver._latest_frame is not None else None

    def result(self):
        with self.lock:
            return self.latest


stream_nodes = {}
stream_nodes_lock = threading.Lock()


class EventSocketHub:
    def __init__(self):
        self._connections = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            self._connections.discard(websocket)

    async def broadcast(self, message):
        async with self._lock:
            connections = list(self._connections)
        for websocket in connections:
            try:
                await websocket.send_json(message)
            except (RuntimeError, WebSocketDisconnect, OSError):
                await self.disconnect(websocket)


event_hub = EventSocketHub()


def hotlist_notifications_after(cursor):
    with SessionLocal() as db:
        if cursor is None:
            latest = db.query(HotlistNotification.id).order_by(HotlistNotification.id.desc()).first()
            return latest[0] if latest else 0, []
        rows = (db.query(HotlistNotification, HotlistAlert)
                .join(HotlistAlert, HotlistAlert.id == HotlistNotification.alert_id)
                .filter(HotlistNotification.id > cursor).order_by(HotlistNotification.id).limit(100).all())
        return (rows[-1][0].id if rows else cursor), [alert_payload(alert) for _, alert in rows]


async def publish_hotlist_notifications(cursor):
    # Durable inbox handles reconnects; each web worker publishes new committed alerts.
    while True:
        try:
            cursor, alerts = await asyncio.to_thread(hotlist_notifications_after, cursor)
            for alert in alerts:
                await event_hub.broadcast({"type": "hotlist_alert", "alert": alert})
        except Exception:
            logging.exception("Hotlist notification delivery failed; retrying")
        await asyncio.sleep(.5)


def is_review_status(status):
    return str(status or "").upper() in {"PENDING_REVIEW", "NEEDS_REVIEW"}


def _is_complete_valid_plate(plate_text, confidence=0.0, status=None, partial=False):
    rule = normalize_plate_text(plate_text)
    return (
        bool(rule.valid_format)
        and not partial
        and not is_review_status(status)
        and float(confidence or 0.0) >= OCR_ACCEPT_CONFIDENCE
    )


def _candidate_is_complete_valid(candidate):
    if not candidate:
        return False
    return _is_complete_valid_plate(
        candidate.get("text"),
        candidate.get("confidence"),
        candidate.get("status"),
        candidate.get("partial"),
    )


def _parse_candidates(raw_candidates):
    try:
        parsed = json.loads(raw_candidates or "[]")
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


def _candidate_metadata(raw_candidates, plate_text=None):
    candidates = _parse_candidates(raw_candidates)
    candidates = [candidate for candidate in candidates if not candidate.get("debug")]
    matching = [candidate for candidate in candidates if candidate.get("text") == plate_text]
    best = max(matching or candidates, key=lambda item: item.get("confidence") or 0, default={})
    bbox = best.get("bbox") or {}
    return {
        "plate_category": best.get("vehicle_category"),
        "layout": best.get("layout"),
        "rule_violations": json.dumps(best.get("violations", [])),
        "needs_review": bool(best.get("needs_review") or is_review_status(best.get("status"))),
        "raw_text": best.get("raw_text"),
        "crop_image_path": best.get("crop_image_path"),
        "bbox_x": bbox.get("x"),
        "bbox_y": bbox.get("y"),
        "bbox_width": bbox.get("width"),
        "bbox_height": bbox.get("height"),
    }


def _event_payload(event, camera_label=None, filename=None):
    return {
        "event_id": event.id,
        "global_vehicle_id": event.vehicle_id,
        "vehicle_id": event.vehicle_id,
        "camera_id": event.camera_id,
        "camera_label": camera_label or event.camera_id,
        "filename": filename,
        "timestamp": iso_utc(event.timestamp),
        "plate_number": event.plate_text,
        "plate_text": event.plate_text,
        "confidence": event.confidence or 0.0,
        "status": event.status,
        "plate_category": event.plate_category,
        "layout": event.layout,
        "vehicle_type": event.vehicle_type,
        "vehicle_color": event.vehicle_color,
        "vehicle_crop_path": event.vehicle_crop_path,
        "appearance_available": bool(event.appearance_embedding),
        "appearance_model": event.appearance_model,
        "appearance_embedding_version": event.appearance_embedding_version,
        "appearance_quality": event.appearance_quality,
        "needs_review": is_review_status(event.status),
        "image_path": event.image_path,
    }


def _appearance_payload(event):
    return {
        "event_id": event.id,
        "global_vehicle_id": event.vehicle_id,
        "vehicle_id": event.vehicle_id,
        "plate_text": event.plate_text,
        "plate_status": event.status,
        "camera_id": event.camera_id,
        "timestamp": iso_utc(event.timestamp),
        "vehicle_type": event.vehicle_type,
        "vehicle_color": event.vehicle_color,
        "vehicle_crop_path": event.vehicle_crop_path,
        "appearance_available": bool(event.appearance_embedding),
        "appearance_model": event.appearance_model,
        "appearance_embedding_version": event.appearance_embedding_version,
        "appearance_quality": event.appearance_quality,
    }


def _scan_result_rank(item):
    bbox = item.get("bbox") or {}
    area = (bbox.get("width") or 0) * (bbox.get("height") or 0)
    rule = normalize_plate_text(item.get("plate_text"))
    return (
        bool(rule.valid_format),
        not bool(item.get("needs_review") or is_review_status(item.get("status"))),
        float(item.get("confidence") or 0),
        area,
    )


def _public_scan_payload(result):
    """Keep API responses small; full OCR candidates remain persisted on events."""
    def strip_debug(item):
        cleaned = dict(item)
        cleaned.pop("raw_ocr_candidates", None)
        return cleaned

    payload = strip_debug(result)
    payload["detections"] = [strip_debug(item) for item in result.get("detections", [])]
    return payload


def _print_scan_debug(endpoint_name, result):
    candidates = result.get("raw_ocr_candidates") or []
    debug = next((item.get("debug") for item in candidates if isinstance(item, dict) and item.get("debug")), {})
    readable_candidates = [
        item for item in candidates
        if isinstance(item, dict) and not item.get("debug") and (item.get("raw_text") or item.get("text"))
    ]
    raw_text = " | ".join(str(item.get("raw_text") or item.get("text")) for item in readable_candidates)
    corrected = result.get("plate_text") or debug.get("regex_corrected_plate") or ""

    print(f"{endpoint_name}", flush=True)
    print(f"YOLO boxes found: {debug.get('yolo_boxes_found', 0)}", flush=True)
    print(f"Box confidence scores: {debug.get('box_confidence_scores', [])}", flush=True)
    print(f"Raw PaddleOCR text output: '{raw_text}'", flush=True)
    print(f"Regex-corrected plate: '{corrected}'", flush=True)
    print(f"Detected OCR raw lines: {debug.get('detected_ocr_raw_lines', [])}", flush=True)
    print(f"Joined & normalized plate string: '{debug.get('joined_normalized_plate') or corrected}'", flush=True)
    print(f"Regex status: {'Passed' if result.get('status') == 'ok' else 'Needs Review'}", flush=True)


def _save_vehicle_crop(image_path, crop_bgr, suffix):
    if crop_bgr is None or getattr(crop_bgr, "size", 0) == 0:
        return None
    upload_dir = os.path.dirname(os.path.abspath(image_path))
    stem = os.path.splitext(os.path.basename(image_path))[0]
    crop_name = f"{stem}_{suffix}.jpg"
    crop_path = os.path.join(upload_dir, crop_name)
    cv2.imwrite(crop_path, crop_bgr)
    if os.path.basename(upload_dir) == "uploads":
        return f"/uploads/{crop_name}"
    return crop_path


async def _appearance_values(filepath, bbox, detection_id):
    appearance = await asyncio.to_thread(analyze_vehicle_appearance, filepath, bbox)
    crop_path = _save_vehicle_crop(filepath, appearance.crop, f"vehicle_{detection_id}") if appearance.crop is not None else None
    return {
        "vehicle_type": appearance.vehicle_type,
        "vehicle_color": appearance.vehicle_color,
        "vehicle_crop_path": crop_path,
        "appearance_embedding": serialize_embedding(appearance.embedding),
        "appearance_model": appearance.embedding_model,
        "appearance_embedding_version": appearance.embedding_version,
        "appearance_quality": appearance.quality,
    }


async def _process_image_safely(filepath):
    try:
        return await asyncio.to_thread(process_image, filepath)
    except Exception as exc:
        return None, 0.0, "failed", json.dumps([{
            "text": None,
            "confidence": 0.0,
            "error": str(exc),
        }])


async def _create_scan_event(db, camera_id, filepath, filename, persist_unreadable=True, selection="all"):
    cam = db.get(Camera, camera_id)
    if not cam:
        raise HTTPException(status_code=400, detail=f"Unknown camera_id '{camera_id}'")
    if selection not in {"all", "largest"}:
        raise HTTPException(status_code=422, detail="selection must be all or largest")
    _, _, _, raw_candidates = await _process_image_safely(filepath)
    groups = {}
    for candidate in _parse_candidates(raw_candidates):
        if not candidate.get("debug"):
            groups.setdefault(candidate.get("detection_id", 0), []).append(candidate)
    if not groups:
        groups = {0: [dict(text=None, confidence=0, status=PENDING_REVIEW_STATUS,
                           violations=["unreadable"], needs_review=True)]}
    winners = [(key, best_plate_candidate(items), items) for key, items in groups.items()]
    winners.sort(key=lambda entry: ((entry[1].get("bbox") or {}).get("width", 0) *
                                   (entry[1].get("bbox") or {}).get("height", 0)), reverse=True)
    if selection == "largest":
        winners = winners[:1]
    results = []
    for detection_id, best, candidates in winners:
        plate_text = best.get("text")
        confidence = best.get("confidence", 0)
        rule = normalize_plate_text(plate_text)
        status = best.get("status") or status_for_plate(confidence, rule)
        if not rule.valid_format or best.get("partial"):
            status = PENDING_REVIEW_STATUS
        encoded = json.dumps(candidates)
        metadata = _candidate_metadata(encoded, plate_text)
        payload = dict(event_id=None, camera_id=camera_id, camera_label=cam.label,
                       plate_text=plate_text, plate_number=plate_text, confidence=confidence,
                       status=status, timestamp=iso_utc(datetime.datetime.utcnow()),
                       image_path=None, duplicate=False)
        should_persist = persist_unreadable or _candidate_is_complete_valid({
            "text": plate_text,
            "confidence": confidence,
            "status": status,
            "partial": bool(best.get("partial")),
        })
        if should_persist:
            values = dict(camera_id=camera_id, image_path=metadata["crop_image_path"] or f"/uploads/{filename}",
                          plate_text=plate_text, confidence=confidence, status=status,
                          raw_ocr_candidates=encoded)
            for field in ("plate_category", "layout", "rule_violations", "bbox_x", "bbox_y", "bbox_width", "bbox_height"):
                values[field] = metadata[field]
            values.update(await _appearance_values(filepath, best.get("bbox") or metadata, detection_id))
            event, duplicate = await asyncio.to_thread(persist_plate_event, values)
            payload.update(_event_payload(event, cam.label, filename), duplicate=duplicate)
            with SessionLocal() as alert_db:
                payload["hotlist_alerts"] = [alert_payload(a) for a in alert_db.query(HotlistAlert)
                    .filter(HotlistAlert.event_id == event.id, HotlistAlert.match_status != "retracted")]
        payload.update(detection_id=detection_id, bbox=best.get("bbox"),
                       raw_ocr_candidates=candidates, raw_text=metadata["raw_text"],
                       crop_image_path=metadata["crop_image_path"],
                       partial=bool(best.get("partial")), quality=best.get("quality"),
                       needs_review=is_review_status(payload["status"]))
        results.append(payload)
    best_result = max(results, key=_scan_result_rank)
    return {**best_result, "detections": results, "selection": selection,
            "raw_ocr_candidates": _parse_candidates(raw_candidates)}


async def _save_request_image(request: Request, fallback_name="frame.jpg"):
    content_type = request.headers.get("content-type", "").lower()
    camera_id = None
    image_bytes = None
    original_name = fallback_name

    if "multipart/form-data" in content_type:
        form = await request.form()
        camera_id = str(form.get("camera_id") or "").strip()
        upload = form.get("file") or form.get("frame") or form.get("image")
        if upload is None:
            for value in form.values():
                if hasattr(value, "filename") and hasattr(value, "read"):
                    upload = value
                    break
        if upload is None:
            raise HTTPException(status_code=400, detail="No image file was supplied")
        original_name = upload.filename or fallback_name
        image_bytes = await upload.read()
        await form.close()
    else:
        try:
            payload = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Expected multipart image or JSON base64 payload") from exc
        camera_id = str(payload.get("camera_id") or "").strip()
        encoded = payload.get("image") or payload.get("frame") or payload.get("image_base64")
        if not encoded:
            raise HTTPException(status_code=400, detail="JSON payload must include image, frame, or image_base64")
        if "," in encoded:
            encoded = encoded.split(",", 1)[1]
        try:
            image_bytes = base64.b64decode(encoded, validate=False)
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid base64 image payload") from exc
        original_name = str(payload.get("filename") or fallback_name)

    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", camera_id):
        raise HTTPException(status_code=400, detail="Invalid camera_id")
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image payload")

    ext = os.path.splitext(original_name)[1] or ".jpg"
    filename = f"{camera_id}_{uuid.uuid4().hex[:8]}{ext}"
    filepath = os.path.join(UPLOAD_DIR, filename)
    with open(filepath, "wb") as output_file:
        output_file.write(image_bytes)
    return camera_id, filepath, filename


def _append_review_feedback(event, corrected_text):
    if not event.image_path:
        return None

    source_path = os.path.join(UPLOAD_DIR, os.path.basename(event.image_path))
    if not os.path.isfile(source_path):
        return None

    _, ext = os.path.splitext(source_path)
    safe_plate = "".join(char for char in corrected_text.upper() if char.isalnum()) or "UNKNOWN"
    feedback_name = f"event_{event.id}_{safe_plate}{ext or '.jpg'}"
    target_path = os.path.join(REVIEW_FEEDBACK_DIR, "images", feedback_name)
    shutil.copy2(source_path, target_path)

    labels_path = os.path.join(REVIEW_FEEDBACK_DIR, "labels.csv")
    is_new_file = not os.path.exists(labels_path)
    with open(labels_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "event_id", "camera_id", "original_text", "corrected_text", "confidence",
            "status_before", "image_path", "saved_at",
        ])
        if is_new_file:
            writer.writeheader()
        writer.writerow({
            "event_id": event.id,
            "camera_id": event.camera_id,
            "original_text": event.plate_text or "",
            "corrected_text": corrected_text,
            "confidence": event.confidence if event.confidence is not None else "",
            "status_before": event.status or "",
            "image_path": target_path,
            "saved_at": datetime.datetime.utcnow().isoformat(),
        })
    return target_path


def _correct_plate_event_sync(event_id: int, corrected_text: str):
    db = SessionLocal()
    try:
        if db.bind.dialect.name == "sqlite":
            db.connection().exec_driver_sql("BEGIN IMMEDIATE")
        event = db.query(PlateEvent).filter(PlateEvent.id == event_id).with_for_update().first()
        if not event:
            raise ValueError("Event not found")
        rule_result = normalize_plate_text(corrected_text)
        normalized_text = rule_result.normalized_text or corrected_text.upper().strip()
        feedback_path = _append_review_feedback(event, normalized_text)
        event.plate_text = normalized_text
        event.status = "ok"
        event.confidence = 1.0
        event.rule_violations = json.dumps(rule_result.violations)
        ensure_vehicle_for_event(db, event)
        match_event(db, event)
        db.commit()
        return {
            "event_id": event.id,
            "global_vehicle_id": event.vehicle_id,
            "vehicle_id": event.vehicle_id,
            "camera_id": event.camera_id,
            "plate_text": event.plate_text,
            "status": "corrected",
            "feedback_image_path": feedback_path,
        }
    finally:
        db.close()


@app.get("/api/cameras")
def get_cameras(db: Session = Depends(get_db)):
    """List all registered 'cameras' (phones) for the demo."""
    cams = db.query(Camera).all()
    return [
        {"camera_id": c.camera_id, "label": c.label, "lat": camera_location(c)[0], "lng": camera_location(c)[1], "location_known": bool(c.location_known),
         "stream_url": c.stream_url, "stream_type": c.stream_type, "active": c.active,
         "connected": c.camera_id in stream_nodes and stream_nodes[c.camera_id].receiver.is_connected()}
        for c in cams
    ]


@app.post("/api/cameras")
def register_camera(camera: dict, db: Session = Depends(get_db)):
    camera_id = str(camera.get("camera_id", "")).strip()
    if not camera_id or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", camera_id):
        raise HTTPException(status_code=400, detail="camera_id must contain 1-64 letters, digits, underscores or hyphens")
    if db.get(Camera, camera_id):
        raise HTTPException(status_code=409, detail="camera_id already exists")
    try:
        lat, lng, known = resolve_location(camera)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    cam = Camera(camera_id=camera_id, label=camera.get("label", camera_id),
                 lat=lat, lng=lng, location_known=known,
                 stream_url=camera.get("stream_url"), stream_type=camera.get("stream_type"))
    db.add(cam)
    db.commit()
    return {"camera_id": camera_id, "active": False}


@app.patch("/api/cameras/{camera_id}")
def update_camera(camera_id: str, camera: dict, db: Session = Depends(get_db)):
    cam = db.get(Camera, camera_id)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    try:
        cam.lat, cam.lng, cam.location_known = resolve_location(camera, cam)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    for field in ("label", "stream_url", "stream_type"):
        if field in camera:
            setattr(cam, field, camera[field])
    db.commit()
    return {"camera_id": camera_id, "updated": True}


@app.delete("/api/cameras/{camera_id}")
def disconnect_camera(camera_id: str, db: Session = Depends(get_db)):
    cam = db.get(Camera, camera_id)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    with stream_nodes_lock:
        node = stream_nodes.pop(camera_id, None)
    if node:
        node.stop()
    cam.active = False
    db.commit()
    return {"camera_id": camera_id, "active": False}


@app.get("/api/camera-road-network")
def get_camera_road_network(include_inactive: bool = Query(False), db: Session = Depends(get_db)):
    """Directed camera road graph; distances are configured road distances when active."""
    return list_road_network(db, include_inactive=include_inactive)


@app.get("/api/cameras/{camera_id}/road-connections")
def get_camera_outgoing_connections(camera_id: str, include_inactive: bool = Query(False),
                                    db: Session = Depends(get_db)):
    try:
        connections = outgoing_connections(db, camera_id, include_inactive=include_inactive)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"camera_id": camera_id, "connections": connections}


@app.get("/api/camera-road-connections/{source_camera_id}/{destination_camera_id}")
def get_camera_road_connection(source_camera_id: str, destination_camera_id: str,
                               include_inactive: bool = Query(False), db: Session = Depends(get_db)):
    connection = get_connection(db, source_camera_id, destination_camera_id, include_inactive=include_inactive)
    if not connection:
        raise HTTPException(status_code=404, detail="Road connection not found")
    source = db.get(Camera, connection.source_camera_id)
    destination = db.get(Camera, connection.destination_camera_id)
    return road_connection_payload(connection, source, destination)


@app.post("/api/camera-road-connections", status_code=201)
def create_camera_road_connection(connection: dict, db: Session = Depends(get_db)):
    try:
        payload = create_connection(db, connection)
    except ValueError as exc:
        detail = str(exc)
        status_code = 404 if detail.startswith("Unknown camera_id") else 422
        if "already exists" in detail:
            status_code = 409
        raise HTTPException(status_code=status_code, detail=detail) from exc
    db.commit()
    return payload


@app.patch("/api/camera-road-connections/{connection_id}")
def update_camera_road_connection(connection_id: int, connection: dict, db: Session = Depends(get_db)):
    try:
        payload = update_connection(db, connection_id, connection)
    except ValueError as exc:
        detail = str(exc)
        status_code = 404 if detail.startswith("Unknown camera_id") else 422
        if "already exists" in detail:
            status_code = 409
        raise HTTPException(status_code=status_code, detail=detail) from exc
    if not payload:
        raise HTTPException(status_code=404, detail="Road connection not found")
    db.commit()
    return payload


@app.delete("/api/camera-road-connections/{connection_id}")
def disable_camera_road_connection(connection_id: int, db: Session = Depends(get_db)):
    payload = disable_connection(db, connection_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Road connection not found")
    db.commit()
    return payload


@app.post("/api/cameras/{camera_id}/connect")
def connect_camera(camera_id: str, db: Session = Depends(get_db)):
    cam = db.get(Camera, camera_id)
    if not cam or not cam.stream_url:
        raise HTTPException(status_code=404, detail="Camera or stream URL not found")
    with stream_nodes_lock:
        old_node = stream_nodes.pop(camera_id, None)
        if old_node:
            old_node.stop()
        node = StreamNode(camera_id, cam.stream_url)
        stream_nodes[camera_id] = node
        node.start()
    cam.active = True
    cam.last_seen = datetime.datetime.utcnow()
    db.commit()
    return {"camera_id": camera_id, "active": True}


@app.get("/api/cameras/{camera_id}/snapshot")
def camera_snapshot(camera_id: str):
    node = stream_nodes.get(camera_id)
    if not node:
        raise HTTPException(status_code=404, detail="Camera is not connected")
    frame = node.snapshot()
    if frame is None:
        raise HTTPException(status_code=503, detail="No frame available")
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        raise HTTPException(status_code=503, detail="Could not encode frame")
    from fastapi.responses import Response
    return Response(content=encoded.tobytes(), media_type="image/jpeg")


@app.websocket("/ws/cameras/{camera_id}")
async def camera_socket(websocket: WebSocket, camera_id: str):
    await websocket.accept()
    try:
        while True:
            node = stream_nodes.get(camera_id)
            if not node:
                await websocket.send_json({"camera_id": camera_id, "status": "disconnected"})
            else:
                await websocket.send_json({"camera_id": camera_id,
                                           "status": "connected" if node.receiver.is_connected() else "reconnecting",
                                           "result": node.result()})
            try:
                message = await asyncio.wait_for(websocket.receive_json(), timeout=.25)
                if message.get("action") == "ping":
                    await websocket.send_json({"type": "pong"})
            except asyncio.TimeoutError:
                pass
    except (WebSocketDisconnect, RuntimeError):
        return


@app.websocket("/ws/events")
async def events_socket(websocket: WebSocket):
    await event_hub.connect(websocket)
    try:
        await websocket.send_json({"type": "connected"})
        while True:
            message = await websocket.receive_json()
            if message.get("action") == "ping":
                await websocket.send_json({"type": "pong"})
                continue
            if message.get("action") == "correct_event":
                try:
                    payload = await asyncio.to_thread(
                        _correct_plate_event_sync,
                        int(message.get("event_id")),
                        str(message.get("corrected_text", "")),
                    )
                except (TypeError, ValueError) as exc:
                    await websocket.send_json({"type": "error", "detail": str(exc)})
                    continue
                await event_hub.broadcast({"type": "event_corrected", **payload})
    except WebSocketDisconnect:
        await event_hub.disconnect(websocket)


@app.post("/api/upload")
async def upload_plate_photo(
    camera_id: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Phone hits this endpoint with a photo + its assigned camera_id.
    Server stamps the time (never trust the phone clock), runs the ANPR
    pipeline synchronously, and returns the read result immediately so the
    phone page can show instant feedback.
    """
    ext = os.path.splitext(file.filename or "photo.jpg")[1] or ".jpg"
    filename = f"{camera_id}_{uuid.uuid4().hex[:8]}{ext}"
    filepath = os.path.join(UPLOAD_DIR, filename)

    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = await _create_scan_event(db, camera_id, filepath, filename, persist_unreadable=True)
    db.commit()
    public_result = _public_scan_payload(result)
    await event_hub.broadcast({"type": "event_created", **public_result})
    return public_result


@app.post("/api/scan")
async def scan_photo(request: Request, db: Session = Depends(get_db)):
    """Canonical scanner endpoint for manual photos; accepts multipart or JSON base64."""
    camera_id, filepath, filename = await _save_request_image(request, "scan.jpg")
    result = await _create_scan_event(db, camera_id, filepath, filename, persist_unreadable=True,
                                      selection=request.query_params.get("selection", "all"))
    _print_scan_debug("/api/scan", result)
    db.commit()
    public_result = _public_scan_payload(result)
    await event_hub.broadcast({"type": "event_created", **public_result})
    return public_result


@app.post("/api/process-frame")
async def process_frame(request: Request, db: Session = Depends(get_db)):
    """Canonical auto-scan endpoint; accepts multipart frames or JSON base64 frames."""
    camera_id, filepath, filename = await _save_request_image(request, "frame.jpg")
    result = await _create_scan_event(db, camera_id, filepath, filename, persist_unreadable=False,
                                      selection=request.query_params.get("selection", "all"))
    db.commit()
    public_result = _public_scan_payload(result)
    if not public_result.get("event_id"):
        public_result.update({
            "plate_text": None,
            "plate_number": None,
            "raw_text": None,
            "confidence": 0.0,
            "status": "scanning",
            "needs_review": False,
            "detections": [],
        })
    if public_result.get("event_id"):
        _print_scan_debug("/api/process-frame", result)
        await event_hub.broadcast({"type": "event_created", **public_result})
    return public_result


@app.post("/api/upload-batch")
async def upload_plate_batch(
    camera_id: str = Form(...),
    files: list[UploadFile] = File(...),
    scan_mode: str = Form("manual"),
    db: Session = Depends(get_db),
):
    """Process several shutter photos and return the strongest readable result."""
    cam = db.get(Camera, camera_id)
    if not cam:
        raise HTTPException(status_code=400, detail=f"Unknown camera_id '{camera_id}'")
    if not files:
        raise HTTPException(status_code=400, detail="At least one image is required")

    results = []
    created_event_ids = []
    for uploaded_file in files:
        ext = os.path.splitext(uploaded_file.filename or "photo.jpg")[1] or ".jpg"
        filename = f"{camera_id}_{uuid.uuid4().hex[:8]}{ext}"
        filepath = os.path.join(UPLOAD_DIR, filename)
        with open(filepath, "wb") as output_file:
            shutil.copyfileobj(uploaded_file.file, output_file)

        result = await _create_scan_event(
            db,
            camera_id,
            filepath,
            filename,
            persist_unreadable=scan_mode != "auto",
        )
        result["filename"] = uploaded_file.filename
        if result.get("event_id"):
            created_event_ids.append(result["event_id"])
        results.append(_public_scan_payload(result))

    db.commit()
    if created_event_ids:
        await event_hub.broadcast({
            "type": "events_created",
            "camera_id": camera_id,
            "event_ids": created_event_ids,
        })
    best = max(results, key=lambda result: result["confidence"] or 0)
    return {"camera_id": camera_id, "camera_label": cam.label, "best": best, "results": results}


@app.post("/api/events/{event_id}/correct")
async def correct_plate_reading(event_id: int, corrected_text: str = Form(...), db: Session = Depends(get_db)):
    """Human-in-the-loop correction — an operator fixes a misread plate."""
    try:
        payload = await asyncio.to_thread(_correct_plate_event_sync, event_id, corrected_text)
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    await event_hub.broadcast({"type": "event_corrected", **payload})
    return payload


@app.delete("/api/events")
async def clear_events(db: Session = Depends(get_db)):
    """Clear scan history while preserving registered camera configuration."""
    events = db.query(PlateEvent).all()
    deleted_files = 0
    for event in events:
        if event.image_path:
            image_path = os.path.join(os.path.dirname(__file__), "..", event.image_path.lstrip("/"))
            if os.path.isfile(image_path):
                os.remove(image_path)
                deleted_files += 1
        if event.vehicle_crop_path:
            crop_path = os.path.join(os.path.dirname(__file__), "..", event.vehicle_crop_path.lstrip("/"))
            if os.path.isfile(crop_path):
                os.remove(crop_path)
                deleted_files += 1
    deleted_events = len(events)
    # Keep alert snapshots without references to IDs SQLite may reuse after a clear.
    db.query(HotlistAlert).update({HotlistAlert.event_id: None}, synchronize_session=False)
    db.query(RouteAnomalyEvent).delete(synchronize_session=False)
    db.query(PlateSuspicionEvent).delete(synchronize_session=False)
    db.query(VehicleAnomaly).delete(synchronize_session=False)
    db.query(VehicleMatchCandidate).delete(synchronize_session=False)
    db.query(PlateEvent).delete(synchronize_session=False)
    db.commit()
    await event_hub.broadcast({"type": "events_cleared"})
    return {"deleted_events": deleted_events, "deleted_files": deleted_files}


@app.get("/api/events")
def get_events(db: Session = Depends(get_db)):
    """Raw event feed — every photo scanned so far, most recent first."""
    events = db.query(PlateEvent).order_by(PlateEvent.timestamp.desc()).limit(200).all()
    return [
        {
            "event_id": e.id,
            "global_vehicle_id": e.vehicle_id,
            "vehicle_id": e.vehicle_id,
            "camera_id": e.camera_id,
            "image_path": e.image_path,
            "plate_text": e.plate_text,
            "confidence": e.confidence,
            "status": e.status,
            "plate_category": e.plate_category,
            "layout": e.layout,
            "vehicle_type": e.vehicle_type,
            "vehicle_color": e.vehicle_color,
            "vehicle_crop_path": e.vehicle_crop_path,
            "appearance_available": bool(e.appearance_embedding),
            "appearance_model": e.appearance_model,
            "appearance_embedding_version": e.appearance_embedding_version,
            "appearance_quality": e.appearance_quality,
            "rule_violations": _parse_candidates(e.rule_violations),
            "timestamp": iso_utc(e.timestamp),
        }
        for e in events
    ]


@app.get("/api/events/{event_id}/appearance")
def get_event_appearance(event_id: int, db: Session = Depends(get_db)):
    """Observation-level appearance metadata; raw embeddings stay server-side."""
    event = db.get(PlateEvent, event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    return _appearance_payload(event)


@app.get("/api/appearance/similarity")
def get_appearance_similarity(left_event_id: int = Query(...), right_event_id: int = Query(...),
                              db: Session = Depends(get_db)):
    """Compare two observation embeddings without declaring an identity match."""
    left = db.get(PlateEvent, left_event_id)
    right = db.get(PlateEvent, right_event_id)
    if not left or not right:
        raise HTTPException(status_code=404, detail="Event not found")
    score = appearance_similarity(left.appearance_embedding, right.appearance_embedding)
    same_model = left.appearance_model if left.appearance_model == right.appearance_model else None
    return {
        "left_event_id": left.id,
        "right_event_id": right.id,
        "left_available": bool(left.appearance_embedding),
        "right_available": bool(right.appearance_embedding),
        "appearance_similarity": score,
        "appearance_similarity_percent": round(score * 100, 1) if score is not None else None,
        "appearance_model": same_model,
        "embedding_versions": [left.appearance_embedding_version, right.appearance_embedding_version],
        "same_vehicle_decision": None,
    }


@app.get("/api/matches/config")
def get_match_config():
    """Expose the active matching thresholds for operators/tests."""
    return {"matching_model": "vehicle-match-v1", "matching_version": "2B-2026-09-12",
            "thresholds": matching_thresholds()}


@app.get("/api/anomaly-policy")
def get_anomaly_policy(road_type: str | None = Query(None)):
    """Expose configured speed plausibility thresholds."""
    return {"anomaly_type": "impossible_travel", "thresholds": speed_policy(road_type)}


@app.get("/api/anomalies")
def get_anomalies(anomaly_type: str | None = Query(None),
                  severity: str | None = Query(None),
                  vehicle_id: str | None = Query(None),
                  plate: str | None = Query(None),
                  status: str | None = Query(None),
                  limit: int = Query(100, ge=1, le=500),
                  db: Session = Depends(get_db)):
    query = db.query(VehicleAnomaly)
    if anomaly_type:
        query = query.filter(VehicleAnomaly.anomaly_type == anomaly_type)
    if severity:
        query = query.filter(VehicleAnomaly.severity == severity)
    if vehicle_id:
        query = query.filter(VehicleAnomaly.vehicle_id == vehicle_id)
    if plate:
        query = query.filter(VehicleAnomaly.plate_text == normalize_plate_text(plate).normalized_text)
    if status:
        query = query.filter(VehicleAnomaly.status == status)
    rows = query.order_by(VehicleAnomaly.detected_at.desc(), VehicleAnomaly.id.desc()).limit(limit).all()
    return {"items": [anomaly_payload(row) for row in rows], "total": len(rows)}


@app.get("/api/plate-suspicion-policy")
def get_plate_suspicion_policy():
    return suspicion_config()


@app.get("/api/plate-suspicions")
def get_plate_suspicions(plate: str | None = Query(None),
                         classification: str | None = Query(None),
                         status: str | None = Query(None),
                         vehicle_id: str | None = Query(None),
                         limit: int = Query(100, ge=1, le=500),
                         db: Session = Depends(get_db)):
    query = db.query(PlateSuspicionEvent)
    if plate:
        normalized = normalize_plate_text(plate).normalized_text
        if normalized:
            plate_suspicion_summary(db, normalized, persist=True)
            db.commit()
            query = query.filter(PlateSuspicionEvent.plate_text == normalized)
        else:
            query = query.filter(PlateSuspicionEvent.plate_text == "__invalid__")
    if classification:
        query = query.filter(PlateSuspicionEvent.classification == classification)
    if status:
        query = query.filter(PlateSuspicionEvent.status == status)
    if vehicle_id:
        query = query.filter(PlateSuspicionEvent.vehicle_id == vehicle_id)
    rows = (
        query.order_by(PlateSuspicionEvent.suspicion_score.desc(), PlateSuspicionEvent.updated_at.desc())
        .limit(limit)
        .all()
    )
    return {"items": [suspicion_payload(row) for row in rows], "total": len(rows)}


@app.get("/api/plates/{plate_text}/suspicion")
def get_plate_suspicion(plate_text: str, db: Session = Depends(get_db)):
    summary = plate_suspicion_summary(db, plate_text, persist=True)
    db.commit()
    return summary


@app.patch("/api/plate-suspicions/{suspicion_id}/review")
def review_plate_suspicion(suspicion_id: int, payload: dict, db: Session = Depends(get_db)):
    row = db.get(PlateSuspicionEvent, suspicion_id)
    if not row:
        raise HTTPException(status_code=404, detail="Plate suspicion not found")
    status = str(payload.get("status", "")).strip().lower()
    if status not in {"open", "acknowledged", "dismissed"}:
        raise HTTPException(status_code=400, detail="status must be open, acknowledged, or dismissed")
    row.status = status
    row.reviewed_by = payload.get("reviewed_by") or row.reviewed_by
    row.reviewed_at = datetime.datetime.utcnow() if status in {"acknowledged", "dismissed"} else None
    row.updated_at = datetime.datetime.utcnow()
    db.commit()
    db.refresh(row)
    return suspicion_payload(row)


@app.get("/api/route-anomaly-policy")
def get_route_anomaly_policy():
    return route_anomaly_config()


def validate_traffic_filters(db, start, end, minutes, bucket="hour",
                             camera_id=None, source_camera_id=None, destination_camera_id=None):
    start, end = validate_search_filters(start, end, minutes, None, None)
    try:
        bucket = validate_bucket(bucket)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    for value, label in ((camera_id, "camera_id"), (source_camera_id, "source_camera_id"),
                         (destination_camera_id, "destination_camera_id")):
        if value and not db.get(Camera, value):
            raise HTTPException(status_code=404, detail=f"Unknown {label}")
    return start, end, bucket


@app.get("/api/traffic/policy")
def get_traffic_policy():
    return traffic_policy()


@app.get("/api/traffic/summary")
def get_traffic_summary(start: datetime.datetime | None = Query(None),
                        end: datetime.datetime | None = Query(None),
                        minutes: int | None = Query(None, ge=1),
                        db: Session = Depends(get_db)):
    start, end, _bucket = validate_traffic_filters(db, start, end, minutes)
    return traffic_summary(db, start=start, end=end)


@app.get("/api/traffic/cameras")
def get_traffic_cameras(start: datetime.datetime | None = Query(None),
                        end: datetime.datetime | None = Query(None),
                        minutes: int | None = Query(None, ge=1),
                        camera_id: str | None = Query(None),
                        db: Session = Depends(get_db)):
    start, end, _bucket = validate_traffic_filters(db, start, end, minutes, camera_id=camera_id)
    return {"items": camera_metrics(db, start=start, end=end, camera_id=camera_id), "metric_basis": "camera_observations"}


@app.get("/api/traffic/flow")
def get_traffic_flow(start: datetime.datetime | None = Query(None),
                     end: datetime.datetime | None = Query(None),
                     minutes: int | None = Query(None, ge=1),
                     source_camera_id: str | None = Query(None),
                     destination_camera_id: str | None = Query(None),
                     db: Session = Depends(get_db)):
    start, end, _bucket = validate_traffic_filters(db, start, end, minutes,
                                                   source_camera_id=source_camera_id,
                                                   destination_camera_id=destination_camera_id)
    return {"items": flow_metrics(db, start=start, end=end, source_camera_id=source_camera_id,
                                  destination_camera_id=destination_camera_id),
            "metric_basis": "direct_consecutive_global_vehicle_transitions"}


@app.get("/api/traffic/od-matrix")
def get_traffic_od_matrix(start: datetime.datetime | None = Query(None),
                          end: datetime.datetime | None = Query(None),
                          minutes: int | None = Query(None, ge=1),
                          bucket: str = Query("hour"),
                          source_camera_id: str | None = Query(None),
                          destination_camera_id: str | None = Query(None),
                          db: Session = Depends(get_db)):
    start, end, bucket = validate_traffic_filters(db, start, end, minutes, bucket=bucket,
                                                  source_camera_id=source_camera_id,
                                                  destination_camera_id=destination_camera_id)
    return {"items": od_matrix(db, start=start, end=end, bucket=bucket,
                               source_camera_id=source_camera_id,
                               destination_camera_id=destination_camera_id),
            "bucket": bucket,
            "metric_basis": "direct_consecutive_global_vehicle_transitions"}


@app.get("/api/traffic/density")
def get_traffic_density(start: datetime.datetime | None = Query(None),
                        end: datetime.datetime | None = Query(None),
                        minutes: int | None = Query(None, ge=1),
                        camera_id: str | None = Query(None),
                        db: Session = Depends(get_db)):
    start, end, _bucket = validate_traffic_filters(db, start, end, minutes, camera_id=camera_id)
    return {"items": density_metrics(db, start=start, end=end, camera_id=camera_id),
            "metric_basis": "camera_observation_density"}


@app.get("/api/traffic/congestion")
def get_traffic_congestion(start: datetime.datetime | None = Query(None),
                           end: datetime.datetime | None = Query(None),
                           minutes: int | None = Query(None, ge=1),
                           source_camera_id: str | None = Query(None),
                           destination_camera_id: str | None = Query(None),
                           db: Session = Depends(get_db)):
    start, end, _bucket = validate_traffic_filters(db, start, end, minutes,
                                                   source_camera_id=source_camera_id,
                                                   destination_camera_id=destination_camera_id)
    return {"items": congestion_metrics(db, start=start, end=end, source_camera_id=source_camera_id,
                                        destination_camera_id=destination_camera_id),
            "metric_basis": "estimated_camera_to_camera_average_speed_and_volume"}


@app.get("/api/traffic/dwell")
def get_traffic_dwell(start: datetime.datetime | None = Query(None),
                      end: datetime.datetime | None = Query(None),
                      minutes: int | None = Query(None, ge=1),
                      camera_id: str | None = Query(None),
                      bucket: str = Query("hour"),
                      db: Session = Depends(get_db)):
    start, end, bucket = validate_traffic_filters(db, start, end, minutes, bucket=bucket, camera_id=camera_id)
    return {"items": dwell_metrics(db, start=start, end=end, camera_id=camera_id, bucket=bucket),
            "bucket": bucket,
            "metric_basis": "camera_observation_dwell_estimate"}


@app.get("/api/traffic/roads")
def get_traffic_roads(start: datetime.datetime | None = Query(None),
                      end: datetime.datetime | None = Query(None),
                      minutes: int | None = Query(None, ge=1),
                      source_camera_id: str | None = Query(None),
                      destination_camera_id: str | None = Query(None),
                      db: Session = Depends(get_db)):
    start, end, _bucket = validate_traffic_filters(db, start, end, minutes,
                                                   source_camera_id=source_camera_id,
                                                   destination_camera_id=destination_camera_id)
    return {"items": road_metrics(db, start=start, end=end, source_camera_id=source_camera_id,
                                  destination_camera_id=destination_camera_id),
            "metric_basis": "configured_road_connections"}


@app.get("/api/traffic/heatmap")
def get_traffic_heatmap(start: datetime.datetime | None = Query(None),
                        end: datetime.datetime | None = Query(None),
                        minutes: int | None = Query(None, ge=1),
                        db: Session = Depends(get_db)):
    start, end, _bucket = validate_traffic_filters(db, start, end, minutes)
    return {"points": heatmap_points(db, start=start, end=end), "metric_basis": "camera_observation_activity"}


@app.get("/api/traffic/timeseries")
def get_traffic_timeseries(start: datetime.datetime | None = Query(None),
                           end: datetime.datetime | None = Query(None),
                           minutes: int | None = Query(None, ge=1),
                           camera_id: str | None = Query(None),
                           bucket: str = Query("hour"),
                           db: Session = Depends(get_db)):
    start, end, bucket = validate_traffic_filters(db, start, end, minutes, bucket=bucket, camera_id=camera_id)
    return {"items": timeseries(db, start=start, end=end, camera_id=camera_id, bucket=bucket), "bucket": bucket}


@app.get("/api/traffic/lanes")
def get_traffic_lanes():
    return lane_metrics()


@app.get("/api/traffic/dashboard")
def get_traffic_dashboard(start: datetime.datetime | None = Query(None),
                          end: datetime.datetime | None = Query(None),
                          minutes: int | None = Query(None, ge=1),
                          bucket: str = Query("hour"),
                          db: Session = Depends(get_db)):
    start, end, bucket = validate_traffic_filters(db, start, end, minutes, bucket=bucket)
    return traffic_dashboard(db, start=start, end=end, bucket=bucket)


@app.get("/api/route-anomalies")
def get_route_anomalies(vehicle_id: str | None = Query(None),
                        plate: str | None = Query(None),
                        classification: str | None = Query(None),
                        status: str | None = Query(None),
                        limit: int = Query(100, ge=1, le=500),
                        db: Session = Depends(get_db)):
    query = db.query(RouteAnomalyEvent)
    if vehicle_id:
        query = query.filter(RouteAnomalyEvent.vehicle_id == vehicle_id)
    if plate:
        normalized = normalize_plate_text(plate).normalized_text
        query = query.filter(RouteAnomalyEvent.plate_text == normalized) if normalized else query.filter(RouteAnomalyEvent.plate_text == "__invalid__")
    if classification:
        query = query.filter(RouteAnomalyEvent.classification == classification)
    if status:
        query = query.filter(RouteAnomalyEvent.status == status)
    rows = (
        query.order_by(RouteAnomalyEvent.route_anomaly_score.desc(), RouteAnomalyEvent.updated_at.desc())
        .limit(limit)
        .all()
    )
    return {"items": [route_anomaly_payload(row) for row in rows], "total": len(rows)}


@app.get("/api/vehicles/{vehicle_id}/route-anomalies")
def get_vehicle_route_anomalies(vehicle_id: str, db: Session = Depends(get_db)):
    vehicle = get_vehicle_by_id(db, vehicle_id)
    if not vehicle:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    summary = route_anomaly_summary(db, vehicle_id, persist=True)
    db.commit()
    return summary


@app.patch("/api/route-anomalies/{route_anomaly_id}/review")
def review_route_anomaly(route_anomaly_id: int, payload: dict, db: Session = Depends(get_db)):
    row = db.get(RouteAnomalyEvent, route_anomaly_id)
    if not row:
        raise HTTPException(status_code=404, detail="Route anomaly not found")
    status = str(payload.get("status", "")).strip().lower()
    if status not in {"open", "acknowledged", "dismissed"}:
        raise HTTPException(status_code=400, detail="status must be open, acknowledged, or dismissed")
    row.status = status
    row.reviewed_by = payload.get("reviewed_by") or row.reviewed_by
    row.reviewed_at = datetime.datetime.utcnow() if status in {"acknowledged", "dismissed"} else None
    row.updated_at = datetime.datetime.utcnow()
    db.commit()
    db.refresh(row)
    return route_anomaly_payload(row)


@app.get("/api/matches/review")
def get_pending_match_reviews(limit: int = Query(20, ge=1, le=100), db: Session = Depends(get_db)):
    rows = (
        db.query(VehicleMatchCandidate)
        .filter(VehicleMatchCandidate.review_status == "pending")
        .order_by(VehicleMatchCandidate.final_confidence.desc(), VehicleMatchCandidate.created_at.desc())
        .limit(limit)
        .all()
    )
    return {"items": [match_payload(db, row) for row in rows], "total": len(rows)}


@app.get("/api/matches/observation/{event_id}")
def get_observation_matches(event_id: int, db: Session = Depends(get_db)):
    matches = matches_for_observation(db, event_id)
    if matches is None:
        raise HTTPException(status_code=404, detail="Event not found")
    db.commit()
    return {"event_id": event_id, "matches": matches}


@app.get("/api/matches/{match_id}")
def get_match_evidence(match_id: int, db: Session = Depends(get_db)):
    match = db.get(VehicleMatchCandidate, match_id)
    if not match:
        raise HTTPException(status_code=404, detail="Match candidate not found")
    return match_payload(db, match)


@app.post("/api/matches/{match_id}/review")
async def review_match_candidate(match_id: int, request: Request, db: Session = Depends(get_db)):
    content_type = request.headers.get("content-type", "")
    action = None
    reviewer = "operator"
    if "application/json" in content_type:
        payload = await request.json()
        action = payload.get("action")
        reviewer = payload.get("reviewed_by") or reviewer
    else:
        form = await request.form()
        action = form.get("action")
        reviewer = form.get("reviewed_by") or reviewer
        await form.close()
    try:
        match = review_match(db, match_id, action, reviewer)
    except ValueError as exc:
        raise HTTPException(status_code=409 if "Global Vehicle IDs" in str(exc) or "No existing" in str(exc) else 422,
                            detail=str(exc)) from exc
    if not match:
        raise HTTPException(status_code=404, detail="Match candidate not found")
    db.commit()
    payload = match_payload(db, match)
    await event_hub.broadcast({"type": "match_reviewed", "match": payload})
    return payload


@app.get("/api/vehicles")
def get_vehicles(db: Session = Depends(get_db)):
    """Distinct vehicles seen so far — powers the dashboard's vehicle list."""
    return list_all_vehicles(db)


@app.get("/api/vehicles/lookup")
def lookup_vehicle(global_vehicle_id: str | None = Query(None),
                   plate_text: str | None = Query(None),
                   db: Session = Depends(get_db)):
    """Resolve either a Global Vehicle ID or a valid plate to the vehicle record."""
    if global_vehicle_id:
        vehicle = get_vehicle_by_id(db, global_vehicle_id.strip())
    elif plate_text:
        vehicle = get_vehicle_by_plate(db, plate_text.strip())
    else:
        raise HTTPException(status_code=422, detail="Provide global_vehicle_id or plate_text")
    if not vehicle:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    return vehicle_payload(db, vehicle)


@app.get("/api/vehicles/by-plate/{plate_text}")
def get_vehicle_for_plate(plate_text: str, db: Session = Depends(get_db)):
    """Look up the Global Vehicle ID for a confirmed valid plate."""
    vehicle = get_vehicle_by_plate(db, plate_text)
    if not vehicle:
        raise HTTPException(status_code=404, detail="Vehicle not found for this plate")
    return vehicle_payload(db, vehicle)


@app.get("/api/vehicles/{vehicle_id}/history")
def get_vehicle_history(vehicle_id: str, db: Session = Depends(get_db)):
    """Chronological observations for one Global Vehicle ID."""
    history = vehicle_history(db, vehicle_id)
    if not history:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    if history.get("primary_plate_text"):
        history["plate_suspicion"] = plate_suspicion_summary(db, history["primary_plate_text"], persist=True)
    history["route_anomaly"] = route_anomaly_summary(db, vehicle_id, persist=True)
    db.commit()
    return history


@app.get("/api/vehicles/{vehicle_id}/investigation")
def get_vehicle_investigation(vehicle_id: str, db: Session = Depends(get_db)):
    """Unified explainable investigation view for one Global Vehicle ID."""
    investigation = build_vehicle_investigation(db, vehicle_id)
    if not investigation:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    db.commit()
    return investigation


@app.get("/api/plates/{plate_text}/investigation")
def get_plate_investigation(plate_text: str, db: Session = Depends(get_db)):
    """Open an investigation by confirmed valid plate text."""
    investigation = build_plate_investigation(db, plate_text)
    if not investigation:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    db.commit()
    return investigation


@app.get("/api/vehicles/{vehicle_id}/trajectory")
def get_vehicle_trajectory(vehicle_id: str, start: datetime.datetime | None = Query(None),
                           end: datetime.datetime | None = Query(None),
                           minutes: int | None = Query(None, ge=1),
                           db: Session = Depends(get_db)):
    """Complete chronological multi-camera trajectory for one Global Vehicle ID."""
    start, end = validate_search_filters(start, end, minutes, None, None)
    trajectory = build_vehicle_trajectory(db, vehicle_id, start=start, end=end)
    if not trajectory:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    if trajectory.get("plate_text"):
        trajectory["plate_suspicion"] = plate_suspicion_summary(db, trajectory["plate_text"], persist=True)
    trajectory["route_anomaly"] = route_anomaly_summary(db, vehicle_id, persist=True)
    db.commit()
    return trajectory


@app.get("/api/vehicles/{vehicle_id}/observations")
def get_vehicle_observations(vehicle_id: str, db: Session = Depends(get_db)):
    """Observation list for one Global Vehicle ID."""
    history = vehicle_history(db, vehicle_id)
    if not history:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    return {"global_vehicle_id": history["global_vehicle_id"], "vehicle_id": history["vehicle_id"],
            "observations": history["observations"]}


@app.get("/api/vehicles/{vehicle_id}/speed-history")
def get_vehicle_speed_history(vehicle_id: str, db: Session = Depends(get_db)):
    """Chronological camera-to-camera travel-time and estimated average speed segments."""
    vehicle = get_vehicle_by_id(db, vehicle_id)
    if not vehicle:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    history = vehicle_speed_history(db, vehicle_id)
    if history is None:
        raise HTTPException(status_code=404, detail="Vehicle history not found")
    return history


@app.get("/api/vehicles/{vehicle_id}/anomalies")
def get_vehicle_anomalies(vehicle_id: str, status: str | None = Query(None),
                          limit: int = Query(100, ge=1, le=500),
                          db: Session = Depends(get_db)):
    vehicle = get_vehicle_by_id(db, vehicle_id)
    if not vehicle:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    query = db.query(VehicleAnomaly).filter(VehicleAnomaly.vehicle_id == vehicle_id)
    if status:
        query = query.filter(VehicleAnomaly.status == status)
    rows = query.order_by(VehicleAnomaly.detected_at.desc(), VehicleAnomaly.id.desc()).limit(limit).all()
    return {"global_vehicle_id": vehicle_id, "vehicle_id": vehicle_id,
            "items": [anomaly_payload(row) for row in rows], "total": len(rows)}


@app.get("/api/vehicles/{vehicle_id}/matches")
def get_vehicle_matches(vehicle_id: str, db: Session = Depends(get_db)):
    vehicle = get_vehicle_by_id(db, vehicle_id)
    if not vehicle:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    return {"global_vehicle_id": vehicle_id, "vehicle_id": vehicle_id,
            "matches": matches_for_vehicle(db, vehicle_id)}


@app.get("/api/trajectory/{plate_text}")
def get_trajectory(plate_text: str, start: datetime.datetime | None = None,
                   end: datetime.datetime | None = None, lat: float | None = None, lng: float | None = None,
                   radius_m: float = Query(5000, ge=1), minutes: int | None = Query(None, ge=1),
                   db: Session = Depends(get_db)):
    """The stitched path for one vehicle across all cameras that saw it."""
    start, end = validate_search_filters(start, end, minutes, lat, lng)
    hops = build_trajectory(db, plate_text.upper().strip(), start=start, end=end, lat=lat, lng=lng, radius_m=radius_m)
    if not hops:
        raise HTTPException(status_code=404, detail="No sightings for this plate")
    segments = [hop for hop in hops[1:]]
    plate_suspicion = plate_suspicion_summary(db, plate_text.upper().strip(), persist=True)
    complete = build_plate_complete_trajectory(db, plate_text.upper().strip(), start=start, end=end,
                                               lat=lat, lng=lng, radius_m=radius_m)
    route_anomaly = route_anomaly_summary(db, complete["vehicle_id"], persist=True) if complete.get("vehicle_id") else None
    complete = build_plate_complete_trajectory(db, plate_text.upper().strip(), start=start, end=end,
                                               lat=lat, lng=lng, radius_m=radius_m)
    db.commit()
    return {
        "plate_text": plate_text.upper().strip(),
        "hops": hops,
        "speed_summary": speed_summary(segments),
        "plate_suspicion": plate_suspicion,
        "route_anomaly": route_anomaly,
        "vehicle": complete["vehicle"],
        "global_vehicle_id": complete["global_vehicle_id"],
        "vehicle_id": complete["vehicle_id"],
        "observations": complete["observations"],
        "trajectory_hops": complete["trajectory_hops"],
        "segments": complete["segments"],
        "summary": complete["summary"],
    }


def validate_search_filters(start, end, minutes, lat, lng):
    if (lat is not None or lng is not None) and not valid_coordinates(lat, lng):
        raise HTTPException(status_code=422, detail="A valid latitude/longitude pair is required")
    try:
        return time_window(start, end, minutes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/search/plates")
def search_plate_sightings(
    q: str = Query(..., min_length=3),
    start: datetime.datetime | None = Query(None),
    end: datetime.datetime | None = Query(None),
    lat: float | None = Query(None),
    lng: float | None = Query(None),
    radius_m: float = Query(5000, ge=1),
    minutes: int = Query(30, ge=1),
    similarity: float = Query(0.7, ge=0.0, le=1.0),
    db: Session = Depends(get_db),
):
    """Fuzzy + geo + time-window vehicle sighting search."""
    query_plate = normalize_plate_text(q).normalized_text or q.upper().strip()
    since, until = validate_search_filters(start, end, minutes, lat, lng)

    if db.bind.dialect.name == "postgresql" and lat is not None and lng is not None:
        rows = db.execute(
            text(
                """
                SELECT
                    e.id AS event_id,
                    e.vehicle_id AS global_vehicle_id,
                    e.vehicle_id AS vehicle_id,
                    e.camera_id,
                    c.label AS camera_label,
                    e.plate_text,
                    e.confidence,
                    e.status,
                    e.timestamp,
                    e.image_path,
                    e.plate_category,
                    e.layout,
                    e.vehicle_type,
                    e.vehicle_color,
                    e.vehicle_crop_path,
                    e.appearance_model,
                    e.appearance_embedding_version,
                    e.appearance_quality,
                    e.appearance_embedding IS NOT NULL AS appearance_available,
                    similarity(e.plate_text, :plate) AS match_score,
                    ST_Distance(
                        CAST(ST_SetSRID(ST_MakePoint(c.lng, c.lat), 4326) AS geography),
                        CAST(ST_SetSRID(ST_MakePoint(:lng, :lat), 4326) AS geography)
                    ) AS distance_m
                FROM plate_events e
                JOIN cameras c ON c.camera_id = e.camera_id
                WHERE e.plate_text IS NOT NULL
                  AND similarity(e.plate_text, :plate) >= :similarity
                  AND ST_DWithin(
                        CAST(ST_SetSRID(ST_MakePoint(c.lng, c.lat), 4326) AS geography),
                        CAST(ST_SetSRID(ST_MakePoint(:lng, :lat), 4326) AS geography),
                        :radius_m
                      )
                  AND e.timestamp >= :since
                  AND (CAST(:until AS timestamp) IS NULL OR e.timestamp <= CAST(:until AS timestamp))
                  AND c.location_known = TRUE
                ORDER BY match_score DESC, e.timestamp DESC
                LIMIT 200
                """
            ),
            {
                "plate": query_plate,
                "lat": lat,
                "lng": lng,
                "radius_m": radius_m,
                "similarity": similarity,
                "since": since,
                "until": until,
            },
        ).mappings().all()
        return {"query": query_plate, "engine": "postgres_postgis_pg_trgm", "matches": [dict(row) for row in rows]}

    cameras = {camera.camera_id: camera for camera in db.query(Camera).all()}
    events = (
        db.query(PlateEvent)
        .filter(PlateEvent.plate_text.isnot(None))
        .filter(PlateEvent.timestamp >= since)
        .order_by(PlateEvent.timestamp.desc())
    )
    if until is not None:
        events = events.filter(PlateEvent.timestamp <= until)
    events = events.yield_per(500)
    matches = []
    for event in events:
        camera = cameras.get(event.camera_id)
        score = plate_similarity_score(event.plate_text, query_plate)
        if score < similarity:
            continue
        distance_m = None
        if lat is not None and lng is not None:
            position = camera_location(camera)
            if position[0] is None:
                continue
            distance_m = haversine_km(lat, lng, *position) * 1000
            if distance_m > radius_m:
                continue
        matches.append({
            "event_id": event.id,
            "global_vehicle_id": event.vehicle_id,
            "vehicle_id": event.vehicle_id,
            "camera_id": event.camera_id,
            "camera_label": camera.label if camera else event.camera_id,
            "plate_text": event.plate_text,
            "confidence": event.confidence,
            "status": event.status,
            "timestamp": iso_utc(event.timestamp),
            "image_path": event.image_path,
            "plate_category": event.plate_category,
            "layout": event.layout,
            "vehicle_type": event.vehicle_type,
            "vehicle_color": event.vehicle_color,
            "vehicle_crop_path": event.vehicle_crop_path,
            "appearance_available": bool(event.appearance_embedding),
            "appearance_model": event.appearance_model,
            "appearance_embedding_version": event.appearance_embedding_version,
            "appearance_quality": event.appearance_quality,
            "match_score": round(score, 3),
            "distance_m": round(distance_m, 1) if distance_m is not None else None,
        })
    matches.sort(key=lambda item: (item["match_score"], item["timestamp"]), reverse=True)
    return {"query": query_plate, "engine": "sqlite_python_fallback", "matches": matches[:200]}


@app.get("/api/stats")
def get_stats(db: Session = Depends(get_db)):
    """Quick summary numbers for the dashboard header."""
    total_events = db.query(PlateEvent).count()
    needs_review = (
        db.query(PlateEvent)
        .filter(PlateEvent.status.in_(["needs_review", PENDING_REVIEW_STATUS]))
        .count()
    )
    vehicles = list_all_vehicles(db)
    return {
        "total_scans": total_events,
        "unique_vehicles": len(vehicles),
        "needs_review": needs_review,
    }


if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
