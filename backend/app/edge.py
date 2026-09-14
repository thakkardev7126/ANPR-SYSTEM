"""Phase 9 edge ingestion, queue, simulation, health, and retention services.

The edge layer is metadata-first: simulated or real edge agents run the existing
ANPR pipeline near the camera, then send compact observations to the central API.
Central persistence still goes through ``persist_plate_event`` so identity,
matching, trajectory, hotlist, enforcement, analytics, RBAC, and audit behavior
remain shared with the rest of the application.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import random
import shutil
import time
import uuid
from collections import deque
from dataclasses import dataclass

from sqlalchemy.exc import IntegrityError

from app.database import (
    Camera,
    EdgeDevice,
    EdgeObservation,
    PlateEvent,
    RetentionRun,
    SessionLocal,
    persist_plate_event,
)
from app.plate_rules import OCR_ACCEPT_CONFIDENCE, PENDING_REVIEW_STATUS, normalize_plate_text, status_for_plate


EDGE_STATUS_ONLINE = "ONLINE"
EDGE_STATUS_DEGRADED = "DEGRADED"
EDGE_STATUS_OFFLINE = "OFFLINE"
OBS_RECEIVED = "RECEIVED"
OBS_QUEUED = "QUEUED"
OBS_PROCESSED = "PROCESSED"
OBS_FAILED = "FAILED"
OBS_DUPLICATE = "DUPLICATE"
QUEUE_NAME = "edge_observations"


def utcnow():
    return dt.datetime.utcnow()


def parse_timestamp(value, *, field="timestamp"):
    if isinstance(value, dt.datetime):
        return value.replace(tzinfo=None)
    if value is None:
        raise ValueError(f"{field} is required")
    try:
        parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be ISO-8601") from exc
    return parsed.replace(tzinfo=None)


def safe_json(value):
    return json.dumps(value or {}, sort_keys=True, default=str)


@dataclass
class FrameSamplingConfig:
    source_fps: float = float(os.getenv("ANPR_EDGE_SOURCE_FPS", "30"))
    processing_fps: float = float(os.getenv("ANPR_EDGE_PROCESSING_FPS", "2"))
    max_processing_fps: float = float(os.getenv("ANPR_EDGE_MAX_PROCESSING_FPS", "5"))

    @property
    def effective_processing_fps(self):
        source = max(float(self.source_fps or 0.0), 0.01)
        requested = max(float(self.processing_fps or 0.0), 0.01)
        maximum = max(float(self.max_processing_fps or 0.0), 0.01)
        return min(source, requested, maximum)

    @property
    def sampling_interval(self):
        return max(1, int(math.ceil(max(self.source_fps, 0.01) / self.effective_processing_fps)))

    def should_process(self, sequence_number):
        return int(sequence_number or 0) % self.sampling_interval == 0

    def payload(self):
        return {
            "source_fps": self.source_fps,
            "processing_fps": self.processing_fps,
            "max_processing_fps": self.max_processing_fps,
            "effective_processing_fps": self.effective_processing_fps,
            "sampling_interval": self.sampling_interval,
        }


class LocalEdgeQueue:
    def __init__(self):
        self.items = deque()

    @property
    def mode(self):
        return "local"

    def enqueue(self, observation_id):
        self.items.append(observation_id)

    def pop(self):
        return self.items.popleft() if self.items else None

    def depth(self):
        return len(self.items)


class RedisEdgeQueue:
    def __init__(self):
        import redis
        self.client = redis.Redis.from_url(os.getenv("ANPR_REDIS_URL", "redis://localhost:6379/0"))
        self.client.ping()

    @property
    def mode(self):
        return "redis"

    def enqueue(self, observation_id):
        self.client.xadd(QUEUE_NAME, {"observation_id": observation_id})

    def pop(self):
        # Demo worker path remains database-driven; Redis Streams are exposed as
        # the production-style adapter without requiring a local Redis server.
        return None

    def depth(self):
        try:
            return int(self.client.xlen(QUEUE_NAME))
        except Exception:
            return 0


class EdgeQueueManager:
    def __init__(self):
        self.fallback_reason = None
        requested = os.getenv("ANPR_EDGE_QUEUE_MODE", "local").strip().lower()
        if requested == "redis":
            try:
                self.adapter = RedisEdgeQueue()
            except Exception as exc:
                self.adapter = LocalEdgeQueue()
                self.fallback_reason = f"redis_unavailable:{exc.__class__.__name__}"
        else:
            self.adapter = LocalEdgeQueue()

    @property
    def mode(self):
        return self.adapter.mode

    def enqueue(self, observation_id):
        self.adapter.enqueue(observation_id)

    def pop(self):
        return self.adapter.pop()

    def depth(self):
        return self.adapter.depth()


edge_queue = EdgeQueueManager()


class SimulationState:
    def __init__(self):
        self.active = False
        self.started_at = None
        self.completed_at = None
        self.config = {}
        self.metrics = {
            "generated": 0,
            "accepted": 0,
            "duplicates": 0,
            "failed": 0,
            "camera_count": 0,
        }

    def start(self, config):
        self.active = True
        self.started_at = utcnow()
        self.completed_at = None
        self.config = dict(config)
        self.metrics = {"generated": 0, "accepted": 0, "duplicates": 0, "failed": 0,
                        "camera_count": int(config.get("camera_count") or 0)}

    def finish(self):
        self.active = False
        self.completed_at = utcnow()

    def payload(self):
        return {
            "active": self.active,
            "label": "SIMULATION / DEMO",
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "config": self.config,
            "metrics": self.metrics,
        }


simulation_state = SimulationState()


def normalize_observation_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError("observation must be an object")
    observation_id = str(payload.get("observation_id") or "").strip()
    camera_id = str(payload.get("camera_id") or "").strip()
    edge_device_id = str(payload.get("edge_device_id") or "").strip()
    if not observation_id or len(observation_id) > 120:
        raise ValueError("observation_id is required and must be <= 120 characters")
    if not camera_id or len(camera_id) > 64:
        raise ValueError("camera_id is required and must be <= 64 characters")
    if not edge_device_id or len(edge_device_id) > 80:
        raise ValueError("edge_device_id is required and must be <= 80 characters")
    capture_timestamp = parse_timestamp(payload.get("capture_timestamp") or payload.get("timestamp"),
                                        field="capture_timestamp")
    sequence_number = payload.get("sequence_number")
    if sequence_number is not None:
        sequence_number = int(sequence_number)
    confidence = float(payload.get("ocr_confidence", payload.get("confidence", 0.0)) or 0.0)
    plate_text = payload.get("plate_text")
    rule = normalize_plate_text(plate_text)
    status = payload.get("status") or status_for_plate(confidence, rule)
    if not rule.valid_format:
        status = PENDING_REVIEW_STATUS if plate_text else (payload.get("processing_status") or "NO_PLATE")
    return {
        **payload,
        "observation_id": observation_id,
        "camera_id": camera_id,
        "edge_device_id": edge_device_id,
        "capture_timestamp": capture_timestamp,
        "frame_id": str(payload.get("frame_id") or "")[:120] or None,
        "sequence_number": sequence_number,
        "plate_text": rule.normalized_text or plate_text,
        "confidence": confidence,
        "status": status,
    }


def upsert_edge_device_and_camera(db, payload):
    now = utcnow()
    device = db.query(EdgeDevice).filter(EdgeDevice.edge_device_id == payload["edge_device_id"]).first()
    if not device:
        device = EdgeDevice(edge_device_id=payload["edge_device_id"], created_at=now)
        db.add(device)
    device.label = payload.get("edge_label") or device.label
    device.status = EDGE_STATUS_ONLINE
    device.last_seen_at = now
    device.queue_mode = edge_queue.mode
    device.config_json = safe_json(payload.get("sampling") or FrameSamplingConfig().payload())
    secret = os.getenv(f"ANPR_EDGE_TOKEN_{payload['edge_device_id'].replace('-', '_').upper()}") or os.getenv("ANPR_EDGE_TOKEN")
    if secret:
        device.credential_hash = hashlib.sha256(secret.encode("utf-8")).hexdigest()
        device.credential_version = os.getenv("ANPR_EDGE_TOKEN_VERSION", "v1")
    device.frames_sampled = (device.frames_sampled or 0) + int(payload.get("frames_sampled", 1) or 1)
    device.observations_received = (device.observations_received or 0) + 1
    device.updated_at = now

    camera = db.get(Camera, payload["camera_id"])
    if camera:
        camera.edge_device_id = payload["edge_device_id"]
        camera.edge_status = EDGE_STATUS_ONLINE
        camera.last_seen = now
        camera.edge_config_json = device.config_json
        camera.frames_sampled = (camera.frames_sampled or 0) + int(payload.get("frames_sampled", 1) or 1)
        camera.observations_produced = (camera.observations_produced or 0) + 1
    return device, camera


def enqueue_edge_observation(payload, *, max_retries=None):
    normalized = normalize_observation_payload(payload)
    now = utcnow()
    with SessionLocal() as db:
        if not db.get(Camera, normalized["camera_id"]):
            raise ValueError(f"Unknown camera_id '{normalized['camera_id']}'")
        existing = db.query(EdgeObservation).filter(
            EdgeObservation.observation_id == normalized["observation_id"]
        ).first()
        device, camera = upsert_edge_device_and_camera(db, normalized)
        if existing:
            existing.status = existing.status if existing.status == OBS_PROCESSED else OBS_DUPLICATE
            existing.updated_at = now
            device.duplicate_observations = (device.duplicate_observations or 0) + 1
            db.commit()
            return existing, True
        row = EdgeObservation(
            observation_id=normalized["observation_id"],
            camera_id=normalized["camera_id"],
            edge_device_id=normalized["edge_device_id"],
            frame_id=normalized.get("frame_id"),
            sequence_number=normalized.get("sequence_number"),
            capture_timestamp=normalized["capture_timestamp"],
            received_at=now,
            status=OBS_QUEUED,
            max_retries=int(max_retries if max_retries is not None else os.getenv("ANPR_EDGE_MAX_RETRIES", "3")),
            payload_json=safe_json(normalized),
            queue_name=QUEUE_NAME,
            created_at=now,
            updated_at=now,
        )
        db.add(row)
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            existing = db.query(EdgeObservation).filter(
                EdgeObservation.observation_id == normalized["observation_id"]
            ).first()
            if existing:
                device.duplicate_observations = (device.duplicate_observations or 0) + 1
                db.commit()
                return existing, True
            raise
        edge_queue.enqueue(row.observation_id)
        db.refresh(row)
        return row, False


def _event_values_from_observation(row, payload):
    confidence = float(payload.get("confidence") or 0.0)
    rule = normalize_plate_text(payload.get("plate_text"))
    status = payload.get("status") or status_for_plate(confidence, rule)
    if not rule.valid_format:
        status = PENDING_REVIEW_STATUS if payload.get("plate_text") else "NO_PLATE"
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    bbox = payload.get("bbox") if isinstance(payload.get("bbox"), dict) else {}
    return {
        "camera_id": row.camera_id,
        "track_id": payload.get("track_id") or row.frame_id,
        "plate_text": rule.normalized_text or payload.get("plate_text"),
        "confidence": confidence,
        "status": status,
        "raw_ocr_candidates": safe_json(payload.get("raw_ocr_candidates") or [{
            "source": "edge_metadata",
            "observation_id": row.observation_id,
            "raw_text": payload.get("raw_text") or payload.get("plate_text"),
            "text": rule.normalized_text or payload.get("plate_text"),
            "confidence": confidence,
            "valid_format": rule.valid_format,
            "status": status,
            "bbox": bbox,
        }]),
        "plate_category": payload.get("plate_category") or metadata.get("plate_category"),
        "layout": payload.get("layout") or metadata.get("layout"),
        "rule_violations": safe_json(rule.violations),
        "bbox_x": bbox.get("x"),
        "bbox_y": bbox.get("y"),
        "bbox_width": bbox.get("width"),
        "bbox_height": bbox.get("height"),
        "vehicle_type": payload.get("vehicle_type"),
        "vehicle_color": payload.get("vehicle_color"),
        "vehicle_crop_path": payload.get("vehicle_crop_path"),
        "appearance_embedding": payload.get("appearance_embedding"),
        "appearance_model": payload.get("appearance_model"),
        "appearance_embedding_version": payload.get("appearance_embedding_version"),
        "appearance_quality": payload.get("appearance_quality"),
        "image_path": payload.get("privacy_image_path") or payload.get("image_path"),
        "privacy_image_path": payload.get("privacy_image_path"),
        "privacy_status": payload.get("privacy_status"),
        "privacy_metadata": safe_json(payload.get("privacy_metadata") or {}),
    }


def process_edge_observation(observation_id):
    now = utcnow()
    with SessionLocal() as db:
        row = db.query(EdgeObservation).filter(EdgeObservation.observation_id == observation_id).first()
        if not row:
            return None
        if row.status == OBS_PROCESSED:
            return row
        row.status = "PROCESSING"
        row.processing_started_at = now
        row.updated_at = now
        db.commit()
        payload = json.loads(row.payload_json or "{}")
        try:
            rule = normalize_plate_text(payload.get("plate_text"))
            if not payload.get("plate_text") and payload.get("processing_status") in {"NO_PLATE", "SKIPPED"}:
                event = None
            else:
                event, _duplicate = persist_plate_event(_event_values_from_observation(row, payload), window_seconds=0)
            finished = utcnow()
            row = db.query(EdgeObservation).filter(EdgeObservation.observation_id == observation_id).first()
            row.status = OBS_PROCESSED
            row.processing_completed_at = finished
            row.processing_latency_ms = (finished - row.received_at).total_seconds() * 1000.0
            row.plate_event_id = event.id if event else None
            row.updated_at = finished
            device = db.query(EdgeDevice).filter(EdgeDevice.edge_device_id == row.edge_device_id).first()
            if device:
                device.observations_processed = (device.observations_processed or 0) + 1
                device.last_processed_at = finished
                device.updated_at = finished
                device.status = EDGE_STATUS_ONLINE
            camera = db.get(Camera, row.camera_id)
            if camera:
                camera.edge_last_processed = finished
                camera.edge_status = EDGE_STATUS_ONLINE if rule.valid_format or payload.get("processing_status") != "FAILED" else EDGE_STATUS_DEGRADED
            db.commit()
            db.refresh(row)
            return row
        except Exception as exc:
            failed_at = utcnow()
            row = db.query(EdgeObservation).filter(EdgeObservation.observation_id == observation_id).first()
            row.retry_count = (row.retry_count or 0) + 1
            row.error_message = str(exc)[:500]
            if row.retry_count <= row.max_retries:
                backoff = min(60, 2 ** max(row.retry_count - 1, 0))
                row.next_retry_at = failed_at + dt.timedelta(seconds=backoff)
                row.status = OBS_QUEUED
                edge_queue.enqueue(row.observation_id)
            else:
                row.status = OBS_FAILED
            row.updated_at = failed_at
            device = db.query(EdgeDevice).filter(EdgeDevice.edge_device_id == row.edge_device_id).first()
            if device:
                device.observations_failed = (device.observations_failed or 0) + 1
                device.status = EDGE_STATUS_DEGRADED
                device.updated_at = failed_at
            camera = db.get(Camera, row.camera_id)
            if camera:
                camera.processing_errors = (camera.processing_errors or 0) + 1
                camera.edge_status = EDGE_STATUS_DEGRADED
            db.commit()
            return row


def process_due_edge_observations(limit=100):
    processed = []
    now = utcnow()
    with SessionLocal() as db:
        rows = (
            db.query(EdgeObservation)
            .filter(EdgeObservation.status == OBS_QUEUED)
            .filter((EdgeObservation.next_retry_at.is_(None)) | (EdgeObservation.next_retry_at <= now))
            .order_by(EdgeObservation.received_at.asc())
            .limit(limit)
            .all()
        )
        ids = [row.observation_id for row in rows]
    for observation_id in ids:
        processed.append(process_edge_observation(observation_id))
    return processed


def observation_payload(row):
    return {
        "observation_id": row.observation_id,
        "camera_id": row.camera_id,
        "edge_device_id": row.edge_device_id,
        "frame_id": row.frame_id,
        "sequence_number": row.sequence_number,
        "status": row.status,
        "retry_count": row.retry_count,
        "plate_event_id": row.plate_event_id,
        "processing_latency_ms": row.processing_latency_ms,
        "duplicate": row.status == OBS_DUPLICATE,
        "error_message": row.error_message,
    }


def update_health_statuses(db):
    now = utcnow()
    online_after = now - dt.timedelta(seconds=int(os.getenv("ANPR_EDGE_ONLINE_SECONDS", "90")))
    degraded_after = now - dt.timedelta(seconds=int(os.getenv("ANPR_EDGE_DEGRADED_SECONDS", "300")))
    for device in db.query(EdgeDevice).all():
        if not device.last_seen_at or device.last_seen_at < degraded_after:
            device.status = EDGE_STATUS_OFFLINE
        elif device.last_seen_at < online_after or (device.observations_failed or 0) > 0:
            device.status = EDGE_STATUS_DEGRADED
        else:
            device.status = EDGE_STATUS_ONLINE
    for camera in db.query(Camera).all():
        if not camera.last_seen or camera.last_seen < degraded_after:
            camera.edge_status = EDGE_STATUS_OFFLINE
        elif camera.last_seen < online_after or (camera.processing_errors or 0) > 0:
            camera.edge_status = EDGE_STATUS_DEGRADED
        elif camera.edge_device_id:
            camera.edge_status = EDGE_STATUS_ONLINE
    db.commit()


def edge_status_payload(db):
    update_health_statuses(db)
    cameras = db.query(Camera).all()
    devices = db.query(EdgeDevice).all()
    observations = db.query(EdgeObservation).all()
    processed = [row for row in observations if row.status == OBS_PROCESSED]
    failed = [row for row in observations if row.status == OBS_FAILED]
    duplicates = [row for row in observations if row.status == OBS_DUPLICATE]
    latencies = [row.processing_latency_ms for row in processed if row.processing_latency_ms is not None]
    one_minute_ago = utcnow() - dt.timedelta(minutes=1)
    recent_processed = [row for row in processed if row.processing_completed_at and row.processing_completed_at >= one_minute_ago]
    return {
        "architecture": "metadata-first edge ingestion with local demo queue and optional Redis Streams adapter",
        "queue": {
            "mode": edge_queue.mode,
            "name": QUEUE_NAME,
            "depth": edge_queue.depth() + db.query(EdgeObservation).filter(EdgeObservation.status == OBS_QUEUED).count(),
            "fallback_reason": edge_queue.fallback_reason,
            "worker_count": int(os.getenv("ANPR_EDGE_WORKER_COUNT", "1")),
        },
        "sampling": FrameSamplingConfig().payload(),
        "registered_cameras": len(cameras),
        "active_cameras": len([c for c in cameras if c.edge_status == EDGE_STATUS_ONLINE]),
        "online_cameras": len([c for c in cameras if c.edge_status == EDGE_STATUS_ONLINE]),
        "degraded_cameras": len([c for c in cameras if c.edge_status == EDGE_STATUS_DEGRADED]),
        "offline_cameras": len([c for c in cameras if c.edge_status == EDGE_STATUS_OFFLINE]),
        "edge_devices": len(devices),
        "observations_received": len(observations),
        "observations_processed": len(processed),
        "observations_failed": len(failed),
        "duplicate_observations": len(duplicates) + sum(device.duplicate_observations or 0 for device in devices),
        "observations_per_minute": len(recent_processed),
        "average_processing_latency_ms": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        "maximum_processing_latency_ms": round(max(latencies), 3) if latencies else 0.0,
        "processing_throughput_per_minute": len(recent_processed),
        "simulation": simulation_state.payload(),
        "cameras": [
            {
                "camera_id": camera.camera_id,
                "label": camera.label,
                "edge_device_id": camera.edge_device_id,
                "status": camera.edge_status,
                "last_seen": camera.last_seen.isoformat() if camera.last_seen else None,
                "last_processed": camera.edge_last_processed.isoformat() if camera.edge_last_processed else None,
                "frames_sampled": camera.frames_sampled or 0,
                "observations_produced": camera.observations_produced or 0,
                "processing_errors": camera.processing_errors or 0,
            }
            for camera in cameras[:50]
        ],
    }


def simulator_payload(camera_index, sequence, duplicate_of=None, failure_rate=0.0):
    camera_id = f"SIM{camera_index:04d}"
    if duplicate_of:
        observation_id = duplicate_of
    else:
        observation_id = f"sim-{camera_id}-{sequence}"
    failed = random.random() < failure_rate
    plate = None if failed else f"GJ{camera_index % 49 + 1:02d}AA{1000 + sequence % 9000:04d}"
    return {
        "observation_id": observation_id,
        "camera_id": camera_id,
        "edge_device_id": f"EDGE-SIM-{camera_index % 25:02d}",
        "frame_id": f"{camera_id}-frame-{sequence}",
        "sequence_number": sequence,
        "capture_timestamp": utcnow().isoformat(),
        "plate_text": plate,
        "ocr_confidence": 0.94 if plate else 0.0,
        "processing_status": "OK" if plate else "FAILED",
        "vehicle_type": "car",
        "vehicle_color": ["white", "silver", "black", "blue"][camera_index % 4],
        "bbox": {"x": 120, "y": 160, "width": 150, "height": 44},
        "privacy_status": "metadata_only",
        "metadata": {"simulation": True, "label": "SIMULATION / DEMO"},
    }


def ensure_simulated_cameras(db, count):
    now = utcnow()
    existing = {camera.camera_id for camera in db.query(Camera).filter(Camera.camera_id.like("SIM%")).all()}
    for index in range(1, int(count) + 1):
        camera_id = f"SIM{index:04d}"
        if camera_id in existing:
            continue
        db.add(Camera(
            camera_id=camera_id,
            label=f"Simulation Camera {index:04d}",
            lat=23.0 + (index % 50) * 0.001,
            lng=72.0 + (index % 50) * 0.001,
            location_known=True,
            active=False,
            edge_status=EDGE_STATUS_OFFLINE,
            last_seen=now,
        ))
    db.commit()


def run_simulation(config):
    camera_count = max(1, min(int(config.get("camera_count", 10)), 500))
    observation_rate = max(1, int(config.get("observation_rate", 1)))
    duration_seconds = max(0.0, min(float(config.get("duration_seconds", 1.0)), 60.0))
    duplicate_rate = min(max(float(config.get("duplicate_rate", 0.0)), 0.0), 1.0)
    failure_rate = min(max(float(config.get("failure_rate", 0.0)), 0.0), 1.0)
    sampling = FrameSamplingConfig(
        source_fps=float(config.get("source_fps", os.getenv("ANPR_EDGE_SOURCE_FPS", "30"))),
        processing_fps=float(config.get("processing_fps", os.getenv("ANPR_EDGE_PROCESSING_FPS", "2"))),
        max_processing_fps=float(config.get("max_processing_fps", os.getenv("ANPR_EDGE_MAX_PROCESSING_FPS", "5"))),
    )
    normalized_config = {
        "camera_count": camera_count,
        "observation_rate": observation_rate,
        "duration_seconds": duration_seconds,
        "duplicate_rate": duplicate_rate,
        "failure_rate": failure_rate,
        "queue_mode": edge_queue.mode,
        "sampling": sampling.payload(),
    }
    simulation_state.start(normalized_config)
    with SessionLocal() as db:
        ensure_simulated_cameras(db, camera_count)
    generated_ids = []
    total_sequences = max(camera_count, int(camera_count * observation_rate * max(duration_seconds, 0.01)))
    for sequence in range(total_sequences):
        camera_index = sequence % camera_count + 1
        if not sampling.should_process(sequence):
            continue
        duplicate_of = random.choice(generated_ids) if generated_ids and random.random() < duplicate_rate else None
        payload = simulator_payload(camera_index, sequence, duplicate_of=duplicate_of, failure_rate=failure_rate)
        try:
            row, duplicate = enqueue_edge_observation(payload)
            generated_ids.append(row.observation_id)
            simulation_state.metrics["generated"] += 1
            simulation_state.metrics["duplicates" if duplicate else "accepted"] += 1
        except Exception:
            simulation_state.metrics["failed"] += 1
        process_due_edge_observations(limit=100)
    simulation_state.finish()
    return simulation_state.payload()


def retention_policy_payload(raw_image_days=None, action=None):
    days = int(raw_image_days if raw_image_days is not None else os.getenv("ANPR_RETENTION_RAW_IMAGE_DAYS", "30"))
    selected_action = str(action or os.getenv("ANPR_RETENTION_ACTION", "purge")).lower()
    if selected_action not in {"purge", "archive"}:
        selected_action = "purge"
    return {
        "raw_image_days": max(1, days),
        "action": selected_action,
        "archive_dir": os.getenv("ANPR_RETENTION_ARCHIVE_DIR", os.path.join(os.getcwd(), "backend", "evidence_archive")),
        "categories": {
            "raw_high_resolution_image": "purge_or_archive_after_policy_window",
            "privacy_safe_image": "retain_with_metadata_unless_manually_purged",
            "vehicle_crop": "retain_with_observation_for_review",
            "plate_crop": "retain_with_observation_for_review",
            "metadata": "retained according to application database retention policy",
        },
    }


def run_retention(dry_run=True, raw_image_days=None, action=None, actor=None):
    policy = retention_policy_payload(raw_image_days, action)
    cutoff = utcnow() - dt.timedelta(days=policy["raw_image_days"])
    started = utcnow()
    result = {
        "dry_run": bool(dry_run),
        "cutoff": cutoff.isoformat(),
        "eligible_raw_images": 0,
        "deleted_raw_images": 0,
        "archived_raw_images": 0,
        "missing_raw_images": 0,
        "metadata_retained": 0,
        "items": [],
    }
    with SessionLocal() as db:
        events = (
            db.query(PlateEvent)
            .filter(PlateEvent.original_image_path.isnot(None))
            .filter(PlateEvent.timestamp <= cutoff)
            .order_by(PlateEvent.timestamp.asc())
            .all()
        )
        for event in events:
            path = event.original_image_path
            item = {"event_id": event.id, "path": path, "action": "dry_run" if dry_run else policy["action"]}
            result["eligible_raw_images"] += 1
            result["metadata_retained"] += 1
            if not path or not os.path.isfile(path):
                result["missing_raw_images"] += 1
                item["status"] = "missing"
                if not dry_run:
                    event.original_image_path = None
                result["items"].append(item)
                continue
            if dry_run:
                item["status"] = "would_archive" if policy["action"] == "archive" else "would_delete"
            elif policy["action"] == "archive":
                archive_dir = policy["archive_dir"]
                os.makedirs(archive_dir, exist_ok=True)
                destination = os.path.join(archive_dir, f"{event.id}-{uuid.uuid4().hex[:8]}-{os.path.basename(path)}")
                shutil.move(path, destination)
                event.original_image_path = None
                result["archived_raw_images"] += 1
                item["status"] = "archived"
                item["archived_to"] = destination
            else:
                os.remove(path)
                event.original_image_path = None
                result["deleted_raw_images"] += 1
                item["status"] = "deleted"
            result["items"].append(item)
        run = RetentionRun(
            run_id=str(uuid.uuid4()),
            dry_run=bool(dry_run),
            policy_json=safe_json(policy),
            result_json=safe_json(result),
            status="completed",
            started_at=started,
            completed_at=utcnow(),
            actor=actor,
        )
        db.add(run)
        db.commit()
        result["run_id"] = run.run_id
        result["policy"] = policy
    return result
