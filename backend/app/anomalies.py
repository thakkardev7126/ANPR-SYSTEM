"""Impossible-travel / teleportation detection.

This module consumes already-derived trajectory hop data. It does not perform
OCR, Re-ID, plate matching, or identity reassignment.
"""
from __future__ import annotations

import datetime
import json
import os

from app.database import PlateEvent, VehicleAnomaly
from app.location import iso_utc
from app.travel_time import calculate_travel_segment


ANOMALY_TYPE_IMPOSSIBLE_TRAVEL = "impossible_travel"
NORMAL = "normal"
WARNING = "warning"
IMPOSSIBLE = "impossible_travel"
UNAVAILABLE = "unavailable"


def _float_env(name, default):
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return float(default)


def _road_type_key(road_type):
    return "".join(ch if ch.isalnum() else "_" for ch in str(road_type or "").strip().upper()).strip("_")


def speed_policy(road_type=None):
    """Return configurable speed thresholds for a road context."""
    normal_default = _float_env("ANPR_SPEED_NORMAL_MAX_KMH", "80")
    warning_default = _float_env("ANPR_SPEED_WARNING_MAX_KMH", "120")
    impossible_default = _float_env("ANPR_SPEED_IMPOSSIBLE_MAX_KMH", "160")
    key = _road_type_key(road_type)
    normal = _float_env(f"ANPR_SPEED_NORMAL_MAX_KMH_{key}", normal_default) if key else normal_default
    warning = _float_env(f"ANPR_SPEED_WARNING_MAX_KMH_{key}", warning_default) if key else warning_default
    impossible = _float_env(f"ANPR_SPEED_IMPOSSIBLE_MAX_KMH_{key}", impossible_default) if key else impossible_default
    normal = max(1.0, normal)
    warning = max(normal, warning)
    impossible = max(warning, impossible)
    return {
        "normal_max_kmh": normal,
        "warning_max_kmh": warning,
        "impossible_max_kmh": impossible,
        "road_type": road_type,
    }


def evaluate_speed_anomaly(segment, vehicle_id=None, plate_text=None):
    """Classify one trajectory segment as normal/warning/impossible/unavailable."""
    if not segment or not segment.get("speed_available") or segment.get("estimated_speed_kmh") is None:
        return {
            "anomaly_status": UNAVAILABLE,
            "anomaly_type": None,
            "severity": None,
            "status": "not_evaluated",
            "policy": speed_policy(segment.get("road_type") if segment else None),
            "evidence": None,
        }
    policy = speed_policy(segment.get("road_type"))
    speed = float(segment["estimated_speed_kmh"])
    if speed <= policy["normal_max_kmh"]:
        anomaly_status = NORMAL
        severity = None
        status = "normal"
        allowed = policy["normal_max_kmh"]
    elif speed > policy["impossible_max_kmh"]:
        anomaly_status = IMPOSSIBLE
        severity = "critical"
        status = "detected"
        allowed = policy["impossible_max_kmh"]
    else:
        anomaly_status = WARNING
        severity = "warning"
        status = "warning"
        allowed = policy["impossible_max_kmh"]
    excess = round(speed / allowed, 4) if allowed else None
    evidence = {
        "global_vehicle_id": vehicle_id,
        "vehicle_id": vehicle_id,
        "plate_text": plate_text,
        "anomaly_type": ANOMALY_TYPE_IMPOSSIBLE_TRAVEL if anomaly_status in {WARNING, IMPOSSIBLE} else None,
        "status": status,
        "severity": severity,
        "source_event_id": segment.get("source_event_id"),
        "destination_event_id": segment.get("destination_event_id"),
        "source_camera_id": segment.get("source_camera_id"),
        "destination_camera_id": segment.get("destination_camera_id"),
        "source_timestamp": segment.get("start_time"),
        "destination_timestamp": segment.get("end_time"),
        "road_distance_meters": segment.get("distance_meters"),
        "travel_time_seconds": segment.get("travel_time_seconds"),
        "estimated_speed_kmh": speed,
        "allowed_speed_kmh": allowed,
        "excess_ratio": excess,
        "threshold_used": policy,
        "road_type": segment.get("road_type"),
        "road_name": segment.get("road_name"),
    }
    evidence["explanation"] = anomaly_explanation(evidence, anomaly_status)
    return {
        "anomaly_status": anomaly_status,
        "anomaly_type": evidence["anomaly_type"],
        "severity": severity,
        "status": status,
        "policy": policy,
        "evidence": evidence,
    }


def anomaly_explanation(evidence, anomaly_status):
    if anomaly_status == NORMAL:
        return "Observed camera-to-camera travel is within the configured speed policy."
    distance_km = (evidence.get("road_distance_meters") or 0) / 1000.0
    seconds = evidence.get("travel_time_seconds")
    speed = evidence.get("estimated_speed_kmh")
    allowed = evidence.get("allowed_speed_kmh")
    if seconds is None or speed is None or allowed is None:
        return "Insufficient data to evaluate impossible travel."
    return (
        f"Vehicle was observed {distance_km:.1f} km apart in {seconds:.0f} seconds, "
        f"requiring an estimated average speed of {speed:.1f} km/h, above the configured "
        f"{allowed:.1f} km/h plausibility limit."
    )


def anomaly_payload(row):
    policy = json.loads(row.policy or "{}")
    return {
        "id": row.id,
        "anomaly_id": row.id,
        "global_vehicle_id": row.vehicle_id,
        "vehicle_id": row.vehicle_id,
        "plate_text": row.plate_text,
        "anomaly_type": row.anomaly_type,
        "severity": row.severity,
        "status": row.status,
        "source_event_id": row.source_event_id,
        "destination_event_id": row.destination_event_id,
        "source_camera_id": row.source_camera_id,
        "destination_camera_id": row.destination_camera_id,
        "detected_at": iso_utc(row.detected_at) if row.detected_at else None,
        "travel_time_seconds": row.travel_time_seconds,
        "road_distance_meters": row.road_distance_meters,
        "estimated_speed_kmh": row.estimated_speed_kmh,
        "allowed_speed_kmh": row.allowed_speed_kmh,
        "excess_ratio": row.excess_ratio,
        "road_type": row.road_type,
        "road_name": row.road_name,
        "policy": policy,
        "explanation": row.explanation,
        "created_at": iso_utc(row.created_at) if row.created_at else None,
        "acknowledged_at": iso_utc(row.acknowledged_at) if row.acknowledged_at else None,
        "resolved_at": iso_utc(row.resolved_at) if row.resolved_at else None,
    }


def upsert_anomaly(db, evidence):
    if not evidence or evidence.get("anomaly_type") != ANOMALY_TYPE_IMPOSSIBLE_TRAVEL:
        return None
    existing = (
        db.query(VehicleAnomaly)
        .filter(VehicleAnomaly.anomaly_type == ANOMALY_TYPE_IMPOSSIBLE_TRAVEL,
                VehicleAnomaly.source_event_id == evidence["source_event_id"],
                VehicleAnomaly.destination_event_id == evidence["destination_event_id"])
        .first()
    )
    now = datetime.datetime.utcnow()
    row = existing or VehicleAnomaly(
        anomaly_type=ANOMALY_TYPE_IMPOSSIBLE_TRAVEL,
        source_event_id=evidence["source_event_id"],
        destination_event_id=evidence["destination_event_id"],
        created_at=now,
    )
    row.vehicle_id = evidence.get("global_vehicle_id")
    row.plate_text = evidence.get("plate_text")
    row.severity = evidence.get("severity") or "warning"
    row.status = evidence.get("status") or "warning"
    row.source_camera_id = evidence.get("source_camera_id")
    row.destination_camera_id = evidence.get("destination_camera_id")
    row.detected_at = now
    row.travel_time_seconds = evidence.get("travel_time_seconds")
    row.road_distance_meters = evidence.get("road_distance_meters")
    row.estimated_speed_kmh = evidence.get("estimated_speed_kmh")
    row.allowed_speed_kmh = evidence.get("allowed_speed_kmh")
    row.excess_ratio = evidence.get("excess_ratio")
    row.road_type = evidence.get("road_type")
    row.road_name = evidence.get("road_name")
    row.policy = json.dumps(evidence.get("threshold_used") or {})
    row.explanation = evidence.get("explanation") or ""
    if not existing:
        db.add(row)
    db.flush()
    return row


def annotate_segment_anomaly(db, segment, vehicle_id=None, plate_text=None, persist=True):
    result = evaluate_speed_anomaly(segment, vehicle_id=vehicle_id, plate_text=plate_text)
    row = None
    if persist and result["anomaly_status"] in {WARNING, IMPOSSIBLE}:
        row = upsert_anomaly(db, result["evidence"])
    payload = {
        "anomaly_status": result["anomaly_status"],
        "anomaly_type": result["anomaly_type"],
        "anomaly_severity": result["severity"],
        "anomaly_evidence": result["evidence"],
        "speed_policy": result["policy"],
    }
    if row:
        payload["anomaly_id"] = row.id
        payload["anomaly_record"] = anomaly_payload(row)
    return payload


def process_event_anomalies(db, event):
    """Evaluate the new event against the immediately previous same-vehicle event."""
    if not event or not event.vehicle_id or not event.timestamp:
        return []
    previous = (
        db.query(PlateEvent)
        .filter(PlateEvent.vehicle_id == event.vehicle_id,
                PlateEvent.id != event.id,
                PlateEvent.timestamp.isnot(None),
                PlateEvent.timestamp <= event.timestamp)
        .order_by(PlateEvent.timestamp.desc(), PlateEvent.id.desc())
        .first()
    )
    if not previous:
        return []
    segment = calculate_travel_segment(db, previous, event)
    result = annotate_segment_anomaly(db, segment, vehicle_id=event.vehicle_id,
                                      plate_text=event.plate_text, persist=True)
    return [result] if result.get("anomaly_id") else []
