"""Possible cloned-plate suspicion scoring.

This module only evaluates confirmed valid plate observations. It reuses stored
appearance descriptors, Phase 4A impossible-travel evidence, and persisted
PlateEvent history. It never creates plate identities or declares a confirmed
clone.
"""
from __future__ import annotations

import datetime
import json
import os
from itertools import combinations

from app.anomalies import IMPOSSIBLE, annotate_segment_anomaly, anomaly_payload
from app.database import Camera, PlateEvent, PlateSuspicionEvent, VehicleAnomaly, _confirmed_vehicle_plate
from app.location import iso_utc
from app.travel_time import calculate_travel_segment
from app.vehicle_appearance import appearance_similarity


SUSPICION_MODEL = "plate-suspicion-v1"
SUSPICION_VERSION = "4B-2026-09-13"
NORMAL = "normal"
SUSPICIOUS = "suspicious"
HIGH_SUSPICION = "high_suspicion"
EVIDENCE_TYPE_IMPOSSIBLE_TRAVEL = "impossible_travel"
EVIDENCE_TYPE_OVERLAP = "simultaneous_sighting"
EVIDENCE_TYPE_APPEARANCE = "appearance_conflict"
EVIDENCE_TYPE_TYPE = "vehicle_type_conflict"
EVIDENCE_TYPE_COLOR = "vehicle_color_conflict"


def _float_env(name, default):
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return float(default)


def _int_env(name, default):
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return int(default)


def suspicion_config():
    """Centralized, configurable cloned-plate suspicion policy."""
    suspicious = _float_env("ANPR_CLONED_PLATE_SUSPICIOUS_THRESHOLD", "4.0")
    high = _float_env("ANPR_CLONED_PLATE_HIGH_THRESHOLD", "7.0")
    high = max(suspicious, high)
    return {
        "model": SUSPICION_MODEL,
        "version": SUSPICION_VERSION,
        "overlap_seconds": _int_env("ANPR_CLONED_PLATE_OVERLAP_SECONDS", "30"),
        "min_camera_separation_meters": _float_env("ANPR_CLONED_PLATE_MIN_CAMERA_SEPARATION_METERS", "100"),
        "appearance_conflict_max_similarity": _float_env("ANPR_CLONED_PLATE_APPEARANCE_CONFLICT_MAX", "0.45"),
        "suspicious_threshold": suspicious,
        "high_suspicion_threshold": high,
        "weights": {
            "impossible_travel": _float_env("ANPR_CLONED_PLATE_WEIGHT_IMPOSSIBLE_TRAVEL", "5.0"),
            "simultaneous_sighting": _float_env("ANPR_CLONED_PLATE_WEIGHT_OVERLAP", "4.0"),
            "appearance_conflict": _float_env("ANPR_CLONED_PLATE_WEIGHT_APPEARANCE", "3.5"),
            "vehicle_type_conflict": _float_env("ANPR_CLONED_PLATE_WEIGHT_TYPE", "2.0"),
            "vehicle_color_conflict": _float_env("ANPR_CLONED_PLATE_WEIGHT_COLOR", "1.0"),
            "multiple_evidence_bonus": _float_env("ANPR_CLONED_PLATE_WEIGHT_MULTI_BONUS", "1.0"),
        },
    }


def _classification(score, config):
    if score >= config["high_suspicion_threshold"]:
        return HIGH_SUSPICION
    if score >= config["suspicious_threshold"]:
        return SUSPICIOUS
    return NORMAL


def _confirmed_plate(event):
    return _confirmed_vehicle_plate(event.plate_text, event.confidence, event.status) if event else None


def _event_time_key(event):
    return (event.timestamp is None, event.timestamp or datetime.datetime.max, event.id or 0)


def _ordered_pair(left, right):
    return (left, right) if _event_time_key(left) <= _event_time_key(right) else (right, left)


def _camera_distance_meters(left_camera, right_camera):
    if not left_camera or not right_camera:
        return None
    if left_camera.lat is None or left_camera.lng is None or right_camera.lat is None or right_camera.lng is None:
        return None
    from app.trajectory import haversine_km
    return round(haversine_km(left_camera.lat, left_camera.lng, right_camera.lat, right_camera.lng) * 1000, 3)


def _timestamp_delta_seconds(left, right):
    if not left.timestamp or not right.timestamp:
        return None
    return abs((right.timestamp - left.timestamp).total_seconds())


def _color_conflicts(left_color, right_color):
    if not left_color or not right_color or left_color == right_color:
        return False
    soft_neutral = {left_color, right_color}
    return soft_neutral not in ({"white", "gray"}, {"black", "gray"})


def _base_pair_payload(source, destination, plate_text):
    return {
        "plate_text": plate_text,
        "source_event_id": source.id,
        "destination_event_id": destination.id,
        "source_camera_id": source.camera_id,
        "destination_camera_id": destination.camera_id,
        "source_timestamp": iso_utc(source.timestamp) if source.timestamp else None,
        "destination_timestamp": iso_utc(destination.timestamp) if destination.timestamp else None,
        "global_vehicle_id": destination.vehicle_id or source.vehicle_id,
        "vehicle_id": destination.vehicle_id or source.vehicle_id,
    }


def evaluate_suspicion_pair(db, left, right, persist_anomaly=True):
    """Evaluate two observations of the same confirmed plate."""
    source, destination = _ordered_pair(left, right)
    plate_text = _confirmed_plate(source)
    if not plate_text or plate_text != _confirmed_plate(destination):
        return None
    config = suspicion_config()
    evidence = []
    score = 0.0
    similarity = None
    base = _base_pair_payload(source, destination, plate_text)
    segment = calculate_travel_segment(db, source, destination)
    anomaly = annotate_segment_anomaly(
        db,
        segment,
        vehicle_id=base["global_vehicle_id"],
        plate_text=plate_text,
        persist=persist_anomaly,
    )
    if anomaly.get("anomaly_status") == IMPOSSIBLE:
        item = {
            "type": EVIDENCE_TYPE_IMPOSSIBLE_TRAVEL,
            **base,
            "travel_time_seconds": segment.get("travel_time_seconds"),
            "road_distance_meters": segment.get("distance_meters"),
            "estimated_speed_kmh": segment.get("estimated_speed_kmh"),
            "allowed_speed_kmh": (anomaly.get("anomaly_evidence") or {}).get("allowed_speed_kmh"),
            "excess_ratio": (anomaly.get("anomaly_evidence") or {}).get("excess_ratio"),
            "phase4a_anomaly_id": anomaly.get("anomaly_id"),
        }
        evidence.append(item)
        score += config["weights"][EVIDENCE_TYPE_IMPOSSIBLE_TRAVEL]

    if source.camera_id and destination.camera_id and source.camera_id != destination.camera_id:
        delta = _timestamp_delta_seconds(source, destination)
        source_camera = db.get(Camera, source.camera_id)
        destination_camera = db.get(Camera, destination.camera_id)
        camera_distance = _camera_distance_meters(source_camera, destination_camera)
        if (delta is not None
                and delta <= config["overlap_seconds"]
                and camera_distance is not None
                and camera_distance >= config["min_camera_separation_meters"]):
            evidence.append({
                "type": EVIDENCE_TYPE_OVERLAP,
                **base,
                "time_delta_seconds": delta,
                "camera_distance_meters": camera_distance,
                "overlap_threshold_seconds": config["overlap_seconds"],
            })
            score += config["weights"][EVIDENCE_TYPE_OVERLAP]

    similarity = appearance_similarity(source.appearance_embedding, destination.appearance_embedding)
    if similarity is not None and similarity <= config["appearance_conflict_max_similarity"]:
        evidence.append({
            "type": EVIDENCE_TYPE_APPEARANCE,
            **base,
            "similarity": similarity,
            "threshold": config["appearance_conflict_max_similarity"],
            "source_model": source.appearance_model,
            "destination_model": destination.appearance_model,
        })
        score += config["weights"][EVIDENCE_TYPE_APPEARANCE]

    if source.vehicle_type and destination.vehicle_type and source.vehicle_type != destination.vehicle_type:
        evidence.append({
            "type": EVIDENCE_TYPE_TYPE,
            **base,
            "source_vehicle_type": source.vehicle_type,
            "destination_vehicle_type": destination.vehicle_type,
        })
        score += config["weights"][EVIDENCE_TYPE_TYPE]

    if _color_conflicts(source.vehicle_color, destination.vehicle_color):
        evidence.append({
            "type": EVIDENCE_TYPE_COLOR,
            **base,
            "source_vehicle_color": source.vehicle_color,
            "destination_vehicle_color": destination.vehicle_color,
        })
        score += config["weights"][EVIDENCE_TYPE_COLOR]

    if len(evidence) >= 3:
        score += config["weights"]["multiple_evidence_bonus"]

    score = round(score, 3)
    classification = _classification(score, config)
    return {
        **base,
        "suspicion_score": score,
        "classification": classification,
        "evidence": evidence,
        "appearance_similarity": similarity,
        "config": config,
        "explanation": suspicion_explanation(plate_text, classification, score, evidence),
    }


def suspicion_explanation(plate_text, classification, score, evidence):
    if classification == NORMAL:
        return f"Plate {plate_text} has no strong cloned-plate evidence under the configured policy."
    labels = {
        EVIDENCE_TYPE_IMPOSSIBLE_TRAVEL: "physically impossible travel",
        EVIDENCE_TYPE_OVERLAP: "near-simultaneous sightings at different cameras",
        EVIDENCE_TYPE_APPEARANCE: "strong vehicle appearance conflict",
        EVIDENCE_TYPE_TYPE: "vehicle type conflict",
        EVIDENCE_TYPE_COLOR: "vehicle color conflict",
    }
    reasons = [labels.get(item.get("type"), item.get("type")) for item in evidence]
    return (
        f"Possible cloned-plate suspicion for {plate_text}: {', '.join(reasons)}. "
        f"Suspicion score {score:.1f}; classification {classification.replace('_', ' ')}. "
        "This is not a confirmed cloned plate."
    )


def suspicion_payload(row):
    evidence = json.loads(row.evidence or "{}")
    return {
        "id": row.id,
        "suspicion_id": row.id,
        "plate_text": row.plate_text,
        "global_vehicle_id": row.vehicle_id,
        "vehicle_id": row.vehicle_id,
        "source_event_id": row.source_event_id,
        "destination_event_id": row.destination_event_id,
        "source_camera_id": row.source_camera_id,
        "destination_camera_id": row.destination_camera_id,
        "suspicion_score": row.suspicion_score,
        "classification": row.classification,
        "status": row.status,
        "appearance_similarity": row.appearance_similarity,
        "evidence": evidence.get("factors", []),
        "config": evidence.get("config", {}),
        "explanation": evidence.get("explanation"),
        "created_at": iso_utc(row.created_at) if row.created_at else None,
        "updated_at": iso_utc(row.updated_at) if row.updated_at else None,
        "reviewed_at": iso_utc(row.reviewed_at) if row.reviewed_at else None,
        "reviewed_by": row.reviewed_by,
    }


def upsert_plate_suspicion(db, result):
    if not result:
        return None
    existing = (
        db.query(PlateSuspicionEvent)
        .filter(PlateSuspicionEvent.plate_text == result["plate_text"],
                PlateSuspicionEvent.source_event_id == result["source_event_id"],
                PlateSuspicionEvent.destination_event_id == result["destination_event_id"])
        .first()
    )
    if result["classification"] == NORMAL:
        if existing:
            db.delete(existing)
            db.flush()
        return None
    now = datetime.datetime.utcnow()
    row = existing or PlateSuspicionEvent(
        plate_text=result["plate_text"],
        source_event_id=result["source_event_id"],
        destination_event_id=result["destination_event_id"],
        created_at=now,
        status="open",
    )
    row.vehicle_id = result.get("global_vehicle_id")
    row.source_camera_id = result.get("source_camera_id")
    row.destination_camera_id = result.get("destination_camera_id")
    row.suspicion_score = result["suspicion_score"]
    row.classification = result["classification"]
    row.appearance_similarity = result.get("appearance_similarity")
    row.evidence = json.dumps({
        "factors": result["evidence"],
        "config": result["config"],
        "explanation": result["explanation"],
    })
    row.updated_at = now
    if not existing:
        db.add(row)
    db.flush()
    return row


def evaluate_plate_suspicions(db, plate_text, persist=True, limit=100):
    from app.plate_rules import normalize_plate_text
    normalized = normalize_plate_text(plate_text).normalized_text
    if not normalized:
        return []
    events = (
        db.query(PlateEvent)
        .filter(PlateEvent.plate_text == normalized)
        .order_by(PlateEvent.timestamp.asc(), PlateEvent.id.asc())
        .limit(limit)
        .all()
    )
    events = [event for event in events if _confirmed_plate(event) == normalized]
    rows = []
    for left, right in combinations(events, 2):
        result = evaluate_suspicion_pair(db, left, right, persist_anomaly=persist)
        if not result:
            continue
        row = upsert_plate_suspicion(db, result) if persist else None
        if row:
            rows.append(row)
    return rows


def process_plate_suspicions(db, event):
    plate_text = _confirmed_plate(event)
    if not plate_text:
        return []
    rows = evaluate_plate_suspicions(db, plate_text, persist=True)
    return rows


def plate_suspicion_summary(db, plate_text, persist=True):
    rows = evaluate_plate_suspicions(db, plate_text, persist=persist)
    from app.plate_rules import normalize_plate_text
    normalized = normalize_plate_text(plate_text).normalized_text
    evaluated_pairs = []
    evaluated_evidence = []
    if normalized:
        events = (
            db.query(PlateEvent)
            .filter(PlateEvent.plate_text == normalized)
            .order_by(PlateEvent.timestamp.asc(), PlateEvent.id.asc())
            .all()
        )
        events = [event for event in events if _confirmed_plate(event) == normalized]
        for left, right in combinations(events, 2):
            result = evaluate_suspicion_pair(db, left, right, persist_anomaly=False)
            if result:
                evaluated_pairs.append({
                    "source_event_id": result["source_event_id"],
                    "destination_event_id": result["destination_event_id"],
                    "suspicion_score": result["suspicion_score"],
                    "classification": result["classification"],
                    "evidence": result["evidence"],
                    "appearance_similarity": result["appearance_similarity"],
                })
                evaluated_evidence.extend(result["evidence"])
    stored = (
        db.query(PlateSuspicionEvent)
        .filter(PlateSuspicionEvent.plate_text == normalized)
        .order_by(PlateSuspicionEvent.suspicion_score.desc(), PlateSuspicionEvent.updated_at.desc())
        .all()
        if normalized else []
    )
    items = [suspicion_payload(row) for row in stored]
    best = items[0] if items else None
    classification = best["classification"] if best else NORMAL
    score = best["suspicion_score"] if best else 0.0
    related_anomalies = []
    if normalized:
        anomaly_rows = (
            db.query(VehicleAnomaly)
            .filter(VehicleAnomaly.plate_text == normalized)
            .order_by(VehicleAnomaly.detected_at.desc(), VehicleAnomaly.id.desc())
            .all()
        )
        related_anomalies = [anomaly_payload(row) for row in anomaly_rows]
    return {
        "plate_text": normalized or plate_text.upper().strip(),
        "classification": classification,
        "suspicion_score": score,
        "status": best["status"] if best else "normal",
        "items": items,
        "total": len(items),
        "evaluated_pairs": evaluated_pairs,
        "evaluated_evidence": evaluated_evidence,
        "related_impossible_travel_anomalies": related_anomalies,
        "config": suspicion_config(),
        "conclusion": "possible cloned plate" if classification != NORMAL else "normal",
        "confirmed_cloned_plate": False,
    }


def suspicion_for_pair(db, source_event_id, destination_event_id):
    from sqlalchemy import or_
    row = (
        db.query(PlateSuspicionEvent)
        .filter(or_(
            (PlateSuspicionEvent.source_event_id == source_event_id)
            & (PlateSuspicionEvent.destination_event_id == destination_event_id),
            (PlateSuspicionEvent.source_event_id == destination_event_id)
            & (PlateSuspicionEvent.destination_event_id == source_event_id),
        ))
        .order_by(PlateSuspicionEvent.suspicion_score.desc(), PlateSuspicionEvent.updated_at.desc())
        .first()
    )
    return suspicion_payload(row) if row else None
