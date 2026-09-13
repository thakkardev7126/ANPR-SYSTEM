"""Explainable route anomaly detection for persisted vehicle trajectories."""
from __future__ import annotations

import datetime
import json
import os
from collections import Counter

from app.anomalies import ANOMALY_TYPE_IMPOSSIBLE_TRAVEL, anomaly_payload
from app.database import (
    Camera,
    PlateEvent,
    PlateSuspicionEvent,
    RouteAnomalyEvent,
    Vehicle,
    VehicleAnomaly,
)
from app.location import iso_utc
from app.travel_time import calculate_travel_segment


ROUTE_MODEL = "route-anomaly-v1"
ROUTE_VERSION = "4C-2026-09-13"
NORMAL = "normal"
INSUFFICIENT_HISTORY = "insufficient_history"
UNUSUAL_ROUTE = "unusual_route"
HIGH_ROUTE_ANOMALY = "high_route_anomaly"


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


def route_anomaly_config():
    unusual = _float_env("ANPR_ROUTE_UNUSUAL_THRESHOLD", "3.0")
    high = _float_env("ANPR_ROUTE_HIGH_THRESHOLD", "6.0")
    high = max(unusual, high)
    return {
        "model": ROUTE_MODEL,
        "version": ROUTE_VERSION,
        "min_history_transitions": max(0, _int_env("ANPR_ROUTE_MIN_HISTORY", "2")),
        "current_route_hops": max(1, _int_env("ANPR_ROUTE_CURRENT_HOPS", "2")),
        "deviation_threshold": _float_env("ANPR_ROUTE_DEVIATION_THRESHOLD", "0.5"),
        "high_deviation_threshold": _float_env("ANPR_ROUTE_HIGH_DEVIATION_THRESHOLD", "0.8"),
        "unusual_threshold": unusual,
        "high_route_anomaly_threshold": high,
        "weights": {
            "disconnected_transition": _float_env("ANPR_ROUTE_WEIGHT_DISCONNECTED", "3.5"),
            "unseen_transition": _float_env("ANPR_ROUTE_WEIGHT_UNSEEN", "2.0"),
            "high_deviation": _float_env("ANPR_ROUTE_WEIGHT_HIGH_DEVIATION", "2.5"),
            "direction_conflict": _float_env("ANPR_ROUTE_WEIGHT_DIRECTION", "2.0"),
            "impossible_travel_context": _float_env("ANPR_ROUTE_WEIGHT_IMPOSSIBLE", "1.5"),
            "plate_suspicion_context": _float_env("ANPR_ROUTE_WEIGHT_PLATE_SUSPICION", "1.0"),
        },
    }


def _event_sort_key(event):
    return (event.timestamp is None, event.timestamp or datetime.datetime.max, event.id or 0)


def _transition_key(segment):
    return (segment.get("source_camera_id"), segment.get("destination_camera_id"))


def _reverse_key(segment):
    return (segment.get("destination_camera_id"), segment.get("source_camera_id"))


def _continuity(segment, source_camera, destination_camera):
    if not source_camera or not destination_camera:
        return "invalid_observation"
    status = segment.get("speed_status")
    if status in {"calculated", "calculated_approximate"}:
        return "connected"
    if status == "missing_road_connection":
        return "not_connected"
    if status == "invalid_time":
        return "invalid_time"
    return "invalid_observation"


def _segment_payload(db, previous, current, cameras):
    segment = calculate_travel_segment(db, previous, current)
    continuity = _continuity(segment, cameras.get(previous.camera_id), cameras.get(current.camera_id))
    return {
        **segment,
        "continuity": continuity,
        "route_status": continuity,
        "route_transition": f"{previous.camera_id}->{current.camera_id}",
    }


def _vehicle_events(db, vehicle_id):
    return sorted(
        db.query(PlateEvent).filter(PlateEvent.vehicle_id == vehicle_id).all(),
        key=_event_sort_key,
    )


def _route_signature(segments):
    if not segments:
        return ""
    cameras = [segments[0].get("source_camera_id")]
    cameras.extend(segment.get("destination_camera_id") for segment in segments)
    return "->".join(str(camera or "?") for camera in cameras)


def _classification(score, config):
    if score >= config["high_route_anomaly_threshold"]:
        return HIGH_ROUTE_ANOMALY
    if score >= config["unusual_threshold"]:
        return UNUSUAL_ROUTE
    return NORMAL


def _explanation(classification, evidence):
    if classification == INSUFFICIENT_HISTORY:
        return "Not enough previous same-vehicle route history exists for a reliable route-anomaly decision."
    if classification == NORMAL:
        return "Observed route is consistent with the available historical camera-transition baseline."
    parts = []
    if evidence["disconnected_transitions"]:
        parts.append(f"{len(evidence['disconnected_transitions'])} disconnected transition(s)")
    if evidence["unseen_transition_count"]:
        parts.append(f"{evidence['unseen_transition_count']} unseen transition(s)")
    if evidence["direction_conflicts"]:
        parts.append(f"{len(evidence['direction_conflicts'])} direction conflict(s)")
    if evidence["impossible_travel_context"]:
        parts.append("impossible-travel context")
    if evidence["plate_suspicion_context"]:
        parts.append("possible cloned-plate context")
    reason = ", ".join(parts) or "route differs from historical pattern"
    return (
        f"Observed route differs from the vehicle's previous camera-transition pattern: {reason}. "
        "This is explainable route evidence only, not a criminal conclusion."
    )


def _anomaly_payload(row):
    evidence = json.loads(row.evidence or "{}")
    return {
        "id": row.id,
        "route_anomaly_id": row.id,
        "global_vehicle_id": row.vehicle_id,
        "vehicle_id": row.vehicle_id,
        "plate_text": row.plate_text,
        "start_event_id": row.start_event_id,
        "end_event_id": row.end_event_id,
        "start_camera_id": row.start_camera_id,
        "end_camera_id": row.end_camera_id,
        "route_signature": row.route_signature,
        "route_anomaly_score": row.route_anomaly_score,
        "classification": row.classification,
        "status": row.status,
        "evidence": evidence,
        "explanation": row.explanation,
        "created_at": iso_utc(row.created_at) if row.created_at else None,
        "updated_at": iso_utc(row.updated_at) if row.updated_at else None,
        "reviewed_at": iso_utc(row.reviewed_at) if row.reviewed_at else None,
        "reviewed_by": row.reviewed_by,
    }


def route_anomaly_payload(row):
    return _anomaly_payload(row)


def _context_for_segment(db, segment):
    impossible = None
    suspicion = None
    source = db.get(PlateEvent, segment.get("source_event_id")) if segment.get("source_event_id") else None
    destination = db.get(PlateEvent, segment.get("destination_event_id")) if segment.get("destination_event_id") else None
    if source and destination:
        from app.anomalies import IMPOSSIBLE, annotate_segment_anomaly
        fresh_anomaly = annotate_segment_anomaly(
            db,
            segment,
            vehicle_id=destination.vehicle_id or source.vehicle_id,
            plate_text=destination.plate_text or source.plate_text,
            persist=True,
        )
        if fresh_anomaly.get("anomaly_status") == IMPOSSIBLE:
            impossible = (
                db.query(VehicleAnomaly)
                .filter(VehicleAnomaly.anomaly_type == ANOMALY_TYPE_IMPOSSIBLE_TRAVEL,
                        VehicleAnomaly.source_event_id == segment.get("source_event_id"),
                        VehicleAnomaly.destination_event_id == segment.get("destination_event_id"))
                .first()
            )
        from app.plate_suspicion import evaluate_suspicion_pair, upsert_plate_suspicion
        fresh_suspicion = evaluate_suspicion_pair(db, source, destination, persist_anomaly=False)
        if fresh_suspicion and fresh_suspicion.get("classification") != "normal":
            suspicion = upsert_plate_suspicion(db, fresh_suspicion)
    return impossible, suspicion


def evaluate_vehicle_route(db, vehicle_id, persist=True):
    vehicle = db.get(Vehicle, vehicle_id)
    if not vehicle:
        return None
    config = route_anomaly_config()
    events = _vehicle_events(db, vehicle_id)
    cameras = {camera.camera_id: camera for camera in db.query(Camera).all()}
    segments = [
        _segment_payload(db, previous, current, cameras)
        for previous, current in zip(events, events[1:])
    ]
    if not segments:
        return {
            "global_vehicle_id": vehicle_id,
            "vehicle_id": vehicle_id,
            "plate_text": vehicle.primary_plate_text,
            "classification": INSUFFICIENT_HISTORY,
            "route_anomaly_score": 0.0,
            "route_signature": "",
            "evidence": {"reason": "single_or_empty_trajectory", "config": config},
            "explanation": _explanation(INSUFFICIENT_HISTORY, {"disconnected_transitions": [], "unseen_transition_count": 0, "direction_conflicts": [], "impossible_travel_context": False, "plate_suspicion_context": False}),
            "record": None,
        }

    current_hops = min(config["current_route_hops"], len(segments))
    current = segments[-current_hops:]
    baseline = segments[:-current_hops]
    baseline_counts = Counter(_transition_key(segment) for segment in baseline)
    current_keys = [_transition_key(segment) for segment in current]
    unseen = [segment for segment in current if baseline_counts[_transition_key(segment)] == 0]
    disconnected = [segment for segment in current if segment.get("continuity") == "not_connected"]
    direction_conflicts = []
    baseline_by_reverse = {(_reverse_key(segment)): segment for segment in baseline if segment.get("direction")}
    for segment in current:
        reverse_baseline = baseline_by_reverse.get(_transition_key(segment))
        if reverse_baseline and segment.get("direction") and reverse_baseline.get("direction") != segment.get("direction"):
            direction_conflicts.append({
                "source_camera_id": segment.get("source_camera_id"),
                "destination_camera_id": segment.get("destination_camera_id"),
                "observed_direction": segment.get("direction"),
                "previous_reverse_direction": reverse_baseline.get("direction"),
            })

    impossible_context = []
    plate_suspicion_context = []
    for segment in current:
        impossible, suspicion = _context_for_segment(db, segment)
        if impossible:
            impossible_context.append(anomaly_payload(impossible))
        if suspicion:
            plate_suspicion_context.append({
                "id": suspicion.id,
                "plate_text": suspicion.plate_text,
                "classification": suspicion.classification,
                "suspicion_score": suspicion.suspicion_score,
            })

    deviation_ratio = round(len(unseen) / len(current), 4) if current else 0.0
    evidence = {
        "model": ROUTE_MODEL,
        "version": ROUTE_VERSION,
        "route_signature": _route_signature(current),
        "baseline_transition_count": len(baseline),
        "baseline_transitions": [
            {"source_camera_id": source, "destination_camera_id": destination, "count": count}
            for (source, destination), count in baseline_counts.items()
        ],
        "current_transitions": [
            {
                "source_event_id": segment.get("source_event_id"),
                "destination_event_id": segment.get("destination_event_id"),
                "source_camera_id": segment.get("source_camera_id"),
                "destination_camera_id": segment.get("destination_camera_id"),
                "route_status": segment.get("route_status"),
                "direction": segment.get("direction"),
                "travel_time_seconds": segment.get("travel_time_seconds"),
                "distance_meters": segment.get("distance_meters"),
                "estimated_speed_kmh": segment.get("estimated_speed_kmh"),
            }
            for segment in current
        ],
        "route_deviation_ratio": deviation_ratio,
        "unseen_transition_count": len(unseen),
        "total_transition_count": len(current),
        "disconnected_transitions": [
            {
                "source_event_id": segment.get("source_event_id"),
                "destination_event_id": segment.get("destination_event_id"),
                "source_camera_id": segment.get("source_camera_id"),
                "destination_camera_id": segment.get("destination_camera_id"),
                "reason": "No active configured road-network connection exists for this observed camera pair.",
            }
            for segment in disconnected
        ],
        "direction_conflicts": direction_conflicts,
        "impossible_travel_context": bool(impossible_context),
        "impossible_travel_anomalies": impossible_context,
        "plate_suspicion_context": bool(plate_suspicion_context),
        "plate_suspicions": plate_suspicion_context,
        "config": config,
    }

    score = 0.0
    weights = config["weights"]
    score += len(disconnected) * weights["disconnected_transition"]
    score += len(unseen) * weights["unseen_transition"]
    if deviation_ratio >= config["high_deviation_threshold"]:
        score += weights["high_deviation"]
    elif deviation_ratio >= config["deviation_threshold"]:
        score += weights["high_deviation"] / 2.0
    score += len(direction_conflicts) * weights["direction_conflict"]
    if impossible_context:
        score += weights["impossible_travel_context"]
    if plate_suspicion_context:
        score += weights["plate_suspicion_context"]
    score = round(score, 3)

    if len(baseline) < config["min_history_transitions"]:
        classification = INSUFFICIENT_HISTORY
        score = 0.0
    else:
        classification = _classification(score, config)
    explanation = _explanation(classification, evidence)
    result = {
        "global_vehicle_id": vehicle_id,
        "vehicle_id": vehicle_id,
        "plate_text": vehicle.primary_plate_text,
        "classification": classification,
        "route_anomaly_score": score,
        "route_signature": evidence["route_signature"],
        "start_event_id": current[0].get("source_event_id"),
        "end_event_id": current[-1].get("destination_event_id"),
        "start_camera_id": current[0].get("source_camera_id"),
        "end_camera_id": current[-1].get("destination_camera_id"),
        "evidence": evidence,
        "explanation": explanation,
        "record": None,
    }
    if persist:
        result["record"] = upsert_route_anomaly(db, result)
    return result


def upsert_route_anomaly(db, result):
    if result and result.get("vehicle_id") and result.get("start_event_id") and result.get("end_event_id"):
        (
            db.query(RouteAnomalyEvent)
            .filter(RouteAnomalyEvent.vehicle_id == result["vehicle_id"],
                    (RouteAnomalyEvent.start_event_id != result["start_event_id"])
                    | (RouteAnomalyEvent.end_event_id != result["end_event_id"]))
            .delete(synchronize_session=False)
        )
    if not result or result["classification"] in {NORMAL, INSUFFICIENT_HISTORY}:
        existing = None
        if result and result.get("start_event_id") and result.get("end_event_id"):
            existing = (
                db.query(RouteAnomalyEvent)
                .filter(RouteAnomalyEvent.vehicle_id == result["vehicle_id"],
                        RouteAnomalyEvent.start_event_id == result["start_event_id"],
                        RouteAnomalyEvent.end_event_id == result["end_event_id"])
                .first()
            )
        if existing:
            db.delete(existing)
            db.flush()
        return None
    existing = (
        db.query(RouteAnomalyEvent)
        .filter(RouteAnomalyEvent.vehicle_id == result["vehicle_id"],
                RouteAnomalyEvent.start_event_id == result["start_event_id"],
                RouteAnomalyEvent.end_event_id == result["end_event_id"])
        .first()
    )
    now = datetime.datetime.utcnow()
    row = existing or RouteAnomalyEvent(
        vehicle_id=result["vehicle_id"],
        start_event_id=result["start_event_id"],
        end_event_id=result["end_event_id"],
        created_at=now,
        status="open",
    )
    row.plate_text = result.get("plate_text")
    row.start_camera_id = result.get("start_camera_id")
    row.end_camera_id = result.get("end_camera_id")
    row.route_signature = result.get("route_signature") or ""
    row.route_anomaly_score = result["route_anomaly_score"]
    row.classification = result["classification"]
    row.evidence = json.dumps(result["evidence"])
    row.explanation = result["explanation"]
    row.updated_at = now
    if not existing:
        db.add(row)
    db.flush()
    return row


def process_route_anomalies(db, event):
    if not event or not event.vehicle_id:
        return None
    return evaluate_vehicle_route(db, event.vehicle_id, persist=True)


def route_anomaly_summary(db, vehicle_id, persist=True):
    result = evaluate_vehicle_route(db, vehicle_id, persist=persist)
    rows = (
        db.query(RouteAnomalyEvent)
        .filter(RouteAnomalyEvent.vehicle_id == vehicle_id)
        .order_by(RouteAnomalyEvent.route_anomaly_score.desc(), RouteAnomalyEvent.updated_at.desc())
        .all()
    )
    items = [route_anomaly_payload(row) for row in rows]
    best = items[0] if items else None
    if result is None:
        return None
    return {
        **{k: v for k, v in result.items() if k != "record"},
        "id": best["id"] if best else None,
        "route_anomaly_id": best["id"] if best else None,
        "status": best["status"] if best else ("open" if result["classification"] not in {NORMAL, INSUFFICIENT_HISTORY} else result["classification"]),
        "items": items,
        "total": len(items),
        "confirmed_criminal_activity": False,
    }


def route_anomaly_for_pair(db, vehicle_id, source_event_id, destination_event_id):
    row = (
        db.query(RouteAnomalyEvent)
        .filter(RouteAnomalyEvent.vehicle_id == vehicle_id,
                RouteAnomalyEvent.start_event_id <= source_event_id,
                RouteAnomalyEvent.end_event_id >= destination_event_id)
        .order_by(RouteAnomalyEvent.route_anomaly_score.desc(), RouteAnomalyEvent.updated_at.desc())
        .first()
    )
    return route_anomaly_payload(row) if row else None
