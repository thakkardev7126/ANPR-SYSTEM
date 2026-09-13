"""Unified explainable vehicle investigation payloads.

This module composes existing persisted observations, trajectory segments,
matching records, and anomaly/suspicion summaries. It does not run OCR,
appearance extraction, vehicle matching, or create new vehicle identities.
"""
from __future__ import annotations

import datetime
from collections import Counter

from app.anomalies import anomaly_payload
from app.database import PlateEvent, PlateSuspicionEvent, RouteAnomalyEvent, VehicleAnomaly
from app.location import iso_utc
from app.plate_suspicion import NORMAL as PLATE_NORMAL, plate_suspicion_summary, suspicion_payload
from app.route_anomaly import HIGH_ROUTE_ANOMALY, NORMAL as ROUTE_NORMAL, UNUSUAL_ROUTE, route_anomaly_payload, route_anomaly_summary
from app.trajectory import build_vehicle_trajectory, get_vehicle_by_id, get_vehicle_by_plate, vehicle_history, vehicle_payload
from app.vehicle_appearance import appearance_similarity
from app.vehicle_matching import matches_for_vehicle


LOW_VISUAL_SIMILARITY = 0.45
HIGH_VISUAL_SIMILARITY = 0.85


def _safe_iso(value):
    return iso_utc(value) if value else None


def _sort_value(value):
    if not value:
        return datetime.datetime.max
    if isinstance(value, datetime.datetime):
        return value
    try:
        return datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return datetime.datetime.max


def _ordered_vehicle_events(db, vehicle_id):
    return (
        db.query(PlateEvent)
        .filter(PlateEvent.vehicle_id == vehicle_id)
        .order_by(PlateEvent.timestamp.asc(), PlateEvent.id.asc())
        .all()
    )


def _appearance_payload(events):
    observations = [
        {
            "event_id": event.id,
            "camera_id": event.camera_id,
            "timestamp": _safe_iso(event.timestamp),
            "vehicle_type": event.vehicle_type,
            "vehicle_color": event.vehicle_color,
            "vehicle_crop_path": event.vehicle_crop_path,
            "appearance_available": bool(event.appearance_embedding),
            "appearance_quality": event.appearance_quality,
            "appearance_model": event.appearance_model,
            "appearance_embedding_version": event.appearance_embedding_version,
        }
        for event in events
    ]
    comparisons = []
    for left, right in zip(events, events[1:]):
        score = appearance_similarity(left.appearance_embedding, right.appearance_embedding)
        if score is None:
            assessment = "unavailable"
        elif score <= LOW_VISUAL_SIMILARITY:
            assessment = "low_visual_similarity"
        elif score >= HIGH_VISUAL_SIMILARITY:
            assessment = "strong_visual_similarity"
        else:
            assessment = "moderate_visual_similarity"
        comparisons.append({
            "left_event_id": left.id,
            "right_event_id": right.id,
            "left_camera_id": left.camera_id,
            "right_camera_id": right.camera_id,
            "appearance_similarity": score,
            "appearance_similarity_percent": round(score * 100, 1) if score is not None else None,
            "assessment": assessment,
            "model": left.appearance_model if left.appearance_model == right.appearance_model else None,
            "embedding_versions": [left.appearance_embedding_version, right.appearance_embedding_version],
        })
    return {
        "observations": observations,
        "comparisons": comparisons,
        "available_observation_count": sum(1 for item in observations if item["appearance_available"]),
    }


def _status_summary(anomalies, plate_suspicion, route_anomaly, matches):
    concerns = []
    if any(item.get("anomaly_status") == "impossible_travel" or item.get("severity") == "critical" for item in anomalies):
        concerns.append("Impossible travel detected")
    if plate_suspicion and plate_suspicion.get("classification") != PLATE_NORMAL:
        concerns.append("Possible cloned plate")
    if route_anomaly and route_anomaly.get("classification") in {UNUSUAL_ROUTE, HIGH_ROUTE_ANOMALY}:
        concerns.append("Route anomaly detected")
    if any(match.get("review_status") == "pending" for match in matches):
        concerns.append("Under review")
    if len(set(concerns)) > 1:
        return "Multiple concerns"
    return concerns[0] if concerns else "No concerns"


def _summary(vehicle, history, trajectory, anomalies, plate_suspicion, route_anomaly, matches):
    observations = history.get("observations", []) if history else []
    cameras = {item.get("camera_id") for item in observations if item.get("camera_id")}
    return {
        "global_vehicle_id": vehicle.vehicle_id,
        "vehicle_id": vehicle.vehicle_id,
        "primary_plate_text": vehicle.primary_plate_text,
        "first_seen": history.get("first_seen") if history else None,
        "last_seen": history.get("last_seen") if history else None,
        "observation_count": len(observations),
        "camera_count": len(cameras),
        "trajectory_segment_count": len(trajectory.get("trajectory_hops", [])) if trajectory else 0,
        "valid_speed_segment_count": (trajectory.get("summary") or {}).get("valid_segments") if trajectory else 0,
        "anomaly_count": len(anomalies),
        "match_count": len(matches),
        "plate_suspicion": (plate_suspicion or {}).get("classification", PLATE_NORMAL),
        "route_anomaly": (route_anomaly or {}).get("classification", "insufficient_history"),
        "status_summary": _status_summary(anomalies, plate_suspicion, route_anomaly, matches),
        "confirmed_cloned_plate": False,
        "confirmed_criminal_activity": False,
    }


def _timeline(history, anomalies, plate_suspicion, route_anomaly, matches):
    items = []
    for observation in history.get("observations", []) if history else []:
        items.append({
            "type": "observation",
            "timestamp": observation.get("timestamp"),
            "event_id": observation.get("event_id"),
            "camera_id": observation.get("camera_id"),
            "label": "Plate observation",
            "summary": f"{observation.get('camera_id') or 'Camera'} observed {observation.get('plate_text') or 'unreadable plate'}",
            "details": observation,
        })
    for anomaly in anomalies:
        items.append({
            "type": "impossible_travel",
            "timestamp": anomaly.get("detected_at") or anomaly.get("created_at"),
            "event_id": anomaly.get("destination_event_id"),
            "camera_id": anomaly.get("destination_camera_id"),
            "label": "Impossible travel evidence" if anomaly.get("severity") == "critical" else "Speed warning evidence",
            "summary": anomaly.get("explanation"),
            "details": anomaly,
        })
    for suspicion in (plate_suspicion or {}).get("items", []):
        items.append({
            "type": "plate_suspicion",
            "timestamp": suspicion.get("created_at") or suspicion.get("updated_at"),
            "event_id": suspicion.get("destination_event_id"),
            "camera_id": suspicion.get("destination_camera_id"),
            "label": "Possible cloned plate evidence",
            "summary": suspicion.get("explanation"),
            "details": suspicion,
        })
    for anomaly in (route_anomaly or {}).get("items", []):
        items.append({
            "type": "route_anomaly",
            "timestamp": anomaly.get("created_at") or anomaly.get("updated_at"),
            "event_id": anomaly.get("end_event_id"),
            "camera_id": anomaly.get("end_camera_id"),
            "label": "Route anomaly evidence",
            "summary": anomaly.get("explanation"),
            "details": anomaly,
        })
    for match in matches:
        items.append({
            "type": "match_evidence",
            "timestamp": match.get("created_at"),
            "event_id": match.get("observation_b_id"),
            "camera_id": match.get("camera_b_id"),
            "label": "Multi-camera match evidence",
            "summary": f"Match confidence {match.get('final_confidence')}; review status {match.get('review_status')}.",
            "details": match,
        })
    return sorted(items, key=lambda item: (_sort_value(item.get("timestamp")), item.get("event_id") or 0, item.get("type") or ""))


def _explanations(anomalies, plate_suspicion, route_anomaly, matches, appearance):
    items = []
    for anomaly in anomalies:
        if anomaly.get("status") in {"detected", "warning"} or anomaly.get("severity"):
            items.append({
                "type": "impossible_travel",
                "severity": anomaly.get("severity"),
                "title": "Impossible travel" if anomaly.get("severity") == "critical" else "Unusually fast travel",
                "summary": anomaly.get("explanation"),
                "evidence": anomaly,
            })
    for suspicion in (plate_suspicion or {}).get("items", []):
        reasons = [item.get("type") for item in suspicion.get("evidence", []) if item.get("type")]
        items.append({
            "type": "plate_suspicion",
            "severity": "warning",
            "title": "Possible cloned plate",
            "summary": suspicion.get("explanation"),
            "reasons": reasons,
            "evidence": suspicion,
        })
    for anomaly in (route_anomaly or {}).get("items", []):
        items.append({
            "type": "route_anomaly",
            "severity": "warning",
            "title": "Route anomaly",
            "summary": anomaly.get("explanation"),
            "evidence": anomaly,
        })
    for comparison in appearance.get("comparisons", []):
        if comparison.get("assessment") == "low_visual_similarity":
            items.append({
                "type": "appearance_conflict",
                "severity": "info",
                "title": "Low visual similarity",
                "summary": (
                    f"Observation {comparison['left_event_id']} and {comparison['right_event_id']} have "
                    f"{comparison['appearance_similarity_percent']}% appearance similarity."
                ),
                "evidence": comparison,
            })
    for match in matches:
        if match.get("review_status") == "pending" or match.get("decision") == "review_required":
            items.append({
                "type": "match_review",
                "severity": "info",
                "title": "Match review required",
                "summary": f"Candidate match {match.get('match_id')} needs operator review.",
                "evidence": match,
            })
    return items


def build_vehicle_investigation(db, vehicle_id):
    """Build one vehicle investigation payload from existing stored evidence."""
    vehicle = get_vehicle_by_id(db, vehicle_id)
    if not vehicle:
        return None
    history = vehicle_history(db, vehicle.vehicle_id)
    trajectory = build_vehicle_trajectory(db, vehicle.vehicle_id)
    if vehicle.primary_plate_text and trajectory:
        trajectory["plate_suspicion"] = plate_suspicion_summary(db, vehicle.primary_plate_text, persist=True)
    if trajectory:
        trajectory["route_anomaly"] = route_anomaly_summary(db, vehicle.vehicle_id, persist=True)
    anomalies = [
        anomaly_payload(row)
        for row in (
            db.query(VehicleAnomaly)
            .filter(VehicleAnomaly.vehicle_id == vehicle.vehicle_id)
            .order_by(VehicleAnomaly.detected_at.desc(), VehicleAnomaly.id.desc())
            .all()
        )
    ]
    plate_suspicion = plate_suspicion_summary(db, vehicle.primary_plate_text, persist=True) if vehicle.primary_plate_text else None
    route_anomaly = route_anomaly_summary(db, vehicle.vehicle_id, persist=True)
    suspicion_rows = (
        db.query(PlateSuspicionEvent)
        .filter(PlateSuspicionEvent.vehicle_id == vehicle.vehicle_id)
        .order_by(PlateSuspicionEvent.suspicion_score.desc(), PlateSuspicionEvent.updated_at.desc())
        .all()
    )
    if plate_suspicion is not None and not plate_suspicion.get("items") and suspicion_rows:
        plate_suspicion["items"] = [suspicion_payload(row) for row in suspicion_rows]
        plate_suspicion["total"] = len(plate_suspicion["items"])
    route_rows = (
        db.query(RouteAnomalyEvent)
        .filter(RouteAnomalyEvent.vehicle_id == vehicle.vehicle_id)
        .order_by(RouteAnomalyEvent.route_anomaly_score.desc(), RouteAnomalyEvent.updated_at.desc())
        .all()
    )
    if route_anomaly is not None and not route_anomaly.get("items") and route_rows:
        route_anomaly["items"] = [route_anomaly_payload(row) for row in route_rows]
        route_anomaly["total"] = len(route_anomaly["items"])
    events = _ordered_vehicle_events(db, vehicle.vehicle_id)
    appearance = _appearance_payload(events)
    matches = matches_for_vehicle(db, vehicle.vehicle_id)
    explanation_items = _explanations(anomalies, plate_suspicion, route_anomaly, matches, appearance)
    timeline = _timeline(history, anomalies, plate_suspicion, route_anomaly, matches)
    db.flush()
    return {
        "vehicle": vehicle_payload(db, vehicle),
        "summary": _summary(vehicle, history, trajectory or {}, anomalies, plate_suspicion, route_anomaly, matches),
        "history": history,
        "trajectory": trajectory,
        "matches": matches,
        "appearance": appearance,
        "anomalies": {"items": anomalies, "total": len(anomalies)},
        "plate_suspicions": plate_suspicion or {"classification": PLATE_NORMAL, "items": [], "total": 0},
        "route_anomalies": route_anomaly or {"classification": "insufficient_history", "items": [], "total": 0},
        "timeline": timeline,
        "explanations": explanation_items,
        "why_flagged": explanation_items,
        "evidence_counts": dict(Counter(item["type"] for item in explanation_items)),
        "review_actions": {
            "plate_suspicion": "PATCH /api/plate-suspicions/{suspicion_id}/review",
            "route_anomaly": "PATCH /api/route-anomalies/{route_anomaly_id}/review",
            "match": "POST /api/matches/{match_id}/review",
        },
    }


def build_plate_investigation(db, plate_text):
    vehicle = get_vehicle_by_plate(db, plate_text)
    if not vehicle:
        return None
    return build_vehicle_investigation(db, vehicle.vehicle_id)
