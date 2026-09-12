"""Conservative multi-camera vehicle matching.

This phase records explainable candidate matches. It does not merge existing
Global Vehicle IDs based on appearance.
"""
from __future__ import annotations

import datetime
import json
import math
import os

from sqlalchemy import or_

from app.database import Camera, PlateEvent, VehicleMatchCandidate, plate_similarity_score
from app.location import camera_location, iso_utc
from app.plate_rules import OCR_ACCEPT_CONFIDENCE, PENDING_REVIEW_STATUS, normalize_plate_text
from app.vehicle_appearance import appearance_similarity


MATCHING_MODEL = "vehicle-match-v1"
MATCHING_VERSION = "2B-2026-09-12"
HIGH_CONFIDENCE = "HIGH_CONFIDENCE"
MEDIUM_CONFIDENCE = "MEDIUM_CONFIDENCE"
LOW_CONFIDENCE = "LOW_CONFIDENCE"


def matching_thresholds():
    return {
        "high_confidence": float(os.getenv("ANPR_MATCH_HIGH_CONFIDENCE", "0.88")),
        "medium_confidence": float(os.getenv("ANPR_MATCH_MEDIUM_CONFIDENCE", "0.62")),
        "lookback_minutes": int(os.getenv("ANPR_MATCH_LOOKBACK_MINUTES", "180")),
        "max_candidates": int(os.getenv("ANPR_MATCH_MAX_CANDIDATES", "80")),
        "plate_similarity_min": float(os.getenv("ANPR_MATCH_PLATE_SIMILARITY_MIN", "0.85")),
        "appearance_similarity_min": float(os.getenv("ANPR_MATCH_APPEARANCE_MIN", "0.72")),
        "appearance_review": float(os.getenv("ANPR_MATCH_APPEARANCE_REVIEW", "0.88")),
    }


def _confirmed_plate(event):
    rule = normalize_plate_text(event.plate_text)
    if not rule.valid_format or not rule.normalized_text:
        return None
    if event.status == PENDING_REVIEW_STATUS or (event.confidence or 0.0) < OCR_ACCEPT_CONFIDENCE:
        return None
    return rule.normalized_text


def _haversine_km(lat1, lng1, lat2, lng2):
    radius_km = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lng2 - lng1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * radius_km * math.asin(math.sqrt(a))


def _camera_relationship(camera_a, camera_b):
    lat_a, lng_a = camera_location(camera_a)
    lat_b, lng_b = camera_location(camera_b)
    relationship = {
        "camera_a_id": camera_a.camera_id if camera_a else None,
        "camera_b_id": camera_b.camera_id if camera_b else None,
        "same_camera": bool(camera_a and camera_b and camera_a.camera_id == camera_b.camera_id),
        "location_known": False,
        "distance_km": None,
    }
    if lat_a is not None and lng_a is not None and lat_b is not None and lng_b is not None:
        relationship["location_known"] = True
        relationship["distance_km"] = round(_haversine_km(lat_a, lng_a, lat_b, lng_b), 3)
    return relationship


def _time_score(seconds):
    if seconds is None or seconds < 0:
        return 0.0
    minutes = seconds / 60.0
    if minutes <= 30:
        return 1.0
    if minutes <= 180:
        return 0.75
    if minutes <= 720:
        return 0.45
    return 0.2


def _camera_score(relationship):
    if relationship["same_camera"]:
        return 0.65
    if relationship["location_known"]:
        return 0.8
    if relationship["camera_a_id"] and relationship["camera_b_id"]:
        return 0.6
    return None


def compare_observations(left, right, camera_left=None, camera_right=None):
    """Return a transparent score between two observations."""
    thresholds = matching_thresholds()
    left_plate = _confirmed_plate(left)
    right_plate = _confirmed_plate(right)
    factor_scores = {}
    plate_evidence = {
        "left_plate": left_plate,
        "right_plate": right_plate,
        "used": False,
        "exact": False,
        "similarity": None,
    }

    if left_plate and right_plate:
        score = plate_similarity_score(left_plate, right_plate)
        if score >= thresholds["plate_similarity_min"]:
            factor_scores["plate_similarity"] = score
            plate_evidence.update(used=True, exact=left_plate == right_plate, similarity=score)

    appearance = appearance_similarity(left.appearance_embedding, right.appearance_embedding)
    if appearance is not None and appearance >= thresholds["appearance_similarity_min"]:
        factor_scores["appearance_similarity"] = appearance

    if left.vehicle_type and right.vehicle_type:
        factor_scores["vehicle_type_similarity"] = 1.0 if left.vehicle_type == right.vehicle_type else 0.0
    if left.vehicle_color and right.vehicle_color:
        factor_scores["vehicle_color_similarity"] = 1.0 if left.vehicle_color == right.vehicle_color else 0.0

    time_delta = None
    if left.timestamp and right.timestamp:
        time_delta = (right.timestamp - left.timestamp).total_seconds()
        factor_scores["time_consistency"] = _time_score(time_delta)

    relationship = _camera_relationship(camera_left, camera_right)
    camera_factor = _camera_score(relationship)
    if camera_factor is not None:
        factor_scores["camera_relationship"] = camera_factor

    weights = {
        "plate_similarity": 0.50,
        "appearance_similarity": 0.30,
        "vehicle_color_similarity": 0.08,
        "vehicle_type_similarity": 0.07,
        "time_consistency": 0.03,
        "camera_relationship": 0.02,
    }
    used_weight = sum(weights[key] for key in factor_scores)
    final_confidence = 0.0 if used_weight <= 0 else sum(factor_scores[key] * weights[key] for key in factor_scores) / used_weight
    final_confidence = round(float(final_confidence), 4)

    conflicting_vehicle_ids = bool(left.vehicle_id and right.vehicle_id and left.vehicle_id != right.vehicle_id)
    has_valid_plate_evidence = bool(plate_evidence["used"])
    has_exact_plate = bool(plate_evidence["exact"])
    has_attribute_evidence = any(key in factor_scores for key in ("vehicle_color_similarity", "vehicle_type_similarity"))
    appearance_only = "appearance_similarity" in factor_scores and not has_valid_plate_evidence and not has_attribute_evidence

    if time_delta is not None and time_delta < 0:
        state = LOW_CONFIDENCE
    elif appearance_only:
        state = LOW_CONFIDENCE
    elif conflicting_vehicle_ids:
        state = MEDIUM_CONFIDENCE if final_confidence >= thresholds["medium_confidence"] else LOW_CONFIDENCE
    elif has_exact_plate and final_confidence >= thresholds["high_confidence"]:
        state = HIGH_CONFIDENCE
    elif final_confidence >= thresholds["medium_confidence"]:
        state = MEDIUM_CONFIDENCE
    elif appearance is not None and appearance >= thresholds["appearance_review"] and not appearance_only:
        state = MEDIUM_CONFIDENCE
    else:
        state = LOW_CONFIDENCE

    return {
        "plate_evidence": plate_evidence,
        "appearance_similarity": appearance,
        "vehicle_type_similarity": factor_scores.get("vehicle_type_similarity"),
        "vehicle_color_similarity": factor_scores.get("vehicle_color_similarity"),
        "time_delta_seconds": time_delta,
        "spatial_relationship": relationship,
        "factor_scores": factor_scores,
        "factors_used": sorted(factor_scores.keys()),
        "final_confidence": final_confidence,
        "state": state,
        "conflicting_vehicle_ids": conflicting_vehicle_ids,
        "matching_model": MATCHING_MODEL,
        "matching_version": MATCHING_VERSION,
    }


def _candidate_vehicle_id(left, right):
    if left.vehicle_id and right.vehicle_id and left.vehicle_id == right.vehicle_id:
        return left.vehicle_id
    if left.vehicle_id and not right.vehicle_id:
        return left.vehicle_id
    if right.vehicle_id and not left.vehicle_id:
        return right.vehicle_id
    return None


def _automatic_decision(current, previous, comparison):
    if comparison["state"] != HIGH_CONFIDENCE:
        return "review_required" if comparison["state"] == MEDIUM_CONFIDENCE else "no_association"
    if comparison["conflicting_vehicle_ids"]:
        return "review_required"
    if current.vehicle_id and previous.vehicle_id and current.vehicle_id == previous.vehicle_id:
        return "confirmed_existing"
    if previous.vehicle_id and not current.vehicle_id:
        current.vehicle_id = previous.vehicle_id
        return "auto_associated"
    return "candidate_recorded"


def _upsert_match_candidate(db, previous, current, comparison):
    left_id, right_id = previous.id, current.id
    existing = (
        db.query(VehicleMatchCandidate)
        .filter(VehicleMatchCandidate.observation_a_id == left_id,
                VehicleMatchCandidate.observation_b_id == right_id)
        .first()
    )
    row = existing or VehicleMatchCandidate(observation_a_id=left_id, observation_b_id=right_id)
    decision = _automatic_decision(current, previous, comparison)
    row.candidate_vehicle_id = _candidate_vehicle_id(previous, current)
    row.camera_a_id = previous.camera_id
    row.camera_b_id = current.camera_id
    row.plate_evidence = json.dumps(comparison["plate_evidence"])
    row.appearance_similarity = comparison["appearance_similarity"]
    row.vehicle_type_similarity = comparison["vehicle_type_similarity"]
    row.vehicle_color_similarity = comparison["vehicle_color_similarity"]
    row.time_delta_seconds = comparison["time_delta_seconds"]
    row.spatial_relationship = json.dumps(comparison["spatial_relationship"])
    row.factor_scores = json.dumps(comparison["factor_scores"])
    row.factors_used = json.dumps(comparison["factors_used"])
    row.final_confidence = comparison["final_confidence"]
    row.state = comparison["state"]
    row.decision = decision
    row.review_status = "pending" if decision == "review_required" else "not_required"
    row.matching_model = comparison["matching_model"]
    row.matching_version = comparison["matching_version"]
    if not existing:
        row.created_at = datetime.datetime.utcnow()
        db.add(row)
    return row


def _candidate_query(db, event):
    thresholds = matching_thresholds()
    since = event.timestamp - datetime.timedelta(minutes=thresholds["lookback_minutes"])
    query = (
        db.query(PlateEvent)
        .filter(PlateEvent.id != event.id)
        .filter(PlateEvent.timestamp >= since)
        .filter(PlateEvent.timestamp <= event.timestamp)
    )
    current_plate = _confirmed_plate(event)
    if current_plate:
        query = query.filter(or_(PlateEvent.plate_text.isnot(None), PlateEvent.appearance_embedding.isnot(None)))
    elif event.appearance_embedding:
        query = query.filter(PlateEvent.appearance_embedding.isnot(None))
        if event.vehicle_color:
            query = query.filter(or_(PlateEvent.vehicle_color == event.vehicle_color, PlateEvent.vehicle_color.is_(None)))
    return (
        query
        .order_by(PlateEvent.timestamp.desc(), PlateEvent.id.desc())
        .limit(max(thresholds["max_candidates"] * 5, thresholds["max_candidates"]))
        .all()
    )


def process_observation_matches(db, event):
    """Generate and persist candidate matches for a newly persisted observation."""
    if event is None or event.id is None or event.timestamp is None:
        return []
    cameras = {camera.camera_id: camera for camera in db.query(Camera).all()}
    matches = []
    for previous in _candidate_query(db, event):
        comparison = compare_observations(previous, event, cameras.get(previous.camera_id), cameras.get(event.camera_id))
        if comparison["state"] == LOW_CONFIDENCE:
            continue
        matches.append(_upsert_match_candidate(db, previous, event, comparison))
        if len(matches) >= matching_thresholds()["max_candidates"]:
            break
    return matches


def _event_summary(event):
    return {
        "event_id": event.id,
        "global_vehicle_id": event.vehicle_id,
        "vehicle_id": event.vehicle_id,
        "plate_text": event.plate_text,
        "plate_status": event.status,
        "confidence": event.confidence,
        "camera_id": event.camera_id,
        "timestamp": iso_utc(event.timestamp),
        "image_path": event.image_path,
        "vehicle_crop_path": event.vehicle_crop_path,
        "vehicle_type": event.vehicle_type,
        "vehicle_color": event.vehicle_color,
        "appearance_available": bool(event.appearance_embedding),
        "appearance_model": event.appearance_model,
        "appearance_embedding_version": event.appearance_embedding_version,
        "appearance_quality": event.appearance_quality,
    }


def match_payload(db, match):
    left = db.get(PlateEvent, match.observation_a_id)
    right = db.get(PlateEvent, match.observation_b_id)
    return {
        "match_id": match.id,
        "observation_a_id": match.observation_a_id,
        "observation_b_id": match.observation_b_id,
        "candidate_vehicle_id": match.candidate_vehicle_id,
        "state": match.state,
        "decision": match.decision,
        "review_status": match.review_status,
        "final_confidence": match.final_confidence,
        "plate_evidence": json.loads(match.plate_evidence or "{}"),
        "appearance_similarity": match.appearance_similarity,
        "vehicle_type_similarity": match.vehicle_type_similarity,
        "vehicle_color_similarity": match.vehicle_color_similarity,
        "time_delta_seconds": match.time_delta_seconds,
        "camera_a_id": match.camera_a_id,
        "camera_b_id": match.camera_b_id,
        "spatial_relationship": json.loads(match.spatial_relationship or "{}"),
        "factor_scores": json.loads(match.factor_scores or "{}"),
        "factors_used": json.loads(match.factors_used or "[]"),
        "matching_model": match.matching_model,
        "matching_version": match.matching_version,
        "created_at": iso_utc(match.created_at),
        "reviewed_at": iso_utc(match.reviewed_at) if match.reviewed_at else None,
        "observation_a": _event_summary(left) if left else None,
        "observation_b": _event_summary(right) if right else None,
    }


def matches_for_observation(db, event_id):
    event = db.get(PlateEvent, event_id)
    if not event:
        return None
    process_observation_matches(db, event)
    db.flush()
    rows = (
        db.query(VehicleMatchCandidate)
        .filter(or_(VehicleMatchCandidate.observation_a_id == event_id,
                    VehicleMatchCandidate.observation_b_id == event_id))
        .order_by(VehicleMatchCandidate.final_confidence.desc(), VehicleMatchCandidate.created_at.desc())
        .all()
    )
    return [match_payload(db, row) for row in rows]


def matches_for_vehicle(db, vehicle_id):
    event_ids = [row[0] for row in db.query(PlateEvent.id).filter(PlateEvent.vehicle_id == vehicle_id).all()]
    if not event_ids:
        return []
    rows = (
        db.query(VehicleMatchCandidate)
        .filter(or_(VehicleMatchCandidate.observation_a_id.in_(event_ids),
                    VehicleMatchCandidate.observation_b_id.in_(event_ids),
                    VehicleMatchCandidate.candidate_vehicle_id == vehicle_id))
        .order_by(VehicleMatchCandidate.final_confidence.desc(), VehicleMatchCandidate.created_at.desc())
        .all()
    )
    return [match_payload(db, row) for row in rows]


def review_match(db, match_id, action, reviewed_by="operator"):
    action = str(action or "").strip().lower()
    if action not in {"accept", "reject"}:
        raise ValueError("action must be accept or reject")
    match = db.get(VehicleMatchCandidate, match_id)
    if not match:
        return None
    left = db.get(PlateEvent, match.observation_a_id)
    right = db.get(PlateEvent, match.observation_b_id)
    if not left or not right:
        raise ValueError("Matched observations are missing")
    match.reviewed_at = datetime.datetime.utcnow()
    match.reviewed_by = reviewed_by
    if action == "reject":
        match.review_status = "rejected"
        match.decision = "rejected"
        return match
    if left.vehicle_id and right.vehicle_id and left.vehicle_id != right.vehicle_id:
        raise ValueError("Cannot merge two existing Global Vehicle IDs in this phase")
    target_vehicle_id = left.vehicle_id or right.vehicle_id or match.candidate_vehicle_id
    if not target_vehicle_id:
        raise ValueError("No existing Global Vehicle ID is available for this accepted match")
    if not left.vehicle_id:
        left.vehicle_id = target_vehicle_id
    if not right.vehicle_id:
        right.vehicle_id = target_vehicle_id
    match.candidate_vehicle_id = target_vehicle_id
    match.review_status = "accepted"
    match.decision = "accepted"
    return match
