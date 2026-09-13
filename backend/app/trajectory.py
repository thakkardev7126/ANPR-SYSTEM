"""Trajectory stitching engine.

The legacy `build_trajectory` return shape is preserved for the dashboard. The
Phase 3C complete trajectory helpers below derive richer observation/transition
records from existing plate events, road-network connections, speed estimates,
and recorded match evidence.
"""
import json
import math
import datetime
from sqlalchemy import or_
from app.location import camera_location, time_window, iso_utc
from app.database import PlateEvent, Camera, Vehicle, VehicleMatchCandidate, backfill_vehicle_ids, plate_similarity_score, stable_vehicle_id_for_plate
from app.plate_rules import OCR_ACCEPT_CONFIDENCE, PENDING_REVIEW_STATUS, normalize_plate_text
from app.travel_time import calculate_travel_segment, speed_summary
from app.anomalies import annotate_segment_anomaly
from app.plate_suspicion import suspicion_for_pair
from app.route_anomaly import route_anomaly_for_pair

ASSUMED_AVG_SPEED_KMPH = 25  # conservative city-traffic average
MAX_SORT_TIMESTAMP = datetime.datetime.max


def _safe_iso(value):
    return iso_utc(value) if value else None


def is_confirmed_plate_event(event):
    rule = normalize_plate_text(event.plate_text)
    return (
        bool(rule.valid_format)
        and event.status != PENDING_REVIEW_STATUS
        and (event.confidence or 0.0) >= OCR_ACCEPT_CONFIDENCE
    )


def haversine_km(lat1, lng1, lat2, lng2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lng2 - lng1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def estimate_travel_minutes(cam_a: Camera, cam_b: Camera):
    dist_km = haversine_km(cam_a.lat, cam_a.lng, cam_b.lat, cam_b.lng)
    return (dist_km / ASSUMED_AVG_SPEED_KMPH) * 60


def build_trajectory(db, plate_text: str, similarity_threshold=0.70, start=None, end=None,
                     lat=None, lng=None, radius_m=5000, minutes=None):
    """Return an ordered list of hops for a given plate, each flagged as
    plausible/implausible based on elapsed time vs. estimated travel time."""
    start, end = time_window(start, end, minutes)
    candidate_events = (
        db.query(PlateEvent)
        .filter(PlateEvent.plate_text.isnot(None))
        .order_by(PlateEvent.timestamp.asc())
    )
    if start is not None:
        candidate_events = candidate_events.filter(PlateEvent.timestamp >= start)
    if end is not None:
        candidate_events = candidate_events.filter(PlateEvent.timestamp <= end)
    events = [
        event for event in candidate_events
        if is_confirmed_plate_event(event)
        and plate_similarity_score(event.plate_text, plate_text) >= similarity_threshold
    ]

    cameras = {c.camera_id: c for c in db.query(Camera).all()}
    if lat is not None and lng is not None:
        events = [event for event in events
                  if camera_location(cameras.get(event.camera_id))[0] is not None
                  and haversine_km(lat, lng, *camera_location(cameras.get(event.camera_id))) * 1000 <= radius_m]
    hops = []

    for i, event in enumerate(events):
        cam = cameras.get(event.camera_id)
        hop = {
            "event_id": event.id,
            "global_vehicle_id": event.vehicle_id,
            "vehicle_id": event.vehicle_id,
            "camera_id": event.camera_id,
            "camera_label": cam.label if cam else event.camera_id,
            "lat": camera_location(cam)[0],
            "lng": camera_location(cam)[1],
            "timestamp": _safe_iso(event.timestamp),
            "confidence": event.confidence,
            "status": event.status,
            "match_score": plate_similarity_score(event.plate_text, plate_text),
            "matched_plate_text": event.plate_text,
            "plausible": True,
            "elapsed_minutes": None,
            "min_expected_minutes": None,
            "road_transition": None,
            "road_distance_meters": None,
            "road_distance_source": None,
            "road_direction": None,
            "travel_time_seconds": None,
            "estimated_speed_kmh": None,
            "estimated_speed_mps": None,
            "speed_available": False,
            "speed_status": "unavailable",
            "speed_status_detail": None,
            "distance_is_approximate": False,
        }

        if i > 0:
            prev_event = events[i - 1]
            prev_cam = cameras.get(prev_event.camera_id)
            elapsed = None
            if event.timestamp and prev_event.timestamp:
                elapsed = (event.timestamp - prev_event.timestamp).total_seconds() / 60.0
                hop["elapsed_minutes"] = round(elapsed, 1)
            from app.road_network import camera_transition
            transition = camera_transition(db, prev_event.camera_id, event.camera_id)
            hop["road_transition"] = transition
            hop["road_distance_meters"] = transition.get("distance_meters")
            hop["road_distance_source"] = transition.get("distance_source")
            hop["road_direction"] = transition.get("direction")
            speed_segment = calculate_travel_segment(db, prev_event, event)
            hop["travel_time_seconds"] = speed_segment["travel_time_seconds"]
            hop["estimated_speed_kmh"] = speed_segment["estimated_speed_kmh"]
            hop["estimated_speed_mps"] = speed_segment["estimated_speed_mps"]
            hop["speed_available"] = speed_segment["speed_available"]
            hop["speed_status"] = speed_segment["speed_status"]
            hop["speed_status_detail"] = speed_segment["speed_status_detail"]
            hop["distance_is_approximate"] = speed_segment["distance_is_approximate"]
            if speed_segment["distance_meters"] is not None:
                hop["road_distance_meters"] = speed_segment["distance_meters"]
                hop["road_distance_source"] = speed_segment["distance_source"]
                hop["road_direction"] = speed_segment["direction"]

            if (elapsed is not None and cam and prev_cam and cam.camera_id != prev_cam.camera_id
                    and camera_location(cam)[0] is not None and camera_location(prev_cam)[0] is not None):
                min_expected = estimate_travel_minutes(prev_cam, cam)
                hop["min_expected_minutes"] = round(min_expected, 1)
                # Flag implausible only if the car arrived impossibly fast
                # (allow generous slack for the demo's short mock distances).
                hop["plausible"] = elapsed >= max(min_expected * 0.3, 0.2)

        hops.append(hop)

    return hops


def _bbox_payload(event):
    if event.bbox_x is None or event.bbox_y is None or event.bbox_width is None or event.bbox_height is None:
        return None
    return {
        "x": event.bbox_x,
        "y": event.bbox_y,
        "width": event.bbox_width,
        "height": event.bbox_height,
    }


def vehicle_observation_payload(event, camera=None):
    lat, lng = camera_location(camera)
    return {
        "event_id": event.id,
        "global_vehicle_id": event.vehicle_id,
        "vehicle_id": event.vehicle_id,
        "plate_text": event.plate_text,
        "plate_status": event.status,
        "status": event.status,
        "confidence": event.confidence,
        "camera_id": event.camera_id,
        "camera_label": camera.label if camera else event.camera_id,
        "lat": lat,
        "lng": lng,
        "timestamp": _safe_iso(event.timestamp),
        "image_path": event.image_path,
        "bbox": _bbox_payload(event),
        "track_id": event.track_id,
        "plate_category": event.plate_category,
        "layout": event.layout,
        "vehicle_type": event.vehicle_type,
        "vehicle_color": event.vehicle_color,
        "vehicle_crop_path": event.vehicle_crop_path,
        "appearance_available": bool(event.appearance_embedding),
        "appearance_model": event.appearance_model,
        "appearance_embedding_version": event.appearance_embedding_version,
        "appearance_quality": event.appearance_quality,
    }


def _event_sort_key(event):
    return (event.timestamp is None, event.timestamp or MAX_SORT_TIMESTAMP, event.id or 0)


def _ordered_events(events):
    return sorted(events, key=_event_sort_key)


def _matching_evidence_for_pair(db, previous_event, event):
    if not previous_event or not event or previous_event.id is None or event.id is None:
        return None
    match = (
        db.query(VehicleMatchCandidate)
        .filter(or_(
            (VehicleMatchCandidate.observation_a_id == previous_event.id)
            & (VehicleMatchCandidate.observation_b_id == event.id),
            (VehicleMatchCandidate.observation_a_id == event.id)
            & (VehicleMatchCandidate.observation_b_id == previous_event.id),
        ))
        .order_by(VehicleMatchCandidate.final_confidence.desc(), VehicleMatchCandidate.created_at.desc())
        .first()
    )
    if not match:
        return None
    return {
        "match_id": match.id,
        "confidence": match.final_confidence,
        "state": match.state,
        "decision": match.decision,
        "review_status": match.review_status,
        "candidate_vehicle_id": match.candidate_vehicle_id,
        "matching_model": match.matching_model,
        "matching_version": match.matching_version,
        "factors_used": json.loads(match.factors_used or "[]"),
    }


def _continuity_status(segment, previous_event, event, cameras):
    if not previous_event or not event:
        return "invalid_observation"
    if not previous_event.camera_id or not event.camera_id:
        return "invalid_observation"
    if previous_event.camera_id not in cameras or event.camera_id not in cameras:
        return "invalid_observation"
    status = segment.get("speed_status")
    if status in {"calculated", "calculated_approximate"}:
        return "connected"
    if status == "missing_road_connection":
        return "not_connected"
    if status == "invalid_time":
        return "invalid_time"
    return "invalid_observation"


def _trajectory_segment_payload(db, previous_event, event, cameras, sequence_index):
    segment = calculate_travel_segment(db, previous_event, event)
    continuity = _continuity_status(segment, previous_event, event, cameras)
    source_camera = cameras.get(previous_event.camera_id) if previous_event else None
    destination_camera = cameras.get(event.camera_id) if event else None
    if continuity == "invalid_observation" and segment.get("speed_status") in {"missing_road_connection", "unavailable"}:
        segment["speed_status"] = "invalid_observation"
        segment["speed_status_detail"] = "One or both observations reference a missing or unknown camera."
    vehicle_id = event.vehicle_id or previous_event.vehicle_id if event and previous_event else None
    plate_text = event.plate_text if event else (previous_event.plate_text if previous_event else None)
    anomaly = annotate_segment_anomaly(db, segment, vehicle_id=vehicle_id, plate_text=plate_text, persist=True)
    plate_suspicion = suspicion_for_pair(db, previous_event.id, event.id) if previous_event and event else None
    route_anomaly = route_anomaly_for_pair(db, vehicle_id, previous_event.id, event.id) if previous_event and event and vehicle_id else None
    return {
        **segment,
        **anomaly,
        "route_status": continuity,
        "route_anomaly_status": route_anomaly["classification"] if route_anomaly else "normal",
        "route_anomaly_score": route_anomaly["route_anomaly_score"] if route_anomaly else 0.0,
        "route_anomaly_id": route_anomaly["id"] if route_anomaly else None,
        "route_anomaly_evidence": route_anomaly["evidence"] if route_anomaly else None,
        "route_anomaly_explanation": route_anomaly["explanation"] if route_anomaly else None,
        "plate_suspicion_status": plate_suspicion["classification"] if plate_suspicion else "normal",
        "plate_suspicion_score": plate_suspicion["suspicion_score"] if plate_suspicion else 0.0,
        "plate_suspicion_id": plate_suspicion["id"] if plate_suspicion else None,
        "plate_suspicion_evidence": plate_suspicion["evidence"] if plate_suspicion else [],
        "sequence_index": sequence_index,
        "hop_index": sequence_index,
        "continuity": continuity,
        "source_observation": vehicle_observation_payload(previous_event, source_camera) if previous_event else None,
        "destination_observation": vehicle_observation_payload(event, destination_camera) if event else None,
        "road_connection": {
            "connection_id": segment.get("connection_id"),
            "source_camera_id": segment.get("source_camera_id"),
            "destination_camera_id": segment.get("destination_camera_id"),
            "distance_meters": segment.get("distance_meters"),
            "distance_source": segment.get("distance_source"),
            "direction": segment.get("direction"),
            "road_name": segment.get("road_name"),
            "road_type": segment.get("road_type"),
            "active": bool(segment.get("connection_id")),
        } if segment.get("connection_id") else None,
        "matching_evidence": _matching_evidence_for_pair(db, previous_event, event),
    }


def trajectory_summary(observations, segments):
    valid = [segment for segment in segments if segment.get("speed_available")]
    total_distance = round(sum(segment.get("distance_meters") or 0 for segment in valid), 3) if valid else 0.0
    total_time = round(sum(segment.get("travel_time_seconds") or 0 for segment in valid), 3) if valid else 0.0
    average = round((total_distance / total_time) * 3.6, 3) if total_time > 0 else None
    first = observations[0] if observations else None
    last = observations[-1] if observations else None
    return {
        "total_observations": len(observations),
        "total_hops": len(segments),
        "valid_segments": len(valid),
        "unavailable_segments": len(segments) - len(valid),
        "total_road_distance_meters": total_distance,
        "total_travel_time_seconds": total_time,
        "average_estimated_speed_kmh": average,
        "first_camera_id": first.get("camera_id") if first else None,
        "last_camera_id": last.get("camera_id") if last else None,
        "first_observed_at": first.get("timestamp") if first else None,
        "last_observed_at": last.get("timestamp") if last else None,
        "speed_type": "camera_to_camera_estimated_average",
    }


def complete_trajectory_payload(db, events, vehicle=None, plate_text=None, match_scores=None):
    cameras = {camera.camera_id: camera for camera in db.query(Camera).all()}
    ordered = _ordered_events(events)
    observations = []
    for index, event in enumerate(ordered):
        payload = vehicle_observation_payload(event, cameras.get(event.camera_id))
        payload["sequence_index"] = index
        payload["matched_plate_text"] = event.plate_text
        if match_scores and event.id in match_scores:
            payload["match_score"] = match_scores[event.id]
        observations.append(payload)

    segments = [
        _trajectory_segment_payload(db, previous, current, cameras, index)
        for index, (previous, current) in enumerate(zip(ordered, ordered[1:]), start=1)
    ]
    summary = trajectory_summary(observations, segments)
    vehicle_data = vehicle_payload(db, vehicle) if vehicle else None
    return {
        "vehicle": vehicle_data,
        "plate_text": plate_text or (vehicle.primary_plate_text if vehicle else None),
        "global_vehicle_id": vehicle.vehicle_id if vehicle else (observations[0]["global_vehicle_id"] if observations else None),
        "vehicle_id": vehicle.vehicle_id if vehicle else (observations[0]["vehicle_id"] if observations else None),
        "observations": observations,
        "trajectory_hops": segments,
        "segments": segments,
        "summary": summary,
        "speed_summary": speed_summary(segments),
    }


def build_vehicle_trajectory(db, vehicle_id, start=None, end=None):
    backfill_vehicle_ids(db)
    vehicle = db.get(Vehicle, vehicle_id)
    if not vehicle:
        return None
    query = (
        db.query(PlateEvent)
        .filter(PlateEvent.vehicle_id == vehicle.vehicle_id)
        .order_by(PlateEvent.timestamp.asc(), PlateEvent.id.asc())
    )
    events = query.all()
    if start is not None:
        events = [event for event in events if event.timestamp and event.timestamp >= start]
    if end is not None:
        events = [event for event in events if event.timestamp and event.timestamp <= end]
    return complete_trajectory_payload(db, events, vehicle=vehicle, plate_text=vehicle.primary_plate_text)


def build_plate_complete_trajectory(db, plate_text, similarity_threshold=0.70, start=None, end=None,
                                    lat=None, lng=None, radius_m=5000, minutes=None):
    start, end = time_window(start, end, minutes)
    candidate_events = (
        db.query(PlateEvent)
        .filter(PlateEvent.plate_text.isnot(None))
        .order_by(PlateEvent.timestamp.asc(), PlateEvent.id.asc())
        .all()
    )
    match_scores = {}
    events = []
    for event in candidate_events:
        if not is_confirmed_plate_event(event):
            continue
        if start is not None and (not event.timestamp or event.timestamp < start):
            continue
        if end is not None and (not event.timestamp or event.timestamp > end):
            continue
        score = plate_similarity_score(event.plate_text, plate_text)
        if score < similarity_threshold:
            continue
        match_scores[event.id] = round(score, 3)
        events.append(event)
    cameras = {c.camera_id: c for c in db.query(Camera).all()}
    if lat is not None and lng is not None:
        events = [
            event for event in events
            if camera_location(cameras.get(event.camera_id))[0] is not None
            and haversine_km(lat, lng, *camera_location(cameras.get(event.camera_id))) * 1000 <= radius_m
        ]
    normalized = normalize_plate_text(plate_text).normalized_text
    vehicle = db.get(Vehicle, stable_vehicle_id_for_plate(normalized)) if normalized else None
    return complete_trajectory_payload(db, events, vehicle=vehicle, plate_text=plate_text, match_scores=match_scores)


def vehicle_payload(db, vehicle):
    events = (
        db.query(PlateEvent)
        .filter(PlateEvent.vehicle_id == vehicle.vehicle_id)
        .order_by(PlateEvent.timestamp.asc())
        .all()
    )
    cameras = {c.camera_id: c for c in db.query(Camera).all()}
    last = events[-1] if events else None
    return {
        "global_vehicle_id": vehicle.vehicle_id,
        "vehicle_id": vehicle.vehicle_id,
        "primary_plate_text": vehicle.primary_plate_text,
        "plate_text": vehicle.primary_plate_text,
        "sightings": len(events),
        "first_seen": _safe_iso(events[0].timestamp) if events else None,
        "last_seen": _safe_iso(last.timestamp) if last else None,
        "last_camera": last.camera_id if last else None,
        "last_camera_label": cameras.get(last.camera_id).label if last and cameras.get(last.camera_id) else (last.camera_id if last else None),
        "vehicle_type": last.vehicle_type if last else None,
        "vehicle_color": last.vehicle_color if last else None,
        "appearance_available": bool(last and last.appearance_embedding),
        "appearance_model": last.appearance_model if last else None,
        "appearance_embedding_version": last.appearance_embedding_version if last else None,
        "appearance_quality": last.appearance_quality if last else None,
        "created_at": _safe_iso(vehicle.created_at),
        "updated_at": _safe_iso(vehicle.updated_at),
    }


def get_vehicle_by_id(db, vehicle_id):
    backfill_vehicle_ids(db)
    return db.get(Vehicle, vehicle_id)


def get_vehicle_by_plate(db, plate_text):
    backfill_vehicle_ids(db)
    normalized = normalize_plate_text(plate_text).normalized_text
    if not normalized:
        return None
    vehicle_id = stable_vehicle_id_for_plate(normalized)
    return db.get(Vehicle, vehicle_id)


def vehicle_history(db, vehicle_id):
    backfill_vehicle_ids(db)
    vehicle = db.get(Vehicle, vehicle_id)
    if not vehicle:
        return None
    cameras = {c.camera_id: c for c in db.query(Camera).all()}
    observations = [
        vehicle_observation_payload(event, cameras.get(event.camera_id))
        for event in (
            db.query(PlateEvent)
            .filter(PlateEvent.vehicle_id == vehicle.vehicle_id)
            .order_by(PlateEvent.timestamp.asc(), PlateEvent.id.asc())
            .all()
        )
    ]
    payload = vehicle_payload(db, vehicle)
    payload["observations"] = observations
    complete = build_vehicle_trajectory(db, vehicle_id)
    payload["trajectory_hops"] = complete["trajectory_hops"] if complete else []
    payload["segments"] = complete["segments"] if complete else []
    payload["summary"] = complete["summary"] if complete else trajectory_summary(observations, [])
    return payload


def list_all_vehicles(db):
    """Distinct global vehicles seen so far — powers the dashboard list."""
    backfill_vehicle_ids(db)
    result = [
        payload for payload in (vehicle_payload(db, vehicle) for vehicle in db.query(Vehicle).all())
        if payload["sightings"] > 0
    ]
    return sorted(result, key=lambda r: r["last_seen"] or "", reverse=True)
