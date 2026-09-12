"""
Trajectory stitching engine.

Groups plate_events by plate_text, orders them by timestamp, and computes a
travel-time plausibility flag between consecutive sightings using straight-line
distance between camera GPS points (a stand-in for a real routing-API distance —
swap `estimate_travel_minutes` for a Google Distance Matrix / OSRM call in
production).
"""
import math
from app.location import camera_location, time_window, iso_utc
from app.database import PlateEvent, Camera, Vehicle, backfill_vehicle_ids, plate_similarity_score, stable_vehicle_id_for_plate
from app.plate_rules import OCR_ACCEPT_CONFIDENCE, PENDING_REVIEW_STATUS, normalize_plate_text
from app.travel_time import calculate_travel_segment

ASSUMED_AVG_SPEED_KMPH = 25  # conservative city-traffic average


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
            "timestamp": iso_utc(event.timestamp),
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

            if (cam and prev_cam and cam.camera_id != prev_cam.camera_id
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
        "timestamp": iso_utc(event.timestamp),
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
        "first_seen": iso_utc(events[0].timestamp) if events else None,
        "last_seen": iso_utc(last.timestamp) if last else None,
        "last_camera": last.camera_id if last else None,
        "last_camera_label": cameras.get(last.camera_id).label if last and cameras.get(last.camera_id) else (last.camera_id if last else None),
        "vehicle_type": last.vehicle_type if last else None,
        "vehicle_color": last.vehicle_color if last else None,
        "appearance_available": bool(last and last.appearance_embedding),
        "appearance_model": last.appearance_model if last else None,
        "appearance_embedding_version": last.appearance_embedding_version if last else None,
        "appearance_quality": last.appearance_quality if last else None,
        "created_at": iso_utc(vehicle.created_at),
        "updated_at": iso_utc(vehicle.updated_at),
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
    return payload


def list_all_vehicles(db):
    """Distinct global vehicles seen so far — powers the dashboard list."""
    backfill_vehicle_ids(db)
    result = [
        payload for payload in (vehicle_payload(db, vehicle) for vehicle in db.query(Vehicle).all())
        if payload["sightings"] > 0
    ]
    return sorted(result, key=lambda r: r["last_seen"] or "", reverse=True)
