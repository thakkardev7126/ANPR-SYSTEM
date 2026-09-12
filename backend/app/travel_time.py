"""Camera-to-camera travel-time and estimated average speed helpers.

Phase 3B deliberately calculates these values from persisted observations and
configured road connections. Nothing here runs per video frame and nothing is
treated as instantaneous or authoritative vehicle speed.
"""
from __future__ import annotations

from app.database import Camera, PlateEvent
from app.location import iso_utc
from app.road_network import get_connection


CALCULATED = "calculated"
CALCULATED_APPROXIMATE = "calculated_approximate"
MISSING_ROAD_CONNECTION = "missing_road_connection"
INVALID_TIME = "invalid_time"
INVALID_DISTANCE = "invalid_distance"
UNAVAILABLE = "unavailable"


def _event_time(value):
    return iso_utc(value) if value else None


def _base_segment(previous_event, event):
    return {
        "source_event_id": previous_event.id if previous_event else None,
        "destination_event_id": event.id if event else None,
        "start_event_id": previous_event.id if previous_event else None,
        "end_event_id": event.id if event else None,
        "source_camera_id": previous_event.camera_id if previous_event else None,
        "destination_camera_id": event.camera_id if event else None,
        "start_camera_id": previous_event.camera_id if previous_event else None,
        "end_camera_id": event.camera_id if event else None,
        "start_time": _event_time(previous_event.timestamp) if previous_event else None,
        "end_time": _event_time(event.timestamp) if event else None,
        "distance_meters": None,
        "distance_km": None,
        "distance_source": None,
        "distance_is_approximate": False,
        "travel_time_seconds": None,
        "estimated_speed_mps": None,
        "estimated_speed_kmh": None,
        "speed_available": False,
        "speed_status": UNAVAILABLE,
        "speed_status_detail": None,
        "connection_id": None,
        "road_name": None,
        "road_type": None,
        "direction": None,
    }


def calculate_travel_segment(db, previous_event: PlateEvent | None, event: PlateEvent | None):
    """Return a safe travel-time/speed payload for one directed event pair."""
    segment = _base_segment(previous_event, event)
    if previous_event is None or event is None:
        segment["speed_status_detail"] = "Two observations are required."
        return segment

    if not previous_event.timestamp or not event.timestamp:
        segment["speed_status"] = INVALID_TIME
        segment["speed_status_detail"] = "One or both observations are missing timestamps."
        return segment

    travel_time = (event.timestamp - previous_event.timestamp).total_seconds()
    segment["travel_time_seconds"] = round(travel_time, 3)
    if travel_time <= 0:
        segment["speed_status"] = INVALID_TIME
        segment["speed_status_detail"] = "Destination observation must be later than source observation."
        return segment

    connection = get_connection(db, previous_event.camera_id, event.camera_id)
    if not connection:
        segment["speed_status"] = MISSING_ROAD_CONNECTION
        segment["speed_status_detail"] = "No active directed road connection is configured for this camera pair."
        return segment

    segment.update({
        "distance_meters": connection.distance_meters,
        "distance_km": round(connection.distance_meters / 1000.0, 3) if connection.distance_meters else None,
        "distance_source": connection.distance_source,
        "distance_is_approximate": connection.distance_source == "fallback_straight_line",
        "connection_id": connection.id,
        "road_name": connection.road_name,
        "road_type": connection.road_type,
        "direction": connection.direction,
    })
    try:
        distance = float(connection.distance_meters)
    except (TypeError, ValueError):
        distance = 0.0
    if distance <= 0:
        segment["speed_status"] = INVALID_DISTANCE
        segment["speed_status_detail"] = "Configured road distance is missing or not positive."
        return segment

    speed_mps = distance / travel_time
    segment.update({
        "estimated_speed_mps": round(speed_mps, 3),
        "estimated_speed_kmh": round(speed_mps * 3.6, 3),
        "speed_available": True,
        "speed_status": CALCULATED_APPROXIMATE
        if connection.distance_source == "fallback_straight_line" else CALCULATED,
        "speed_status_detail": (
            "Estimated camera-to-camera average speed from approximate straight-line fallback distance."
            if connection.distance_source == "fallback_straight_line"
            else "Estimated camera-to-camera average speed from configured road distance."
        ),
    })
    return segment


def speed_summary(segments):
    valid = [segment for segment in segments if segment.get("speed_available")]
    unavailable = [segment for segment in segments if not segment.get("speed_available")]
    average = None
    if valid:
        average = round(
            sum(segment["estimated_speed_kmh"] for segment in valid) / len(valid),
            3,
        )
    return {
        "average_estimated_speed_kmh": average,
        "valid_speed_segments": len(valid),
        "unavailable_speed_segments": len(unavailable),
        "speed_type": "camera_to_camera_estimated_average",
    }


def vehicle_speed_history(db, vehicle_id: str):
    events = (
        db.query(PlateEvent)
        .filter(PlateEvent.vehicle_id == vehicle_id)
        .order_by(PlateEvent.timestamp.asc(), PlateEvent.id.asc())
        .all()
    )
    if not events:
        return None
    cameras = {camera.camera_id: camera for camera in db.query(Camera).all()}
    segments = []
    for previous, current in zip(events, events[1:]):
        segment = calculate_travel_segment(db, previous, current)
        source_camera = cameras.get(previous.camera_id)
        destination_camera = cameras.get(current.camera_id)
        segment.update({
            "global_vehicle_id": vehicle_id,
            "vehicle_id": vehicle_id,
            "source_camera_label": source_camera.label if source_camera else previous.camera_id,
            "destination_camera_label": destination_camera.label if destination_camera else current.camera_id,
            "start_camera_label": source_camera.label if source_camera else previous.camera_id,
            "end_camera_label": destination_camera.label if destination_camera else current.camera_id,
            "source_plate_text": previous.plate_text,
            "destination_plate_text": current.plate_text,
        })
        segments.append(segment)
    return {
        "global_vehicle_id": vehicle_id,
        "vehicle_id": vehicle_id,
        "segments": segments,
        "summary": speed_summary(segments),
    }
