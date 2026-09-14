"""Explainable urban traffic analytics from persisted ANPR observations.

Phase 6 is intentionally derived data. It reuses PlateEvent observations,
Global Vehicle IDs, configured road connections, and the existing travel-time
service. No OCR, detection, appearance extraction, or identity matching runs
while building these aggregates.
"""
from __future__ import annotations

import datetime
import os
from collections import defaultdict
from statistics import median

from app.database import Camera, CameraRoadConnection, PlateEvent
from app.location import camera_location, iso_utc
from app.travel_time import calculate_travel_segment, speed_summary


VALID_BUCKETS = {"hour", "day"}
LANE_STATUS = "NO_DATA"


def _float_env(name, default):
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return float(default)


def traffic_policy():
    low = _float_env("ANPR_TRAFFIC_LOW_THRESHOLD", "5")
    medium = _float_env("ANPR_TRAFFIC_MEDIUM_THRESHOLD", "15")
    medium = max(low, medium)
    road_low = _float_env("ANPR_TRAFFIC_ROAD_LOW_THRESHOLD", str(low))
    road_medium = _float_env("ANPR_TRAFFIC_ROAD_MEDIUM_THRESHOLD", str(medium))
    road_medium = max(road_low, road_medium)
    return {
        "density_low_threshold": low,
        "density_medium_threshold": medium,
        "road_low_threshold": road_low,
        "road_medium_threshold": road_medium,
        "congestion_watch_speed_kmh": _float_env("ANPR_CONGESTION_WATCH_SPEED_KMH", "25"),
        "congestion_congested_speed_kmh": _float_env("ANPR_CONGESTION_CONGESTED_SPEED_KMH", "12"),
        "congestion_watch_volume": _float_env("ANPR_CONGESTION_WATCH_VOLUME", "5"),
        "congestion_congested_volume": _float_env("ANPR_CONGESTION_CONGESTED_VOLUME", "12"),
    }


def validate_bucket(bucket):
    value = str(bucket or "hour").strip().lower()
    if value not in VALID_BUCKETS:
        raise ValueError("bucket must be hour or day")
    return value


def _safe_iso(value):
    return iso_utc(value) if value else None


def _bucket_start(timestamp, bucket):
    if not timestamp:
        return None
    if bucket == "day":
        return timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
    return timestamp.replace(minute=0, second=0, microsecond=0)


def _bucket_key(timestamp, bucket):
    start = _bucket_start(timestamp, bucket)
    return _safe_iso(start) if start else None


def _level(count, low, medium):
    if count >= medium:
        return "HIGH"
    if count >= low:
        return "MEDIUM"
    return "LOW"


def _events_query(db, start=None, end=None, camera_id=None):
    query = db.query(PlateEvent)
    if start is not None:
        query = query.filter(PlateEvent.timestamp >= start)
    if end is not None:
        query = query.filter(PlateEvent.timestamp <= end)
    if camera_id:
        query = query.filter(PlateEvent.camera_id == camera_id)
    return query


def _events(db, start=None, end=None, camera_id=None):
    return (
        _events_query(db, start=start, end=end, camera_id=camera_id)
        .order_by(PlateEvent.timestamp.asc(), PlateEvent.id.asc())
        .all()
    )


def _camera_payload(camera):
    lat, lng = camera_location(camera)
    return {
        "camera_id": camera.camera_id,
        "camera_label": camera.label,
        "lat": lat,
        "lng": lng,
        "location_known": bool(camera.location_known and lat is not None and lng is not None),
    }


def _camera_map(db):
    return {camera.camera_id: camera for camera in db.query(Camera).all()}


def _filtered_connection_query(db, source_camera_id=None, destination_camera_id=None):
    query = db.query(CameraRoadConnection).filter(CameraRoadConnection.active.is_(True))
    if source_camera_id:
        query = query.filter(CameraRoadConnection.source_camera_id == source_camera_id)
    if destination_camera_id:
        query = query.filter(CameraRoadConnection.destination_camera_id == destination_camera_id)
    return query


def camera_metrics(db, start=None, end=None, camera_id=None):
    cameras = _camera_map(db)
    if camera_id and camera_id not in cameras:
        return None
    grouped = {
        cid: {
            **_camera_payload(camera),
            "observation_count": 0,
            "unique_vehicle_count": 0,
            "valid_vehicle_observation_count": 0,
            "missing_identity_observation_count": 0,
        }
        for cid, camera in cameras.items()
        if camera_id is None or cid == camera_id
    }
    vehicle_sets = defaultdict(set)
    for event in _events(db, start=start, end=end, camera_id=camera_id):
        if event.camera_id not in grouped:
            continue
        item = grouped[event.camera_id]
        item["observation_count"] += 1
        if event.vehicle_id:
            item["valid_vehicle_observation_count"] += 1
            vehicle_sets[event.camera_id].add(event.vehicle_id)
        else:
            item["missing_identity_observation_count"] += 1
    for cid, item in grouped.items():
        item["unique_vehicle_count"] = len(vehicle_sets[cid])
        item["metric_note"] = (
            "observation_count counts persisted camera observations; unique_vehicle_count counts distinct Global Vehicle IDs only."
        )
    return sorted(grouped.values(), key=lambda item: (-item["observation_count"], item["camera_id"]))


def density_metrics(db, start=None, end=None, camera_id=None):
    policy = traffic_policy()
    rows = []
    for item in camera_metrics(db, start=start, end=end, camera_id=camera_id) or []:
        level = _level(item["observation_count"], policy["density_low_threshold"], policy["density_medium_threshold"])
        rows.append({
            **item,
            "density_level": level,
            "density_status": level,
            "low_threshold": policy["density_low_threshold"],
            "medium_threshold": policy["density_medium_threshold"],
            "explanation": (
                f"{item['observation_count']} camera observation(s) and {item['unique_vehicle_count']} distinct "
                "Global Vehicle ID(s) in the selected window."
            ),
        })
    return rows


def timeseries(db, start=None, end=None, camera_id=None, bucket="hour"):
    bucket = validate_bucket(bucket)
    grouped = {}
    vehicles = defaultdict(set)
    for event in _events(db, start=start, end=end, camera_id=camera_id):
        key = _bucket_key(event.timestamp, bucket)
        if not key:
            continue
        group_key = (key, event.camera_id)
        row = grouped.setdefault(group_key, {
            "time_bucket": key,
            "bucket": bucket,
            "camera_id": event.camera_id,
            "observation_count": 0,
            "unique_vehicle_count": 0,
        })
        row["observation_count"] += 1
        if event.vehicle_id:
            vehicles[group_key].add(event.vehicle_id)
    for key, row in grouped.items():
        row["unique_vehicle_count"] = len(vehicles[key])
    return sorted(grouped.values(), key=lambda item: (item["time_bucket"], item["camera_id"] or ""))


def _vehicle_events(db, start=None, end=None):
    query = (
        db.query(PlateEvent)
        .filter(PlateEvent.vehicle_id.isnot(None))
        .filter(PlateEvent.timestamp.isnot(None))
    )
    if start is not None:
        query = query.filter(PlateEvent.timestamp >= start)
    if end is not None:
        query = query.filter(PlateEvent.timestamp <= end)
    rows = query.order_by(PlateEvent.vehicle_id.asc(), PlateEvent.timestamp.asc(), PlateEvent.id.asc()).all()
    grouped = defaultdict(list)
    for event in rows:
        grouped[event.vehicle_id].append(event)
    return grouped


def movement_segments(db, start=None, end=None, source_camera_id=None, destination_camera_id=None):
    segments = []
    for vehicle_id, events in _vehicle_events(db, start=start, end=end).items():
        for previous, current in zip(events, events[1:]):
            if source_camera_id and previous.camera_id != source_camera_id:
                continue
            if destination_camera_id and current.camera_id != destination_camera_id:
                continue
            segment = calculate_travel_segment(db, previous, current)
            segment.update({
                "global_vehicle_id": vehicle_id,
                "vehicle_id": vehicle_id,
                "source_plate_text": previous.plate_text,
                "destination_plate_text": current.plate_text,
            })
            segments.append(segment)
    return segments


def flow_metrics(db, start=None, end=None, source_camera_id=None, destination_camera_id=None):
    grouped = {}
    vehicles = defaultdict(set)
    for segment in movement_segments(db, start=start, end=end,
                                     source_camera_id=source_camera_id,
                                     destination_camera_id=destination_camera_id):
        origin = segment.get("source_camera_id")
        destination = segment.get("destination_camera_id")
        if not origin or not destination:
            continue
        key = (origin, destination)
        row = grouped.setdefault(key, {
            "origin": origin,
            "destination": destination,
            "source_camera_id": origin,
            "destination_camera_id": destination,
            "movement_count": 0,
            "vehicle_count": 0,
            "unique_vehicle_count": 0,
            "valid_speed_segment_count": 0,
            "unavailable_segment_count": 0,
            "average_estimated_speed_kmh": None,
            "direction": segment.get("direction"),
            "road_connection_id": segment.get("connection_id"),
            "road_distance_meters": segment.get("distance_meters"),
            "distance_source": segment.get("distance_source"),
            "metric_note": "movement_count counts consecutive same-vehicle camera transitions.",
        })
        row["movement_count"] += 1
        vehicles[key].add(segment.get("vehicle_id"))
        if segment.get("speed_available"):
            row["valid_speed_segment_count"] += 1
            row.setdefault("_speeds", []).append(segment["estimated_speed_kmh"])
        else:
            row["unavailable_segment_count"] += 1
    for key, row in grouped.items():
        row["unique_vehicle_count"] = len({vehicle for vehicle in vehicles[key] if vehicle})
        row["vehicle_count"] = row["unique_vehicle_count"]
        speeds = row.pop("_speeds", [])
        if speeds:
            row["average_estimated_speed_kmh"] = round(sum(speeds) / len(speeds), 3)
    return sorted(grouped.values(), key=lambda item: (-item["movement_count"], item["origin"], item["destination"]))


def od_matrix(db, start=None, end=None, bucket="hour", source_camera_id=None, destination_camera_id=None):
    bucket = validate_bucket(bucket)
    grouped = {}
    vehicles = defaultdict(set)
    for segment in movement_segments(db, start=start, end=end,
                                     source_camera_id=source_camera_id,
                                     destination_camera_id=destination_camera_id):
        origin = segment.get("source_camera_id")
        destination = segment.get("destination_camera_id")
        if not origin or not destination:
            continue
        bucket_key = _bucket_key(_parse_segment_time(segment.get("start_time")), bucket)
        if not bucket_key:
            continue
        key = (bucket_key, origin, destination)
        row = grouped.setdefault(key, {
            "time_bucket": bucket_key,
            "bucket": bucket,
            "origin": origin,
            "destination": destination,
            "source_camera_id": origin,
            "destination_camera_id": destination,
            "vehicle_count": 0,
            "movement_count": 0,
            "metric_basis": "direct_consecutive_global_vehicle_transitions",
        })
        row["movement_count"] += 1
        vehicles[key].add(segment.get("vehicle_id"))
    for key, row in grouped.items():
        row["vehicle_count"] = len({vehicle for vehicle in vehicles[key] if vehicle})
    return sorted(grouped.values(), key=lambda item: (item["time_bucket"], item["origin"], item["destination"]))


def _parse_segment_time(value):
    if not value:
        return None
    if isinstance(value, datetime.datetime):
        return value
    try:
        return datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def road_metrics(db, start=None, end=None, source_camera_id=None, destination_camera_id=None):
    segment_groups = defaultdict(list)
    for segment in movement_segments(db, start=start, end=end,
                                     source_camera_id=source_camera_id,
                                     destination_camera_id=destination_camera_id):
        segment_groups[(segment.get("source_camera_id"), segment.get("destination_camera_id"))].append(segment)
    rows = []
    for connection in _filtered_connection_query(db, source_camera_id=source_camera_id,
                                                 destination_camera_id=destination_camera_id).all():
        key = (connection.source_camera_id, connection.destination_camera_id)
        segments = segment_groups.get(key, [])
        valid = [segment for segment in segments if segment.get("speed_available")]
        speeds = [segment["estimated_speed_kmh"] for segment in valid]
        vehicles = {segment.get("vehicle_id") for segment in segments if segment.get("vehicle_id")}
        summary = speed_summary(segments)
        row = {
            "connection_id": connection.id,
            "source_camera_id": connection.source_camera_id,
            "destination_camera_id": connection.destination_camera_id,
            "direction": connection.direction,
            "road_name": connection.road_name,
            "road_type": connection.road_type,
            "distance_meters": connection.distance_meters,
            "distance_source": connection.distance_source,
            "movement_count": len(segments),
            "observation_count": len(segments),
            "unique_vehicle_count": len(vehicles),
            "vehicle_count": len(vehicles),
            "valid_speed_segment_count": len(valid),
            "unavailable_segment_count": len(segments) - len(valid),
            "average_estimated_speed_kmh": summary["average_estimated_speed_kmh"],
            "minimum_estimated_speed_kmh": round(min(speeds), 3) if speeds else None,
            "maximum_estimated_speed_kmh": round(max(speeds), 3) if speeds else None,
            "speed_type": summary["speed_type"],
        }
        row["utilization"] = road_utilization(row)
        row["congestion"] = congestion_for_road(row)
        rows.append(row)
    return sorted(rows, key=lambda item: (-item["movement_count"], item["source_camera_id"], item["destination_camera_id"]))


def road_utilization(row):
    policy = traffic_policy()
    level = _level(row.get("movement_count") or 0, policy["road_low_threshold"], policy["road_medium_threshold"])
    score = min(1.0, round((row.get("movement_count") or 0) / max(policy["road_medium_threshold"], 1.0), 3))
    return {
        "status": level,
        "relative_flow_score": score,
        "capacity_configured": False,
        "metric_note": "Relative utilization/activity; no physical road capacity is configured.",
    }


def congestion_for_road(row):
    policy = traffic_policy()
    avg_speed = row.get("average_estimated_speed_kmh")
    volume = row.get("movement_count") or 0
    if avg_speed is None:
        return {
            "status": "NO_DATA",
            "score": 0.0,
            "label": "Insufficient road-speed data",
            "explanation": "No valid camera-to-camera speed segments are available for this road connection.",
            "evidence": {"movement_count": volume, "valid_speed_segment_count": row.get("valid_speed_segment_count", 0)},
        }
    congested = avg_speed <= policy["congestion_congested_speed_kmh"] and volume >= policy["congestion_congested_volume"]
    watch = avg_speed <= policy["congestion_watch_speed_kmh"] or volume >= policy["congestion_watch_volume"]
    status = "CONGESTED" if congested else ("WATCH" if watch else "NORMAL")
    score = 0.0
    if status == "WATCH":
        score = 0.5
    elif status == "CONGESTED":
        score = 1.0
    return {
        "status": status,
        "score": score,
        "explanation": (
            f"Possible congestion status {status.lower()} from {volume} movement(s) and "
            f"{avg_speed:.1f} km/h average camera-to-camera estimated speed."
        ),
        "evidence": {
            "average_estimated_speed_kmh": avg_speed,
            "movement_count": volume,
            "watch_speed_kmh": policy["congestion_watch_speed_kmh"],
            "congested_speed_kmh": policy["congestion_congested_speed_kmh"],
            "watch_volume": policy["congestion_watch_volume"],
            "congested_volume": policy["congestion_congested_volume"],
        },
    }


def congestion_metrics(db, start=None, end=None, source_camera_id=None, destination_camera_id=None):
    return [
        {**row, "congestion_status": row["congestion"]["status"]}
        for row in road_metrics(db, start=start, end=end,
                                source_camera_id=source_camera_id,
                                destination_camera_id=destination_camera_id)
    ]


def dwell_metrics(db, start=None, end=None, camera_id=None, bucket="hour"):
    bucket = validate_bucket(bucket)
    groups = defaultdict(list)
    observation_counts = defaultdict(int)
    for event in _events(db, start=start, end=end, camera_id=camera_id):
        key = _bucket_key(event.timestamp, bucket)
        if key:
            observation_counts[(key, event.camera_id)] += 1
        if event.vehicle_id and event.timestamp:
            groups[(key, event.camera_id, event.vehicle_id)].append(event.timestamp)
    dwell_groups = defaultdict(list)
    for (bucket_key, cam_id, _vehicle_id), stamps in groups.items():
        if not bucket_key or len(stamps) < 2:
            continue
        stamps = sorted(stamps)
        seconds = (stamps[-1] - stamps[0]).total_seconds()
        if seconds > 0:
            dwell_groups[(bucket_key, cam_id)].append(seconds)
    rows = []
    all_keys = set(observation_counts) | set(dwell_groups)
    for key in sorted(all_keys):
        values = dwell_groups.get(key, [])
        rows.append({
            "time_bucket": key[0],
            "bucket": bucket,
            "camera_id": key[1],
            "observation_count": observation_counts.get(key, 0),
            "dwell_sample_count": len(values),
            "average_dwell_seconds": round(sum(values) / len(values), 3) if values else None,
            "median_dwell_seconds": round(median(values), 3) if values else None,
            "maximum_dwell_seconds": round(max(values), 3) if values else None,
            "dwell_status": "estimated" if values else "unavailable",
            "metric_note": "Camera observation dwell estimate, not exact physical queue waiting time.",
        })
    return rows


def heatmap_points(db, start=None, end=None):
    cameras = {item["camera_id"]: item for item in camera_metrics(db, start=start, end=end)}
    density_by_camera = {item["camera_id"]: item for item in density_metrics(db, start=start, end=end)}
    rows = []
    for camera_id, item in cameras.items():
        if item["lat"] is None or item["lng"] is None:
            continue
        density = density_by_camera.get(camera_id, {})
        rows.append({
            "camera_id": camera_id,
            "camera_label": item["camera_label"],
            "lat": item["lat"],
            "lng": item["lng"],
            "traffic_count": item["observation_count"],
            "unique_vehicle_count": item["unique_vehicle_count"],
            "density_level": density.get("density_level", "LOW"),
            "congestion_status": _camera_congestion_status(camera_id, db, start, end),
            "weight": item["observation_count"],
        })
    return rows


def _camera_congestion_status(camera_id, db, start, end):
    statuses = [
        row["congestion"]["status"]
        for row in road_metrics(db, start=start, end=end, source_camera_id=camera_id)
    ]
    if "CONGESTED" in statuses:
        return "CONGESTED"
    if "WATCH" in statuses:
        return "WATCH"
    if statuses:
        return "NORMAL"
    return "NO_DATA"


def lane_metrics():
    return {
        "status": LANE_STATUS,
        "available": False,
        "label": "Lane analysis not configured",
        "items": [],
        "explanation": "Lane-level IDs, lane bounding boxes, and reliable per-lane tracking are not configured in the current project.",
    }


def traffic_summary(db, start=None, end=None):
    events = _events(db, start=start, end=end)
    unique_vehicles = {event.vehicle_id for event in events if event.vehicle_id}
    cameras = camera_metrics(db, start=start, end=end)
    flows = flow_metrics(db, start=start, end=end)
    roads = road_metrics(db, start=start, end=end)
    busiest_camera = cameras[0] if cameras else None
    busiest_road = roads[0] if roads else None
    valid_road_speeds = [
        row["average_estimated_speed_kmh"]
        for row in roads
        if row.get("average_estimated_speed_kmh") is not None
    ]
    return {
        "total_observations": len(events),
        "unique_vehicle_count": len(unique_vehicles),
        "camera_count": len(cameras),
        "road_connection_count": len(roads),
        "movement_count": sum(item["movement_count"] for item in flows),
        "busiest_camera": busiest_camera,
        "busiest_road": busiest_road,
        "average_road_speed_kmh": round(sum(valid_road_speeds) / len(valid_road_speeds), 3) if valid_road_speeds else None,
        "average_road_speed_available": bool(valid_road_speeds),
        "lane_analysis": None,
        "lane_analysis_available": False,
        "lane_analysis_label": None,
        "metric_note": "Traffic analytics are derived from persisted ANPR observations and Global Vehicle IDs.",
        "time_window": {"start": _safe_iso(start), "end": _safe_iso(end)},
    }


def traffic_dashboard(db, start=None, end=None, bucket="hour"):
    return {
        "summary": traffic_summary(db, start=start, end=end),
        "cameras": camera_metrics(db, start=start, end=end),
        "flow": flow_metrics(db, start=start, end=end),
        "od_matrix": od_matrix(db, start=start, end=end, bucket=bucket),
        "density": density_metrics(db, start=start, end=end),
        "congestion": congestion_metrics(db, start=start, end=end),
        "dwell": dwell_metrics(db, start=start, end=end, bucket=bucket),
        "roads": road_metrics(db, start=start, end=end),
        "heatmap": heatmap_points(db, start=start, end=end),
        "timeseries": timeseries(db, start=start, end=end, bucket=bucket),
        "lane_analysis": lane_metrics(),
        "policy": traffic_policy(),
    }
