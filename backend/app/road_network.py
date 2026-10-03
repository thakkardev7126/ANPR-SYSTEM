"""Directed camera-to-camera road-network support.

The SIH demo can run fully offline with manually configured connections. The
provider interface leaves room for OpenStreetMap/OSRM integration later without
hardcoding external routing into trajectory logic.
"""
from __future__ import annotations

import datetime
import json
import logging
import math
import os
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.database import Camera, CameraRoadConnection
from app.location import camera_location, iso_utc


VALID_DIRECTIONS = {"N", "NE", "E", "SE", "S", "SW", "W", "NW"}
VALID_DISTANCE_SOURCES = {"manual", "osm", "osrm", "fallback_straight_line"}
LOGGER = logging.getLogger(__name__)


@dataclass
class RouteEstimate:
    distance_meters: float
    distance_source: str
    provider: str
    provider_reference: str | None = None
    duration_seconds: float | None = None
    route_geometry: list[dict] | None = None


class ManualRoadNetworkProvider:
    name = "manual"

    def estimate(self, *_args, **_kwargs):
        return None


class StraightLineReferenceProvider:
    name = "fallback_straight_line"

    def estimate(self, source_camera, destination_camera):
        source = camera_location(source_camera)
        destination = camera_location(destination_camera)
        if source[0] is None or destination[0] is None:
            return None
        distance = _haversine_km(source[0], source[1], destination[0], destination[1]) * 1000
        return RouteEstimate(round(distance, 1), "fallback_straight_line", self.name)


def _float_env(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def osrm_routing_config():
    base_url = os.environ.get("ANPR_OSRM_BASE_URL", "").strip().rstrip("/")
    return {
        "enabled": bool(base_url),
        "base_url": base_url,
        "profile": os.environ.get("ANPR_OSRM_PROFILE", "driving").strip() or "driving",
        "timeout_seconds": _float_env("ANPR_OSRM_TIMEOUT_SECONDS", 2.0),
    }


def build_osrm_route_url(source_camera, destination_camera, base_url=None, profile=None):
    config = osrm_routing_config()
    base = (base_url or config["base_url"]).strip().rstrip("/")
    route_profile = (profile or config["profile"] or "driving").strip()
    if not base:
        raise ValueError("OSRM base URL is not configured")
    source_lat, source_lng = camera_location(source_camera)
    destination_lat, destination_lng = camera_location(destination_camera)
    if source_lat is None or source_lng is None or destination_lat is None or destination_lng is None:
        raise ValueError("Both cameras need valid coordinates for OSRM routing")
    return (
        f"{base}/route/v1/{route_profile}/"
        f"{source_lng},{source_lat};{destination_lng},{destination_lat}"
        "?overview=full&geometries=geojson&alternatives=false&steps=false"
    )


def parse_osrm_route(data):
    if not isinstance(data, dict) or data.get("code") != "Ok":
        return None
    routes = data.get("routes") or []
    if not routes or not isinstance(routes[0], dict):
        return None
    route = routes[0]
    try:
        distance = float(route.get("distance"))
    except (TypeError, ValueError):
        return None
    if distance <= 0:
        return None
    duration = None
    try:
        duration = float(route.get("duration")) if route.get("duration") is not None else None
    except (TypeError, ValueError):
        duration = None
    geometry = []
    coordinates = ((route.get("geometry") or {}).get("coordinates") or [])
    for coordinate in coordinates:
        if not isinstance(coordinate, (list, tuple)) or len(coordinate) < 2:
            continue
        try:
            lng = float(coordinate[0])
            lat = float(coordinate[1])
        except (TypeError, ValueError):
            continue
        geometry.append({"lat": lat, "lng": lng})
    return RouteEstimate(
        distance_meters=round(distance, 3),
        distance_source="osrm",
        provider="osrm",
        provider_reference="OSRM route/v1",
        duration_seconds=round(duration, 3) if duration is not None else None,
        route_geometry=geometry or None,
    )


class OSRMRoutingProvider:
    name = "osrm"

    def __init__(self, base_url=None, profile=None, timeout_seconds=None):
        config = osrm_routing_config()
        self.base_url = (base_url if base_url is not None else config["base_url"]).strip().rstrip("/")
        self.profile = profile or config["profile"]
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else config["timeout_seconds"]

    @property
    def enabled(self):
        return bool(self.base_url)

    def estimate(self, source_camera, destination_camera):
        if not self.enabled:
            return None
        try:
            url = build_osrm_route_url(
                source_camera, destination_camera, base_url=self.base_url, profile=self.profile
            )
            payload = self._fetch_json(url)
            return parse_osrm_route(payload)
        except (ValueError, TimeoutError, HTTPError, URLError, OSError, json.JSONDecodeError) as exc:
            LOGGER.info("OSRM routing unavailable for %s -> %s: %s",
                        getattr(source_camera, "camera_id", None),
                        getattr(destination_camera, "camera_id", None), exc)
            return None

    def _fetch_json(self, url):
        request = Request(url, headers={
            "Accept": "application/json",
            "User-Agent": "ANPR-SIH-Demo/1.0",
        })
        with urlopen(request, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))


def _haversine_km(lat1, lng1, lat2, lng2):
    radius_km = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lng2 - lng1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * radius_km * math.asin(math.sqrt(a))


def normalize_direction(value):
    if value is None or value == "":
        return None
    direction = str(value).strip().upper().replace("-", "").replace("_", "")
    aliases = {
        "NORTH": "N", "NORTHEAST": "NE", "EAST": "E", "SOUTHEAST": "SE",
        "SOUTH": "S", "SOUTHWEST": "SW", "WEST": "W", "NORTHWEST": "NW",
    }
    direction = aliases.get(direction, direction)
    if direction not in VALID_DIRECTIONS:
        raise ValueError(f"direction must be one of {', '.join(sorted(VALID_DIRECTIONS))}")
    return direction


def _require_camera(db, camera_id):
    camera = db.get(Camera, camera_id)
    if not camera:
        raise ValueError(f"Unknown camera_id '{camera_id}'")
    return camera


def straight_line_reference(source_camera, destination_camera):
    estimate = StraightLineReferenceProvider().estimate(source_camera, destination_camera)
    if not estimate:
        return None
    return {
        "distance_meters": estimate.distance_meters,
        "distance_source": estimate.distance_source,
        "provider": estimate.provider,
    }


def osrm_route_estimate(source_camera, destination_camera):
    return OSRMRoutingProvider().estimate(source_camera, destination_camera)


def effective_route_estimate(db, connection):
    """Return OSRM distance when available, otherwise the configured connection.

    OSRM is only evaluated for an already configured directed connection. This
    preserves the manual road graph as the source of camera-to-camera topology.
    """
    if not connection:
        return None
    source = db.get(Camera, connection.source_camera_id)
    destination = db.get(Camera, connection.destination_camera_id)
    if source and destination:
        estimate = osrm_route_estimate(source, destination)
        if estimate:
            return estimate
    return RouteEstimate(
        distance_meters=connection.distance_meters,
        distance_source=connection.distance_source,
        provider=connection.provider or connection.distance_source,
        provider_reference=connection.provider_reference,
    )


def road_connection_payload(connection, source_camera=None, destination_camera=None):
    return {
        "id": connection.id,
        "source_camera_id": connection.source_camera_id,
        "destination_camera_id": connection.destination_camera_id,
        "distance_meters": connection.distance_meters,
        "distance_km": round((connection.distance_meters or 0) / 1000.0, 3),
        "distance_source": connection.distance_source,
        "direction": connection.direction,
        "road_name": connection.road_name,
        "road_type": connection.road_type,
        "provider": connection.provider,
        "provider_reference": connection.provider_reference,
        "active": bool(connection.active),
        "created_at": iso_utc(connection.created_at),
        "updated_at": iso_utc(connection.updated_at),
        "straight_line_reference": straight_line_reference(source_camera, destination_camera)
        if source_camera and destination_camera else None,
    }


def list_road_network(db, include_inactive=False):
    cameras = {camera.camera_id: camera for camera in db.query(Camera).all()}
    query = db.query(CameraRoadConnection).order_by(CameraRoadConnection.source_camera_id.asc(),
                                                    CameraRoadConnection.destination_camera_id.asc())
    if not include_inactive:
        query = query.filter(CameraRoadConnection.active.is_(True))
    connections = [
        road_connection_payload(row, cameras.get(row.source_camera_id), cameras.get(row.destination_camera_id))
        for row in query.all()
    ]
    return {"nodes": [
        {
            "camera_id": camera.camera_id,
            "label": camera.label,
            "lat": camera_location(camera)[0],
            "lng": camera_location(camera)[1],
            "location_known": bool(camera.location_known),
            "active": bool(camera.active),
        }
        for camera in cameras.values()
    ], "connections": connections}


def get_connection(db, source_camera_id, destination_camera_id, include_inactive=False):
    query = db.query(CameraRoadConnection).filter(
        CameraRoadConnection.source_camera_id == source_camera_id,
        CameraRoadConnection.destination_camera_id == destination_camera_id,
    )
    if not include_inactive:
        query = query.filter(CameraRoadConnection.active.is_(True))
    return query.first()


def outgoing_connections(db, camera_id, include_inactive=False):
    _require_camera(db, camera_id)
    query = db.query(CameraRoadConnection).filter(CameraRoadConnection.source_camera_id == camera_id)
    if not include_inactive:
        query = query.filter(CameraRoadConnection.active.is_(True))
    cameras = {camera.camera_id: camera for camera in db.query(Camera).all()}
    return [
        road_connection_payload(row, cameras.get(row.source_camera_id), cameras.get(row.destination_camera_id))
        for row in query.order_by(CameraRoadConnection.destination_camera_id.asc()).all()
    ]


def _connection_values(db, data, existing=None):
    source_id = str(data.get("source_camera_id", existing.source_camera_id if existing else "")).strip()
    destination_id = str(data.get("destination_camera_id", existing.destination_camera_id if existing else "")).strip()
    if not source_id or not destination_id:
        raise ValueError("source_camera_id and destination_camera_id are required")
    if source_id == destination_id:
        raise ValueError("source_camera_id and destination_camera_id must be different")
    source = _require_camera(db, source_id)
    destination = _require_camera(db, destination_id)
    try:
        distance = float(data.get("distance_meters", existing.distance_meters if existing else 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("distance_meters must be a positive number") from exc
    if distance <= 0:
        raise ValueError("distance_meters must be greater than 0")
    distance_source = str(data.get("distance_source", existing.distance_source if existing else "manual")).strip().lower()
    if distance_source not in VALID_DISTANCE_SOURCES:
        raise ValueError(f"distance_source must be one of {', '.join(sorted(VALID_DISTANCE_SOURCES))}")
    active = bool(data.get("active", existing.active if existing else True))
    direction = normalize_direction(data.get("direction", existing.direction if existing else None))
    provider = str(data.get("provider", existing.provider if existing else distance_source)).strip() or "manual"
    duplicate = get_connection(db, source_id, destination_id, include_inactive=True)
    if duplicate and (existing is None or duplicate.id != existing.id):
        if active or duplicate.active:
            raise ValueError("A connection between these cameras already exists")
    return source, destination, {
        "source_camera_id": source_id,
        "destination_camera_id": destination_id,
        "distance_meters": round(distance, 3),
        "distance_source": distance_source,
        "direction": direction,
        "road_name": data.get("road_name", existing.road_name if existing else None),
        "road_type": data.get("road_type", existing.road_type if existing else None),
        "provider": provider,
        "provider_reference": data.get("provider_reference", existing.provider_reference if existing else None),
        "active": active,
    }


def create_connection(db, data):
    source, destination, values = _connection_values(db, data)
    now = datetime.datetime.utcnow()
    connection = CameraRoadConnection(**values, created_at=now, updated_at=now)
    db.add(connection)
    db.flush()
    return road_connection_payload(connection, source, destination)


def update_connection(db, connection_id, data):
    connection = db.get(CameraRoadConnection, connection_id)
    if not connection:
        return None
    source, destination, values = _connection_values(db, data, connection)
    for field, value in values.items():
        setattr(connection, field, value)
    connection.updated_at = datetime.datetime.utcnow()
    db.flush()
    return road_connection_payload(connection, source, destination)


def disable_connection(db, connection_id):
    connection = db.get(CameraRoadConnection, connection_id)
    if not connection:
        return None
    connection.active = False
    connection.updated_at = datetime.datetime.utcnow()
    db.flush()
    source = db.get(Camera, connection.source_camera_id)
    destination = db.get(Camera, connection.destination_camera_id)
    return road_connection_payload(connection, source, destination)


def camera_transition(db, source_camera_id, destination_camera_id):
    connection = get_connection(db, source_camera_id, destination_camera_id)
    if not connection:
        return {
            "configured": False,
            "can_transition": False,
            "distance_meters": None,
            "distance_source": None,
            "direction": None,
            "connection_id": None,
        }
    return {
        "configured": True,
        "can_transition": True,
        "distance_meters": connection.distance_meters,
        "distance_source": connection.distance_source,
        "direction": connection.direction,
        "connection_id": connection.id,
        "road_name": connection.road_name,
        "road_type": connection.road_type,
        "provider": connection.provider,
    }
