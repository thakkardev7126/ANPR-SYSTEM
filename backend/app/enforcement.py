"""Phase 7 law-enforcement incident-command helpers.

This module is intentionally assistive: it recommends a nearby available PCR
from the latest reported demo GPS location and never records an interception.
"""
import datetime as dt
import math
import os

from sqlalchemy.exc import IntegrityError

from app.database import Camera, EnforcementIncident, HotlistAlert, PCRVehicle, PlateEvent
from app.location import camera_location, iso_utc, utc_naive, valid_coordinates


PCR_AVAILABLE = "AVAILABLE"
PCR_BUSY = "BUSY"
PCR_OFFLINE = "OFFLINE"
PCR_STATUSES = {PCR_AVAILABLE, PCR_BUSY, PCR_OFFLINE}

LOCATION_LIVE = "LIVE"
LOCATION_STALE = "STALE"
LOCATION_UNKNOWN = "UNKNOWN"

INCIDENT_NEW = "NEW"
INCIDENT_ACKNOWLEDGED = "ACKNOWLEDGED"
INCIDENT_DISPATCH_RECOMMENDED = "DISPATCH_RECOMMENDED"
INCIDENT_DISMISSED = "DISMISSED"
INCIDENT_STATUSES = {
    INCIDENT_NEW,
    INCIDENT_ACKNOWLEDGED,
    INCIDENT_DISPATCH_RECOMMENDED,
    INCIDENT_DISMISSED,
}

RECOMMEND_DISPATCH = "DISPATCH_RECOMMENDED"
RECOMMEND_NO_AVAILABLE = "NO_AVAILABLE_PCR"
RECOMMEND_NO_CAMERA_LOCATION = "NO_CAMERA_COORDINATES"

DEFAULT_PCR_LOCATION_MAX_AGE_SECONDS = 300

DEMO_PCR_VEHICLES = [
    {
        "pcr_id": "PCR01",
        "name": "PCR01",
        "call_sign": "PCR01",
        "vehicle_identifier": "DEMO-PCR-01",
        "lat": 23.0413,
        "lng": 72.5310,
        "status": PCR_AVAILABLE,
        "active": True,
        "contact_channel": "Demo PCR GPS",
    },
    {
        "pcr_id": "PCR02",
        "name": "PCR02",
        "call_sign": "PCR02",
        "vehicle_identifier": "DEMO-PCR-02",
        "lat": 23.0334,
        "lng": 72.5242,
        "status": PCR_AVAILABLE,
        "active": True,
        "contact_channel": "Demo PCR GPS",
    },
    {
        "pcr_id": "PCR03",
        "name": "PCR03",
        "call_sign": "PCR03",
        "vehicle_identifier": "DEMO-PCR-03",
        "lat": 23.0301,
        "lng": 72.5162,
        "status": PCR_BUSY,
        "active": True,
        "contact_channel": "Demo PCR GPS",
    },
]


def utcnow():
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


def pcr_location_max_age_seconds():
    try:
        return max(1, int(os.getenv("ANPR_PCR_LOCATION_MAX_AGE_SECONDS", str(DEFAULT_PCR_LOCATION_MAX_AGE_SECONDS))))
    except (TypeError, ValueError):
        return DEFAULT_PCR_LOCATION_MAX_AGE_SECONDS


def normalize_pcr_id(value):
    pcr_id = str(value or "").strip().upper()
    if not pcr_id or len(pcr_id) > 32 or not all(ch.isalnum() or ch in "-_" for ch in pcr_id):
        raise ValueError("pcr_id must contain 1-32 letters, digits, underscores or hyphens")
    return pcr_id


def normalize_pcr_status(value):
    status = str(value or PCR_AVAILABLE).strip().upper()
    if status not in PCR_STATUSES:
        raise ValueError("status must be AVAILABLE, BUSY or OFFLINE")
    return status


def normalize_incident_status(value):
    status = str(value or "").strip().upper()
    if status not in INCIDENT_STATUSES:
        raise ValueError("Only NEW, DISPATCH_RECOMMENDED, ACKNOWLEDGED and DISMISSED are supported in this prototype")
    return status


def parse_timestamp(value, *, required=False):
    if value in (None, ""):
        if required:
            raise ValueError("timestamp is required")
        return None
    if isinstance(value, dt.datetime):
        return utc_naive(value)
    try:
        return utc_naive(dt.datetime.fromisoformat(str(value).replace("Z", "+00:00")))
    except ValueError as exc:
        raise ValueError("timestamp must be a valid ISO-8601 datetime") from exc


def validate_lat_lng(lat, lng):
    if not valid_coordinates(lat, lng):
        raise ValueError("latitude and longitude must be finite values within -90..90 and -180..180")
    return float(lat), float(lng)


def haversine_meters(lat1, lng1, lat2, lng2):
    radius_m = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lng2 - lng1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * radius_m * math.asin(math.sqrt(a))


def location_status(pcr, now=None, max_age_seconds=None):
    if pcr.lat is None or pcr.lng is None or not valid_coordinates(pcr.lat, pcr.lng) or not pcr.last_seen_at:
        return LOCATION_UNKNOWN
    now = now or utcnow()
    max_age_seconds = max_age_seconds or pcr_location_max_age_seconds()
    age = (now - utc_naive(pcr.last_seen_at)).total_seconds()
    return LOCATION_LIVE if age <= max_age_seconds else LOCATION_STALE


def pcr_payload(pcr, now=None):
    status = location_status(pcr, now=now)
    age_seconds = None
    if pcr.last_seen_at:
        age_seconds = max(0, round(((now or utcnow()) - utc_naive(pcr.last_seen_at)).total_seconds(), 3))
    return {
        "pcr_id": pcr.pcr_id,
        "name": pcr.name,
        "call_sign": pcr.call_sign,
        "vehicle_identifier": pcr.vehicle_identifier,
        "latitude": pcr.lat,
        "longitude": pcr.lng,
        "status": pcr.status,
        "active": bool(pcr.active),
        "last_seen_at": iso_utc(pcr.last_seen_at) if pcr.last_seen_at else None,
        "location_status": status,
        "location_age_seconds": age_seconds,
        "contact_channel": pcr.contact_channel,
        "demo_only": True,
        "created_at": iso_utc(pcr.created_at),
        "updated_at": iso_utc(pcr.updated_at),
    }


def seed_demo_pcr_vehicles(db):
    now = utcnow()
    changed = False
    for values in DEMO_PCR_VEHICLES:
        if db.get(PCRVehicle, values["pcr_id"]):
            continue
        db.add(PCRVehicle(**values, last_seen_at=now, created_at=now, updated_at=now))
        changed = True
    if changed:
        db.commit()


def create_pcr_vehicle(db, values):
    now = utcnow()
    lat = values.get("latitude", values.get("lat"))
    lng = values.get("longitude", values.get("lng"))
    if (lat is None) != (lng is None):
        raise ValueError("latitude and longitude must be supplied together")
    if lat is not None:
        lat, lng = validate_lat_lng(lat, lng)
    last_seen_at = parse_timestamp(values.get("last_seen_at") or values.get("timestamp"))
    if lat is not None and last_seen_at is None:
        last_seen_at = now
    pcr = PCRVehicle(
        pcr_id=normalize_pcr_id(values.get("pcr_id")),
        name=str(values.get("name") or values.get("call_sign") or values.get("pcr_id")).strip()[:80],
        call_sign=str(values.get("call_sign") or values.get("name") or values.get("pcr_id")).strip()[:80],
        vehicle_identifier=str(values.get("vehicle_identifier") or "").strip()[:80],
        lat=lat,
        lng=lng,
        status=normalize_pcr_status(values.get("status")),
        active=bool(values.get("active", True)),
        last_seen_at=last_seen_at,
        contact_channel=str(values.get("contact_channel") or values.get("channel") or "Demo PCR GPS").strip()[:120],
        created_at=now,
        updated_at=now,
    )
    db.add(pcr)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise ValueError("pcr_id already exists") from exc
    return pcr


def update_pcr_location(db, pcr_id, values):
    pcr = db.get(PCRVehicle, normalize_pcr_id(pcr_id))
    if not pcr:
        return None
    lat, lng = validate_lat_lng(values.get("latitude", values.get("lat")), values.get("longitude", values.get("lng")))
    timestamp = parse_timestamp(values.get("timestamp"), required=True)
    pcr.lat = lat
    pcr.lng = lng
    pcr.last_seen_at = timestamp
    pcr.updated_at = utcnow()
    db.commit()
    return pcr


def update_pcr_vehicle(db, pcr_id, values):
    pcr = db.get(PCRVehicle, normalize_pcr_id(pcr_id))
    if not pcr:
        return None
    if "status" in values:
        pcr.status = normalize_pcr_status(values["status"])
    if "active" in values:
        pcr.active = bool(values["active"])
    for source, target, limit in (
        ("name", "name", 80),
        ("call_sign", "call_sign", 80),
        ("vehicle_identifier", "vehicle_identifier", 80),
        ("contact_channel", "contact_channel", 120),
    ):
        if source in values:
            setattr(pcr, target, str(values[source] or "").strip()[:limit])
    if "latitude" in values or "longitude" in values or "lat" in values or "lng" in values:
        lat, lng = validate_lat_lng(values.get("latitude", values.get("lat")), values.get("longitude", values.get("lng")))
        pcr.lat = lat
        pcr.lng = lng
        pcr.last_seen_at = parse_timestamp(values.get("timestamp")) or utcnow()
    pcr.updated_at = utcnow()
    db.commit()
    return pcr


def nearest_available_pcr(db, camera, now=None):
    cam_lat, cam_lng = camera_location(camera)
    calculated_at = now or utcnow()
    if cam_lat is None or cam_lng is None:
        return {
            "recommended_pcr": None,
            "reason": RECOMMEND_NO_CAMERA_LOCATION,
            "calculated_at": iso_utc(calculated_at),
            "candidates": [],
        }

    candidates = []
    best = None
    for pcr in db.query(PCRVehicle).filter(PCRVehicle.active.is_(True)).all():
        state = location_status(pcr, now=calculated_at)
        distance_m = None
        eligible = pcr.status == PCR_AVAILABLE and state == LOCATION_LIVE
        reason = None
        if pcr.status != PCR_AVAILABLE:
            reason = f"PCR status is {pcr.status}"
        elif state != LOCATION_LIVE:
            reason = f"PCR location is {state}"
        if state != LOCATION_UNKNOWN:
            distance_m = haversine_meters(cam_lat, cam_lng, pcr.lat, pcr.lng)
        item = {
            "pcr_id": pcr.pcr_id,
            "latitude": pcr.lat,
            "longitude": pcr.lng,
            "status": pcr.status,
            "active": bool(pcr.active),
            "location_status": state,
            "last_seen_at": iso_utc(pcr.last_seen_at) if pcr.last_seen_at else None,
            "distance_meters": round(distance_m, 3) if distance_m is not None else None,
            "eligible": eligible,
            "reason": reason,
        }
        candidates.append(item)
        if eligible and (best is None or item["distance_meters"] < best["distance_meters"]):
            best = item

    candidates.sort(key=lambda item: (not item["eligible"], item["distance_meters"] is None, item["distance_meters"] or 0))
    return {
        "recommended_pcr": best,
        "reason": RECOMMEND_DISPATCH if best else RECOMMEND_NO_AVAILABLE,
        "calculated_at": iso_utc(calculated_at),
        "candidates": candidates,
    }


def _reference_paths(event, vehicle_id, plate_text):
    if vehicle_id:
        return f"/api/vehicles/{vehicle_id}/trajectory", f"/api/vehicles/{vehicle_id}/investigation"
    return f"/api/trajectory/{plate_text}", f"/api/plates/{plate_text}/investigation"


def ensure_incident_for_alert(db, alert):
    if not alert or alert.match_status != "matched":
        return None
    existing = db.query(EnforcementIncident).filter(EnforcementIncident.hotlist_alert_id == alert.id).first()
    if existing:
        return existing

    db.flush()
    event = db.get(PlateEvent, alert.event_id) if alert.event_id else None
    vehicle_id = event.vehicle_id if event else None
    camera = db.get(Camera, alert.camera_id)
    recommendation = nearest_available_pcr(db, camera)
    selected = recommendation["recommended_pcr"]
    trajectory_reference, investigation_reference = _reference_paths(event, vehicle_id, alert.plate_text)
    incident_status = INCIDENT_DISPATCH_RECOMMENDED if selected else INCIDENT_NEW
    incident = EnforcementIncident(
        hotlist_alert_id=alert.id,
        vehicle_id=vehicle_id,
        plate_text=alert.plate_text,
        camera_id=alert.camera_id,
        event_id=alert.event_id,
        detected_at=alert.seen_at,
        recommended_pcr_id=selected["pcr_id"] if selected else None,
        pcr_distance_meters=selected["distance_meters"] if selected else None,
        pcr_location_status=selected["location_status"] if selected else LOCATION_UNKNOWN,
        recommendation_status=recommendation["reason"],
        recommendation_reason=(
            f"Distance-based recommendation from latest reported demo PCR GPS: {selected['pcr_id']}"
            if selected else recommendation["reason"]
        ),
        recommendation_calculated_at=utcnow(),
        status=incident_status,
        trajectory_reference=trajectory_reference,
        investigation_reference=investigation_reference,
        created_at=utcnow(),
        updated_at=utcnow(),
    )
    db.add(incident)
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        return db.query(EnforcementIncident).filter(EnforcementIncident.hotlist_alert_id == alert.id).first()
    return incident


def incident_payload(db, incident, alert=None, pcr=None, camera=None):
    alert = alert if alert is not None else db.get(HotlistAlert, incident.hotlist_alert_id)
    pcr = pcr if pcr is not None else (db.get(PCRVehicle, incident.recommended_pcr_id) if incident.recommended_pcr_id else None)
    camera = camera if camera is not None else db.get(Camera, incident.camera_id)
    cam_lat, cam_lng = camera_location(camera)
    return {
        "incident_id": incident.id,
        "hotlist_alert_id": incident.hotlist_alert_id,
        "alert_id": incident.hotlist_alert_id,
        "plate_text": incident.plate_text,
        "category": alert.category if alert else None,
        "reason": alert.reason if alert else None,
        "reference": alert.reference if alert else None,
        "match_status": alert.match_status if alert else None,
        "confidence": alert.confidence if alert else None,
        "global_vehicle_id": incident.vehicle_id,
        "vehicle_id": incident.vehicle_id,
        "camera_id": incident.camera_id,
        "camera_label": alert.camera_label if alert else incident.camera_id,
        "camera": {
            "camera_id": incident.camera_id,
            "label": alert.camera_label if alert else (camera.label if camera else incident.camera_id),
            "latitude": cam_lat,
            "longitude": cam_lng,
            "location_known": cam_lat is not None and cam_lng is not None,
        },
        "event_id": incident.event_id,
        "detected_at": iso_utc(incident.detected_at),
        "recommended_pcr_id": incident.recommended_pcr_id,
        "recommended_pcr": pcr_payload(pcr) if pcr else None,
        "pcr_distance_meters": incident.pcr_distance_meters,
        "distance_meters": incident.pcr_distance_meters,
        "distance_km": round(incident.pcr_distance_meters / 1000, 3) if incident.pcr_distance_meters is not None else None,
        "pcr_location_status": incident.pcr_location_status,
        "recommendation_status": incident.recommendation_status,
        "recommendation_reason": incident.recommendation_reason,
        "recommendation_calculated_at": iso_utc(incident.recommendation_calculated_at) if incident.recommendation_calculated_at else None,
        "recommendation": f"Recommended PCR: {incident.recommended_pcr_id}" if incident.recommended_pcr_id else "No available PCR recommendation",
        "recommendation_basis": "Distance-based recommendation from latest available demo PCR GPS.",
        "status": incident.status,
        "trajectory_reference": incident.trajectory_reference,
        "vehicle_trajectory_reference": incident.trajectory_reference,
        "investigation_reference": incident.investigation_reference,
        "created_at": iso_utc(incident.created_at),
        "updated_at": iso_utc(incident.updated_at),
        "updated_by": incident.updated_by,
        "demo_only": True,
        "limitations": [
            "PCR GPS is seeded or manually updated for the SIH demo.",
            "Nearest PCR is geographic distance, not road travel time.",
            "No police dispatch/control-room integration is confirmed.",
        ],
    }


def incident_for_alert_payload(db, alert_id):
    incident = db.query(EnforcementIncident).filter(EnforcementIncident.hotlist_alert_id == alert_id).first()
    return incident_payload(db, incident) if incident else None


def update_incident_status(db, incident_id, status, updated_by=None):
    incident = db.get(EnforcementIncident, incident_id)
    if not incident:
        return None
    incident.status = normalize_incident_status(status)
    incident.updated_at = utcnow()
    incident.updated_by = str(updated_by or "").strip()[:80] or None
    db.commit()
    return incident


def acknowledge_incident_for_alert(db, alert_id, updated_by=None):
    incident = db.query(EnforcementIncident).filter(EnforcementIncident.hotlist_alert_id == alert_id).first()
    if not incident:
        return None
    incident.status = INCIDENT_ACKNOWLEDGED
    incident.updated_at = utcnow()
    incident.updated_by = str(updated_by or "hotlist_acknowledge").strip()[:80]
    return incident
