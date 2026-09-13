"""Local operator-managed hotlist and transactional alert creation."""
import datetime as dt
from sqlalchemy import or_
from app.database import Camera, HotlistEntry, HotlistAlert, HotlistNotification
from app.location import camera_location, iso_utc
from app.plate_rules import normalize_plate_text, OCR_ACCEPT_CONFIDENCE


def utcnow():
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


def alert_payload(alert):
    return {key: getattr(alert, key) for key in (
        "id", "event_id", "hotlist_id", "plate_text", "camera_id", "camera_label",
        "lat", "lng", "category", "reason", "reference", "confidence", "match_status", "revision"
    )} | {"seen_at": iso_utc(alert.seen_at), "created_at": iso_utc(alert.created_at),
          "acknowledged_at": iso_utc(alert.acknowledged_at) if alert.acknowledged_at else None}


def entry_payload(entry):
    return {key: getattr(entry, key) for key in ("id", "plate_text", "category", "reason", "reference", "active")} | {
        "expires_at": iso_utc(entry.expires_at) if entry.expires_at else None,
        "created_at": iso_utc(entry.created_at), "updated_at": iso_utc(entry.updated_at),
        "expired": entry.expires_at is not None and entry.expires_at <= utcnow()}


def notify(db, alert):
    db.flush()
    db.add(HotlistNotification(alert_id=alert.id))


def match_event(db, event):
    """Called inside the scan/correction transaction; never fuzzy-match a hotlist."""
    rule = normalize_plate_text(event.plate_text)
    plate = rule.normalized_text
    existing = db.query(HotlistAlert).filter(HotlistAlert.event_id == event.id).with_for_update().all()
    # Corrections retract old matches without erasing the alert record.
    for alert in existing:
        if alert.plate_text != plate and alert.match_status != "retracted":
            alert.match_status = "retracted"
            alert.revision += 1
            notify(db, alert)
    if not plate:
        return
    entry = (db.query(HotlistEntry).filter(HotlistEntry.plate_text == plate,
             HotlistEntry.active.is_(True), or_(HotlistEntry.expires_at.is_(None),
                                              HotlistEntry.expires_at > utcnow())).first())
    if not entry:
        return
    status = "matched" if event.status == "ok" and (event.confidence or 0) >= OCR_ACCEPT_CONFIDENCE and rule.valid_format else "review"
    alert = next((a for a in existing if a.hotlist_id == entry.id), None)
    if alert:
        if alert.match_status == status:
            if status == "matched":
                from app.enforcement import ensure_incident_for_alert
                ensure_incident_for_alert(db, alert)
            return
        alert.match_status = status
        alert.confidence = event.confidence or 0
        alert.acknowledged_at = None
        alert.revision += 1
    else:
        camera = db.get(Camera, event.camera_id)
        lat, lng = camera_location(camera)
        alert = HotlistAlert(event_id=event.id, hotlist_id=entry.id, plate_text=plate,
            camera_id=event.camera_id, camera_label=camera.label if camera else event.camera_id,
            lat=lat, lng=lng, category=entry.category, reason=entry.reason, reference=entry.reference,
            confidence=event.confidence or 0, match_status=status, seen_at=event.timestamp)
        db.add(alert)
    if status == "matched":
        db.flush()
        from app.enforcement import ensure_incident_for_alert
        ensure_incident_for_alert(db, alert)
    notify(db, alert)
