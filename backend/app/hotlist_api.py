"""Hotlist management and durable alert inbox APIs."""
import datetime as dt
from typing import Literal
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from app.database import get_db, HotlistEntry, HotlistAlert
from app.hotlist import entry_payload, alert_payload, notify, utcnow
from app.location import utc_naive
from app.plate_rules import normalize_plate_text

router = APIRouter(prefix="/api", tags=["Hotlist"])


class EntryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    plate_text: str = Field(min_length=4, max_length=24)
    category: Literal["stolen", "wanted", "watchlist"] = "stolen"
    reason: str = Field(min_length=1, max_length=500)
    reference: str = Field(default="", max_length=120)
    active: bool = True
    expires_at: dt.datetime | None = None

    @field_validator("plate_text")
    @classmethod
    def plate(cls, value):
        rule = normalize_plate_text(value)
        if not rule.normalized_text:
            raise ValueError("Enter a complete supported plate number")
        return rule.normalized_text

    @field_validator("reason", "reference")
    @classmethod
    def trim(cls, value, info):
        value = value.strip()
        if info.field_name == "reason" and not value:
            raise ValueError("Reason is required")
        return value

    @field_validator("expires_at")
    @classmethod
    def expiry(cls, value):
        return utc_naive(value)


@router.get("/hotlist")
def list_entries(q: str = Query("", max_length=120), offset: int = Query(0, ge=0),
                 limit: int = Query(50, ge=1, le=200), db: Session = Depends(get_db)):
    query = db.query(HotlistEntry)
    if q.strip():
        query = query.filter(HotlistEntry.plate_text.contains(q.strip().upper(), autoescape=True))
    return {"total": query.count(), "items": [entry_payload(e) for e in
            query.order_by(HotlistEntry.id.desc()).offset(offset).limit(limit)]}


@router.post("/hotlist", status_code=201)
def add_entry(body: EntryInput, db: Session = Depends(get_db)):
    entry = HotlistEntry(**body.model_dump())
    db.add(entry)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "This plate is already listed; edit or reactivate its entry")
    return entry_payload(entry)


@router.put("/hotlist/{entry_id}")
def edit_entry(entry_id: int, body: EntryInput, db: Session = Depends(get_db)):
    entry = db.get(HotlistEntry, entry_id)
    if not entry:
        raise HTTPException(404, "Hotlist entry not found")
    if entry.plate_text != body.plate_text:
        raise HTTPException(422, "Plate numbers are immutable; deactivate this entry and add a new plate")
    for key, value in body.model_dump().items():
        setattr(entry, key, value)
    entry.updated_at = utcnow()
    db.commit()
    return entry_payload(entry)


@router.delete("/hotlist/{entry_id}")
def deactivate_entry(entry_id: int, db: Session = Depends(get_db)):
    entry = db.get(HotlistEntry, entry_id)
    if not entry:
        raise HTTPException(404, "Hotlist entry not found")
    entry.active = False
    entry.updated_at = utcnow()
    db.commit()
    return entry_payload(entry)


@router.get("/hotlist-alerts")
def list_alerts(unacknowledged: bool = False, camera_id: str | None = None,
                offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200),
                db: Session = Depends(get_db)):
    query = db.query(HotlistAlert)
    if unacknowledged:
        query = query.filter(HotlistAlert.acknowledged_at.is_(None), HotlistAlert.match_status != "retracted")
    if camera_id:
        query = query.filter(HotlistAlert.camera_id == camera_id)
    return {"total": query.count(), "items": [alert_payload(a) for a in
            query.order_by(HotlistAlert.id.desc()).offset(offset).limit(limit)]}


@router.post("/hotlist-alerts/{alert_id}/acknowledge")
def acknowledge(alert_id: int, db: Session = Depends(get_db)):
    if db.bind.dialect.name == "sqlite":
        db.connection().exec_driver_sql("BEGIN IMMEDIATE")
    alert = db.query(HotlistAlert).filter(HotlistAlert.id == alert_id).with_for_update().first()
    if not alert:
        raise HTTPException(404, "Alert not found")
    if not alert.acknowledged_at:
        alert.acknowledged_at = utcnow()
        alert.revision += 1
        notify(db, alert)
        db.commit()
    return alert_payload(alert)
