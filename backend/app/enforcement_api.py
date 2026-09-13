"""Law enforcement / PCR incident-command APIs for Phase 7."""
import datetime as dt
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.orm import Session

from app.database import EnforcementIncident, HotlistAlert, PCRVehicle, get_db
from app.enforcement import (
    INCIDENT_STATUSES,
    create_pcr_vehicle,
    incident_payload,
    nearest_available_pcr,
    normalize_pcr_id,
    pcr_location_max_age_seconds,
    pcr_payload,
    seed_demo_pcr_vehicles,
    update_incident_status,
    update_pcr_location,
    update_pcr_vehicle,
)
from app.database import Camera


router = APIRouter(prefix="/api", tags=["Law enforcement"])


class PCRInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pcr_id: str = Field(min_length=1, max_length=32)
    name: str | None = Field(default=None, max_length=80)
    call_sign: str | None = Field(default=None, max_length=80)
    vehicle_identifier: str | None = Field(default="", max_length=80)
    latitude: float | None = None
    longitude: float | None = None
    status: Literal["AVAILABLE", "BUSY", "OFFLINE"] = "AVAILABLE"
    active: bool = True
    last_seen_at: dt.datetime | None = None
    contact_channel: str | None = Field(default="Demo PCR GPS", max_length=120)

    @field_validator("pcr_id")
    @classmethod
    def pcr_identifier(cls, value):
        return normalize_pcr_id(value)

    @field_validator("status")
    @classmethod
    def pcr_status(cls, value):
        return value.upper()


class PCRUpdateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, max_length=80)
    call_sign: str | None = Field(default=None, max_length=80)
    vehicle_identifier: str | None = Field(default=None, max_length=80)
    latitude: float | None = None
    longitude: float | None = None
    status: Literal["AVAILABLE", "BUSY", "OFFLINE"] | None = None
    active: bool | None = None
    timestamp: dt.datetime | None = None
    contact_channel: str | None = Field(default=None, max_length=120)

    @field_validator("status")
    @classmethod
    def pcr_status(cls, value):
        return value.upper() if value else value


class PCRLocationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    latitude: float
    longitude: float
    timestamp: dt.datetime


class IncidentStatusInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    updated_by: str | None = Field(default=None, max_length=80)

    @field_validator("status")
    @classmethod
    def incident_status(cls, value):
        status = value.upper()
        if status not in INCIDENT_STATUSES:
            raise ValueError("Only NEW, DISPATCH_RECOMMENDED, ACKNOWLEDGED and DISMISSED are supported")
        return status


@router.post("/pcr", status_code=201)
def create_pcr(body: PCRInput, db: Session = Depends(get_db)):
    try:
        pcr = create_pcr_vehicle(db, body.model_dump())
    except ValueError as exc:
        status = 409 if "already exists" in str(exc) else 422
        raise HTTPException(status, str(exc)) from exc
    return pcr_payload(pcr)


@router.get("/pcr")
def list_pcr(db: Session = Depends(get_db), include_inactive: bool = Query(False)):
    query = db.query(PCRVehicle)
    if not include_inactive:
        query = query.filter(PCRVehicle.active.is_(True))
    now = dt.datetime.utcnow()
    return {
        "location_max_age_seconds": pcr_location_max_age_seconds(),
        "demo_only": True,
        "items": [pcr_payload(pcr, now=now) for pcr in query.order_by(PCRVehicle.pcr_id.asc()).all()],
    }


@router.post("/pcr/seed-demo")
def seed_demo_pcr(db: Session = Depends(get_db)):
    seed_demo_pcr_vehicles(db)
    now = dt.datetime.utcnow()
    rows = db.query(PCRVehicle).filter(PCRVehicle.active.is_(True)).order_by(PCRVehicle.pcr_id.asc()).all()
    return {
        "location_max_age_seconds": pcr_location_max_age_seconds(),
        "demo_only": True,
        "items": [pcr_payload(pcr, now=now) for pcr in rows],
    }


@router.get("/pcr/nearest")
def get_nearest_pcr(camera_id: str = Query(..., min_length=1), db: Session = Depends(get_db)):
    camera = db.get(Camera, camera_id)
    if not camera:
        raise HTTPException(404, "Camera not found")
    return nearest_available_pcr(db, camera)


@router.get("/pcr/{pcr_id}")
def get_pcr(pcr_id: str, db: Session = Depends(get_db)):
    pcr = db.get(PCRVehicle, normalize_pcr_id(pcr_id))
    if not pcr:
        raise HTTPException(404, "PCR vehicle not found")
    return pcr_payload(pcr)


@router.patch("/pcr/{pcr_id}")
def patch_pcr(pcr_id: str, body: PCRUpdateInput, db: Session = Depends(get_db)):
    try:
        pcr = update_pcr_vehicle(db, pcr_id, body.model_dump(exclude_unset=True))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if not pcr:
        raise HTTPException(404, "PCR vehicle not found")
    return pcr_payload(pcr)


@router.post("/pcr/{pcr_id}/location")
def post_pcr_location(pcr_id: str, body: PCRLocationInput, db: Session = Depends(get_db)):
    try:
        pcr = update_pcr_location(db, pcr_id, body.model_dump())
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if not pcr:
        raise HTTPException(404, "PCR vehicle not found")
    return pcr_payload(pcr)


@router.get("/incidents")
def list_incidents(status: str | None = None, offset: int = Query(0, ge=0),
                   limit: int = Query(50, ge=1, le=200), db: Session = Depends(get_db)):
    query = db.query(EnforcementIncident)
    if status:
        query = query.filter(EnforcementIncident.status == status.upper())
    total = query.count()
    rows = query.order_by(EnforcementIncident.created_at.desc(), EnforcementIncident.id.desc()).offset(offset).limit(limit).all()
    alerts = {
        alert.id: alert for alert in db.query(HotlistAlert)
        .filter(HotlistAlert.id.in_([row.hotlist_alert_id for row in rows] or [0]))
        .all()
    }
    pcrs = {
        pcr.pcr_id: pcr for pcr in db.query(PCRVehicle)
        .filter(PCRVehicle.pcr_id.in_([row.recommended_pcr_id for row in rows if row.recommended_pcr_id] or [""]))
        .all()
    }
    cameras = {
        camera.camera_id: camera for camera in db.query(Camera)
        .filter(Camera.camera_id.in_([row.camera_id for row in rows] or [""]))
        .all()
    }
    return {"total": total, "items": [
        incident_payload(db, row, alert=alerts.get(row.hotlist_alert_id),
                         pcr=pcrs.get(row.recommended_pcr_id), camera=cameras.get(row.camera_id))
        for row in rows
    ]}


@router.get("/incidents/{incident_id}")
def get_incident(incident_id: int, db: Session = Depends(get_db)):
    incident = db.get(EnforcementIncident, incident_id)
    if not incident:
        raise HTTPException(404, "Incident not found")
    return incident_payload(db, incident)


@router.post("/incidents/{incident_id}/status")
def post_incident_status(incident_id: int, body: IncidentStatusInput, db: Session = Depends(get_db)):
    try:
        incident = update_incident_status(db, incident_id, body.status, body.updated_by)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if not incident:
        raise HTTPException(404, "Incident not found")
    return incident_payload(db, incident)
