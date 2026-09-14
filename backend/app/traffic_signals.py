"""Phase 10 smart traffic signal recommendations and safe simulation.

This module recommends timing changes from existing Phase 6 traffic analytics.
It never controls physical traffic signals by default; the controller surface is
an explicit adapter that reports SIMULATION / NOT CONNECTED unless configured.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import uuid

from app.database import (
    Camera,
    PlateEvent,
    SignalPhase,
    SignalRecommendation,
    SignalSimulationState,
    TrafficJunction,
    SessionLocal,
)
from app.location import camera_location
from app.traffic_analytics import camera_metrics, density_metrics, road_metrics


DATA_WINDOW_MINUTES = int(os.getenv("ANPR_SIGNAL_DEMAND_WINDOW_MINUTES", "60"))
DEFAULT_CYCLE_SECONDS = int(os.getenv("ANPR_SIGNAL_DEFAULT_CYCLE_SECONDS", "90"))
MAX_WAIT_BONUS_SECONDS = int(os.getenv("ANPR_SIGNAL_MAX_WAIT_BONUS_SECONDS", "12"))
MAX_WAIT_SECONDS = int(os.getenv("ANPR_SIGNAL_MAX_WAIT_SECONDS", "180"))


def utcnow():
    return dt.datetime.utcnow()


def _safe_json(value):
    return json.dumps(value, sort_keys=True, default=str)


def _load_json(value, default):
    try:
        parsed = json.loads(value or "")
        return parsed if parsed is not None else default
    except Exception:
        return default


def _camera_ids(value):
    if isinstance(value, str):
        value = _load_json(value, [])
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item or "").strip()]


def _validate_junction_payload(db, payload, existing=None):
    if not isinstance(payload, dict):
        raise ValueError("junction payload must be an object")
    junction_id = str(payload.get("junction_id") or getattr(existing, "junction_id", "")).strip().upper()
    name = str(payload.get("name") or getattr(existing, "name", "")).strip()
    if not junction_id or len(junction_id) > 80:
        raise ValueError("junction_id is required and must be <= 80 characters")
    if not name or len(name) > 160:
        raise ValueError("name is required and must be <= 160 characters")
    camera_ids = _camera_ids(payload.get("camera_ids", _load_json(getattr(existing, "camera_ids_json", "[]"), [])))
    known = {camera.camera_id for camera in db.query(Camera).all()}
    missing = [camera_id for camera_id in camera_ids if camera_id not in known]
    if missing:
        raise ValueError(f"unknown camera_id(s): {', '.join(missing)}")
    lat = payload.get("lat", getattr(existing, "lat", None))
    lng = payload.get("lng", getattr(existing, "lng", None))
    if lat is not None:
        lat = float(lat)
        if not -90 <= lat <= 90:
            raise ValueError("lat must be between -90 and 90")
    if lng is not None:
        lng = float(lng)
        if not -180 <= lng <= 180:
            raise ValueError("lng must be between -180 and 180")
    mode = str(payload.get("controller_mode", getattr(existing, "controller_mode", "simulation")) or "simulation").lower()
    if mode not in {"simulation", "rest"}:
        raise ValueError("controller_mode must be simulation or rest")
    controller_url = payload.get("controller_url", getattr(existing, "controller_url", None))
    if mode == "rest" and not controller_url:
        mode = "simulation"
    return {
        "junction_id": junction_id,
        "name": name,
        "lat": lat,
        "lng": lng,
        "camera_ids": camera_ids,
        "active": bool(payload.get("active", getattr(existing, "active", True))),
        "controller_mode": mode,
        "controller_url": str(controller_url).strip() if controller_url else None,
    }


def _validate_phase_payload(db, junction, payload, existing=None):
    if not isinstance(payload, dict):
        raise ValueError("phase payload must be an object")
    phase_id = str(payload.get("phase_id") or getattr(existing, "phase_id", "")).strip().upper()
    movement = str(payload.get("movement") or getattr(existing, "movement", "")).strip().upper()
    if not phase_id or len(phase_id) > 80:
        raise ValueError("phase_id is required and must be <= 80 characters")
    if not movement or len(movement) > 120:
        raise ValueError("movement is required and must be <= 120 characters")
    camera_ids = _camera_ids(payload.get("movement_camera_ids", _load_json(getattr(existing, "movement_camera_ids_json", "[]"), [])))
    junction_camera_ids = set(_load_json(junction.camera_ids_json, []))
    if camera_ids and not set(camera_ids).issubset(junction_camera_ids):
        raise ValueError("movement_camera_ids must belong to the junction")
    min_green = int(payload.get("min_green_seconds", getattr(existing, "min_green_seconds", 15)))
    max_green = int(payload.get("max_green_seconds", getattr(existing, "max_green_seconds", 90)))
    yellow = int(payload.get("yellow_seconds", getattr(existing, "yellow_seconds", 4)))
    all_red = int(payload.get("all_red_seconds", getattr(existing, "all_red_seconds", 2)))
    if min_green < 5 or max_green < min_green or max_green > 180:
        raise ValueError("green times must satisfy 5 <= min_green <= max_green <= 180")
    if yellow < 3 or all_red < 1:
        raise ValueError("yellow must be >= 3 and all-red must be >= 1")
    return {
        "phase_id": phase_id,
        "movement": movement,
        "movement_camera_ids": camera_ids,
        "min_green_seconds": min_green,
        "max_green_seconds": max_green,
        "yellow_seconds": yellow,
        "all_red_seconds": all_red,
        "enabled": bool(payload.get("enabled", getattr(existing, "enabled", True))),
        "display_order": int(payload.get("display_order", getattr(existing, "display_order", 0))),
    }


def junction_payload(junction, phases=None):
    if phases is None:
        phases = []
    return {
        "junction_id": junction.junction_id,
        "name": junction.name,
        "lat": junction.lat,
        "lng": junction.lng,
        "camera_ids": _load_json(junction.camera_ids_json, []),
        "active": bool(junction.active),
        "controller_mode": junction.controller_mode,
        "controller_status": controller_status(junction),
        "phases": [phase_payload(phase) for phase in phases],
        "created_at": junction.created_at.isoformat() if junction.created_at else None,
        "updated_at": junction.updated_at.isoformat() if junction.updated_at else None,
    }


def phase_payload(phase):
    return {
        "phase_id": phase.phase_id,
        "junction_id": phase.junction_id,
        "movement": phase.movement,
        "movement_camera_ids": _load_json(phase.movement_camera_ids_json, []),
        "min_green_seconds": phase.min_green_seconds,
        "max_green_seconds": phase.max_green_seconds,
        "yellow_seconds": phase.yellow_seconds,
        "all_red_seconds": phase.all_red_seconds,
        "current_state": phase.current_state,
        "enabled": bool(phase.enabled),
        "display_order": phase.display_order,
        "last_served_at": phase.last_served_at.isoformat() if phase.last_served_at else None,
    }


def ensure_demo_junction(db):
    junction = db.query(TrafficJunction).filter(TrafficJunction.junction_id == "JUNC-DEMO-01").first()
    if junction:
        return junction
    cameras = db.query(Camera).order_by(Camera.camera_id.asc()).limit(4).all()
    camera_ids = [camera.camera_id for camera in cameras]
    lat, lng = (camera_location(cameras[0]) if cameras else (None, None))
    now = utcnow()
    junction = TrafficJunction(
        junction_id="JUNC-DEMO-01",
        name="Demo Two-Direction Junction",
        lat=lat,
        lng=lng,
        camera_ids_json=_safe_json(camera_ids),
        active=True,
        created_at=now,
        updated_at=now,
    )
    db.add(junction)
    db.flush()
    defaults = [
        ("NORTH_SOUTH_GREEN", "NORTH_SOUTH", camera_ids[:2]),
        ("EAST_WEST_GREEN", "EAST_WEST", camera_ids[2:4] or camera_ids[:2]),
    ]
    for order, (phase_id, movement, ids) in enumerate(defaults):
        db.add(SignalPhase(
            phase_id=phase_id,
            junction_id=junction.junction_id,
            movement=movement,
            movement_camera_ids_json=_safe_json(ids),
            min_green_seconds=15,
            max_green_seconds=75,
            yellow_seconds=4,
            all_red_seconds=2,
            enabled=True,
            display_order=order,
            created_at=now,
            updated_at=now,
        ))
    db.commit()
    return junction


def create_junction(db, payload):
    values = _validate_junction_payload(db, payload)
    if db.query(TrafficJunction).filter(TrafficJunction.junction_id == values["junction_id"]).first():
        raise ValueError("junction_id already exists")
    now = utcnow()
    junction = TrafficJunction(
        junction_id=values["junction_id"],
        name=values["name"],
        lat=values["lat"],
        lng=values["lng"],
        camera_ids_json=_safe_json(values["camera_ids"]),
        active=values["active"],
        controller_mode=values["controller_mode"],
        controller_url=values["controller_url"],
        created_at=now,
        updated_at=now,
    )
    db.add(junction)
    db.commit()
    return junction


def update_junction(db, junction_id, payload):
    junction = get_junction_or_raise(db, junction_id)
    values = _validate_junction_payload(db, payload, junction)
    junction.name = values["name"]
    junction.lat = values["lat"]
    junction.lng = values["lng"]
    junction.camera_ids_json = _safe_json(values["camera_ids"])
    junction.active = values["active"]
    junction.controller_mode = values["controller_mode"]
    junction.controller_url = values["controller_url"]
    junction.updated_at = utcnow()
    db.commit()
    return junction


def get_junction_or_raise(db, junction_id):
    junction = db.query(TrafficJunction).filter(TrafficJunction.junction_id == str(junction_id).upper()).first()
    if not junction:
        raise ValueError("junction not found")
    return junction


def phases_for_junction(db, junction_id):
    return (
        db.query(SignalPhase)
        .filter(SignalPhase.junction_id == junction_id)
        .order_by(SignalPhase.display_order.asc(), SignalPhase.id.asc())
        .all()
    )


def upsert_phase(db, junction_id, payload):
    junction = get_junction_or_raise(db, junction_id)
    values = _validate_phase_payload(db, junction, payload)
    phase = (
        db.query(SignalPhase)
        .filter(SignalPhase.junction_id == junction.junction_id, SignalPhase.phase_id == values["phase_id"])
        .first()
    )
    now = utcnow()
    if phase is None:
        phase = SignalPhase(phase_id=values["phase_id"], junction_id=junction.junction_id, created_at=now)
        db.add(phase)
    phase.movement = values["movement"]
    phase.movement_camera_ids_json = _safe_json(values["movement_camera_ids"])
    phase.min_green_seconds = values["min_green_seconds"]
    phase.max_green_seconds = values["max_green_seconds"]
    phase.yellow_seconds = values["yellow_seconds"]
    phase.all_red_seconds = values["all_red_seconds"]
    phase.enabled = values["enabled"]
    phase.display_order = values["display_order"]
    phase.updated_at = now
    db.commit()
    return phase


def demand_for_junction(db, junction_id, window_minutes=DATA_WINDOW_MINUTES):
    junction = get_junction_or_raise(db, junction_id)
    phases = [phase for phase in phases_for_junction(db, junction.junction_id) if phase.enabled]
    end = utcnow()
    start = end - dt.timedelta(minutes=max(5, int(window_minutes or DATA_WINDOW_MINUTES)))
    camera_rows = {row["camera_id"]: row for row in camera_metrics(db, start=start, end=end)}
    density_rows = {row["camera_id"]: row for row in density_metrics(db, start=start, end=end)}
    road_rows = road_metrics(db, start=start, end=end)
    max_observations = max([row.get("observation_count", 0) for row in camera_rows.values()] + [1])
    max_movements = max([row.get("movement_count", 0) for row in road_rows] + [1])
    items = []
    total_observations = 0
    for phase in phases:
        movement_cameras = _load_json(phase.movement_camera_ids_json, []) or _load_json(junction.camera_ids_json, [])
        observation_count = sum((camera_rows.get(cid) or {}).get("observation_count", 0) for cid in movement_cameras)
        unique_count = sum((camera_rows.get(cid) or {}).get("unique_vehicle_count", 0) for cid in movement_cameras)
        flow_count = sum(
            row.get("movement_count", 0)
            for row in road_rows
            if row.get("source_camera_id") in movement_cameras or row.get("destination_camera_id") in movement_cameras
        )
        congestion_score = max([
            (row.get("congestion") or {}).get("score", 0.0)
            for row in road_rows
            if row.get("source_camera_id") in movement_cameras or row.get("destination_camera_id") in movement_cameras
        ] + [0.0])
        utilization = max([
            (row.get("utilization") or {}).get("relative_flow_score", 0.0)
            for row in road_rows
            if row.get("source_camera_id") in movement_cameras or row.get("destination_camera_id") in movement_cameras
        ] + [0.0])
        density_score = max([
            {"LOW": 0.25, "MEDIUM": 0.60, "HIGH": 1.0}.get((density_rows.get(cid) or {}).get("density_level"), 0.0)
            for cid in movement_cameras
        ] + [0.0])
        flow_score = min(1.0, flow_count / max(max_movements, 1))
        volume_score = min(1.0, observation_count / max(max_observations, 1))
        wait_seconds = (end - phase.last_served_at).total_seconds() if phase.last_served_at else MAX_WAIT_SECONDS
        fairness_score = min(1.0, max(0.0, wait_seconds) / max(MAX_WAIT_SECONDS, 1))
        demand_score = round(min(1.0, 0.35 * flow_score + 0.25 * volume_score + 0.20 * density_score +
                                 0.15 * congestion_score + 0.05 * utilization + 0.10 * fairness_score), 3)
        reasons = []
        if flow_score >= 0.65:
            reasons.append("high_flow")
        if density_score >= 0.60:
            reasons.append("high_density")
        if congestion_score >= 0.50:
            reasons.append("congestion")
        if utilization >= 0.65:
            reasons.append("high_road_utilization")
        if fairness_score >= 0.80:
            reasons.append("waiting_time")
        total_observations += observation_count
        items.append({
            "phase_id": phase.phase_id,
            "movement": phase.movement,
            "camera_ids": movement_cameras,
            "observation_count": observation_count,
            "unique_vehicle_count": unique_count,
            "flow_count": flow_count,
            "density_score": round(density_score, 3),
            "congestion_score": round(congestion_score, 3),
            "road_utilization_score": round(utilization, 3),
            "waiting_seconds": round(wait_seconds, 1),
            "fairness_score": round(fairness_score, 3),
            "demand_score": demand_score,
            "reasons": reasons or ["low_recent_demand"],
        })
    quality = "INSUFFICIENT_DATA" if total_observations < 2 else ("LOW" if total_observations < 5 else "GOOD")
    return {
        "junction_id": junction.junction_id,
        "window_minutes": window_minutes,
        "timestamp": end.isoformat(),
        "data_quality": quality,
        "formula": "0.35*flow + 0.25*volume + 0.20*density + 0.15*congestion + 0.05*utilization + 0.10*fairness",
        "movements": items,
    }


def generate_recommendation(db, junction_id, actor=None, cycle_seconds=DEFAULT_CYCLE_SECONDS, persist=True):
    junction = get_junction_or_raise(db, junction_id)
    if not junction.active:
        raise ValueError("cannot recommend timings for inactive junction")
    phases = [phase for phase in phases_for_junction(db, junction.junction_id) if phase.enabled]
    if len(phases) < 2:
        raise ValueError("at least two enabled phases are required")
    demand = demand_for_junction(db, junction.junction_id)
    fixed_safety = sum(phase.yellow_seconds + phase.all_red_seconds for phase in phases)
    available_green = max(sum(phase.min_green_seconds for phase in phases), int(cycle_seconds) - fixed_safety)
    demand_by_phase = {item["phase_id"]: item for item in demand["movements"]}
    total_score = sum(max(0.05, item["demand_score"]) for item in demand["movements"]) or 1.0
    plan = []
    for phase in phases:
        item = demand_by_phase.get(phase.phase_id, {"demand_score": 0.0, "reasons": ["insufficient_data"]})
        share = max(0.05, item["demand_score"]) / total_score
        proposed = round(available_green * share)
        if "waiting_time" in item.get("reasons", []):
            proposed += min(MAX_WAIT_BONUS_SECONDS, max(0, phase.max_green_seconds - proposed))
        if demand["data_quality"] == "INSUFFICIENT_DATA":
            proposed = max(phase.min_green_seconds, min(phase.max_green_seconds, DEFAULT_CYCLE_SECONDS // len(phases)))
        green = int(max(phase.min_green_seconds, min(phase.max_green_seconds, proposed)))
        plan.append({
            "phase_id": phase.phase_id,
            "movement": phase.movement,
            "recommended_green_seconds": green,
            "current_green_seconds": phase.min_green_seconds,
            "min_green_seconds": phase.min_green_seconds,
            "max_green_seconds": phase.max_green_seconds,
            "yellow_seconds": phase.yellow_seconds,
            "all_red_seconds": phase.all_red_seconds,
            "demand_score": item.get("demand_score", 0.0),
            "reasons": item.get("reasons", ["insufficient_data"]),
        })
    plan.sort(key=lambda item: item["demand_score"], reverse=True)
    cycle_length = sum(item["recommended_green_seconds"] + item["yellow_seconds"] + item["all_red_seconds"] for item in plan)
    reliability = 0.25 if demand["data_quality"] == "INSUFFICIENT_DATA" else (0.6 if demand["data_quality"] == "LOW" else 0.85)
    explanation = _recommendation_explanation(plan, demand)
    recommendation = {
        "recommendation_id": str(uuid.uuid4()),
        "junction_id": junction.junction_id,
        "timestamp": utcnow().isoformat(),
        "mode": "RECOMMENDATION_ONLY",
        "controller_status": controller_status(junction),
        "current_phase": current_simulation_state(db, junction.junction_id).get("current_phase_id"),
        "current_timing": current_simulation_state(db, junction.junction_id),
        "demand": demand,
        "recommended_phase_order": [item["phase_id"] for item in plan],
        "phases": plan,
        "cycle_length_seconds": cycle_length,
        "data_quality": demand["data_quality"],
        "recommendation_reliability": reliability,
        "explanation": explanation,
        "safety": {
            "bounded_by_min_max_green": True,
            "yellow_preserved": all(item["yellow_seconds"] >= 3 for item in plan),
            "all_red_preserved": all(item["all_red_seconds"] >= 1 for item in plan),
            "physical_control": False,
        },
    }
    if persist:
        row = SignalRecommendation(
            recommendation_id=recommendation["recommendation_id"],
            junction_id=junction.junction_id,
            current_phase_id=recommendation["current_phase"],
            demand_json=_safe_json(demand),
            recommendation_json=_safe_json(recommendation),
            data_quality=recommendation["data_quality"],
            reliability=reliability,
            actor=actor,
            created_at=utcnow(),
        )
        db.add(row)
        db.commit()
    return recommendation


def _recommendation_explanation(plan, demand):
    if demand["data_quality"] == "INSUFFICIENT_DATA":
        return {
            "summary": "Insufficient recent observations; using safe balanced timing within configured phase limits.",
            "reason_codes": ["insufficient_data"],
        }
    lead = plan[0]
    return {
        "summary": (
            f"{lead['movement']} has the highest transparent demand score "
            f"({lead['demand_score']}); recommended green is {lead['recommended_green_seconds']}s."
        ),
        "reason_codes": sorted({reason for item in plan for reason in item.get("reasons", [])}),
    }


def recommendation_payload(row):
    body = _load_json(row.recommendation_json, {})
    body.update({
        "recommendation_id": row.recommendation_id,
        "junction_id": row.junction_id,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "applied": bool(row.applied),
        "applied_at": row.applied_at.isoformat() if row.applied_at else None,
    })
    return body


def current_simulation_state(db, junction_id):
    state = db.query(SignalSimulationState).filter(SignalSimulationState.junction_id == junction_id).first()
    if not state:
        return {
            "junction_id": junction_id,
            "active": False,
            "mode": "SIMULATION MODE",
            "current_phase_id": None,
            "phase_kind": "STOPPED",
            "remaining_seconds": 0,
            "cycle_plan": [],
            "physical_control": False,
        }
    return {
        "junction_id": state.junction_id,
        "active": bool(state.active),
        "mode": state.mode,
        "current_phase_id": state.current_phase_id,
        "phase_kind": state.phase_kind,
        "remaining_seconds": state.remaining_seconds,
        "cycle_plan": _load_json(state.cycle_plan_json, []),
        "cursor_index": state.cursor_index,
        "updated_at": state.updated_at.isoformat() if state.updated_at else None,
        "physical_control": False,
    }


def _cycle_steps_from_plan(plan):
    steps = []
    for item in plan:
        steps.append({"phase_id": item["phase_id"], "phase_kind": "GREEN", "duration_seconds": item["recommended_green_seconds"]})
        steps.append({"phase_id": item["phase_id"], "phase_kind": "YELLOW", "duration_seconds": item["yellow_seconds"]})
        steps.append({"phase_id": "ALL_RED", "phase_kind": "ALL_RED", "duration_seconds": item["all_red_seconds"]})
    return steps


def start_simulation(db, junction_id, recommendation_id=None, actor=None):
    junction = get_junction_or_raise(db, junction_id)
    if not junction.active:
        raise ValueError("cannot start simulation for inactive junction")
    if recommendation_id:
        row = db.query(SignalRecommendation).filter(SignalRecommendation.recommendation_id == recommendation_id).first()
        if not row:
            raise ValueError("recommendation not found")
        recommendation = recommendation_payload(row)
    else:
        recommendation = generate_recommendation(db, junction.junction_id, actor=actor)
    steps = _cycle_steps_from_plan(recommendation["phases"])
    if not steps:
        raise ValueError("recommendation has no phases")
    state = db.query(SignalSimulationState).filter(SignalSimulationState.junction_id == junction.junction_id).first()
    now = utcnow()
    if state is None:
        state = SignalSimulationState(junction_id=junction.junction_id)
        db.add(state)
    state.active = True
    state.current_phase_id = steps[0]["phase_id"]
    state.phase_kind = steps[0]["phase_kind"]
    state.remaining_seconds = int(steps[0]["duration_seconds"])
    state.cycle_plan_json = _safe_json(steps)
    state.cursor_index = 0
    state.mode = "SIMULATION MODE - NO REAL TRAFFIC SIGNAL CONTROL"
    state.updated_at = now
    db.commit()
    return current_simulation_state(db, junction.junction_id)


def tick_simulation(db, junction_id, seconds=1):
    state = db.query(SignalSimulationState).filter(SignalSimulationState.junction_id == str(junction_id).upper()).first()
    if not state:
        raise ValueError("simulation not found")
    if not state.active:
        return current_simulation_state(db, state.junction_id)
    steps = _load_json(state.cycle_plan_json, [])
    if not steps:
        raise ValueError("simulation has no cycle plan")
    state.remaining_seconds -= max(1, int(seconds or 1))
    while state.remaining_seconds <= 0:
        state.cursor_index = (state.cursor_index + 1) % len(steps)
        step = steps[state.cursor_index]
        state.current_phase_id = step["phase_id"]
        state.phase_kind = step["phase_kind"]
        state.remaining_seconds += int(step["duration_seconds"])
        if state.phase_kind == "GREEN":
            phase = db.query(SignalPhase).filter(
                SignalPhase.junction_id == state.junction_id,
                SignalPhase.phase_id == state.current_phase_id,
            ).first()
            if phase:
                phase.last_served_at = utcnow()
    state.updated_at = utcnow()
    db.commit()
    return current_simulation_state(db, state.junction_id)


def stop_simulation(db, junction_id):
    state = db.query(SignalSimulationState).filter(SignalSimulationState.junction_id == str(junction_id).upper()).first()
    if not state:
        raise ValueError("simulation not found")
    state.active = False
    state.phase_kind = "STOPPED"
    state.remaining_seconds = 0
    state.updated_at = utcnow()
    db.commit()
    return current_simulation_state(db, state.junction_id)


def reset_simulation(db, junction_id):
    state = db.query(SignalSimulationState).filter(SignalSimulationState.junction_id == str(junction_id).upper()).first()
    if state:
        db.delete(state)
        db.commit()
    return current_simulation_state(db, str(junction_id).upper())


def controller_status(junction):
    if not junction or junction.controller_mode != "rest" or not junction.controller_url:
        return {"mode": "simulation", "status": "SIMULATION / NOT CONNECTED", "physical_control": False}
    return {"mode": "rest", "status": "CONFIGURED / NOT AUTO-APPLIED", "physical_control": False}


class SignalController:
    def get_state(self, db, junction_id):
        raise NotImplementedError

    def apply_recommendation(self, db, recommendation_id):
        raise NotImplementedError


class SimulationSignalController(SignalController):
    def get_state(self, db, junction_id):
        return current_simulation_state(db, junction_id)

    def apply_recommendation(self, db, recommendation_id):
        row = db.query(SignalRecommendation).filter(SignalRecommendation.recommendation_id == recommendation_id).first()
        if not row:
            raise ValueError("recommendation not found")
        row.applied = True
        row.applied_at = utcnow()
        db.commit()
        return start_simulation(db, row.junction_id, recommendation_id=recommendation_id, actor=row.actor)


class RestSignalControllerAdapter(SignalController):
    def get_state(self, db, junction_id):
        junction = get_junction_or_raise(db, junction_id)
        return {"junction_id": junction.junction_id, **controller_status(junction)}

    def apply_recommendation(self, db, recommendation_id):
        row = db.query(SignalRecommendation).filter(SignalRecommendation.recommendation_id == recommendation_id).first()
        if not row:
            raise ValueError("recommendation not found")
        junction = get_junction_or_raise(db, row.junction_id)
        if not junction.controller_url:
            return {"status": "SIMULATION / NOT CONNECTED", "applied": False, "physical_control": False}
        return {"status": "CONFIGURED / NOT SENT AUTOMATICALLY", "applied": False, "physical_control": False}


def controller_for_junction(junction):
    if junction.controller_mode == "rest" and junction.controller_url:
        return RestSignalControllerAdapter()
    return SimulationSignalController()
