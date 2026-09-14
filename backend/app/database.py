"""
Database setup (SQLite for the demo — same schema would run on Postgres/TimescaleDB
in production, just swap the connection string).
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, UniqueConstraint, text
from sqlalchemy.orm import declarative_base, sessionmaker
import datetime
import asyncio
import os
import hashlib
import logging
import uuid
from difflib import SequenceMatcher

try:
    from rapidfuzz.distance import Levenshtein
except ImportError:  # pragma: no cover - optional production dependency
    Levenshtein = None

DEFAULT_SQLITE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "anpr_demo.db"))
DATABASE_PATH = os.getenv(
    "ANPR_DATABASE_PATH",
    DEFAULT_SQLITE_PATH,
)
DATABASE_URL = os.getenv("ANPR_DATABASE_URL", f"sqlite:///{DATABASE_PATH}")

engine_kwargs = {"connect_args": {"check_same_thread": False}} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, **engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, expire_on_commit=False)
Base = declarative_base()


class Camera(Base):
    """A registered 'camera' — in the demo, one phone = one camera."""
    __tablename__ = "cameras"

    camera_id = Column(String, primary_key=True, index=True)
    label = Column(String, nullable=False)          # e.g. "College Gate"
    lat = Column(Float, nullable=False)
    lng = Column(Float, nullable=False)
    location_known = Column(Boolean, default=True, nullable=False)
    stream_url = Column(String, nullable=True)
    stream_type = Column(String, nullable=True)       # http | rtsp
    active = Column(Boolean, default=False, nullable=False)
    last_seen = Column(DateTime, nullable=True)
    edge_device_id = Column(String(80), nullable=True, index=True)
    edge_status = Column(String(20), nullable=False, default="OFFLINE", index=True)
    edge_last_processed = Column(DateTime, nullable=True)
    edge_config_json = Column(Text, nullable=True)
    frames_sampled = Column(Integer, nullable=False, default=0)
    observations_produced = Column(Integer, nullable=False, default=0)
    processing_errors = Column(Integer, nullable=False, default=0)


class CameraRoadConnection(Base):
    """Directed road relationship between two configured cameras."""
    __tablename__ = "camera_road_connections"
    __table_args__ = (UniqueConstraint("source_camera_id", "destination_camera_id", name="uq_camera_road_pair"),)

    id = Column(Integer, primary_key=True)
    source_camera_id = Column(String, nullable=False, index=True)
    destination_camera_id = Column(String, nullable=False, index=True)
    distance_meters = Column(Float, nullable=False)
    distance_source = Column(String, nullable=False, default="manual")
    direction = Column(String, nullable=True)
    road_name = Column(String, nullable=True)
    road_type = Column(String, nullable=True)
    provider = Column(String, nullable=False, default="manual")
    provider_reference = Column(String, nullable=True)
    active = Column(Boolean, default=True, nullable=False, index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)


class PlateEvent(Base):
    """One detection event: a plate read at a camera at a point in time."""
    __tablename__ = "plate_events"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String, index=True)
    image_path = Column(String)
    original_image_path = Column(String, nullable=True)
    original_image_path_ciphertext = Column(Text, nullable=True)
    original_image_path_nonce = Column(String(64), nullable=True)
    original_image_path_key_version = Column(String(40), nullable=True)
    privacy_image_path = Column(String, nullable=True)
    privacy_status = Column(String, nullable=True)
    privacy_metadata = Column(Text, nullable=True)
    privacy_metadata_ciphertext = Column(Text, nullable=True)
    privacy_metadata_nonce = Column(String(64), nullable=True)
    privacy_metadata_key_version = Column(String(40), nullable=True)
    privacy_processed_at = Column(DateTime, nullable=True)
    plate_text = Column(String, index=True, nullable=True)
    confidence = Column(Float, nullable=True)
    status = Column(String, default="processing")   # processing | ok | PENDING_REVIEW | failed
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    raw_ocr_candidates = Column(Text, nullable=True)  # JSON string of all frame-level reads
    track_id = Column(String, nullable=True, index=True)
    plate_category = Column(String, nullable=True)
    layout = Column(String, nullable=True)
    rule_violations = Column(Text, nullable=True)
    bbox_x = Column(Float, nullable=True)
    bbox_y = Column(Float, nullable=True)
    bbox_width = Column(Float, nullable=True)
    bbox_height = Column(Float, nullable=True)
    velocity_x = Column(Float, nullable=True)
    velocity_y = Column(Float, nullable=True)
    vehicle_id = Column(String, nullable=True, index=True)
    vehicle_type = Column(String, nullable=True)
    vehicle_color = Column(String, nullable=True)
    vehicle_crop_path = Column(String, nullable=True)
    appearance_embedding = Column(Text, nullable=True)
    appearance_model = Column(String, nullable=True)
    appearance_embedding_version = Column(String, nullable=True)
    appearance_quality = Column(Float, nullable=True)


class Vehicle(Base):
    """Persistent global vehicle identity, currently anchored to a confirmed plate."""
    __tablename__ = "vehicles"

    vehicle_id = Column(String, primary_key=True, index=True)
    primary_plate_text = Column(String(24), nullable=False, unique=True, index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)


class VehicleMatchCandidate(Base):
    """Explainable comparison between two persisted vehicle observations."""
    __tablename__ = "vehicle_match_candidates"
    __table_args__ = (UniqueConstraint("observation_a_id", "observation_b_id", name="uq_vehicle_match_pair"),)

    id = Column(Integer, primary_key=True)
    observation_a_id = Column(Integer, nullable=False, index=True)
    observation_b_id = Column(Integer, nullable=False, index=True)
    candidate_vehicle_id = Column(String, nullable=True, index=True)
    camera_a_id = Column(String, nullable=True)
    camera_b_id = Column(String, nullable=True)
    plate_evidence = Column(Text, nullable=True)
    appearance_similarity = Column(Float, nullable=True)
    vehicle_type_similarity = Column(Float, nullable=True)
    vehicle_color_similarity = Column(Float, nullable=True)
    time_delta_seconds = Column(Float, nullable=True)
    spatial_relationship = Column(Text, nullable=True)
    factor_scores = Column(Text, nullable=True)
    factors_used = Column(Text, nullable=True)
    final_confidence = Column(Float, nullable=False, default=0.0)
    state = Column(String, nullable=False, default="LOW_CONFIDENCE")
    decision = Column(String, nullable=False, default="no_association")
    review_status = Column(String, nullable=False, default="not_required")
    matching_model = Column(String, nullable=False, default="vehicle-match-v1")
    matching_version = Column(String, nullable=False, default="2B-2026-09-12")
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    reviewed_at = Column(DateTime, nullable=True)
    reviewed_by = Column(String, nullable=True)


class VehicleAnomaly(Base):
    """Persisted explainable trajectory anomaly such as impossible travel."""
    __tablename__ = "vehicle_anomalies"
    __table_args__ = (
        UniqueConstraint("anomaly_type", "source_event_id", "destination_event_id", name="uq_vehicle_anomaly_pair"),
    )

    id = Column(Integer, primary_key=True)
    vehicle_id = Column(String, nullable=True, index=True)
    plate_text = Column(String(24), nullable=True, index=True)
    anomaly_type = Column(String, nullable=False, index=True)
    severity = Column(String, nullable=False, index=True)
    status = Column(String, nullable=False, default="active", index=True)
    source_event_id = Column(Integer, nullable=False, index=True)
    destination_event_id = Column(Integer, nullable=False, index=True)
    source_camera_id = Column(String, nullable=False)
    destination_camera_id = Column(String, nullable=False)
    detected_at = Column(DateTime, nullable=False, index=True)
    travel_time_seconds = Column(Float, nullable=True)
    road_distance_meters = Column(Float, nullable=True)
    estimated_speed_kmh = Column(Float, nullable=True)
    allowed_speed_kmh = Column(Float, nullable=True)
    excess_ratio = Column(Float, nullable=True)
    road_type = Column(String, nullable=True)
    road_name = Column(String, nullable=True)
    policy = Column(Text, nullable=True)
    explanation = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    acknowledged_at = Column(DateTime, nullable=True)
    resolved_at = Column(DateTime, nullable=True)


class PlateSuspicionEvent(Base):
    """Explainable possible cloned-plate evidence for one valid plate observation pair."""
    __tablename__ = "plate_suspicion_events"
    __table_args__ = (
        UniqueConstraint("plate_text", "source_event_id", "destination_event_id", name="uq_plate_suspicion_pair"),
    )

    id = Column(Integer, primary_key=True)
    plate_text = Column(String(24), nullable=False, index=True)
    vehicle_id = Column(String, nullable=True, index=True)
    source_event_id = Column(Integer, nullable=False, index=True)
    destination_event_id = Column(Integer, nullable=False, index=True)
    source_camera_id = Column(String, nullable=True)
    destination_camera_id = Column(String, nullable=True)
    suspicion_score = Column(Float, nullable=False, default=0.0)
    classification = Column(String, nullable=False, default="normal", index=True)
    status = Column(String, nullable=False, default="open", index=True)
    evidence = Column(Text, nullable=False)
    appearance_similarity = Column(Float, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow, index=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    reviewed_at = Column(DateTime, nullable=True)
    reviewed_by = Column(String, nullable=True)


class RouteAnomalyEvent(Base):
    """Explainable anomaly over a vehicle trajectory window."""
    __tablename__ = "route_anomaly_events"
    __table_args__ = (
        UniqueConstraint("vehicle_id", "start_event_id", "end_event_id", name="uq_route_anomaly_window"),
    )

    id = Column(Integer, primary_key=True)
    vehicle_id = Column(String, nullable=False, index=True)
    plate_text = Column(String(24), nullable=True, index=True)
    start_event_id = Column(Integer, nullable=False, index=True)
    end_event_id = Column(Integer, nullable=False, index=True)
    start_camera_id = Column(String, nullable=True)
    end_camera_id = Column(String, nullable=True)
    route_signature = Column(String, nullable=False)
    route_anomaly_score = Column(Float, nullable=False, default=0.0)
    classification = Column(String, nullable=False, default="normal", index=True)
    status = Column(String, nullable=False, default="open", index=True)
    evidence = Column(Text, nullable=False)
    explanation = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow, index=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    reviewed_at = Column(DateTime, nullable=True)
    reviewed_by = Column(String, nullable=True)


class HotlistEntry(Base):
    __tablename__ = "hotlist_entries"
    id = Column(Integer, primary_key=True)
    plate_text = Column(String(24), nullable=False, unique=True, index=True)
    category = Column(String(20), nullable=False, default="stolen")
    reason = Column(String(500), nullable=False)
    reference = Column(String(120), nullable=False, default="")
    active = Column(Boolean, nullable=False, default=True)
    expires_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)


class HotlistAlert(Base):
    __tablename__ = "hotlist_alerts"
    __table_args__ = (UniqueConstraint("event_id", "hotlist_id", name="uq_hotlist_event"),)
    id = Column(Integer, primary_key=True)
    event_id = Column(Integer, nullable=True, index=True)
    hotlist_id = Column(Integer, nullable=False, index=True)
    plate_text = Column(String(24), nullable=False)
    camera_id = Column(String, nullable=False)
    camera_label = Column(String, nullable=False)
    lat = Column(Float, nullable=True)
    lng = Column(Float, nullable=True)
    category = Column(String(20), nullable=False)
    reason = Column(String(500), nullable=False)
    reference = Column(String(120), nullable=False)
    confidence = Column(Float, nullable=False)
    match_status = Column(String(20), nullable=False)
    seen_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    acknowledged_at = Column(DateTime, nullable=True)
    revision = Column(Integer, nullable=False, default=1)


class HotlistNotification(Base):
    __tablename__ = "hotlist_notifications"
    __table_args__ = {"sqlite_autoincrement": True}
    id = Column(Integer, primary_key=True)
    alert_id = Column(Integer, nullable=False, index=True)


class PCRVehicle(Base):
    """Demo patrol/PCR vehicle location reported by a manual or simulated feed."""
    __tablename__ = "pcr_vehicles"

    pcr_id = Column(String(32), primary_key=True, index=True)
    name = Column(String(80), nullable=False)
    call_sign = Column(String(80), nullable=True)
    vehicle_identifier = Column(String(80), nullable=False, default="")
    lat = Column(Float, nullable=True)
    lng = Column(Float, nullable=True)
    status = Column(String(20), nullable=False, default="AVAILABLE", index=True)
    active = Column(Boolean, nullable=False, default=True, index=True)
    last_seen_at = Column(DateTime, nullable=True, index=True)
    contact_channel = Column(String(120), nullable=False, default="")
    metadata_json = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)


class EnforcementIncident(Base):
    """Assistive incident-command record linked one-to-one with a hotlist alert."""
    __tablename__ = "enforcement_incidents"
    __table_args__ = (UniqueConstraint("hotlist_alert_id", name="uq_enforcement_hotlist_alert"),)

    id = Column(Integer, primary_key=True)
    hotlist_alert_id = Column(Integer, nullable=False, index=True)
    vehicle_id = Column(String, nullable=True, index=True)
    plate_text = Column(String(24), nullable=False, index=True)
    camera_id = Column(String, nullable=False, index=True)
    event_id = Column(Integer, nullable=True, index=True)
    detected_at = Column(DateTime, nullable=False, index=True)
    recommended_pcr_id = Column(String(32), nullable=True, index=True)
    pcr_distance_meters = Column(Float, nullable=True)
    pcr_location_status = Column(String(20), nullable=False, default="UNKNOWN")
    recommendation_status = Column(String(40), nullable=False, default="NO_AVAILABLE_PCR", index=True)
    recommendation_reason = Column(String(240), nullable=False, default="")
    recommendation_calculated_at = Column(DateTime, nullable=True)
    status = Column(String(40), nullable=False, default="NEW", index=True)
    trajectory_reference = Column(String(240), nullable=False, default="")
    investigation_reference = Column(String(240), nullable=False, default="")
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    updated_by = Column(String(80), nullable=True)


class User(Base):
    """Local demo user for Phase 8 RBAC."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String(80), nullable=False, unique=True, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(40), nullable=False, index=True)
    active = Column(Boolean, nullable=False, default=True, index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)


class AuditLog(Base):
    """Append-only, hash-chained application audit record."""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True)
    audit_id = Column(String(36), nullable=False, unique=True, index=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.datetime.utcnow, index=True)
    user_id = Column(Integer, nullable=True, index=True)
    username = Column(String(80), nullable=True, index=True)
    role = Column(String(40), nullable=True, index=True)
    action = Column(String(80), nullable=False, index=True)
    resource_type = Column(String(80), nullable=True, index=True)
    resource_id = Column(String(120), nullable=True, index=True)
    source = Column(Text, nullable=True)
    success = Column(Boolean, nullable=False, default=True, index=True)
    reason = Column(String(240), nullable=True)
    details = Column(Text, nullable=True)
    previous_hash = Column(String(64), nullable=False)
    current_hash = Column(String(64), nullable=False)


class EdgeDevice(Base):
    """Logical edge agent/device feeding sampled ANPR metadata into the backend."""
    __tablename__ = "edge_devices"

    id = Column(Integer, primary_key=True)
    edge_device_id = Column(String(80), nullable=False, unique=True, index=True)
    label = Column(String(120), nullable=True)
    status = Column(String(20), nullable=False, default="OFFLINE", index=True)
    last_seen_at = Column(DateTime, nullable=True, index=True)
    last_processed_at = Column(DateTime, nullable=True)
    config_json = Column(Text, nullable=True)
    credential_hash = Column(String(128), nullable=True)
    credential_version = Column(String(40), nullable=True)
    queue_mode = Column(String(30), nullable=False, default="local")
    frames_sampled = Column(Integer, nullable=False, default=0)
    observations_received = Column(Integer, nullable=False, default=0)
    observations_processed = Column(Integer, nullable=False, default=0)
    observations_failed = Column(Integer, nullable=False, default=0)
    duplicate_observations = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)


class EdgeObservation(Base):
    """Idempotent edge observation envelope before/after central persistence."""
    __tablename__ = "edge_observations"

    id = Column(Integer, primary_key=True)
    observation_id = Column(String(120), nullable=False, unique=True, index=True)
    camera_id = Column(String, nullable=False, index=True)
    edge_device_id = Column(String(80), nullable=False, index=True)
    frame_id = Column(String(120), nullable=True)
    sequence_number = Column(Integer, nullable=True)
    capture_timestamp = Column(DateTime, nullable=False, index=True)
    received_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow, index=True)
    processing_started_at = Column(DateTime, nullable=True)
    processing_completed_at = Column(DateTime, nullable=True)
    status = Column(String(30), nullable=False, default="RECEIVED", index=True)
    retry_count = Column(Integer, nullable=False, default=0)
    max_retries = Column(Integer, nullable=False, default=3)
    next_retry_at = Column(DateTime, nullable=True)
    error_message = Column(String(500), nullable=True)
    payload_json = Column(Text, nullable=False)
    plate_event_id = Column(Integer, nullable=True, index=True)
    queue_name = Column(String(80), nullable=True)
    processing_latency_ms = Column(Float, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)


class RetentionRun(Base):
    """Manual retention lifecycle run for raw/high-resolution evidence."""
    __tablename__ = "retention_runs"

    id = Column(Integer, primary_key=True)
    run_id = Column(String(36), nullable=False, unique=True, index=True)
    dry_run = Column(Boolean, nullable=False, default=True, index=True)
    policy_json = Column(Text, nullable=False)
    result_json = Column(Text, nullable=False)
    status = Column(String(30), nullable=False, default="completed", index=True)
    started_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow, index=True)
    completed_at = Column(DateTime, nullable=True)
    actor = Column(String(80), nullable=True)


class TrafficJunction(Base):
    """Configured junction for Phase 10 signal recommendations."""
    __tablename__ = "traffic_junctions"

    id = Column(Integer, primary_key=True)
    junction_id = Column(String(80), nullable=False, unique=True, index=True)
    name = Column(String(160), nullable=False)
    lat = Column(Float, nullable=True)
    lng = Column(Float, nullable=True)
    camera_ids_json = Column(Text, nullable=False, default="[]")
    active = Column(Boolean, nullable=False, default=True, index=True)
    controller_mode = Column(String(30), nullable=False, default="simulation")
    controller_url = Column(String(500), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)


class SignalPhase(Base):
    """One configurable signal phase/movement for a junction."""
    __tablename__ = "signal_phases"
    __table_args__ = (UniqueConstraint("junction_id", "phase_id", name="uq_signal_phase_junction"),)

    id = Column(Integer, primary_key=True)
    phase_id = Column(String(80), nullable=False, index=True)
    junction_id = Column(String(80), nullable=False, index=True)
    movement = Column(String(120), nullable=False)
    movement_camera_ids_json = Column(Text, nullable=False, default="[]")
    min_green_seconds = Column(Integer, nullable=False, default=15)
    max_green_seconds = Column(Integer, nullable=False, default=90)
    yellow_seconds = Column(Integer, nullable=False, default=4)
    all_red_seconds = Column(Integer, nullable=False, default=2)
    current_state = Column(String(30), nullable=False, default="RED")
    enabled = Column(Boolean, nullable=False, default=True, index=True)
    display_order = Column(Integer, nullable=False, default=0)
    last_served_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)


class SignalRecommendation(Base):
    """Persisted explainable rule-based signal timing recommendation."""
    __tablename__ = "signal_recommendations"

    id = Column(Integer, primary_key=True)
    recommendation_id = Column(String(36), nullable=False, unique=True, index=True)
    junction_id = Column(String(80), nullable=False, index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow, index=True)
    current_phase_id = Column(String(80), nullable=True)
    demand_json = Column(Text, nullable=False)
    recommendation_json = Column(Text, nullable=False)
    data_quality = Column(String(40), nullable=False, default="INSUFFICIENT_DATA", index=True)
    reliability = Column(Float, nullable=False, default=0.0)
    applied = Column(Boolean, nullable=False, default=False, index=True)
    applied_at = Column(DateTime, nullable=True)
    actor = Column(String(80), nullable=True)


class SignalSimulationState(Base):
    """Safe local simulation state; no physical controller is driven."""
    __tablename__ = "signal_simulation_state"

    id = Column(Integer, primary_key=True)
    junction_id = Column(String(80), nullable=False, unique=True, index=True)
    active = Column(Boolean, nullable=False, default=False, index=True)
    current_phase_id = Column(String(80), nullable=True)
    phase_kind = Column(String(20), nullable=False, default="GREEN")
    remaining_seconds = Column(Integer, nullable=False, default=0)
    cycle_plan_json = Column(Text, nullable=False, default="[]")
    cursor_index = Column(Integer, nullable=False, default=0)
    mode = Column(String(40), nullable=False, default="SIMULATION MODE")
    updated_at = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)


def _sanitize_plate(value):
    from app.plate_rules import strip_hsrp_noise
    return strip_hsrp_noise(value)


def trigram_similarity(left, right):
    """PostgreSQL pg_trgm-style fallback for local SQLite/demo mode."""
    left = _sanitize_plate(left)
    right = _sanitize_plate(right)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0

    def trigrams(value):
        padded = f"  {value} "
        return {padded[index:index + 3] for index in range(len(padded) - 2)}

    left_grams = trigrams(left)
    right_grams = trigrams(right)
    if not left_grams or not right_grams:
        return 0.0
    return len(left_grams & right_grams) / float(len(left_grams | right_grams))


def levenshtein_similarity(left, right):
    left = _sanitize_plate(left)
    right = _sanitize_plate(right)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    if Levenshtein is not None:
        return Levenshtein.normalized_similarity(left, right)
    return SequenceMatcher(None, left, right).ratio()


def plate_similarity_score(left, right):
    """Return a fuzzy plate score compatible with the 0.70 review threshold."""
    return round(max(trigram_similarity(left, right), levenshtein_similarity(left, right)), 4)


def plates_match(left, right, threshold=0.70):
    return plate_similarity_score(left, right) >= threshold


def stable_vehicle_id_for_plate(plate_text):
    from app.plate_rules import normalize_plate_text
    normalized = normalize_plate_text(plate_text).normalized_text
    if not normalized:
        return None
    return "GVH-" + uuid.uuid5(uuid.NAMESPACE_URL, f"anpr.vehicle:{normalized}").hex[:12].upper()


def _confirmed_vehicle_plate(plate_text, confidence, status):
    from app.plate_rules import normalize_plate_text, OCR_ACCEPT_CONFIDENCE, PENDING_REVIEW_STATUS
    rule = normalize_plate_text(plate_text)
    if not rule.valid_format or not rule.normalized_text:
        return None
    if status == PENDING_REVIEW_STATUS or float(confidence or 0.0) < OCR_ACCEPT_CONFIDENCE:
        return None
    return rule.normalized_text


def ensure_vehicle_for_event(db, event):
    """Attach a global vehicle only to confirmed, complete plate observations."""
    normalized = _confirmed_vehicle_plate(event.plate_text, event.confidence, event.status)
    if not normalized:
        event.vehicle_id = None
        return None
    vehicle_id = stable_vehicle_id_for_plate(normalized)
    vehicle = db.get(Vehicle, vehicle_id)
    now = datetime.datetime.utcnow()
    if vehicle is None:
        vehicle = Vehicle(vehicle_id=vehicle_id, primary_plate_text=normalized,
                          created_at=now, updated_at=now)
        db.add(vehicle)
        db.flush()
    else:
        vehicle.updated_at = now
    event.vehicle_id = vehicle.vehicle_id
    return vehicle


def backfill_vehicle_ids(db):
    """Populate vehicle links for old confirmed events after the schema is upgraded."""
    changed = False
    events = (
        db.query(PlateEvent)
        .filter(PlateEvent.vehicle_id.is_(None), PlateEvent.plate_text.isnot(None))
        .all()
    )
    for event in events:
        if ensure_vehicle_for_event(db, event):
            changed = True
    if changed:
        db.commit()


Base.metadata.create_all(bind=engine)


def _ensure_sqlite_columns():
    """Add fields introduced after the original demo database was created."""
    if not DATABASE_URL.startswith("sqlite"):
        return
    columns = {
        "stream_url": "VARCHAR", "stream_type": "VARCHAR", "active": "BOOLEAN DEFAULT 0",
        "last_seen": "DATETIME", "location_known": "BOOLEAN DEFAULT 1",
        "edge_device_id": "VARCHAR", "edge_status": "VARCHAR DEFAULT 'OFFLINE'",
        "edge_last_processed": "DATETIME", "edge_config_json": "TEXT",
        "frames_sampled": "INTEGER DEFAULT 0", "observations_produced": "INTEGER DEFAULT 0",
        "processing_errors": "INTEGER DEFAULT 0",
    }
    event_columns = {
        "track_id": "VARCHAR", "plate_category": "VARCHAR", "layout": "VARCHAR",
        "rule_violations": "TEXT", "bbox_x": "FLOAT", "bbox_y": "FLOAT",
        "bbox_width": "FLOAT", "bbox_height": "FLOAT", "velocity_x": "FLOAT",
        "velocity_y": "FLOAT", "vehicle_id": "VARCHAR", "vehicle_type": "VARCHAR",
        "vehicle_color": "VARCHAR", "vehicle_crop_path": "VARCHAR", "appearance_embedding": "TEXT",
        "appearance_model": "VARCHAR", "appearance_embedding_version": "VARCHAR",
        "appearance_quality": "FLOAT", "original_image_path": "VARCHAR", "privacy_image_path": "VARCHAR",
        "privacy_status": "VARCHAR", "privacy_metadata": "TEXT", "privacy_processed_at": "DATETIME",
        "original_image_path_ciphertext": "TEXT", "original_image_path_nonce": "VARCHAR",
        "original_image_path_key_version": "VARCHAR", "privacy_metadata_ciphertext": "TEXT",
        "privacy_metadata_nonce": "VARCHAR", "privacy_metadata_key_version": "VARCHAR",
    }
    edge_device_columns = {
        "credential_hash": "VARCHAR", "credential_version": "VARCHAR",
    }
    with engine.begin() as connection:
        connection.exec_driver_sql("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                username VARCHAR(80) NOT NULL UNIQUE,
                password_hash VARCHAR(255) NOT NULL,
                role VARCHAR(40) NOT NULL,
                active BOOLEAN NOT NULL DEFAULT 1,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL
            )
        """)
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_users_role ON users(role)")
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_users_active ON users(active)")
        connection.exec_driver_sql("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY,
                audit_id VARCHAR(36) NOT NULL UNIQUE,
                timestamp DATETIME NOT NULL,
                user_id INTEGER,
                username VARCHAR(80),
                role VARCHAR(40),
                action VARCHAR(80) NOT NULL,
                resource_type VARCHAR(80),
                resource_id VARCHAR(120),
                source TEXT,
                success BOOLEAN NOT NULL DEFAULT 1,
                reason VARCHAR(240),
                details TEXT,
                previous_hash VARCHAR(64) NOT NULL,
                current_hash VARCHAR(64) NOT NULL
            )
        """)
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp)")
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id)")
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_audit_logs_username ON audit_logs(username)")
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_audit_logs_role ON audit_logs(role)")
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs(action)")
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_audit_logs_resource ON audit_logs(resource_type, resource_id)")
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_audit_logs_success ON audit_logs(success)")
        connection.exec_driver_sql("""
            CREATE TABLE IF NOT EXISTS edge_devices (
                id INTEGER PRIMARY KEY,
                edge_device_id VARCHAR(80) NOT NULL UNIQUE,
                label VARCHAR(120),
                status VARCHAR(20) NOT NULL DEFAULT 'OFFLINE',
                last_seen_at DATETIME,
                last_processed_at DATETIME,
                config_json TEXT,
                queue_mode VARCHAR(30) NOT NULL DEFAULT 'local',
                frames_sampled INTEGER NOT NULL DEFAULT 0,
                observations_received INTEGER NOT NULL DEFAULT 0,
                observations_processed INTEGER NOT NULL DEFAULT 0,
                observations_failed INTEGER NOT NULL DEFAULT 0,
                duplicate_observations INTEGER NOT NULL DEFAULT 0,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL
            )
        """)
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_edge_devices_device ON edge_devices(edge_device_id)")
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_edge_devices_status ON edge_devices(status)")
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_edge_devices_last_seen ON edge_devices(last_seen_at)")
        existing = {row[1] for row in connection.exec_driver_sql("PRAGMA table_info(edge_devices)")}
        for name, definition in edge_device_columns.items():
            if name not in existing:
                connection.exec_driver_sql(f"ALTER TABLE edge_devices ADD COLUMN {name} {definition}")
        connection.exec_driver_sql("""
            CREATE TABLE IF NOT EXISTS edge_observations (
                id INTEGER PRIMARY KEY,
                observation_id VARCHAR(120) NOT NULL UNIQUE,
                camera_id VARCHAR NOT NULL,
                edge_device_id VARCHAR(80) NOT NULL,
                frame_id VARCHAR(120),
                sequence_number INTEGER,
                capture_timestamp DATETIME NOT NULL,
                received_at DATETIME NOT NULL,
                processing_started_at DATETIME,
                processing_completed_at DATETIME,
                status VARCHAR(30) NOT NULL DEFAULT 'RECEIVED',
                retry_count INTEGER NOT NULL DEFAULT 0,
                max_retries INTEGER NOT NULL DEFAULT 3,
                next_retry_at DATETIME,
                error_message VARCHAR(500),
                payload_json TEXT NOT NULL,
                plate_event_id INTEGER,
                queue_name VARCHAR(80),
                processing_latency_ms FLOAT,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL
            )
        """)
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_edge_observations_id ON edge_observations(observation_id)")
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_edge_observations_camera ON edge_observations(camera_id)")
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_edge_observations_device ON edge_observations(edge_device_id)")
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_edge_observations_capture ON edge_observations(capture_timestamp)")
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_edge_observations_status ON edge_observations(status)")
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_edge_observations_event ON edge_observations(plate_event_id)")
        connection.exec_driver_sql("""
            CREATE TABLE IF NOT EXISTS retention_runs (
                id INTEGER PRIMARY KEY,
                run_id VARCHAR(36) NOT NULL UNIQUE,
                dry_run BOOLEAN NOT NULL DEFAULT 1,
                policy_json TEXT NOT NULL,
                result_json TEXT NOT NULL,
                status VARCHAR(30) NOT NULL DEFAULT 'completed',
                started_at DATETIME NOT NULL,
                completed_at DATETIME,
                actor VARCHAR(80)
            )
        """)
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_retention_runs_run ON retention_runs(run_id)")
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_retention_runs_dry_run ON retention_runs(dry_run)")
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_retention_runs_status ON retention_runs(status)")
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_retention_runs_started ON retention_runs(started_at)")
        connection.exec_driver_sql("""
            CREATE TABLE IF NOT EXISTS traffic_junctions (
                id INTEGER PRIMARY KEY,
                junction_id VARCHAR(80) NOT NULL UNIQUE,
                name VARCHAR(160) NOT NULL,
                lat FLOAT,
                lng FLOAT,
                camera_ids_json TEXT NOT NULL DEFAULT '[]',
                active BOOLEAN NOT NULL DEFAULT 1,
                controller_mode VARCHAR(30) NOT NULL DEFAULT 'simulation',
                controller_url VARCHAR(500),
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL
            )
        """)
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_traffic_junctions_junction ON traffic_junctions(junction_id)")
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_traffic_junctions_active ON traffic_junctions(active)")
        connection.exec_driver_sql("""
            CREATE TABLE IF NOT EXISTS signal_phases (
                id INTEGER PRIMARY KEY,
                phase_id VARCHAR(80) NOT NULL,
                junction_id VARCHAR(80) NOT NULL,
                movement VARCHAR(120) NOT NULL,
                movement_camera_ids_json TEXT NOT NULL DEFAULT '[]',
                min_green_seconds INTEGER NOT NULL DEFAULT 15,
                max_green_seconds INTEGER NOT NULL DEFAULT 90,
                yellow_seconds INTEGER NOT NULL DEFAULT 4,
                all_red_seconds INTEGER NOT NULL DEFAULT 2,
                current_state VARCHAR(30) NOT NULL DEFAULT 'RED',
                enabled BOOLEAN NOT NULL DEFAULT 1,
                display_order INTEGER NOT NULL DEFAULT 0,
                last_served_at DATETIME,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL,
                UNIQUE(junction_id, phase_id)
            )
        """)
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_signal_phases_junction ON signal_phases(junction_id)")
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_signal_phases_phase ON signal_phases(phase_id)")
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_signal_phases_enabled ON signal_phases(enabled)")
        connection.exec_driver_sql("""
            CREATE TABLE IF NOT EXISTS signal_recommendations (
                id INTEGER PRIMARY KEY,
                recommendation_id VARCHAR(36) NOT NULL UNIQUE,
                junction_id VARCHAR(80) NOT NULL,
                created_at DATETIME NOT NULL,
                current_phase_id VARCHAR(80),
                demand_json TEXT NOT NULL,
                recommendation_json TEXT NOT NULL,
                data_quality VARCHAR(40) NOT NULL DEFAULT 'INSUFFICIENT_DATA',
                reliability FLOAT NOT NULL DEFAULT 0,
                applied BOOLEAN NOT NULL DEFAULT 0,
                applied_at DATETIME,
                actor VARCHAR(80)
            )
        """)
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_signal_recommendations_id ON signal_recommendations(recommendation_id)")
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_signal_recommendations_junction ON signal_recommendations(junction_id)")
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_signal_recommendations_created ON signal_recommendations(created_at)")
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_signal_recommendations_quality ON signal_recommendations(data_quality)")
        connection.exec_driver_sql("""
            CREATE TABLE IF NOT EXISTS signal_simulation_state (
                id INTEGER PRIMARY KEY,
                junction_id VARCHAR(80) NOT NULL UNIQUE,
                active BOOLEAN NOT NULL DEFAULT 0,
                current_phase_id VARCHAR(80),
                phase_kind VARCHAR(20) NOT NULL DEFAULT 'GREEN',
                remaining_seconds INTEGER NOT NULL DEFAULT 0,
                cycle_plan_json TEXT NOT NULL DEFAULT '[]',
                cursor_index INTEGER NOT NULL DEFAULT 0,
                mode VARCHAR(40) NOT NULL DEFAULT 'SIMULATION MODE',
                updated_at DATETIME NOT NULL
            )
        """)
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_signal_simulation_junction ON signal_simulation_state(junction_id)")
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_signal_simulation_active ON signal_simulation_state(active)")
        connection.exec_driver_sql("""
            CREATE TABLE IF NOT EXISTS vehicles (
                vehicle_id VARCHAR PRIMARY KEY,
                primary_plate_text VARCHAR(24) NOT NULL UNIQUE,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL
            )
        """)
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS ix_vehicles_primary_plate_text ON vehicles(primary_plate_text)")
        connection.exec_driver_sql("""
            CREATE TABLE IF NOT EXISTS vehicle_match_candidates (
                id INTEGER PRIMARY KEY,
                observation_a_id INTEGER NOT NULL,
                observation_b_id INTEGER NOT NULL,
                candidate_vehicle_id VARCHAR,
                camera_a_id VARCHAR,
                camera_b_id VARCHAR,
                plate_evidence TEXT,
                appearance_similarity FLOAT,
                vehicle_type_similarity FLOAT,
                vehicle_color_similarity FLOAT,
                time_delta_seconds FLOAT,
                spatial_relationship TEXT,
                factor_scores TEXT,
                factors_used TEXT,
                final_confidence FLOAT NOT NULL DEFAULT 0,
                state VARCHAR NOT NULL DEFAULT 'LOW_CONFIDENCE',
                decision VARCHAR NOT NULL DEFAULT 'no_association',
                review_status VARCHAR NOT NULL DEFAULT 'not_required',
                matching_model VARCHAR NOT NULL DEFAULT 'vehicle-match-v1',
                matching_version VARCHAR NOT NULL DEFAULT '2B-2026-09-12',
                created_at DATETIME NOT NULL,
                reviewed_at DATETIME,
                reviewed_by VARCHAR,
                CONSTRAINT uq_vehicle_match_pair UNIQUE (observation_a_id, observation_b_id)
            )
        """)
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_vehicle_match_a ON vehicle_match_candidates(observation_a_id)")
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_vehicle_match_b ON vehicle_match_candidates(observation_b_id)")
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_vehicle_match_vehicle ON vehicle_match_candidates(candidate_vehicle_id)")
        connection.exec_driver_sql("""
            CREATE TABLE IF NOT EXISTS camera_road_connections (
                id INTEGER PRIMARY KEY,
                source_camera_id VARCHAR NOT NULL,
                destination_camera_id VARCHAR NOT NULL,
                distance_meters FLOAT NOT NULL,
                distance_source VARCHAR NOT NULL DEFAULT 'manual',
                direction VARCHAR,
                road_name VARCHAR,
                road_type VARCHAR,
                provider VARCHAR NOT NULL DEFAULT 'manual',
                provider_reference VARCHAR,
                active BOOLEAN NOT NULL DEFAULT 1,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL,
                CONSTRAINT uq_camera_road_pair UNIQUE (source_camera_id, destination_camera_id)
            )
        """)
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_camera_road_source ON camera_road_connections(source_camera_id)")
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_camera_road_destination ON camera_road_connections(destination_camera_id)")
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_camera_road_active ON camera_road_connections(active)")
        connection.exec_driver_sql("""
            CREATE TABLE IF NOT EXISTS vehicle_anomalies (
                id INTEGER PRIMARY KEY,
                vehicle_id VARCHAR,
                plate_text VARCHAR(24),
                anomaly_type VARCHAR NOT NULL,
                severity VARCHAR NOT NULL,
                status VARCHAR NOT NULL DEFAULT 'active',
                source_event_id INTEGER NOT NULL,
                destination_event_id INTEGER NOT NULL,
                source_camera_id VARCHAR NOT NULL,
                destination_camera_id VARCHAR NOT NULL,
                detected_at DATETIME NOT NULL,
                travel_time_seconds FLOAT,
                road_distance_meters FLOAT,
                estimated_speed_kmh FLOAT,
                allowed_speed_kmh FLOAT,
                excess_ratio FLOAT,
                road_type VARCHAR,
                road_name VARCHAR,
                policy TEXT,
                explanation TEXT NOT NULL,
                created_at DATETIME NOT NULL,
                acknowledged_at DATETIME,
                resolved_at DATETIME,
                CONSTRAINT uq_vehicle_anomaly_pair UNIQUE (anomaly_type, source_event_id, destination_event_id)
            )
        """)
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_vehicle_anomaly_vehicle ON vehicle_anomalies(vehicle_id)")
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_vehicle_anomaly_plate ON vehicle_anomalies(plate_text)")
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_vehicle_anomaly_type ON vehicle_anomalies(anomaly_type)")
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_vehicle_anomaly_severity ON vehicle_anomalies(severity)")
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_vehicle_anomaly_status ON vehicle_anomalies(status)")
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_vehicle_anomaly_detected ON vehicle_anomalies(detected_at)")
        connection.exec_driver_sql("""
            CREATE TABLE IF NOT EXISTS plate_suspicion_events (
                id INTEGER PRIMARY KEY,
                plate_text VARCHAR(24) NOT NULL,
                vehicle_id VARCHAR,
                source_event_id INTEGER NOT NULL,
                destination_event_id INTEGER NOT NULL,
                source_camera_id VARCHAR,
                destination_camera_id VARCHAR,
                suspicion_score FLOAT NOT NULL DEFAULT 0.0,
                classification VARCHAR NOT NULL DEFAULT 'normal',
                status VARCHAR NOT NULL DEFAULT 'open',
                evidence TEXT NOT NULL,
                appearance_similarity FLOAT,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL,
                reviewed_at DATETIME,
                reviewed_by VARCHAR,
                CONSTRAINT uq_plate_suspicion_pair UNIQUE (plate_text, source_event_id, destination_event_id)
            )
        """)
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_plate_suspicion_plate ON plate_suspicion_events(plate_text)")
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_plate_suspicion_vehicle ON plate_suspicion_events(vehicle_id)")
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_plate_suspicion_classification ON plate_suspicion_events(classification)")
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_plate_suspicion_status ON plate_suspicion_events(status)")
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_plate_suspicion_created ON plate_suspicion_events(created_at)")
        connection.exec_driver_sql("""
            CREATE TABLE IF NOT EXISTS route_anomaly_events (
                id INTEGER PRIMARY KEY,
                vehicle_id VARCHAR NOT NULL,
                plate_text VARCHAR(24),
                start_event_id INTEGER NOT NULL,
                end_event_id INTEGER NOT NULL,
                start_camera_id VARCHAR,
                end_camera_id VARCHAR,
                route_signature VARCHAR NOT NULL,
                route_anomaly_score FLOAT NOT NULL DEFAULT 0.0,
                classification VARCHAR NOT NULL DEFAULT 'normal',
                status VARCHAR NOT NULL DEFAULT 'open',
                evidence TEXT NOT NULL,
                explanation TEXT NOT NULL,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL,
                reviewed_at DATETIME,
                reviewed_by VARCHAR,
                CONSTRAINT uq_route_anomaly_window UNIQUE (vehicle_id, start_event_id, end_event_id)
            )
        """)
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_route_anomaly_vehicle ON route_anomaly_events(vehicle_id)")
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_route_anomaly_plate ON route_anomaly_events(plate_text)")
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_route_anomaly_classification ON route_anomaly_events(classification)")
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_route_anomaly_status ON route_anomaly_events(status)")
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_route_anomaly_created ON route_anomaly_events(created_at)")
        existing = {row[1] for row in connection.exec_driver_sql("PRAGMA table_info(cameras)")}
        for name, definition in columns.items():
            if name not in existing:
                connection.exec_driver_sql(f"ALTER TABLE cameras ADD COLUMN {name} {definition}")
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_cameras_edge_device ON cameras(edge_device_id)")
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_cameras_edge_status ON cameras(edge_status)")
        existing = {row[1] for row in connection.exec_driver_sql("PRAGMA table_info(plate_events)")}
        for name, definition in event_columns.items():
            if name not in existing:
                connection.exec_driver_sql(f"ALTER TABLE plate_events ADD COLUMN {name} {definition}")
        # Older stream writers used ISO 'T'; keep SQLite's timestamp ordering consistent.
        connection.exec_driver_sql("UPDATE plate_events SET timestamp = REPLACE(timestamp, 'T', ' ') WHERE timestamp LIKE '%T%'")
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_plate_event_debounce ON plate_events(camera_id, plate_text, timestamp)")
        connection.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_plate_events_vehicle_id ON plate_events(vehicle_id)")


_ensure_sqlite_columns()


def persist_plate_event(values, window_seconds=None):
    """Serialize duplicate checks with insertion across threads and worker processes."""
    window_seconds = float(os.getenv("ANPR_DEDUP_SECONDS", "12")) if window_seconds is None else window_seconds
    window_seconds = max(0.0, window_seconds)
    values = dict(values)
    values["timestamp"] = datetime.datetime.utcnow()
    if values.get("track_id") is not None:
        values["track_id"] = str(values["track_id"])
    plate = values.get("plate_text")
    camera = values.get("camera_id")
    with SessionLocal() as db:
        if db.bind.dialect.name == "sqlite":
            db.connection().exec_driver_sql("BEGIN IMMEDIATE")
        elif db.bind.dialect.name == "postgresql":
            key = int.from_bytes(hashlib.sha256(str(camera).encode()).digest()[:8], "big", signed=True)
            db.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": key})
        previous = None
        if plate and window_seconds:
            previous = (db.query(PlateEvent)
                        .filter(PlateEvent.camera_id == camera, PlateEvent.plate_text == plate,
                                PlateEvent.timestamp >= values["timestamp"] - datetime.timedelta(seconds=window_seconds))
                        .order_by(PlateEvent.timestamp.desc()).first())
        if previous:
            if ((previous.status != "ok" and values.get("status") == "ok") or
                    (previous.status == values.get("status") and
                     (values.get("confidence") or 0) > (previous.confidence or 0))):
                for field in ("confidence", "status", "raw_ocr_candidates", "image_path", "original_image_path",
                              "original_image_path_ciphertext", "original_image_path_nonce",
                              "original_image_path_key_version",
                              "privacy_image_path",
                              "privacy_status", "privacy_metadata", "privacy_metadata_ciphertext",
                              "privacy_metadata_nonce", "privacy_metadata_key_version",
                              "privacy_processed_at", "rule_violations",
                              "plate_category", "layout", "bbox_x", "bbox_y", "bbox_width", "bbox_height",
                              "vehicle_type", "vehicle_color", "vehicle_crop_path", "appearance_embedding",
                              "appearance_model", "appearance_embedding_version", "appearance_quality"):
                    if values.get(field) is not None:
                        setattr(previous, field, values[field])
            ensure_vehicle_for_event(db, previous)
            from app.hotlist import match_event
            match_event(db, previous)
            from app.vehicle_matching import process_observation_matches
            process_observation_matches(db, previous)
            from app.anomalies import process_event_anomalies
            process_event_anomalies(db, previous)
            from app.plate_suspicion import process_plate_suspicions
            process_plate_suspicions(db, previous)
            from app.route_anomaly import process_route_anomalies
            process_route_anomalies(db, previous)
            db.commit()
            return previous, True
        event = PlateEvent(**values)
        db.add(event)
        db.flush()
        ensure_vehicle_for_event(db, event)
        from app.hotlist import match_event
        match_event(db, event)
        from app.vehicle_matching import process_observation_matches
        process_observation_matches(db, event)
        from app.anomalies import process_event_anomalies
        process_event_anomalies(db, event)
        from app.plate_suspicion import process_plate_suspicions
        process_plate_suspicions(db, event)
        from app.route_anomaly import process_route_anomalies
        process_route_anomalies(db, event)
        db.commit()
        return event, False


class AsyncPlateEventWriter:
    """Use the same transactional debouncer for captured streams and photo uploads."""
    def __init__(self, database_path=None):
        self.queue = None
        self._task = None
        self._loop = None
        self.failures = 0

    async def start(self):
        self._loop = asyncio.get_running_loop()
        self.queue = asyncio.Queue(maxsize=1024)
        self._task = None
        self._task = asyncio.create_task(self._run())

    async def stop(self):
        if self._task:
            await self.queue.put(None)
            await self._task
        self._task = None
        self.queue = None

    def enqueue_from_thread(self, event):
        if self._loop and self.queue is not None and self._task and not self._task.done():
            self._loop.call_soon_threadsafe(self._enqueue, event)

    def _enqueue(self, event):
        try:
            self.queue.put_nowait(event)
        except asyncio.QueueFull:
            self.failures += 1
            logging.warning("Live event queue full; skipping event")

    def _write_event_sync(self, event):
        return persist_plate_event(event)

    async def _run(self):
        while True:
            event = await self.queue.get()
            if event is None:
                return
            try:
                await asyncio.to_thread(self._write_event_sync, event)
            except Exception:
                self.failures += 1
                logging.exception("Live event persistence failed")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
