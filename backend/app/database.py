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
    }
    event_columns = {
        "track_id": "VARCHAR", "plate_category": "VARCHAR", "layout": "VARCHAR",
        "rule_violations": "TEXT", "bbox_x": "FLOAT", "bbox_y": "FLOAT",
        "bbox_width": "FLOAT", "bbox_height": "FLOAT", "velocity_x": "FLOAT",
        "velocity_y": "FLOAT", "vehicle_id": "VARCHAR", "vehicle_type": "VARCHAR",
        "vehicle_color": "VARCHAR", "vehicle_crop_path": "VARCHAR", "appearance_embedding": "TEXT",
        "appearance_model": "VARCHAR", "appearance_embedding_version": "VARCHAR",
        "appearance_quality": "FLOAT",
    }
    with engine.begin() as connection:
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
        existing = {row[1] for row in connection.exec_driver_sql("PRAGMA table_info(cameras)")}
        for name, definition in columns.items():
            if name not in existing:
                connection.exec_driver_sql(f"ALTER TABLE cameras ADD COLUMN {name} {definition}")
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
                for field in ("confidence", "status", "raw_ocr_candidates", "image_path", "rule_violations",
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
