-- Phase 9: Edge processing, scalable metadata ingestion, and retention lifecycle.
-- This migration is additive and preserves Phases 1-8 data.

ALTER TABLE cameras ADD COLUMN edge_device_id VARCHAR;
ALTER TABLE cameras ADD COLUMN edge_status VARCHAR DEFAULT 'OFFLINE';
ALTER TABLE cameras ADD COLUMN edge_last_processed DATETIME;
ALTER TABLE cameras ADD COLUMN edge_config_json TEXT;
ALTER TABLE cameras ADD COLUMN frames_sampled INTEGER DEFAULT 0;
ALTER TABLE cameras ADD COLUMN observations_produced INTEGER DEFAULT 0;
ALTER TABLE cameras ADD COLUMN processing_errors INTEGER DEFAULT 0;

CREATE INDEX IF NOT EXISTS idx_cameras_edge_device ON cameras(edge_device_id);
CREATE INDEX IF NOT EXISTS idx_cameras_edge_status ON cameras(edge_status);

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
);

CREATE INDEX IF NOT EXISTS idx_edge_devices_device ON edge_devices(edge_device_id);
CREATE INDEX IF NOT EXISTS idx_edge_devices_status ON edge_devices(status);
CREATE INDEX IF NOT EXISTS idx_edge_devices_last_seen ON edge_devices(last_seen_at);

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
);

CREATE INDEX IF NOT EXISTS idx_edge_observations_id ON edge_observations(observation_id);
CREATE INDEX IF NOT EXISTS idx_edge_observations_camera ON edge_observations(camera_id);
CREATE INDEX IF NOT EXISTS idx_edge_observations_device ON edge_observations(edge_device_id);
CREATE INDEX IF NOT EXISTS idx_edge_observations_capture ON edge_observations(capture_timestamp);
CREATE INDEX IF NOT EXISTS idx_edge_observations_status ON edge_observations(status);
CREATE INDEX IF NOT EXISTS idx_edge_observations_event ON edge_observations(plate_event_id);

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
);

CREATE INDEX IF NOT EXISTS idx_retention_runs_run ON retention_runs(run_id);
CREATE INDEX IF NOT EXISTS idx_retention_runs_dry_run ON retention_runs(dry_run);
CREATE INDEX IF NOT EXISTS idx_retention_runs_status ON retention_runs(status);
CREATE INDEX IF NOT EXISTS idx_retention_runs_started ON retention_runs(started_at);
