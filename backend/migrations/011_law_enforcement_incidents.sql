-- Phase 7: assistive law-enforcement / incident-command workflow.
-- New tables only; HotlistAlert remains the source of truth for hotlist hits.
CREATE TABLE IF NOT EXISTS pcr_vehicles (
    pcr_id VARCHAR(32) PRIMARY KEY,
    name VARCHAR(80) NOT NULL,
    call_sign VARCHAR(80),
    vehicle_identifier VARCHAR(80) NOT NULL DEFAULT '',
    lat FLOAT,
    lng FLOAT,
    status VARCHAR(20) NOT NULL DEFAULT 'AVAILABLE',
    active BOOLEAN NOT NULL DEFAULT TRUE,
    last_seen_at TIMESTAMP,
    contact_channel VARCHAR(120) NOT NULL DEFAULT '',
    metadata_json TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS enforcement_incidents (
    id SERIAL PRIMARY KEY,
    hotlist_alert_id INTEGER NOT NULL,
    vehicle_id VARCHAR,
    plate_text VARCHAR(24) NOT NULL,
    camera_id VARCHAR NOT NULL,
    event_id INTEGER,
    detected_at TIMESTAMP NOT NULL,
    recommended_pcr_id VARCHAR(32),
    pcr_distance_meters FLOAT,
    pcr_location_status VARCHAR(20) NOT NULL DEFAULT 'UNKNOWN',
    recommendation_status VARCHAR(40) NOT NULL DEFAULT 'NO_AVAILABLE_PCR',
    recommendation_reason VARCHAR(240) NOT NULL DEFAULT '',
    recommendation_calculated_at TIMESTAMP,
    status VARCHAR(40) NOT NULL DEFAULT 'NEW',
    trajectory_reference VARCHAR(240) NOT NULL DEFAULT '',
    investigation_reference VARCHAR(240) NOT NULL DEFAULT '',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_by VARCHAR(80),
    CONSTRAINT uq_enforcement_hotlist_alert UNIQUE (hotlist_alert_id)
);

CREATE INDEX IF NOT EXISTS idx_pcr_status ON pcr_vehicles(status);
CREATE INDEX IF NOT EXISTS idx_pcr_active ON pcr_vehicles(active);
CREATE INDEX IF NOT EXISTS idx_pcr_last_seen ON pcr_vehicles(last_seen_at);
CREATE INDEX IF NOT EXISTS idx_enforcement_alert ON enforcement_incidents(hotlist_alert_id);
CREATE INDEX IF NOT EXISTS idx_enforcement_vehicle ON enforcement_incidents(vehicle_id);
CREATE INDEX IF NOT EXISTS idx_enforcement_plate ON enforcement_incidents(plate_text);
CREATE INDEX IF NOT EXISTS idx_enforcement_camera ON enforcement_incidents(camera_id);
CREATE INDEX IF NOT EXISTS idx_enforcement_event ON enforcement_incidents(event_id);
CREATE INDEX IF NOT EXISTS idx_enforcement_detected ON enforcement_incidents(detected_at);
CREATE INDEX IF NOT EXISTS idx_enforcement_pcr ON enforcement_incidents(recommended_pcr_id);
CREATE INDEX IF NOT EXISTS idx_enforcement_recommendation ON enforcement_incidents(recommendation_status);
CREATE INDEX IF NOT EXISTS idx_enforcement_status ON enforcement_incidents(status);
