-- New tables only; existing camera and scan data are preserved.
CREATE TABLE IF NOT EXISTS hotlist_entries (
    id SERIAL PRIMARY KEY,
    plate_text VARCHAR(24) NOT NULL UNIQUE,
    category VARCHAR(20) NOT NULL DEFAULT 'stolen',
    reason VARCHAR(500) NOT NULL,
    reference VARCHAR(120) NOT NULL DEFAULT '',
    active BOOLEAN NOT NULL DEFAULT TRUE,
    expires_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS hotlist_alerts (
    id SERIAL PRIMARY KEY,
    event_id INTEGER,
    hotlist_id INTEGER NOT NULL,
    plate_text VARCHAR(24) NOT NULL,
    camera_id VARCHAR NOT NULL,
    camera_label VARCHAR NOT NULL,
    lat FLOAT,
    lng FLOAT,
    category VARCHAR(20) NOT NULL,
    reason VARCHAR(500) NOT NULL,
    reference VARCHAR(120) NOT NULL,
    confidence FLOAT NOT NULL,
    match_status VARCHAR(20) NOT NULL,
    seen_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    acknowledged_at TIMESTAMP,
    revision INTEGER NOT NULL DEFAULT 1,
    CONSTRAINT uq_hotlist_event UNIQUE (event_id, hotlist_id)
);
CREATE TABLE IF NOT EXISTS hotlist_notifications (
    id SERIAL PRIMARY KEY,
    alert_id INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_hotlist_entries_plate_text ON hotlist_entries(plate_text);
CREATE INDEX IF NOT EXISTS ix_hotlist_alerts_event_id ON hotlist_alerts(event_id);
CREATE INDEX IF NOT EXISTS ix_hotlist_alerts_hotlist_id ON hotlist_alerts(hotlist_id);
CREATE INDEX IF NOT EXISTS ix_hotlist_notifications_alert_id ON hotlist_notifications(alert_id);
