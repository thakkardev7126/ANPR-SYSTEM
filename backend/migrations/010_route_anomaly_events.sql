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
);

CREATE INDEX IF NOT EXISTS idx_route_anomaly_vehicle ON route_anomaly_events(vehicle_id);
CREATE INDEX IF NOT EXISTS idx_route_anomaly_plate ON route_anomaly_events(plate_text);
CREATE INDEX IF NOT EXISTS idx_route_anomaly_classification ON route_anomaly_events(classification);
CREATE INDEX IF NOT EXISTS idx_route_anomaly_status ON route_anomaly_events(status);
CREATE INDEX IF NOT EXISTS idx_route_anomaly_created ON route_anomaly_events(created_at);
