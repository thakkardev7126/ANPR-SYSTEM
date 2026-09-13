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
);

CREATE INDEX IF NOT EXISTS idx_vehicle_anomaly_vehicle ON vehicle_anomalies(vehicle_id);
CREATE INDEX IF NOT EXISTS idx_vehicle_anomaly_plate ON vehicle_anomalies(plate_text);
CREATE INDEX IF NOT EXISTS idx_vehicle_anomaly_type ON vehicle_anomalies(anomaly_type);
CREATE INDEX IF NOT EXISTS idx_vehicle_anomaly_severity ON vehicle_anomalies(severity);
CREATE INDEX IF NOT EXISTS idx_vehicle_anomaly_status ON vehicle_anomalies(status);
CREATE INDEX IF NOT EXISTS idx_vehicle_anomaly_detected ON vehicle_anomalies(detected_at);
