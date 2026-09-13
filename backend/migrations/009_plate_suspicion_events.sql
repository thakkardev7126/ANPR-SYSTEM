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
);

CREATE INDEX IF NOT EXISTS idx_plate_suspicion_plate ON plate_suspicion_events(plate_text);
CREATE INDEX IF NOT EXISTS idx_plate_suspicion_vehicle ON plate_suspicion_events(vehicle_id);
CREATE INDEX IF NOT EXISTS idx_plate_suspicion_classification ON plate_suspicion_events(classification);
CREATE INDEX IF NOT EXISTS idx_plate_suspicion_status ON plate_suspicion_events(status);
CREATE INDEX IF NOT EXISTS idx_plate_suspicion_created ON plate_suspicion_events(created_at);
