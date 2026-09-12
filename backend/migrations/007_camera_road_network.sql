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
);

CREATE INDEX IF NOT EXISTS idx_camera_road_source ON camera_road_connections(source_camera_id);
CREATE INDEX IF NOT EXISTS idx_camera_road_destination ON camera_road_connections(destination_camera_id);
CREATE INDEX IF NOT EXISTS idx_camera_road_active ON camera_road_connections(active);
