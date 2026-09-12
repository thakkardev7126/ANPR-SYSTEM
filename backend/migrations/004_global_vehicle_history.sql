CREATE TABLE IF NOT EXISTS vehicles (
    vehicle_id VARCHAR PRIMARY KEY,
    primary_plate_text VARCHAR(24) NOT NULL UNIQUE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE plate_events ADD COLUMN IF NOT EXISTS vehicle_id VARCHAR;

CREATE INDEX IF NOT EXISTS ix_vehicles_primary_plate_text ON vehicles(primary_plate_text);
CREATE INDEX IF NOT EXISTS idx_plate_events_vehicle_id ON plate_events(vehicle_id);
