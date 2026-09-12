-- Run before starting the updated app against PostgreSQL.
ALTER TABLE cameras ADD COLUMN IF NOT EXISTS location_known BOOLEAN NOT NULL DEFAULT TRUE;
CREATE INDEX IF NOT EXISTS idx_plate_event_debounce ON plate_events(camera_id, plate_text, timestamp);
