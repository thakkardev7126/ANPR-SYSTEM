-- PostgreSQL/PostGIS production migration for ANPR fuzzy + spatial search.
-- Run this against a PostgreSQL database, then start the app with:
--   ANPR_DATABASE_URL=postgresql+psycopg://user:password@host:5432/anpr

CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

ALTER TABLE IF EXISTS plate_events
    ADD COLUMN IF NOT EXISTS plate_category VARCHAR,
    ADD COLUMN IF NOT EXISTS layout VARCHAR,
    ADD COLUMN IF NOT EXISTS rule_violations TEXT;

CREATE INDEX IF NOT EXISTS idx_plate_events_plate_trgm
    ON plate_events USING gin (plate_text gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_plate_events_timestamp
    ON plate_events (timestamp);

CREATE INDEX IF NOT EXISTS idx_cameras_location_geog
    ON cameras USING gist (
        (CAST(ST_SetSRID(ST_MakePoint(lng, lat), 4326) AS geography))
    );

CREATE OR REPLACE VIEW vehicle_sightings AS
SELECT
    e.id AS event_id,
    e.camera_id,
    c.label AS camera_label,
    e.plate_text AS plate_number,
    e.confidence,
    e.status,
    e.timestamp,
    e.image_path,
    e.plate_category,
    e.layout,
    CAST(ST_SetSRID(ST_MakePoint(c.lng, c.lat), 4326) AS geography) AS location
FROM plate_events e
JOIN cameras c ON c.camera_id = e.camera_id
WHERE e.plate_text IS NOT NULL;
