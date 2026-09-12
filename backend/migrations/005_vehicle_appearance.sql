ALTER TABLE plate_events ADD COLUMN IF NOT EXISTS vehicle_type VARCHAR;
ALTER TABLE plate_events ADD COLUMN IF NOT EXISTS vehicle_color VARCHAR;
ALTER TABLE plate_events ADD COLUMN IF NOT EXISTS vehicle_crop_path VARCHAR;
ALTER TABLE plate_events ADD COLUMN IF NOT EXISTS appearance_embedding TEXT;
ALTER TABLE plate_events ADD COLUMN IF NOT EXISTS appearance_model VARCHAR;
ALTER TABLE plate_events ADD COLUMN IF NOT EXISTS appearance_embedding_version VARCHAR;
ALTER TABLE plate_events ADD COLUMN IF NOT EXISTS appearance_quality FLOAT;
