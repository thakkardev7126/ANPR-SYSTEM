-- Phase 11: final security hardening.
-- Adds metadata needed for AES-256-GCM at-rest protection and edge credential versioning.

ALTER TABLE plate_events ADD COLUMN IF NOT EXISTS original_image_path_ciphertext TEXT;
ALTER TABLE plate_events ADD COLUMN IF NOT EXISTS original_image_path_nonce VARCHAR(64);
ALTER TABLE plate_events ADD COLUMN IF NOT EXISTS original_image_path_key_version VARCHAR(40);
ALTER TABLE plate_events ADD COLUMN IF NOT EXISTS privacy_metadata_ciphertext TEXT;
ALTER TABLE plate_events ADD COLUMN IF NOT EXISTS privacy_metadata_nonce VARCHAR(64);
ALTER TABLE plate_events ADD COLUMN IF NOT EXISTS privacy_metadata_key_version VARCHAR(40);

ALTER TABLE edge_devices ADD COLUMN IF NOT EXISTS credential_hash VARCHAR(128);
ALTER TABLE edge_devices ADD COLUMN IF NOT EXISTS credential_version VARCHAR(40);

CREATE INDEX IF NOT EXISTS idx_plate_events_original_key_version
    ON plate_events(original_image_path_key_version);
CREATE INDEX IF NOT EXISTS idx_edge_devices_credential_version
    ON edge_devices(credential_version);
