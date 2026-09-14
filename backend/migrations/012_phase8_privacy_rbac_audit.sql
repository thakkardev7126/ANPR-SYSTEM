-- Phase 8: privacy-safe image references, local RBAC users, and
-- tamper-evident application-level audit logging with SHA-256 hash chaining.

CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(80) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(40) NOT NULL,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id SERIAL PRIMARY KEY,
    audit_id VARCHAR(36) NOT NULL UNIQUE,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    user_id INTEGER,
    username VARCHAR(80),
    role VARCHAR(40),
    action VARCHAR(80) NOT NULL,
    resource_type VARCHAR(80),
    resource_id VARCHAR(120),
    source TEXT,
    success BOOLEAN NOT NULL DEFAULT TRUE,
    reason VARCHAR(240),
    details TEXT,
    previous_hash VARCHAR(64) NOT NULL,
    current_hash VARCHAR(64) NOT NULL
);

ALTER TABLE plate_events ADD COLUMN IF NOT EXISTS privacy_image_path VARCHAR;
ALTER TABLE plate_events ADD COLUMN IF NOT EXISTS original_image_path VARCHAR;
ALTER TABLE plate_events ADD COLUMN IF NOT EXISTS privacy_status VARCHAR;
ALTER TABLE plate_events ADD COLUMN IF NOT EXISTS privacy_metadata TEXT;
ALTER TABLE plate_events ADD COLUMN IF NOT EXISTS privacy_processed_at TIMESTAMP;

CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
CREATE INDEX IF NOT EXISTS idx_users_active ON users(active);
CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_username ON audit_logs(username);
CREATE INDEX IF NOT EXISTS idx_audit_logs_role ON audit_logs(role);
CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs(action);
CREATE INDEX IF NOT EXISTS idx_audit_logs_resource ON audit_logs(resource_type, resource_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_success ON audit_logs(success);
