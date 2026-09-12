CREATE TABLE IF NOT EXISTS vehicle_match_candidates (
    id INTEGER PRIMARY KEY,
    observation_a_id INTEGER NOT NULL,
    observation_b_id INTEGER NOT NULL,
    candidate_vehicle_id VARCHAR,
    camera_a_id VARCHAR,
    camera_b_id VARCHAR,
    plate_evidence TEXT,
    appearance_similarity FLOAT,
    vehicle_type_similarity FLOAT,
    vehicle_color_similarity FLOAT,
    time_delta_seconds FLOAT,
    spatial_relationship TEXT,
    factor_scores TEXT,
    factors_used TEXT,
    final_confidence FLOAT NOT NULL DEFAULT 0,
    state VARCHAR NOT NULL DEFAULT 'LOW_CONFIDENCE',
    decision VARCHAR NOT NULL DEFAULT 'no_association',
    review_status VARCHAR NOT NULL DEFAULT 'not_required',
    matching_model VARCHAR NOT NULL DEFAULT 'vehicle-match-v1',
    matching_version VARCHAR NOT NULL DEFAULT '2B-2026-09-12',
    created_at DATETIME NOT NULL,
    reviewed_at DATETIME,
    reviewed_by VARCHAR,
    CONSTRAINT uq_vehicle_match_pair UNIQUE (observation_a_id, observation_b_id)
);

CREATE INDEX IF NOT EXISTS idx_vehicle_match_a ON vehicle_match_candidates(observation_a_id);
CREATE INDEX IF NOT EXISTS idx_vehicle_match_b ON vehicle_match_candidates(observation_b_id);
CREATE INDEX IF NOT EXISTS idx_vehicle_match_vehicle ON vehicle_match_candidates(candidate_vehicle_id);
