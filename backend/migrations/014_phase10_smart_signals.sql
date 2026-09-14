-- Phase 10: Smart traffic signal recommendation + safe signal simulation.
-- This is an SIH/demo recommendation layer. It does not control physical traffic lights.

CREATE TABLE IF NOT EXISTS traffic_junctions (
    id INTEGER PRIMARY KEY,
    junction_id VARCHAR(80) NOT NULL UNIQUE,
    name VARCHAR(160) NOT NULL,
    lat FLOAT,
    lng FLOAT,
    camera_ids_json TEXT NOT NULL DEFAULT '[]',
    active BOOLEAN NOT NULL DEFAULT 1,
    controller_mode VARCHAR(30) NOT NULL DEFAULT 'simulation',
    controller_url VARCHAR(500),
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_traffic_junctions_junction ON traffic_junctions(junction_id);
CREATE INDEX IF NOT EXISTS idx_traffic_junctions_active ON traffic_junctions(active);

CREATE TABLE IF NOT EXISTS signal_phases (
    id INTEGER PRIMARY KEY,
    phase_id VARCHAR(80) NOT NULL,
    junction_id VARCHAR(80) NOT NULL,
    movement VARCHAR(120) NOT NULL,
    movement_camera_ids_json TEXT NOT NULL DEFAULT '[]',
    min_green_seconds INTEGER NOT NULL DEFAULT 15,
    max_green_seconds INTEGER NOT NULL DEFAULT 90,
    yellow_seconds INTEGER NOT NULL DEFAULT 4,
    all_red_seconds INTEGER NOT NULL DEFAULT 2,
    current_state VARCHAR(30) NOT NULL DEFAULT 'RED',
    enabled BOOLEAN NOT NULL DEFAULT 1,
    display_order INTEGER NOT NULL DEFAULT 0,
    last_served_at DATETIME,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    UNIQUE(junction_id, phase_id)
);

CREATE INDEX IF NOT EXISTS idx_signal_phases_junction ON signal_phases(junction_id);
CREATE INDEX IF NOT EXISTS idx_signal_phases_phase ON signal_phases(phase_id);
CREATE INDEX IF NOT EXISTS idx_signal_phases_enabled ON signal_phases(enabled);

CREATE TABLE IF NOT EXISTS signal_recommendations (
    id INTEGER PRIMARY KEY,
    recommendation_id VARCHAR(36) NOT NULL UNIQUE,
    junction_id VARCHAR(80) NOT NULL,
    created_at DATETIME NOT NULL,
    current_phase_id VARCHAR(80),
    demand_json TEXT NOT NULL,
    recommendation_json TEXT NOT NULL,
    data_quality VARCHAR(40) NOT NULL DEFAULT 'INSUFFICIENT_DATA',
    reliability FLOAT NOT NULL DEFAULT 0,
    applied BOOLEAN NOT NULL DEFAULT 0,
    applied_at DATETIME,
    actor VARCHAR(80)
);

CREATE INDEX IF NOT EXISTS idx_signal_recommendations_id ON signal_recommendations(recommendation_id);
CREATE INDEX IF NOT EXISTS idx_signal_recommendations_junction ON signal_recommendations(junction_id);
CREATE INDEX IF NOT EXISTS idx_signal_recommendations_created ON signal_recommendations(created_at);
CREATE INDEX IF NOT EXISTS idx_signal_recommendations_quality ON signal_recommendations(data_quality);

CREATE TABLE IF NOT EXISTS signal_simulation_state (
    id INTEGER PRIMARY KEY,
    junction_id VARCHAR(80) NOT NULL UNIQUE,
    active BOOLEAN NOT NULL DEFAULT 0,
    current_phase_id VARCHAR(80),
    phase_kind VARCHAR(20) NOT NULL DEFAULT 'GREEN',
    remaining_seconds INTEGER NOT NULL DEFAULT 0,
    cycle_plan_json TEXT NOT NULL DEFAULT '[]',
    cursor_index INTEGER NOT NULL DEFAULT 0,
    mode VARCHAR(40) NOT NULL DEFAULT 'SIMULATION MODE',
    updated_at DATETIME NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_signal_simulation_junction ON signal_simulation_state(junction_id);
CREATE INDEX IF NOT EXISTS idx_signal_simulation_active ON signal_simulation_state(active);
