"""Release sanity checks for the ANPR SIH demo repository.

The checks are intentionally lightweight and local:
- verify migration filenames are ordered and unique
- check obvious PostgreSQL dialect regressions in migration files
- verify a temporary SQLite database can initialize from the current models
- report tracked runtime artifacts that should be removed from Git tracking

This script never modifies the user's working demo database.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIGRATION_RE = re.compile(r"^(\d{3})_.+\.sql$")
POSTGRESQL_INCOMPATIBLE_PATTERNS = (
    ("DATETIME", re.compile(r"\bDATETIME\b", re.IGNORECASE)),
    ("BOOLEAN DEFAULT 1/0", re.compile(r"\bBOOLEAN\b[^,\n;]*\bDEFAULT\s+[01]\b", re.IGNORECASE)),
    ("SQLite integer primary key", re.compile(r"\bid\s+INTEGER\s+PRIMARY\s+KEY\b", re.IGNORECASE)),
)

REQUIRED_TABLES = {
    "cameras",
    "plate_events",
    "vehicles",
    "camera_road_connections",
    "vehicle_match_candidates",
    "vehicle_anomalies",
    "plate_suspicion_events",
    "route_anomaly_events",
    "hotlist_entries",
    "hotlist_alerts",
    "users",
    "audit_logs",
    "edge_devices",
    "edge_observations",
    "traffic_junctions",
    "signal_phases",
    "signal_recommendations",
}

REQUIRED_COLUMNS = {
    "cameras": {"camera_id", "label", "lat", "lng", "location_known", "edge_status"},
    "plate_events": {
        "id",
        "camera_id",
        "plate_text",
        "timestamp",
        "vehicle_id",
        "appearance_embedding",
        "original_image_path_ciphertext",
        "privacy_metadata_ciphertext",
    },
    "vehicles": {"vehicle_id", "primary_plate_text"},
    "camera_road_connections": {"source_camera_id", "destination_camera_id", "distance_meters", "distance_source"},
    "edge_devices": {"edge_device_id", "credential_hash", "credential_version"},
}

RUNTIME_ARTIFACT_PATTERNS = (
    re.compile(r"(^|/)(__pycache__|uploads|evidence)/"),
    re.compile(r"\.pyc$"),
    re.compile(r"\.db$"),
    re.compile(r"^(dashboard|scanner|hotlist-popup)-.*\.png$"),
    re.compile(r"^(runs|saved_checkpoints)/"),
    re.compile(r"^data/browser-test.*\.db$"),
    re.compile(r"^data/review_feedback/"),
    re.compile(r"^data/plate-dataset/(?!SOURCES\.md$)"),
    re.compile(r"\.osrm(\.|$)"),
)


def migration_paths(project_root: Path = PROJECT_ROOT) -> list[Path]:
    return sorted((project_root / "backend" / "migrations").glob("*.sql"))


def migration_sequence_report(project_root: Path = PROJECT_ROOT) -> dict:
    paths = migration_paths(project_root)
    numbers: list[int] = []
    invalid: list[str] = []
    for path in paths:
        match = MIGRATION_RE.match(path.name)
        if not match:
            invalid.append(path.name)
            continue
        numbers.append(int(match.group(1)))
    duplicates = sorted({number for number in numbers if numbers.count(number) > 1})
    expected = list(range(1, max(numbers, default=0) + 1))
    missing = [number for number in expected if number not in numbers]
    return {
        "count": len(paths),
        "numbers": numbers,
        "invalid_names": invalid,
        "duplicates": duplicates,
        "missing": missing,
        "ordered": numbers == sorted(numbers),
    }


def migration_postgresql_compatibility_report(project_root: Path = PROJECT_ROOT) -> dict:
    issues: list[dict[str, str | int]] = []
    for path in migration_paths(project_root):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for label, pattern in POSTGRESQL_INCOMPATIBLE_PATTERNS:
                if pattern.search(line):
                    issues.append({"file": path.name, "line": line_number, "issue": label})
    return {"issue_count": len(issues), "issues": issues}


def tracked_files(project_root: Path = PROJECT_ROOT) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=True,
    )
    return [line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line.strip()]


def tracked_runtime_artifacts(project_root: Path = PROJECT_ROOT) -> list[str]:
    artifacts: list[str] = []
    for filename in tracked_files(project_root):
        normalized = filename.replace("\\", "/")
        if any(pattern.search(normalized) for pattern in RUNTIME_ARTIFACT_PATTERNS):
            artifacts.append(normalized)
    return artifacts


def sqlite_schema_report(database_path: Path) -> dict:
    with sqlite3.connect(database_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        columns = {
            table: {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
            for table in tables
        }
    missing_tables = sorted(REQUIRED_TABLES - tables)
    missing_columns = {
        table: sorted(required - columns.get(table, set()))
        for table, required in REQUIRED_COLUMNS.items()
        if required - columns.get(table, set())
    }
    return {
        "database_path": str(database_path),
        "table_count": len(tables),
        "missing_tables": missing_tables,
        "missing_columns": missing_columns,
    }


def fresh_sqlite_startup_report(project_root: Path = PROJECT_ROOT) -> dict:
    with tempfile.TemporaryDirectory(prefix="anpr-fresh-db-", ignore_cleanup_errors=True) as tmp:
        tmp_path = Path(tmp)
        database_path = tmp_path / "fresh.db"
        env = os.environ.copy()
        env.pop("ANPR_DATABASE_URL", None)
        env.update(
            {
                "ANPR_DATABASE_PATH": str(database_path),
                "ANPR_UPLOAD_DIR": str(tmp_path / "uploads"),
                "ANPR_ORIGINAL_EVIDENCE_DIR": str(tmp_path / "evidence" / "originals"),
                "ANPR_REVIEW_DIR": str(tmp_path / "review"),
                "ANPR_APPEARANCE_BACKEND": "opencv",
                "ANPR_APPEARANCE_DEVICE": "cpu",
                "PYTHONPATH": str(project_root / "backend"),
            }
        )
        code = """
import json
from sqlalchemy import inspect
from app.main import app
from app.database import engine, SessionLocal, Camera, User
with SessionLocal() as db:
    payload = {
        "tables": sorted(inspect(engine).get_table_names()),
        "camera_count": db.query(Camera).count(),
        "user_count": db.query(User).count(),
    }
print("ANPR_FRESH_SCHEMA_JSON:" + json.dumps(payload, sort_keys=True))
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=project_root,
            env=env,
            text=True,
            capture_output=True,
            timeout=120,
        )
        if result.returncode != 0:
            return {
                "ok": False,
                "returncode": result.returncode,
                "stdout": result.stdout[-2000:],
                "stderr": result.stderr[-2000:],
            }
        marker = "ANPR_FRESH_SCHEMA_JSON:"
        lines = [line for line in result.stdout.splitlines() if line.startswith(marker)]
        if not lines:
            return {"ok": False, "returncode": 0, "stdout": result.stdout[-2000:], "stderr": result.stderr[-2000:]}
        payload = json.loads(lines[-1][len(marker):])
        schema = sqlite_schema_report(database_path)
        payload.update(schema)
        payload["ok"] = not schema["missing_tables"] and not schema["missing_columns"]
        return payload


def main() -> int:
    migrations = migration_sequence_report()
    postgresql_compatibility = migration_postgresql_compatibility_report()
    fresh = fresh_sqlite_startup_report()
    artifacts = tracked_runtime_artifacts()
    report = {
        "migrations": migrations,
        "postgresql_migration_compatibility": postgresql_compatibility,
        "fresh_sqlite": fresh,
        "tracked_runtime_artifacts": artifacts[:200],
        "tracked_runtime_artifact_count": len(artifacts),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    failed = bool(
        migrations["invalid_names"]
        or migrations["duplicates"]
        or migrations["missing"]
        or not migrations["ordered"]
        or postgresql_compatibility["issues"]
        or not fresh.get("ok")
        or artifacts
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
