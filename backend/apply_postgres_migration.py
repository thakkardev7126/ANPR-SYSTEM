from pathlib import Path
import os

from sqlalchemy import create_engine


project_root = Path(__file__).resolve().parent.parent
migration_paths = sorted((project_root / "backend" / "migrations").glob("*.sql"))
database_url = os.getenv("ANPR_DATABASE_URL")

if not database_url:
    raise SystemExit("Set ANPR_DATABASE_URL before running this migration.")
if not database_url.startswith("postgresql"):
    raise SystemExit("This migration is only for PostgreSQL/PostGIS databases.")

engine = create_engine(database_url)
with engine.begin() as connection:
    for migration_path in migration_paths:
        # Execute each file intact so SQL bodies and comments may contain semicolons.
        connection.exec_driver_sql(migration_path.read_text(encoding="utf-8"))

for migration_path in migration_paths:
    print(f"Applied migration: {migration_path}")
