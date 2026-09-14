"""Tamper-evident application-level audit logging."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import uuid

from sqlalchemy import text

from app.database import AuditLog, SessionLocal

GENESIS_HASH = "0" * 64


def _iso(value):
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.replace(microsecond=value.microsecond).isoformat()
    return str(value)


def canonical_audit_payload(values):
    payload = {
        "audit_id": values.get("audit_id"),
        "timestamp": _iso(values.get("timestamp")),
        "user_id": values.get("user_id"),
        "username": values.get("username"),
        "role": values.get("role"),
        "action": values.get("action"),
        "resource_type": values.get("resource_type"),
        "resource_id": values.get("resource_id"),
        "source": values.get("source"),
        "success": bool(values.get("success")),
        "reason": values.get("reason"),
        "details": values.get("details"),
        "previous_hash": values.get("previous_hash") or GENESIS_HASH,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def calculate_hash(values):
    return hashlib.sha256(canonical_audit_payload(values).encode("utf-8")).hexdigest()


def _source_from_request(request):
    if request is None:
        return None
    client = getattr(request, "client", None)
    source = {
        "method": request.method,
        "path": request.url.path,
        "client": client.host if client else None,
    }
    user_agent = request.headers.get("user-agent")
    if user_agent:
        source["user_agent"] = user_agent[:160]
    return json.dumps(source, sort_keys=True, separators=(",", ":"))


def _safe_details(details):
    if details is None:
        return None
    if isinstance(details, str):
        return details[:1000]
    scrubbed = {}
    for key, value in dict(details).items():
        lowered = str(key).lower()
        if any(secret in lowered for secret in ("password", "token", "secret", "authorization")):
            scrubbed[key] = "[redacted]"
        else:
            scrubbed[key] = value
    return json.dumps(scrubbed, sort_keys=True, default=str)[:1000]


def append_audit_log(db, *, actor=None, action, resource_type=None, resource_id=None,
                     request=None, success=True, reason=None, details=None):
    """Append one hash-chained audit record using the caller's database session."""
    if db.bind.dialect.name == "sqlite" and not db.in_transaction():
        db.connection().exec_driver_sql("BEGIN IMMEDIATE")
    elif db.bind.dialect.name == "postgresql":
        db.execute(text("LOCK TABLE audit_logs IN EXCLUSIVE MODE"))

    previous = db.query(AuditLog).order_by(AuditLog.id.desc()).first()
    timestamp = dt.datetime.utcnow()
    values = {
        "audit_id": str(uuid.uuid4()),
        "timestamp": timestamp,
        "user_id": getattr(actor, "id", None),
        "username": getattr(actor, "username", None) if actor else None,
        "role": getattr(actor, "role", None) if actor else None,
        "action": action,
        "resource_type": resource_type,
        "resource_id": str(resource_id) if resource_id is not None else None,
        "source": _source_from_request(request),
        "success": bool(success),
        "reason": reason,
        "details": _safe_details(details),
        "previous_hash": previous.current_hash if previous else GENESIS_HASH,
    }
    values["current_hash"] = calculate_hash(values)
    record = AuditLog(**values)
    db.add(record)
    db.flush()
    return record


def append_audit_log_safe(*, actor=None, action, resource_type=None, resource_id=None,
                          request=None, success=True, reason=None, details=None):
    try:
        with SessionLocal() as db:
            record = append_audit_log(
                db,
                actor=actor,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                request=request,
                success=success,
                reason=reason,
                details=details,
            )
            db.commit()
            return record
    except Exception:
        return None


def audit_payload(row):
    return {
        "id": row.id,
        "audit_id": row.audit_id,
        "timestamp": _iso(row.timestamp),
        "user_id": row.user_id,
        "username": row.username,
        "role": row.role,
        "action": row.action,
        "resource_type": row.resource_type,
        "resource_id": row.resource_id,
        "source": json.loads(row.source) if row.source else None,
        "success": bool(row.success),
        "reason": row.reason,
        "details": json.loads(row.details) if row.details and row.details[:1] in "[{" else row.details,
        "previous_hash": row.previous_hash,
        "current_hash": row.current_hash,
    }


def verify_audit_chain(db):
    rows = db.query(AuditLog).order_by(AuditLog.id.asc()).all()
    previous_hash = GENESIS_HASH
    for index, row in enumerate(rows, start=1):
        values = {
            "audit_id": row.audit_id,
            "timestamp": row.timestamp,
            "user_id": row.user_id,
            "username": row.username,
            "role": row.role,
            "action": row.action,
            "resource_type": row.resource_type,
            "resource_id": row.resource_id,
            "source": row.source,
            "success": bool(row.success),
            "reason": row.reason,
            "details": row.details,
            "previous_hash": row.previous_hash,
        }
        expected = calculate_hash(values)
        if row.previous_hash != previous_hash:
            return {
                "valid": False,
                "records_checked": index,
                "first_invalid_record": row.id,
                "reason": "previous_hash_mismatch",
                "expected_previous_hash": previous_hash,
                "actual_previous_hash": row.previous_hash,
            }
        if row.current_hash != expected:
            return {
                "valid": False,
                "records_checked": index,
                "first_invalid_record": row.id,
                "reason": "current_hash_mismatch",
                "expected_current_hash": expected,
                "actual_current_hash": row.current_hash,
            }
        previous_hash = row.current_hash
    return {
        "valid": True,
        "records_checked": len(rows),
        "first_invalid_record": None,
        "head_hash": previous_hash,
        "description": "tamper-evident application-level audit logging using SHA-256 hash chaining",
    }
