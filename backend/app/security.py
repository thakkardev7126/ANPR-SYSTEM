"""HTTP/API hardening helpers for Phase 11."""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from app.security_config import get_security_settings


_RATE_LIMIT_STATE: dict[tuple[str, str], list[float]] = {}
_RATE_LIMIT_LOCK = threading.Lock()


def request_scheme(request: Request) -> str:
    settings = get_security_settings()
    client_host = request.client.host if request.client else ""
    if client_host in settings.trusted_proxies:
        forwarded = request.headers.get("x-forwarded-proto", "").split(",", 1)[0].strip().lower()
        if forwarded in {"http", "https"}:
            return forwarded
    return request.url.scheme


def resolve_under_root(root: str, filename_or_path: str) -> str | None:
    root_path = Path(root).resolve()
    candidate = Path(filename_or_path)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        if ".." in candidate.parts:
            return None
        resolved = (root_path / candidate).resolve()
    try:
        resolved.relative_to(root_path)
    except ValueError:
        return None
    return str(resolved) if resolved.is_file() else None


def security_headers_for(request: Request) -> dict[str, str]:
    settings = get_security_settings()
    headers = {
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "same-origin",
        "Permissions-Policy": "camera=(self), microphone=(), geolocation=(self)",
        "Content-Security-Policy": (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://unpkg.com; "
            "style-src 'self' 'unsafe-inline' https://unpkg.com; "
            "img-src 'self' data: blob: https://*.tile.openstreetmap.org https://*.basemaps.cartocdn.com https://server.arcgisonline.com https://unpkg.com; "
            "connect-src 'self' ws: wss:; "
            "font-src 'self' data:; "
            "object-src 'none'; "
            "base-uri 'self'; "
            "frame-ancestors 'none'"
        ),
    }
    if settings.hsts_enabled and request_scheme(request) == "https":
        headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return headers


async def security_headers_middleware(request, call_next):
    response = await call_next(request)
    for name, value in security_headers_for(request).items():
        response.headers.setdefault(name, value)
    return response


async def body_size_limit_middleware(request: Request, call_next):
    settings = get_security_settings()
    content_type = request.headers.get("content-type", "").lower()
    limit = settings.max_upload_size if "multipart/form-data" in content_type else settings.max_json_body_size
    length = request.headers.get("content-length")
    if length:
        try:
            if int(length) > limit:
                return JSONResponse({"detail": "Request body too large"}, status_code=413)
        except ValueError:
            return JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)
    return await call_next(request)


def reset_rate_limit_state() -> None:
    with _RATE_LIMIT_LOCK:
        _RATE_LIMIT_STATE.clear()


def client_rate_identity(request: Request) -> str:
    settings = get_security_settings()
    client_host = request.client.host if request.client else "unknown"
    if client_host in settings.trusted_proxies:
        forwarded_for = request.headers.get("x-forwarded-for", "").split(",", 1)[0].strip()
        if forwarded_for:
            return forwarded_for
    return client_host or "unknown"


def rate_limit_policy(method: str, path: str) -> tuple[str, int, int] | None:
    settings = get_security_settings()
    method = method.upper()
    if method == "POST" and path == "/api/auth/login":
        return "login", settings.rate_limit_login_max, settings.rate_limit_login_window_seconds
    if path in {"/api/auth/me", "/api/auth/logout"}:
        return "auth", settings.rate_limit_auth_max, settings.rate_limit_auth_window_seconds
    if method == "POST" and path in {"/api/upload", "/api/scan", "/api/process-frame", "/api/upload-batch"}:
        return "processing", settings.rate_limit_processing_max, settings.rate_limit_processing_window_seconds
    if method == "POST" and path in {"/api/edge/observations", "/api/edge/observations/batch"}:
        return "edge", settings.rate_limit_edge_max, settings.rate_limit_edge_window_seconds
    return None


def _rate_limit_decision(identity: str, category: str, limit: int, window_seconds: int, now: float | None = None) -> int | None:
    if limit <= 0 or window_seconds <= 0:
        return None
    now = time.monotonic() if now is None else now
    key = (category, identity)
    with _RATE_LIMIT_LOCK:
        history = [stamp for stamp in _RATE_LIMIT_STATE.get(key, []) if now - stamp < window_seconds]
        if len(history) >= limit:
            oldest = min(history)
            retry_after = max(1, int(window_seconds - (now - oldest)))
            _RATE_LIMIT_STATE[key] = history
            return retry_after
        history.append(now)
        _RATE_LIMIT_STATE[key] = history
    return None


async def rate_limit_middleware(request: Request, call_next):
    settings = get_security_settings()
    if not settings.rate_limit_enabled:
        return await call_next(request)
    policy = rate_limit_policy(request.method, request.url.path)
    if not policy:
        return await call_next(request)
    category, limit, window_seconds = policy
    retry_after = _rate_limit_decision(client_rate_identity(request), category, limit, window_seconds)
    if retry_after is not None:
        return JSONResponse(
            {"detail": "Too many requests"},
            status_code=429,
            headers={"Retry-After": str(retry_after)},
        )
    return await call_next(request)


def validate_image_size(size: int) -> None:
    if size > get_security_settings().max_upload_size:
        raise HTTPException(status_code=413, detail="Image payload too large")


def public_security_status() -> dict:
    from app.audit import verify_audit_chain
    from app.crypto import is_encryption_configured
    from app.database import SessionLocal

    with SessionLocal() as db:
        audit_valid = verify_audit_chain(db).get("valid")
    settings = get_security_settings()
    return {
        "authentication": "ENABLED",
        "rbac": "ENABLED",
        "privacy_masking": "ENABLED",
        "audit_chain": "VALID" if audit_valid else "INVALID",
        "encryption_at_rest": "CONFIGURED" if is_encryption_configured() else "NOT CONFIGURED",
        "https_tls": "TLS 1.3 PRODUCTION" if settings.production and settings.tls_enabled else "DEVELOPMENT",
        "secure_evidence_access": "ENABLED",
        "environment": settings.env,
    }


def production_requires_encryption() -> bool:
    return os.getenv("ANPR_ENV", "development").strip().lower() == "production"
