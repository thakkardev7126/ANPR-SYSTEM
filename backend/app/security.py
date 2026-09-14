"""HTTP/API hardening helpers for Phase 11."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from app.security_config import get_security_settings


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
            "img-src 'self' data: blob: https://*.tile.openstreetmap.org; "
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
