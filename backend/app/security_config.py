"""Centralized Phase 11 security configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _csv_env(name: str, default: list[str] | None = None) -> list[str]:
    raw = os.getenv(name)
    if raw is None:
        return list(default or [])
    return [item.strip() for item in raw.split(",") if item.strip()]


def _production_default(production_value: int, development_value: int) -> int:
    return production_value if os.getenv("ANPR_ENV", "development").strip().lower() == "production" else development_value


@dataclass(frozen=True)
class SecuritySettings:
    env: str = field(default_factory=lambda: os.getenv("ANPR_ENV", "development").strip().lower())
    cookie_secure: bool = field(default_factory=lambda: _bool_env("ANPR_COOKIE_SECURE", False))
    cookie_samesite: str = field(default_factory=lambda: os.getenv("ANPR_COOKIE_SAMESITE", "lax").strip().lower())
    tls_enabled: bool = field(default_factory=lambda: _bool_env("ANPR_TLS_ENABLED", False))
    allowed_origins: tuple[str, ...] = field(default_factory=lambda: tuple(_csv_env(
        "ANPR_ALLOWED_ORIGINS",
        ["http://127.0.0.1:8000", "http://localhost:8000", "http://127.0.0.1:5500", "http://localhost:5500"],
    )))
    trusted_proxies: tuple[str, ...] = field(default_factory=lambda: tuple(_csv_env("ANPR_TRUSTED_PROXIES", ["127.0.0.1", "::1", "testclient"])))
    max_upload_size: int = field(default_factory=lambda: _int_env("ANPR_MAX_UPLOAD_SIZE", 10 * 1024 * 1024))
    max_json_body_size: int = field(default_factory=lambda: _int_env("ANPR_MAX_JSON_BODY_SIZE", 2 * 1024 * 1024))
    max_edge_batch_size: int = field(default_factory=lambda: _int_env("ANPR_MAX_EDGE_BATCH_SIZE", 250))
    edge_auth_required: bool = field(default_factory=lambda: _bool_env("ANPR_EDGE_AUTH_REQUIRED", os.getenv("ANPR_ENV", "development").lower() == "production"))
    edge_timestamp_skew_seconds: int = field(default_factory=lambda: _int_env("ANPR_EDGE_TIMESTAMP_SKEW_SECONDS", 300))
    auth_query_tokens: bool = field(default_factory=lambda: _bool_env("ANPR_ALLOW_QUERY_TOKENS", False))
    rate_limit_enabled: bool = field(default_factory=lambda: _bool_env("ANPR_RATE_LIMIT_ENABLED", True))
    rate_limit_login_max: int = field(default_factory=lambda: _int_env("ANPR_RATE_LIMIT_LOGIN_MAX", _production_default(20, 300)))
    rate_limit_login_window_seconds: int = field(default_factory=lambda: _int_env("ANPR_RATE_LIMIT_LOGIN_WINDOW_SECONDS", 60))
    rate_limit_auth_max: int = field(default_factory=lambda: _int_env("ANPR_RATE_LIMIT_AUTH_MAX", _production_default(120, 600)))
    rate_limit_auth_window_seconds: int = field(default_factory=lambda: _int_env("ANPR_RATE_LIMIT_AUTH_WINDOW_SECONDS", 60))
    rate_limit_processing_max: int = field(default_factory=lambda: _int_env("ANPR_RATE_LIMIT_PROCESSING_MAX", _production_default(60, 600)))
    rate_limit_processing_window_seconds: int = field(default_factory=lambda: _int_env("ANPR_RATE_LIMIT_PROCESSING_WINDOW_SECONDS", 60))
    rate_limit_edge_max: int = field(default_factory=lambda: _int_env("ANPR_RATE_LIMIT_EDGE_MAX", _production_default(120, 1000)))
    rate_limit_edge_window_seconds: int = field(default_factory=lambda: _int_env("ANPR_RATE_LIMIT_EDGE_WINDOW_SECONDS", 60))

    @property
    def production(self) -> bool:
        return self.env == "production"

    @property
    def hsts_enabled(self) -> bool:
        return self.production and (self.cookie_secure or self.tls_enabled)

    @property
    def cors_origins(self) -> list[str]:
        if self.production and not self.allowed_origins:
            raise RuntimeError("ANPR_ALLOWED_ORIGINS must be configured in production")
        if "*" in self.allowed_origins and self.production:
            raise RuntimeError("Wildcard CORS origins are not allowed in production")
        return list(self.allowed_origins)


def get_security_settings() -> SecuritySettings:
    return SecuritySettings()


def validate_production_security_config() -> list[str]:
    settings = get_security_settings()
    if not settings.production:
        return []
    issues: list[str] = []
    auth_secret = os.getenv("ANPR_AUTH_SECRET", "")
    if not auth_secret or auth_secret == "local-sih-demo-change-me":
        issues.append("ANPR_AUTH_SECRET must be set to a non-default value")
    if not settings.cookie_secure:
        issues.append("ANPR_COOKIE_SECURE=true is required for production HTTPS")
    if settings.cookie_samesite not in {"lax", "strict", "none"}:
        issues.append("ANPR_COOKIE_SAMESITE must be lax, strict or none")
    if settings.cookie_samesite == "none" and not settings.cookie_secure:
        issues.append("SameSite=None requires Secure cookies")
    if not settings.allowed_origins:
        issues.append("ANPR_ALLOWED_ORIGINS must list explicit production origins")
    if "*" in settings.allowed_origins:
        issues.append("Wildcard CORS origins are not allowed in production")
    if not settings.edge_auth_required:
        issues.append("ANPR_EDGE_AUTH_REQUIRED=true is required in production")
    return issues


def require_production_security_config() -> None:
    issues = validate_production_security_config()
    if issues:
        raise RuntimeError("Invalid production security configuration: " + "; ".join(issues))
