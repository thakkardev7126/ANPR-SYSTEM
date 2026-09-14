"""Minimal local authentication and centralized RBAC for the demo API."""
from __future__ import annotations

import base64
import datetime as dt
import hashlib
import hmac
import json
import os
import re
import secrets
from dataclasses import dataclass

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from app.audit import append_audit_log, append_audit_log_safe
from app.database import AuditLog, SessionLocal, User, get_db
from app.security_config import get_security_settings

ROLE_SYSTEM_ADMIN = "SYSTEM_ADMIN"
ROLE_TRAFFIC_OPERATOR = "TRAFFIC_OPERATOR"
ROLE_PCR_OFFICER = "PCR_OFFICER"
ROLES = {ROLE_SYSTEM_ADMIN, ROLE_TRAFFIC_OPERATOR, ROLE_PCR_OFFICER}

PERMISSIONS_BY_ROLE = {
    ROLE_SYSTEM_ADMIN: {"*"},
    ROLE_TRAFFIC_OPERATOR: {
        "camera:read", "traffic:read", "vehicle:investigate", "trajectory:read",
        "hotlist:read", "incident:read", "event:read", "review:write",
        "evidence:privacy", "appearance:read", "anomaly:read", "edge:read",
        "signal:read",
    },
    ROLE_PCR_OFFICER: {
        "pcr:read", "pcr:write", "hotlist:read", "incident:read", "incident:write",
        "vehicle:investigate", "trajectory:read", "event:read", "evidence:privacy",
        "anomaly:read",
    },
}

DEMO_USERS = [
    ("admin_demo", "AdminDemo!2026", ROLE_SYSTEM_ADMIN),
    ("traffic_demo", "TrafficDemo!2026", ROLE_TRAFFIC_OPERATOR),
    ("pcr_demo", "PcrDemo!2026", ROLE_PCR_OFFICER),
]

TOKEN_TTL_SECONDS = int(os.getenv("ANPR_AUTH_TOKEN_TTL_SECONDS", "28800"))
COOKIE_NAME = "anpr_session"
SECRET = os.getenv("ANPR_AUTH_SECRET", "local-sih-demo-change-me")
if os.getenv("ANPR_ENV", "development").strip().lower() == "production" and SECRET == "local-sih-demo-change-me":
    raise RuntimeError("ANPR_AUTH_SECRET must be configured in production")
PBKDF2_ITERATIONS = 180_000


class LoginInput(BaseModel):
    username: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=1, max_length=256)


class UserInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    username: str = Field(min_length=3, max_length=80)
    password: str | None = Field(default=None, min_length=8, max_length=256)
    role: str
    active: bool = True


class UserUpdateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    password: str | None = Field(default=None, min_length=8, max_length=256)
    role: str | None = None
    active: bool | None = None


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def hash_password(password, *, salt=None):
    salt_bytes = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${_b64url(salt_bytes)}${_b64url(digest)}"


def verify_password(password, password_hash):
    try:
        algorithm, iterations, salt, digest = password_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        expected = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            _b64url_decode(salt),
            int(iterations),
        )
        return hmac.compare_digest(_b64url(expected), digest)
    except Exception:
        return False


def create_token(user):
    now = int(dt.datetime.utcnow().timestamp())
    payload = {
        "sub": user.username,
        "uid": user.id,
        "role": user.role,
        "iat": now,
        "exp": now + TOKEN_TTL_SECONDS,
        "jti": secrets.token_hex(8),
    }
    body = _b64url(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    signature = _b64url(hmac.new(SECRET.encode("utf-8"), body.encode("ascii"), hashlib.sha256).digest())
    return f"{body}.{signature}", payload["exp"]


def decode_token(token):
    try:
        body, signature = token.split(".", 1)
        expected = _b64url(hmac.new(SECRET.encode("utf-8"), body.encode("ascii"), hashlib.sha256).digest())
        if not hmac.compare_digest(signature, expected):
            raise ValueError("invalid_signature")
        payload = json.loads(_b64url_decode(body))
        if int(payload.get("exp", 0)) < int(dt.datetime.utcnow().timestamp()):
            raise ValueError("expired_token")
        return payload
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("invalid_token") from exc


def sanitize_username(username):
    username = str(username or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]{3,80}", username):
        raise ValueError("username must contain 3-80 letters, digits, dots, underscores or hyphens")
    return username


def normalize_role(role):
    role = str(role or "").strip().upper()
    if role not in ROLES:
        raise ValueError("role must be SYSTEM_ADMIN, TRAFFIC_OPERATOR or PCR_OFFICER")
    return role


def user_payload(user):
    return {
        "id": user.id,
        "username": user.username,
        "role": user.role,
        "active": bool(user.active),
        "created_at": user.created_at.isoformat() if user.created_at else None,
        "updated_at": user.updated_at.isoformat() if user.updated_at else None,
    }


def seed_demo_users(db):
    now = dt.datetime.utcnow()
    for username, password, role in DEMO_USERS:
        existing = db.query(User).filter(User.username == username).first()
        if existing:
            if existing.role != role or not existing.active:
                existing.role = role
                existing.active = True
                existing.updated_at = now
            continue
        db.add(User(
            username=username,
            password_hash=hash_password(password),
            role=role,
            active=True,
            created_at=now,
            updated_at=now,
        ))
    db.commit()


def permission_allowed(user, permission):
    if not user or not user.active:
        return False
    permissions = PERMISSIONS_BY_ROLE.get(user.role, set())
    return "*" in permissions or permission in permissions


def token_from_request(request):
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth.split(None, 1)[1].strip()
    if get_security_settings().auth_query_tokens and request.query_params.get("token"):
        return request.query_params.get("token")
    return request.cookies.get(COOKIE_NAME)


def authenticate_request_user(request, db):
    token = token_from_request(request)
    if not token:
        return None, "missing_token"
    try:
        payload = decode_token(token)
    except ValueError as exc:
        return None, str(exc)
    user = db.get(User, int(payload.get("uid")))
    if not user or user.username != payload.get("sub"):
        return None, "unknown_user"
    if not user.active:
        return None, "inactive_user"
    return user, None


def require_permission(permission):
    def dependency(request: Request, db: Session = Depends(get_db)):
        user, reason = authenticate_request_user(request, db)
        if not user:
            raise HTTPException(status_code=401, detail=reason or "Authentication required")
        if not permission_allowed(user, permission):
            raise HTTPException(status_code=403, detail="Permission denied")
        return user
    return dependency


def require_role(*roles):
    allowed = {normalize_role(role) for role in roles}

    def dependency(request: Request, db: Session = Depends(get_db)):
        user, reason = authenticate_request_user(request, db)
        if not user:
            raise HTTPException(status_code=401, detail=reason or "Authentication required")
        if user.role not in allowed:
            raise HTTPException(status_code=403, detail="Permission denied")
        return user
    return dependency


@dataclass
class RoutePolicy:
    method: str
    pattern: re.Pattern
    permission: str
    action: str
    resource_type: str


def _policy(method, pattern, permission, action, resource_type):
    return RoutePolicy(method, re.compile(pattern), permission, action, resource_type)


ROUTE_POLICIES = [
    _policy("GET", r"^/api/auth/me$", "event:read", "session_lookup", "auth"),
    _policy("POST", r"^/api/auth/logout$", "event:read", "logout", "auth"),
    _policy("GET", r"^/api/audit/?$", "audit:read", "audit_log_access", "audit_log"),
    _policy("GET", r"^/api/audit/verify$", "audit:read", "audit_chain_verify", "audit_log"),
    _policy("GET", r"^/api/users/?$", "rbac:write", "user_list", "user"),
    _policy("POST", r"^/api/users/?$", "rbac:write", "user_create", "user"),
    _policy("PATCH", r"^/api/users/(?P<id>\d+)$", "rbac:write", "user_update", "user"),
    _policy("GET", r"^/api/edge(?:/.*)?$", "edge:read", "edge_status_lookup", "edge"),
    _policy("POST", r"^/api/edge/observations(?:/batch)?$", "edge:write", "edge_observation_ingest", "edge_observation"),
    _policy("POST", r"^/api/edge/workers/run$", "edge:write", "edge_worker_run", "edge_queue"),
    _policy("POST", r"^/api/edge/simulation/(?:start|stop)$", "edge:write", "edge_simulation_change", "edge_simulation"),
    _policy("GET", r"^/api/retention/policy$", "retention:write", "retention_policy_lookup", "retention"),
    _policy("POST", r"^/api/retention/run$", "retention:write", "retention_execution", "retention"),
    _policy("POST", r"^/api/signals/junctions(?:/[^/]+/phases)?$", "signal:write", "signal_configuration_change", "signal"),
    _policy("PATCH", r"^/api/signals/junctions/[^/]+$", "signal:write", "signal_configuration_change", "signal"),
    _policy("DELETE", r"^/api/signals/junctions/[^/]+$", "signal:write", "signal_configuration_change", "signal"),
    _policy("POST", r"^/api/signals/controller/apply$", "signal:write", "signal_controller_apply", "signal_controller"),
    _policy("POST", r"^/api/signals/recommendation$", "signal:read", "signal_recommendation_generate", "signal_recommendation"),
    _policy("POST", r"^/api/signals/simulation/[^/]+/(?:start|stop|reset|tick)$", "signal:read", "signal_simulation_change", "signal_simulation"),
    _policy("GET", r"^/api/signals(?:/.*)?$", "signal:read", "signal_lookup", "signal"),
    _policy("GET", r"^/api/cameras(?:/[^/]+/(?:snapshot|road-connections))?$", "camera:read", "camera_access", "camera"),
    _policy("POST", r"^/api/cameras(?:/[^/]+/connect)?$", "camera:write", "camera_change", "camera"),
    _policy("PATCH", r"^/api/cameras/[^/]+$", "camera:write", "camera_change", "camera"),
    _policy("DELETE", r"^/api/cameras/[^/]+$", "camera:write", "camera_change", "camera"),
    _policy("GET", r"^/api/camera-road", "traffic:read", "road_network_lookup", "road_network"),
    _policy("POST", r"^/api/camera-road", "camera:write", "road_network_change", "road_network"),
    _policy("PATCH", r"^/api/camera-road", "camera:write", "road_network_change", "road_network"),
    _policy("DELETE", r"^/api/camera-road", "camera:write", "road_network_change", "road_network"),
    _policy("POST", r"^/api/(?:upload|scan|process-frame|upload-batch)$", "camera:read", "scan_ingest", "plate_event"),
    _policy("GET", r"^/api/events/?$", "event:read", "event_lookup", "plate_event"),
    _policy("DELETE", r"^/api/events/?$", "rbac:write", "event_clear", "plate_event"),
    _policy("POST", r"^/api/events/(?P<id>\d+)/correct$", "review:write", "review_correction", "plate_event"),
    _policy("GET", r"^/api/events/(?P<id>\d+)/appearance$", "appearance:read", "appearance_lookup", "plate_event"),
    _policy("GET", r"^/api/appearance/similarity$", "appearance:read", "appearance_matching_lookup", "plate_event"),
    _policy("GET", r"^/api/matches", "appearance:read", "appearance_matching_lookup", "vehicle_match"),
    _policy("POST", r"^/api/matches/(?P<id>\d+)/review$", "review:write", "review_correction", "vehicle_match"),
    _policy("GET", r"^/api/(?:anomalies|anomaly-policy|plate-suspicion|plate-suspicions|route-anomaly|route-anomalies)", "anomaly:read", "anomaly_lookup", "anomaly"),
    _policy("PATCH", r"^/api/(?:plate-suspicions|route-anomalies)/(?P<id>\d+)/review$", "review:write", "review_correction", "anomaly"),
    _policy("GET", r"^/api/traffic", "traffic:read", "traffic_analytics_lookup", "traffic"),
    _policy("GET", r"^/api/hotlist", "hotlist:read", "hotlist_access", "hotlist"),
    _policy("POST", r"^/api/hotlist-alerts/(?P<id>\d+)/acknowledge$", "incident:write", "hotlist_acknowledgement", "hotlist_alert"),
    _policy("POST", r"^/api/hotlist/?$", "hotlist:write", "hotlist_creation", "hotlist"),
    _policy("PUT", r"^/api/hotlist/(?P<id>\d+)$", "hotlist:write", "hotlist_update", "hotlist"),
    _policy("DELETE", r"^/api/hotlist/(?P<id>\d+)$", "hotlist:write", "hotlist_deletion", "hotlist"),
    _policy("GET", r"^/api/pcr", "pcr:read", "pcr_access", "pcr_vehicle"),
    _policy("POST", r"^/api/pcr(?:/[^/]+/location|/seed-demo)?$", "pcr:write", "pcr_update", "pcr_vehicle"),
    _policy("PATCH", r"^/api/pcr/[^/]+$", "pcr:write", "pcr_update", "pcr_vehicle"),
    _policy("GET", r"^/api/incidents", "incident:read", "incident_access", "incident"),
    _policy("POST", r"^/api/incidents/(?P<id>\d+)/status$", "incident:write", "incident_status_change", "incident"),
    _policy("GET", r"^/api/(?:vehicles|trajectory|search/plates|plates)", "vehicle:investigate", "vehicle_lookup", "vehicle"),
    _policy("GET", r"^/api/stats$", "traffic:read", "traffic_analytics_lookup", "traffic"),
    _policy("GET", r"^/api/evidence/(?P<id>\d+)/image$", "evidence:privacy", "evidence_access", "evidence"),
    _policy("GET", r"^/uploads/", "evidence:privacy", "evidence_access", "evidence"),
]

PUBLIC_API = {
    ("GET", "/api/system/status"),
    ("POST", "/api/auth/login"),
}


def route_policy_for(method, path):
    for policy in ROUTE_POLICIES:
        if policy.method == method and policy.pattern.match(path):
            return policy, policy.pattern.match(path)
    if path.startswith("/api/"):
        return _policy(method, r"^/api/", "event:read", "api_access", "api"), None
    return None, None


def needs_original_permission(path, query_params):
    if path.startswith("/api/evidence/") and query_params.get("kind") == "original":
        return True
    if path.startswith("/uploads/"):
        name = os.path.basename(path)
        return "_privacy" not in name and not any(marker in name for marker in ("_crop_", "_vehicle_", "_plate_band", "_contour_"))
    return False


async def rbac_middleware(request, call_next):
    method = request.method.upper()
    path = request.url.path
    if (
        method == "POST"
        and path in {"/api/edge/observations", "/api/edge/observations/batch"}
        and get_security_settings().edge_auth_required
    ):
        return await call_next(request)
    if method == "OPTIONS" or (method, path) in PUBLIC_API or not (path.startswith("/api/") or path.startswith("/uploads/")):
        return await call_next(request)

    policy, match = route_policy_for(method, path)
    with SessionLocal() as db:
        user, reason = authenticate_request_user(request, db)
        if not user:
            append_audit_log_safe(
                actor=None,
                action=policy.action if policy else "unauthenticated_access",
                resource_type=policy.resource_type if policy else "api",
                request=request,
                success=False,
                reason=reason or "authentication_required",
                details={"path": path, "method": method},
            )
            return JSONResponse({"detail": reason or "Authentication required"}, status_code=401)
        permission = "evidence:original" if needs_original_permission(path, request.query_params) else policy.permission
        if not permission_allowed(user, permission):
            append_audit_log_safe(
                actor=user,
                action=policy.action if policy else "unauthorized_access",
                resource_type=policy.resource_type if policy else "api",
                resource_id=(match.groupdict().get("id") if match else None),
                request=request,
                success=False,
                reason=f"missing_permission:{permission}",
                details={"path": path, "method": method, "required_permission": permission},
            )
            return JSONResponse({"detail": "Permission denied"}, status_code=403)
        request.state.current_user = user_payload(user)
        request.state.current_user_model = user

    response = await call_next(request)
    if policy:
        append_audit_log_safe(
            actor=getattr(request.state, "current_user_model", None),
            action=policy.action,
            resource_type=policy.resource_type,
            resource_id=(match.groupdict().get("id") if match else None),
            request=request,
            success=response.status_code < 400,
            reason=None if response.status_code < 400 else f"http_{response.status_code}",
        )
    return response


async def authenticate_websocket(websocket):
    auth = websocket.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        token = auth.split(None, 1)[1].strip()
    else:
        token = None
    token = token or websocket.query_params.get("token") or websocket.cookies.get(COOKIE_NAME)
    if not token:
        await websocket.close(code=1008)
        return None
    try:
        payload = decode_token(token)
    except ValueError:
        await websocket.close(code=1008)
        return None
    db = next(get_db())
    try:
        user = db.get(User, int(payload.get("uid")))
        if not user or not user.active:
            await websocket.close(code=1008)
            return None
        return user
    finally:
        db.close()


router = APIRouter(prefix="/api", tags=["Auth/RBAC"])


@router.post("/auth/login")
def login(body: LoginInput, request: Request, response: Response, db: Session = Depends(get_db)):
    username = body.username.strip()
    user = db.query(User).filter(User.username == username).first()
    if not user or not user.active or not verify_password(body.password, user.password_hash):
        append_audit_log(db, actor=user, action="login_failure", resource_type="auth",
                         request=request, success=False, reason="invalid_credentials")
        db.commit()
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token, expires_at = create_token(user)
    append_audit_log(db, actor=user, action="login_success", resource_type="auth",
                     request=request, success=True)
    db.commit()
    settings = get_security_settings()
    response.set_cookie(
        COOKIE_NAME,
        token,
        httponly=True,
        secure=settings.cookie_secure,
        samesite=settings.cookie_samesite,
        max_age=TOKEN_TTL_SECONDS,
    )
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_at": dt.datetime.utcfromtimestamp(expires_at).isoformat() + "Z",
        "user": user_payload(user),
        "demo_credentials": [
            {"username": username, "role": role}
            for username, password, role in DEMO_USERS
        ],
        "demo_only": True,
    }


@router.post("/auth/logout")
def logout(request: Request, response: Response, db: Session = Depends(get_db)):
    user, _ = authenticate_request_user(request, db)
    append_audit_log(db, actor=user, action="logout", resource_type="auth", request=request, success=True)
    db.commit()
    settings = get_security_settings()
    response.delete_cookie(COOKIE_NAME, secure=settings.cookie_secure, samesite=settings.cookie_samesite)
    return {"ok": True}


@router.get("/auth/me")
def me(request: Request, db: Session = Depends(get_db)):
    user, reason = authenticate_request_user(request, db)
    if not user:
        raise HTTPException(status_code=401, detail=reason or "Authentication required")
    return {"user": user_payload(user), "permissions": sorted(PERMISSIONS_BY_ROLE.get(user.role, set()))}


@router.get("/audit")
def list_audit_logs(offset: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    limit = min(max(int(limit), 1), 500)
    rows = db.query(AuditLog).order_by(AuditLog.id.desc()).offset(max(offset, 0)).limit(limit).all()
    from app.audit import audit_payload
    return {"total": db.query(AuditLog).count(), "items": [audit_payload(row) for row in rows]}


@router.get("/audit/verify")
def verify_audit_logs(db: Session = Depends(get_db)):
    from app.audit import verify_audit_chain
    return verify_audit_chain(db)


@router.get("/users")
def list_users(db: Session = Depends(get_db)):
    return {"items": [user_payload(user) for user in db.query(User).order_by(User.username.asc()).all()]}


@router.post("/users", status_code=201)
def create_user(body: UserInput, db: Session = Depends(get_db)):
    try:
        username = sanitize_username(body.username)
        role = normalize_role(body.role)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not body.password:
        raise HTTPException(status_code=422, detail="password is required")
    if db.query(User).filter(User.username == username).first():
        raise HTTPException(status_code=409, detail="username already exists")
    now = dt.datetime.utcnow()
    user = User(username=username, password_hash=hash_password(body.password), role=role,
                active=body.active, created_at=now, updated_at=now)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user_payload(user)


@router.patch("/users/{user_id}")
def update_user(user_id: int, body: UserUpdateInput, db: Session = Depends(get_db)):
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if body.role is not None:
        try:
            user.role = normalize_role(body.role)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    if body.active is not None:
        user.active = body.active
    if body.password:
        user.password_hash = hash_password(body.password)
    user.updated_at = dt.datetime.utcnow()
    db.commit()
    db.refresh(user)
    return user_payload(user)
