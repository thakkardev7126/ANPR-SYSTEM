# ANPR Security Controls

Security controls implemented for the SIH prototype. This document does not claim legal or DPDP compliance.

## Authentication and RBAC

The backend uses local demo users with PBKDF2-SHA256 password hashes and role-based permissions. Passwords are not encrypted, returned by APIs, logged, or stored in plaintext. Demo user names are still available for local SIH demonstration, but demo passwords are documented only for local operators and should be changed for production.

Roles currently include `SYSTEM_ADMIN`, `TRAFFIC_OPERATOR`, and `PCR_OFFICER`. API access is enforced centrally by the RBAC middleware in `backend/app/auth.py`.

## Sessions and Cookies

Login sets the `anpr_session` cookie with `HttpOnly`. Cookie hardening is configurable:

- `ANPR_COOKIE_SECURE=true` for HTTPS production
- `ANPR_COOKIE_SAMESITE=lax` or `strict`
- `ANPR_AUTH_SECRET` must be set in production
- `ANPR_AUTH_TOKEN_TTL_SECONDS` controls token lifetime

The frontend now works with HttpOnly cookies for same-origin requests and only keeps bearer tokens as a compatibility fallback for local/cross-origin demos. Normal HTTP APIs do not accept token query parameters unless `ANPR_ALLOW_QUERY_TOKENS=true` is explicitly set.

## Privacy Masking

Original frames are stored separately from privacy-safe derivatives. Operators see privacy-safe evidence by default. Original evidence requires the existing `evidence:original` permission through RBAC.

## Audit Logging

Phase 8 tamper-evident audit logging is preserved. Audit records are SHA-256 hash chained and redact details containing password, token, secret, or authorization fields. Phase 11 adds auditing for authentication failures, unauthorized API access, original evidence access, and edge authentication failures without logging secrets.

## AES-256-GCM At-Rest Protection

Phase 11 adds centralized AES-256-GCM helpers in `backend/app/crypto.py` using the established `cryptography` package. It does not implement custom cryptography.

Encrypted data:

- `plate_events.original_image_path` for newly persisted sensitive original evidence references when encryption is configured
- `plate_events.privacy_metadata` for privacy-processing metadata when encryption is configured

Not encrypted:

- Passwords, which remain one-way PBKDF2 hashes
- Plate/trajectory/analytics columns that must remain queryable for the prototype
- Image bytes on local disk; production deployments should use encrypted volumes/object storage

Encrypted records store ciphertext, nonce, and key version. Keys are never stored in the database.

## Key Management and Rotation

Configure the active key with:

- `ANPR_ENCRYPTION_KEY_VERSION=v1`
- `ANPR_ENCRYPTION_KEY=<base64url 32-byte key>`

Generate a local key:

```powershell
cd backend
python -c "from app.crypto import generate_key; print(generate_key())"
```

Additional old keys can be supplied as `ANPR_ENCRYPTION_KEY_V1`, `ANPR_ENCRYPTION_KEY_V2`, etc. New writes use `ANPR_ENCRYPTION_KEY_VERSION`; decrypt selects the stored key version. Rotation does not require immediate bulk re-encryption.

Production should store keys in the deployment secret manager and inject them as environment variables. Do not commit keys.

## TLS 1.3 Deployment

The FastAPI application does not implement TLS cryptography. Production TLS should terminate at a standard reverse proxy:

Client -> HTTPS/TLS 1.3 reverse proxy -> FastAPI on localhost/private network

An Nginx example is provided at `deploy/nginx_tls13.conf`. It enables TLS 1.3, redirects HTTP to HTTPS, forwards trusted proxy headers, and supports WebSockets.

Local demo mode may run over HTTP. Do not claim TLS 1.3 is active unless the reverse proxy and HTTPS deployment are actually in use. Set `ANPR_TLS_ENABLED=true` only for production HTTPS deployments.

## Trusted Proxy Headers

The app only interprets forwarded protocol data from configured trusted proxies:

- `ANPR_TRUSTED_PROXIES=127.0.0.1,::1`

Nginx should set `Host`, `X-Forwarded-For`, and `X-Forwarded-Proto` as shown in the deployment example.

## CORS

CORS is configured by `ANPR_ALLOWED_ORIGINS` as a comma-separated list. Local defaults allow localhost demo origins. Production must set explicit HTTPS origins and must not use wildcard origins with credentials.

## Security Headers

Every response receives:

- `Content-Security-Policy`
- `X-Content-Type-Options: nosniff`
- `Referrer-Policy: same-origin`
- `Permissions-Policy`
- HSTS only in HTTPS/production mode

The CSP remains compatible with the existing dashboard's inline JavaScript and the Leaflet CDN while blocking object embeds and external frame ancestors.

## File and Evidence Protection

Raw static `/uploads` exposure has been replaced with an RBAC-protected route. File serving rejects absolute paths, traversal, missing files, and files outside the configured evidence roots. Original evidence remains protected; privacy-safe derivatives are the default.

## Edge Ingestion Security

Production edge ingestion should set:

- `ANPR_EDGE_AUTH_REQUIRED=true`
- `ANPR_EDGE_TOKEN=<shared edge secret>` or per-device `ANPR_EDGE_TOKEN_<DEVICE_ID>`
- `ANPR_EDGE_TOKEN_VERSION=v1`

Edge agents can use `Authorization: Bearer <edge-token>` or an HMAC header pair `X-ANPR-Edge-Timestamp` and `X-ANPR-Edge-Signature`. The app stores only a SHA-256 fingerprint/version for observed edge credentials.

Idempotency still uses `observation_id`. Unknown cameras, malformed observations, oversized batches, and unauthenticated production ingestion are rejected.

## Request Limits

Configurable limits:

- `ANPR_MAX_UPLOAD_SIZE`
- `ANPR_MAX_JSON_BODY_SIZE`
- `ANPR_MAX_EDGE_BATCH_SIZE`

Large requests receive clean `413` responses where possible.

## WebSocket Security

WebSocket endpoints authenticate using the session cookie or compatibility bearer query token. Unauthenticated clients are closed with policy violation. Role permissions are checked before operational data is sent.

## Local Development

Local HTTP demo remains supported:

- HTTP allowed
- Demo users allowed
- Encryption optional, reported as `NOT CONFIGURED`
- Edge auth optional unless explicitly enabled

## Production Requirements

Before production:

- Set `ANPR_ENV=production`
- Set `ANPR_AUTH_SECRET`
- Set `ANPR_COOKIE_SECURE=true`
- Set explicit `ANPR_ALLOWED_ORIGINS`
- Configure TLS 1.3 reverse proxy
- Configure encryption keys
- Configure edge credentials
- Use encrypted storage/volumes for image files
- Restrict network access to the FastAPI backend

## Known Limitations

This prototype does not include enterprise SSO, HSM/KMS integration, malware scanning for uploads, certificate automation, or encrypted image blob storage. Existing local demo certificates under `backend/certs` are development artifacts only and are not production credentials.
