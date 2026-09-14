"""AES-256-GCM helpers for sensitive at-rest application values."""
from __future__ import annotations

import base64
import os
from dataclasses import dataclass

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class EncryptionConfigurationError(RuntimeError):
    pass


class DecryptionError(RuntimeError):
    pass


def _b64decode_key(value: str) -> bytes:
    raw = value.strip()
    try:
        key = base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4))
    except Exception as exc:
        raise EncryptionConfigurationError("encryption key must be base64 encoded") from exc
    if len(key) != 32:
        raise EncryptionConfigurationError("AES-256-GCM requires a 32-byte key")
    return key


def configured_keys() -> dict[str, bytes]:
    keys: dict[str, bytes] = {}
    current_version = os.getenv("ANPR_ENCRYPTION_KEY_VERSION", "v1").strip() or "v1"
    current_key = os.getenv("ANPR_ENCRYPTION_KEY")
    if current_key:
        keys[current_version] = _b64decode_key(current_key)
    for name, value in os.environ.items():
        prefix = "ANPR_ENCRYPTION_KEY_"
        if not name.startswith(prefix) or name in {"ANPR_ENCRYPTION_KEY_VERSION"}:
            continue
        version = name[len(prefix):].lower()
        if version and value:
            keys[version] = _b64decode_key(value)
    return keys


def current_key_version() -> str:
    return os.getenv("ANPR_ENCRYPTION_KEY_VERSION", "v1").strip() or "v1"


def is_encryption_configured() -> bool:
    return bool(configured_keys())


@dataclass(frozen=True)
class EncryptedValue:
    ciphertext: str
    nonce: str
    key_version: str


def _aad(resource_type: str, record_id: str | int | None, purpose: str) -> bytes:
    return f"anpr:{resource_type}:{record_id or 'pending'}:{purpose}".encode("utf-8")


def encrypt_sensitive(value: str | bytes | None, *, resource_type: str, record_id=None, purpose: str) -> EncryptedValue | None:
    if value is None:
        return None
    key_version = current_key_version()
    keys = configured_keys()
    key = keys.get(key_version)
    if not key:
        raise EncryptionConfigurationError(f"missing encryption key for version {key_version}")
    plaintext = value if isinstance(value, bytes) else str(value).encode("utf-8")
    nonce = os.urandom(12)
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, _aad(resource_type, record_id, purpose))
    return EncryptedValue(
        ciphertext=base64.urlsafe_b64encode(ciphertext).decode("ascii"),
        nonce=base64.urlsafe_b64encode(nonce).decode("ascii"),
        key_version=key_version,
    )


def decrypt_sensitive(ciphertext: str | None, nonce: str | None, key_version: str | None, *,
                      resource_type: str, record_id=None, purpose: str) -> str | None:
    if not ciphertext:
        return None
    if not nonce or not key_version:
        raise DecryptionError("encrypted value is missing nonce or key version")
    key = configured_keys().get(key_version)
    if not key:
        raise DecryptionError(f"missing encryption key for version {key_version}")
    try:
        plaintext = AESGCM(key).decrypt(
            base64.urlsafe_b64decode(nonce),
            base64.urlsafe_b64decode(ciphertext),
            _aad(resource_type, record_id, purpose),
        )
    except (InvalidTag, ValueError) as exc:
        raise DecryptionError("encrypted value authentication failed") from exc
    return plaintext.decode("utf-8")


def generate_key() -> str:
    """Return a base64url 32-byte key for local setup documentation/tests."""
    return base64.urlsafe_b64encode(os.urandom(32)).decode("ascii")
