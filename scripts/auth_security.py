"""
auth_security.py
=================
Cryptographic + cookie helpers for passwordless email auth.

Responsibilities:
  - Generate 6-digit auth codes and opaque session tokens (raw values are
    only ever held in memory for the duration of a single request, and are
    never logged).
  - HMAC-hash codes/tokens with a server-side secret (AUTH_SECRET) before
    they are ever written to Postgres — a leaked DB row alone is not
    enough to forge a session or replay a code.
  - Set / clear the sonya_session HttpOnly cookie.

AUTH_SECRET is intentionally separate from WORKER_SECRET: the worker secret
authenticates trusted GPU workers, this secret authenticates end-user
sessions and codes. Rotating one must never invalidate the other.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
import secrets
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

# ── Config ————————————————————————————————————————————————————————————————————

SESSION_COOKIE_NAME = "sonya_session"
SESSION_TTL_SECONDS = 60 * 60 * 24 * 30          # 30 days
AUTH_CODE_TTL_SECONDS = 10 * 60                   # 10 minutes
AUTH_CODE_MAX_ATTEMPTS = 5

VALID_PURPOSES = {"login", "register", "email_verify"}

_EMAIL_RE = re.compile(r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+$")


def is_valid_email(email: str) -> bool:
    if not email or len(email) > 254:
        return False
    return bool(_EMAIL_RE.match(email.strip()))


def _cookie_secure() -> bool:
    """
    Secure=true is mandatory in production. Non-production environments may
    disable it (via AUTH_COOKIE_INSECURE=true) so local http:// dev servers
    and tests can exercise the cookie without TLS.
    """
    app_env = os.environ.get("APP_ENV", "development").lower()
    if app_env == "production":
        return True
    return os.environ.get("AUTH_COOKIE_INSECURE", "false").lower() not in ("1", "true", "yes")


def _get_auth_secret() -> str:
    secret = os.environ.get("AUTH_SECRET", "")
    if not secret:
        raise RuntimeError(
            "AUTH_SECRET not configured. Set AUTH_SECRET to a long random value "
            "(e.g. openssl rand -hex 32) — required to hash auth codes and session tokens."
        )
    return secret


# ── Auth codes ————————————————————————————————————————————————————————————————

def generate_numeric_code() -> str:
    """Cryptographically random 6-digit code, zero-padded."""
    return f"{secrets.randbelow(1_000_000):06d}"


def hash_code(code: str, email: str, purpose: str) -> str:
    """
    HMAC-SHA256(AUTH_SECRET, "email:purpose:code").
    Binding email+purpose into the hash means a leaked hash cannot be
    replayed for a different address or a different flow.
    """
    secret = _get_auth_secret().encode("utf-8")
    msg = f"{email.strip().lower()}:{purpose}:{code}".encode("utf-8")
    return hmac.new(secret, msg, hashlib.sha256).hexdigest()


def code_expiry() -> datetime:
    return datetime.now(timezone.utc) + timedelta(seconds=AUTH_CODE_TTL_SECONDS)


# ── Session tokens ————————————————————————————————————————————————————————————

def generate_session_token() -> str:
    """Opaque random token — this is the ONLY value ever sent to the browser."""
    return secrets.token_urlsafe(32)


def hash_session_token(token: str) -> str:
    """HMAC-SHA256(AUTH_SECRET, token) — this is the ONLY value ever stored in Postgres."""
    secret = _get_auth_secret().encode("utf-8")
    return hmac.new(secret, token.encode("utf-8"), hashlib.sha256).hexdigest()


def session_expiry() -> datetime:
    return datetime.now(timezone.utc) + timedelta(seconds=SESSION_TTL_SECONDS)


# ── Cookie helpers ————————————————————————————————————————————————————————————

def set_session_cookie(response, token: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        secure=_cookie_secure(),
        samesite="lax",
        path="/",
    )


def clear_session_cookie(response) -> None:
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        path="/",
        httponly=True,
        secure=_cookie_secure(),
        samesite="lax",
    )
