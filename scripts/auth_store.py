"""
auth_store.py
=============
PostgreSQL data access layer for passwordless email auth.

Tables (from migration 008):
  users        — account records, plan/quota state
  auth_codes   — hashed one-time email verification codes
  sessions     — hashed opaque session tokens (cookie-based)

No raw auth codes or raw session tokens are ever persisted or returned by
this module — callers pass in pre-computed hashes (see auth_security.py).

All functions open/close their own connection (matches prod_job_store.py
style) since this is a low-QPS path (auth, not the hot job-processing path).
"""
from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DB_AVAILABLE = False
try:
    import psycopg2
    import psycopg2.extras
    _DB_AVAILABLE = True
except ImportError:
    pass


def _get_conn():
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL not set")
    if not _DB_AVAILABLE:
        raise RuntimeError("psycopg2 not installed — run: pip install psycopg2-binary")
    return psycopg2.connect(url, cursor_factory=psycopg2.extras.RealDictCursor)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_email(email: str) -> str:
    return email.strip().lower()


def _row(conn, sql: str, params: tuple) -> Optional[Dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
        return dict(row) if row else None


def _rows(conn, sql: str, params: tuple) -> List[Dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]


# ── Users ——————————————————————————————————————————————————————————————————————

def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    try:
        return _row(
            conn,
            "SELECT * FROM users WHERE LOWER(email) = LOWER(%s)",
            (normalize_email(email),),
        )
    finally:
        conn.close()


def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    try:
        return _row(conn, "SELECT * FROM users WHERE id = %s", (user_id,))
    finally:
        conn.close()


def create_user(email: str) -> Dict[str, Any]:
    """Create a new user with default free-plan fields. Caller must have
    already verified there is no existing user with this email."""
    user_id = str(uuid.uuid4())
    email_norm = normalize_email(email)
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO users (id, email, created_at, updated_at)
                    VALUES (%s, %s, %s, %s)
                    RETURNING *
                    """,
                    (user_id, email_norm, _now(), _now()),
                )
                row = cur.fetchone()
    finally:
        conn.close()
    return dict(row)


def get_or_create_user(email: str) -> Dict[str, Any]:
    """
    Atomically get-or-create a user by email.
    Uses ON CONFLICT to avoid a race between concurrent verify-code calls
    for a brand-new address (e.g. two tabs verifying the same code).
    """
    user_id = str(uuid.uuid4())
    email_norm = normalize_email(email)
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO users (id, email, created_at, updated_at)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (LOWER(email)) DO NOTHING
                    RETURNING *
                    """,
                    (user_id, email_norm, _now(), _now()),
                )
                row = cur.fetchone()
                if row:
                    return dict(row)
            return _row(conn, "SELECT * FROM users WHERE LOWER(email) = LOWER(%s)", (email_norm,))
    finally:
        conn.close()


def mark_email_verified(user_id: str) -> None:
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE users
                    SET email_verified_at = COALESCE(email_verified_at, %s),
                        updated_at = %s
                    WHERE id = %s
                    """,
                    (_now(), _now(), user_id),
                )
    finally:
        conn.close()


def increment_free_video_used(user_id: str) -> None:
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE users
                    SET free_video_used = free_video_used + 1,
                        updated_at = %s
                    WHERE id = %s
                    """,
                    (_now(), user_id),
                )
    finally:
        conn.close()


# ── Auth codes —————————————————————————————————————————————————————————————————

def create_auth_code(
    email: str,
    code_hash: str,
    purpose: str,
    expires_at: datetime,
    ip_address: Optional[str] = None,
) -> str:
    code_id = str(uuid.uuid4())
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO auth_codes
                        (id, email, code_hash, purpose, expires_at, ip_address, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (code_id, normalize_email(email), code_hash, purpose,
                     expires_at, ip_address, _now()),
                )
    finally:
        conn.close()
    return code_id


def get_latest_pending_code(email: str, purpose: str) -> Optional[Dict[str, Any]]:
    """Most recent, not-yet-consumed code for this email+purpose (may be expired)."""
    conn = _get_conn()
    try:
        return _row(
            conn,
            """
            SELECT * FROM auth_codes
            WHERE LOWER(email) = LOWER(%s) AND purpose = %s AND consumed_at IS NULL
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (email, purpose),
        )
    finally:
        conn.close()


def increment_code_attempts(code_id: str) -> int:
    """Increment attempts counter, return new attempts count."""
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE auth_codes SET attempts = attempts + 1 WHERE id = %s RETURNING attempts",
                    (code_id,),
                )
                row = cur.fetchone()
                return int(row["attempts"]) if row else 0
    finally:
        conn.close()


def consume_code(code_id: str) -> None:
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE auth_codes SET consumed_at = %s WHERE id = %s AND consumed_at IS NULL",
                    (_now(), code_id),
                )
    finally:
        conn.close()


def count_recent_codes(email: Optional[str], ip_address: Optional[str], since: datetime) -> Dict[str, int]:
    """Return counts of auth_codes rows created since `since`, by email and by ip.
    Used for request-code rate limiting in addition to the generic sliding-window limiter."""
    conn = _get_conn()
    try:
        by_email = 0
        by_ip = 0
        if email:
            row = _row(
                conn,
                "SELECT COUNT(*) AS n FROM auth_codes WHERE LOWER(email) = LOWER(%s) AND created_at >= %s",
                (email, since),
            )
            by_email = int(row["n"]) if row else 0
        if ip_address:
            row = _row(
                conn,
                "SELECT COUNT(*) AS n FROM auth_codes WHERE ip_address = %s AND created_at >= %s",
                (ip_address, since),
            )
            by_ip = int(row["n"]) if row else 0
        return {"by_email": by_email, "by_ip": by_ip}
    finally:
        conn.close()


# ── Sessions ———————————————————————————————————————————————————————————————————

def create_session(
    user_id: str,
    token_hash: str,
    expires_at: datetime,
    user_agent: Optional[str] = None,
    ip_address: Optional[str] = None,
) -> str:
    session_id = str(uuid.uuid4())
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO sessions
                        (id, user_id, token_hash, expires_at, user_agent, ip_address, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (session_id, user_id, token_hash, expires_at,
                     (user_agent or "")[:400] or None, ip_address, _now()),
                )
    finally:
        conn.close()
    return session_id


def get_active_session_by_token_hash(token_hash: str) -> Optional[Dict[str, Any]]:
    """Return the session row only if it is not revoked and not expired."""
    conn = _get_conn()
    try:
        return _row(
            conn,
            """
            SELECT * FROM sessions
            WHERE token_hash = %s AND revoked_at IS NULL AND expires_at > %s
            """,
            (token_hash, _now()),
        )
    finally:
        conn.close()


def revoke_session_by_token_hash(token_hash: str) -> None:
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE sessions SET revoked_at = %s WHERE token_hash = %s AND revoked_at IS NULL",
                    (_now(), token_hash),
                )
    finally:
        conn.close()
