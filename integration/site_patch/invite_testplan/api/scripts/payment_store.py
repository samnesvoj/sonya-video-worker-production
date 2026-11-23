"""
payment_store.py
=================
PostgreSQL data access layer for payments (migrations 010, 016) -- provider
independent: the payments row carries the plan snapshot and the provider
name; whichever provider created it confirms it through
process_successful_payment().

All functions open/close their own connection (matches auth_store.py style
-- low-QPS path, not the hot job-processing path).

process_successful_payment() is the ONLY place that transitions a payment
to 'paid' and extends a subscription -- and it does both in a single DB
transaction with a row lock, so a crash between "mark paid" and "extend
subscription" cannot happen (Postgres rolls back the whole thing, and a
retried provider callback safely redoes it from scratch), and two
concurrent callbacks for the same invoice_id serialize on the row lock
instead of racing.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, Optional, Tuple

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


def create_pending_payment(
    user_id: str, plan_id: str, amount: Decimal, is_test: bool,
    plan_type: str, duration_days: int,
    plan_mode: Optional[str] = None, ops_limit: Optional[int] = None,
    max_source_sec: Optional[int] = None, *, provider: str,
) -> Dict[str, Any]:
    """
    plan_type/duration_days and the entitlement terms (plan_mode, ops_limit,
    max_source_sec -- migration 016) are a snapshot of PLAN_CATALOG[plan_id]
    at checkout time (caller passes them in -- this module has no
    PLAN_CATALOG dependency), exactly like `amount`.
    process_successful_payment() below reads them back from this row, never
    from a fresh catalog lookup, so a later catalog change can't alter the
    terms of an already-created payment.
    """
    payment_id = str(uuid.uuid4())
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO payments
                        (id, user_id, plan_id, amount, is_test, plan_type, duration_days,
                         plan_mode, ops_limit, max_source_sec, provider, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING *
                    """,
                    (payment_id, user_id, plan_id, amount, is_test, plan_type, duration_days,
                     plan_mode, ops_limit, max_source_sec, provider, _now(), _now()),
                )
                row = cur.fetchone()
    finally:
        conn.close()
    return dict(row)


# ── Payment attempt state machine (migration 018) ────────────────────────
#
#   created ──handoff──> initializing ──provider: pending──> awaiting_payment
#      │                     │   (signed form given        │
#      │                     │    to a browser)            │
#      └─────────────────────┴─────────────────────────────┴──> paid
#                                                               failed / canceled / expired
#
# attempt_state is the fine-grained state; `status` keeps the coarse,
# provider-independent value every older reader understands (pending for
# all open states), so a rollback to pre-018 code still sees valid rows.
# NULL attempt_state = row created before 018 (Robokassa, Selfwork 3-9):
# treated as "open, provider state unknown" while status = 'pending'.
# Only process_successful_payment() moves a row to paid -- also from
# failed/canceled/expired when the provider itself reports success (money
# taken after we gave up must still activate).

ATTEMPT_OPEN = ("created", "initializing", "awaiting_payment")
ATTEMPT_CLOSED = ("failed", "canceled", "expired")
ATTEMPT_STATUS = {"created": "pending", "initializing": "pending", "awaiting_payment": "pending",
                  "paid": "paid", "failed": "failed", "canceled": "cancelled", "expired": "expired"}
# transition_attempt(): any open (or pre-018) attempt may move to one of
# these; created -> initializing is record_checkout_handoff()'s job.
_OPEN_OR_LEGACY = (None,) + ATTEMPT_OPEN
_TRANSITION_TARGETS = frozenset({"awaiting_payment"}) | frozenset(ATTEMPT_CLOSED)

# An open attempt is "fresh" -- the same purchase intent the user may still
# be finishing -- for this long after its last hand-off to a browser.
ATTEMPT_TTL_MINUTES = 60


def is_fresh(payment: Dict[str, Any], now: Optional[datetime] = None) -> bool:
    last = payment.get("last_handoff_at") or payment.get("created_at")
    if last is None:
        return False
    return ((now or _now()) - last).total_seconds() < ATTEMPT_TTL_MINUTES * 60


def get_or_create_checkout(
    user_id: str, plan_id: str, amount: Decimal, is_test: bool,
    plan_type: str, duration_days: int,
    plan_mode: Optional[str], ops_limit: Optional[int], max_source_sec: Optional[int],
    *, provider: str, idempotency_key: Optional[str] = None,
) -> Tuple[Dict[str, Any], str]:
    """
    One open payment attempt per user + plan + provider. Returns
    (payment, outcome), outcome one of:

      "existing_paid" -- this Idempotency-Key bought this plan moments ago
                         (a retry after paying); an older purchase releases
                         the key instead -- the next purchase is new
      "reused_key"    -- the open attempt this Idempotency-Key created
      "reused_open"   -- an open attempt of this user + plan + provider (new
                         key: reload, second tab/device, older client)
      "created"       -- a new attempt (attempt_state 'created')

    A returned open attempt is only a CANDIDATE: the caller checks it with
    the provider and either continues it or closes it and calls again
    (see scripts/selfwork_billing.py). The payment dict carries
    "_reusable": True when it is fresh and has the current terms.

    Runs under the user's row lock: concurrent checkouts of one user
    serialize, so the second always sees the first's attempt. The partial
    unique index uq_payments_one_open_attempt backs this up in the schema.
    """
    terms = {"amount": amount, "is_test": is_test, "plan_type": plan_type, "duration_days": duration_days,
             "plan_mode": plan_mode, "ops_limit": ops_limit, "max_source_sec": max_source_sec}
    now = _now()
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM users WHERE id = %s FOR UPDATE", (user_id,))

                def candidate(row: Dict[str, Any], outcome: str) -> Tuple[Dict[str, Any], str]:
                    row["_reusable"] = (row.get("attempt_state") in ATTEMPT_OPEN and is_fresh(row, now)
                                        and all(row.get(k) == v for k, v in terms.items()))
                    return row, outcome

                if idempotency_key:
                    cur.execute("SELECT * FROM payments WHERE user_id = %s AND idempotency_key = %s",
                                (user_id, idempotency_key))
                    row = cur.fetchone()
                    if row is not None:
                        row = dict(row)
                        if row["plan_id"] == plan_id and row.get("provider") == provider:
                            if row["status"] == "paid" and row.get("paid_at") and \
                                    (now - row["paid_at"]).total_seconds() < ATTEMPT_TTL_MINUTES * 60:
                                return row, "existing_paid"
                            if row["status"] == "pending":
                                return candidate(row, "reused_key")
                        # Closed attempt / older purchase / other plan: the
                        # key no longer names a live intent. Release it so
                        # the attempt made below carries it (a retry of THIS
                        # intent must find the new attempt, not a new one).
                        cur.execute("UPDATE payments SET idempotency_key = NULL WHERE id = %s", (row["id"],))

                cur.execute(
                    """
                    SELECT * FROM payments
                    WHERE user_id = %s AND plan_id = %s AND provider = %s AND status = 'pending'
                    ORDER BY created_at DESC
                    """,
                    (user_id, plan_id, provider),
                )
                for row in cur.fetchall():
                    row = dict(row)
                    if row.get("attempt_state") == "created" and not candidate(row, "")[0]["_reusable"]:
                        # Never handed to a browser -> the provider can't know
                        # it: safe to expire right here, no provider call.
                        _set_attempt_state(cur, row["id"], "expired", now)
                        continue
                    return candidate(row, "reused_open")

                cur.execute(
                    """
                    INSERT INTO payments
                        (id, user_id, plan_id, amount, is_test, plan_type, duration_days,
                         plan_mode, ops_limit, max_source_sec, provider, idempotency_key,
                         attempt_state, checkout_handoffs, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'created', 0, %s, %s)
                    RETURNING *
                    """,
                    (str(uuid.uuid4()), user_id, plan_id, amount, is_test, plan_type, duration_days,
                     plan_mode, ops_limit, max_source_sec, provider,
                     idempotency_key, now, now),
                )
                row = dict(cur.fetchone())
                row["_reusable"] = True
                return row, "created"
    finally:
        conn.close()


def _set_attempt_state(cur, payment_id, state: str, now: datetime) -> None:
    cur.execute(
        "UPDATE payments SET attempt_state = %s, status = %s, updated_at = %s WHERE id = %s",
        (state, ATTEMPT_STATUS[state], now, payment_id),
    )


def record_checkout_handoff(invoice_id: int) -> Optional[Dict[str, Any]]:
    """The signed provider form of this attempt is being given to a browser:
    created/legacy -> initializing, handoffs + 1. Returns
    {"attempt_state", "checkout_handoffs"} or None if no longer open."""
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE payments
                    SET checkout_handoffs = COALESCE(checkout_handoffs, 0) + 1, last_handoff_at = %s,
                        attempt_state = CASE WHEN attempt_state IS NULL OR attempt_state = 'created'
                                             THEN 'initializing' ELSE attempt_state END,
                        updated_at = %s
                    WHERE invoice_id = %s AND status = 'pending'
                    RETURNING attempt_state, checkout_handoffs
                    """,
                    (_now(), _now(), invoice_id),
                )
                row = cur.fetchone()
                return dict(row) if row else None
    finally:
        conn.close()


def transition_attempt(
    invoice_id: int, to_state: str, *,
    provider_status: Optional[str] = None, raw_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Move an OPEN attempt to `to_state` (never to/from paid -- that is
    process_successful_payment's job). Idempotent: an attempt already in
    `to_state`, or no longer open, is left alone.

    Returns {"changed": bool, "before": state, "after": state}."""
    if to_state not in _TRANSITION_TARGETS:
        raise ValueError(f"not a transition target: {to_state}")
    now = _now()
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id, status, attempt_state FROM payments WHERE invoice_id = %s FOR UPDATE",
                            (invoice_id,))
                row = cur.fetchone()
                if row is None:
                    return {"changed": False, "before": None, "after": None}
                before = row["attempt_state"]
                if row["status"] != "pending" or before not in _OPEN_OR_LEGACY or before == to_state:
                    if provider_status:
                        cur.execute("UPDATE payments SET provider_status = %s, provider_checked_at = %s "
                                    "WHERE id = %s", (provider_status, now, row["id"]))
                    return {"changed": False, "before": before, "after": before}
                closing = to_state in ATTEMPT_CLOSED
                cur.execute(
                    """
                    UPDATE payments
                    SET attempt_state = %s, status = %s,
                        provider_status = COALESCE(%s, provider_status),
                        provider_checked_at = CASE WHEN %s IS NULL THEN provider_checked_at ELSE %s END,
                        result_received_at = CASE WHEN %s THEN %s ELSE result_received_at END,
                        raw_result_params = COALESCE(%s, raw_result_params),
                        updated_at = %s
                    WHERE id = %s
                    """,
                    (to_state, ATTEMPT_STATUS[to_state], provider_status, provider_status, now,
                     closing, now, json.dumps(raw_params, default=str) if raw_params is not None else None,
                     now, row["id"]),
                )
                return {"changed": True, "before": before, "after": to_state}
    finally:
        conn.close()


def record_provider_payment(invoice_id: int, provider_payment_id: str,
                            confirmation_url: Optional[str]) -> Optional[Dict[str, Any]]:
    """Bind the provider's payment id (migration 019) to this row. Only ever
    sets it once: a row that already carries ANOTHER id is left alone and
    None is returned (the caller refuses -- one attempt, one provider
    payment). Re-recording the same id refreshes confirmation_url."""
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE payments
                    SET provider_payment_id = %s,
                        confirmation_url = COALESCE(%s, confirmation_url),
                        updated_at = %s
                    WHERE invoice_id = %s
                      AND (provider_payment_id IS NULL OR provider_payment_id = %s)
                    RETURNING *
                    """,
                    (provider_payment_id, confirmation_url, _now(), invoice_id, provider_payment_id),
                )
                row = cur.fetchone()
                return dict(row) if row else None
    finally:
        conn.close()


def get_payment_by_provider_payment_id(provider: str, provider_payment_id: str) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM payments WHERE provider = %s AND provider_payment_id = %s",
                        (provider, provider_payment_id))
            row = cur.fetchone()
            return dict(row) if row else None
    finally:
        conn.close()


def get_payment_by_invoice_id(invoice_id: int) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM payments WHERE invoice_id = %s", (invoice_id,))
            row = cur.fetchone()
            return dict(row) if row else None
    finally:
        conn.close()


def process_successful_payment(invoice_id: int, out_sum: Decimal, raw_params: Dict[str, Any],
                               *, reopen_closed: bool = False,
                               fiscal: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Atomically: lock the payment row, verify it's pending and the amount
    matches, mark it paid, and extend the user's subscription -- all in one
    transaction. Returns {"result": ..., "payment": {...}} where result is
    one of: "activated", "already_processed", "not_found", "invalid_state",
    "amount_mismatch".

    Caller (payment_routes.py) must call this ONLY after verify_result_signature
    has already passed -- this function does not check the signature.

    reopen_closed=True (Selfwork, after its own GET /status said
    succeeded): a failed/cancelled/expired attempt is still activated --
    the provider took the money, our close was a guess.

    fiscal (YooKassa, when FISCAL_OUTBOX_ENABLED): data for the receipt
    outbox row, inserted in this same transaction right after activation
    (scripts/fiscal_store.py; a refused insert never blocks the payment).
    Only on "activated" -- a repeated confirmation adds nothing.
    """
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM payments WHERE invoice_id = %s FOR UPDATE",
                    (invoice_id,),
                )
                payment = cur.fetchone()
                if not payment:
                    return {"result": "not_found"}
                payment = dict(payment)

                if payment["status"] == "paid":
                    return {"result": "already_processed", "payment": payment}
                reopened = reopen_closed and payment["status"] in ("failed", "cancelled", "expired")
                if payment["status"] != "pending" and not reopened:
                    return {"result": "invalid_state", "payment": payment}
                if reopened:
                    logger.warning("[payment] paid_after_close invoice_id=%s closed_status=%s",
                                   invoice_id, payment["status"])
                if payment["amount"] != out_sum:
                    logger.warning(
                        "[payment] amount_mismatch invoice_id=%s expected=%s received=%s",
                        invoice_id, payment["amount"], out_sum,
                    )
                    return {"result": "amount_mismatch", "payment": payment}

                cur.execute(
                    """
                    UPDATE payments
                    SET status = 'paid', attempt_state = 'paid', paid_at = %s, result_received_at = %s,
                        raw_result_params = %s, updated_at = %s
                    WHERE id = %s
                    RETURNING *
                    """,
                    (_now(), _now(), json.dumps(raw_params, default=str), _now(), payment["id"]),
                )
                updated = dict(cur.fetchone())

                # Terms come from THIS payment's own snapshot (set at
                # checkout time) -- never a fresh PLAN_CATALOG lookup, so a
                # catalog change made after checkout can't alter what an
                # already-pending payment activates.
                if payment.get("plan_mode") in ("gen", "beep"):
                    # One-off pack (SONYA Gen images / BEEP processings): the paid row IS the
                    # entitlement (balance = paid packs - usage: prod_generation_api._gen_quota,
                    # prod_job_store.pack_left). No user_subscriptions row.
                    logger.info("[payment] %s_pack_paid invoice_id=%s user_id=%s units=%s", payment.get("plan_mode"),
                                invoice_id, payment["user_id"], payment.get("ops_limit"))
                    return {"result": "activated", "payment": updated, "subscription": None,
                            "fiscal_receipt": _fiscal_outbox(cur, updated, fiscal)}
                if payment.get("plan_mode") == "test":
                    # «Тест-драйв»: one counter per video mode (plan_mode test:<slot>, migration 026).
                    rows = _activate_test_plan(cur, payment)
                    return {"result": "activated", "payment": updated, "subscription": rows[0] if rows else None,
                            "fiscal_receipt": _fiscal_outbox(cur, updated, fiscal)}
                if payment.get("plan_mode"):
                    subscription = _activate_subscription(cur, payment)
                    return {"result": "activated", "payment": updated, "subscription": subscription,
                            "fiscal_receipt": _fiscal_outbox(cur, updated, fiscal)}

                # Legacy payment (created before migration 016, e.g. a
                # pending pro_30d): activate exactly what was sold then.
                cur.execute(
                    """
                    UPDATE users
                    SET plan_type = %s,
                        plan_status = 'active',
                        plan_active_until = GREATEST(NOW(), COALESCE(plan_active_until, NOW()))
                                             + (%s || ' days')::interval,
                        updated_at = %s
                    WHERE id = %s
                    """,
                    (payment["plan_type"], payment["duration_days"], _now(), payment["user_id"]),
                )
                fiscal_outcome = _fiscal_outbox(cur, updated, fiscal)
            return {"result": "activated", "payment": updated, "fiscal_receipt": fiscal_outcome}
    finally:
        conn.close()


def _fiscal_outbox(cur, payment: Dict[str, Any], fiscal: Optional[Dict[str, Any]]) -> Optional[str]:
    if not fiscal:
        return None
    from scripts import fiscal_store  # fiscal_store imports this module
    return fiscal_store.insert_pending_in_tx(cur, payment, fiscal)


def _activate_test_plan(cur, payment: Dict[str, Any]) -> list:
    """«Тест-драйв» (pricing.TEST_PLAN): inside process_successful_payment's transaction, under the user's row
    lock, open ops_limit processings for EACH slot in pricing.TEST_SLOTS -- one user_subscriptions row per slot,
    plan_mode "test:<slot>", all with this payment's period and length cap. Idempotent per payment: a replayed
    callback finds the rows of this payment_id and opens nothing new."""
    from scripts.pricing import TEST_SLOTS
    now = _now()
    cur.execute("SELECT id FROM users WHERE id = %s FOR UPDATE", (payment["user_id"],))
    cur.execute("SELECT * FROM user_subscriptions WHERE payment_id = %s", (payment["id"],))
    existing = [dict(r) for r in cur.fetchall()]
    if existing:
        return existing
    rows = []
    for slot in TEST_SLOTS:
        cur.execute(
            """
            INSERT INTO user_subscriptions
                (id, user_id, plan_id, plan_mode, ops_limit, ops_used, max_source_sec,
                 period_start, period_end, payment_id, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, 0, %s, %s, %s + (%s || ' days')::interval, %s, %s, %s)
            RETURNING *
            """,
            (str(uuid.uuid4()), payment["user_id"], payment["plan_id"], "test:" + slot,
             payment["ops_limit"], payment["max_source_sec"], now, now, payment["duration_days"],
             payment["id"], now, now),
        )
        rows.append(dict(cur.fetchone()))
    logger.info("[payment] test_plan_activated user_id=%s payment_id=%s slots=%s", payment["user_id"],
                payment["id"], len(rows))
    return rows


def _activate_subscription(cur, payment: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inside process_successful_payment's transaction. Serializes on the
    user's row (same lock create_job_with_quota takes), closes any still
    running period for the same mode (checkout refuses a second purchase
    while that period has operations left -- see payment_routes.checkout --
    so in practice this only ends an exhausted period), and opens a fresh
    period with the purchased plan's own terms. users.plan_type is left
    alone: a new plan never grants the legacy unlimited "pro".
    """
    now = _now()
    cur.execute("SELECT id FROM users WHERE id = %s FOR UPDATE", (payment["user_id"],))
    cur.execute(
        """
        UPDATE user_subscriptions
        SET period_end = %s
        WHERE user_id = %s AND plan_mode = %s AND period_end > %s
        RETURNING id, plan_id, ops_used, ops_limit
        """,
        (now, payment["user_id"], payment["plan_mode"], now),
    )
    for closed in cur.fetchall():
        if closed["ops_used"] < closed["ops_limit"]:
            logger.warning(
                "[payment] subscription_superseded_with_ops_left subscription_id=%s plan_id=%s "
                "ops_used=%s ops_limit=%s new_payment_id=%s",
                closed["id"], closed["plan_id"], closed["ops_used"], closed["ops_limit"], payment["id"],
            )
    cur.execute(
        """
        INSERT INTO user_subscriptions
            (id, user_id, plan_id, plan_mode, ops_limit, ops_used, max_source_sec,
             period_start, period_end, payment_id, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, 0, %s, %s, %s + (%s || ' days')::interval, %s, %s, %s)
        RETURNING *
        """,
        (str(uuid.uuid4()), payment["user_id"], payment["plan_id"], payment["plan_mode"],
         payment["ops_limit"], payment["max_source_sec"], now, now, payment["duration_days"],
         payment["id"], now, now),
    )
    return dict(cur.fetchone())
