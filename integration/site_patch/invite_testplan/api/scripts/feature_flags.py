"""
feature_flags.py
================
Server-side switches that temporarily take a product feature offline
WITHOUT deleting any of its code, data or plan definitions.

SONYA_TRAILER_ENABLED  (default: OFF)
    Trailer ("Filmbreaker", job mode `trailer_film_breaker`, plan mode
    `trailer`) is unavailable to every user until the owner explicitly
    allows it. Unset, empty or any value other than 1/true/yes/on means OFF,
    so a server that was deployed without the variable fails CLOSED.

    OFF blocks only the creation of NEW things:
      * new Trailer jobs   -- POST /api/generation/jobs (mode=trailer_film_breaker),
                              POST /api/generation/jobs/from-url, POST /api/trailer/jobs
      * new Trailer checkouts -- POST /api/billing/checkout with a trailer_* plan

    It deliberately does NOT touch:
      * PLAN_CATALOG / pricing.py -- the catalog is also how a payment that
        was already created is confirmed and priced (selfwork_billing,
        payment webhooks, reconcile), and how the account menu names an
        existing subscription;
      * existing jobs, outputs, payments, user_subscriptions, entitlements;
      * the worker / dispatcher path for jobs that are already queued.

    To re-enable: set SONYA_TRAILER_ENABLED=true in the API environment and
    restart the API. No code has to be restored.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

TRAILER_JOB_MODES = frozenset({"trailer_film_breaker"})
TRAILER_PLAN_MODE = "trailer"

TRAILER_DISABLED_CODE = "TRAILER_DISABLED"
TRAILER_DISABLED_MESSAGE = "Режим Trailer временно недоступен."

_TRUE = {"1", "true", "yes", "on"}


def trailer_enabled() -> bool:
    """Read on every call (not cached) so a restart with the new env is the
    only thing needed, and tests can flip it with monkeypatch.setenv."""
    return os.environ.get("SONYA_TRAILER_ENABLED", "").strip().lower() in _TRUE


def job_mode_disabled(job_mode: Optional[str]) -> bool:
    """True if a NEW job of this mode must be refused right now."""
    return job_mode in TRAILER_JOB_MODES and not trailer_enabled()


def plan_disabled(plan: Any) -> bool:
    """True if a NEW checkout of this plan must be refused right now.
    `plan` is a pricing.Plan (or anything with .plan_mode)."""
    if getattr(plan, "plan_mode", None) == "test":
        from scripts import pricing   # «Тест-драйв»: off until the owner approves the price
        return not pricing.test_plan_enabled()
    return getattr(plan, "plan_mode", None) == TRAILER_PLAN_MODE and not trailer_enabled()


def disabled_detail(trace_id: str, plan: Any = None) -> Dict[str, Any]:
    """`detail` of the 403 returned for a blocked Trailer job / checkout (or the «Тест-драйв» checkout)."""
    if getattr(plan, "plan_mode", None) == "test":
        return {"error": "feature_disabled", "code": "TEST_PLAN_DISABLED", "feature": "test_plan",
                "message": "Тариф «Тест-драйв» скоро появится.", "trace_id": trace_id}
    return {
        "error": "feature_disabled",
        "code": TRAILER_DISABLED_CODE,
        "feature": "trailer",
        "message": TRAILER_DISABLED_MESSAGE,
        "trace_id": trace_id,
    }


def public_flags() -> Dict[str, bool]:
    """What GET /api/features reports. Booleans only -- no config values."""
    from scripts import pricing
    return {"trailer": trailer_enabled(), "test_plan": pricing.test_plan_enabled()}
