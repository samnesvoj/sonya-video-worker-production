"""
payment_routes.py
==================
Billing endpoints for SONYA.

  POST /api/billing/checkout                  (browser, cookie auth, optional Idempotency-Key) -- provider independent
  GET  /api/billing/payment-status             (browser, cookie auth) -- provider independent
  GET  /api/billing/checkout-availability      (public) -- is any provider configured
  POST /api/billing/robokassa/result           LEGACY Robokassa ResultURL (server-to-server)
  GET  /api/billing/selfwork/pay/{invoice_id}  (browser, cookie auth) -- fallback: same signed form as an auto-submit page
  POST /api/billing/selfwork/callback          Самозанятые.рф callback (server-to-server)
  POST /api/billing/yookassa/webhook           ЮKassa HTTP notification (server-to-server)

Checkout takes only a plan_id: price and terms come from the server catalog
(scripts/pricing.py) and are snapshotted on the payments row; the active
provider (scripts/payment_providers.py, env PAYMENT_PROVIDER) only turns
that row into a redirect URL. A provider's webhook confirms the payment via
payment_store.process_successful_payment(), which activates exactly the
snapshotted plan. No provider configured -> checkout returns 503
payment_unavailable before creating anything.

Same pattern as scripts/auth_routes.py: a plain APIRouter() with no
prefix, each path spelled out in full ("/api/...") in the decorator, and
included in the app with app.include_router(payment_router) -- no
prefix= argument. This keeps the actual route exactly
"/api/billing/robokassa/result", never "/billing/..." or "/api/api/...".

The return pages (payment/success.html, payment/fail.html) are static and
only ever call GET /api/billing/payment-status -- they never hit an endpoint
that can change payment/subscription state. Only a provider's verified
server-to-server notification confirms a payment; see
scripts/payment_store.py::process_successful_payment.
"""
from __future__ import annotations

import hashlib
import html
import logging
import re
from decimal import Decimal, InvalidOperation

from fastapi import APIRouter, Depends, Form, Header, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse
from typing import Optional

from pydantic import BaseModel

from scripts import entitlements, payment_store, selfwork, selfwork_billing, yookassa, yookassa_billing
from scripts.selfwork_billing import plog
from scripts.payment_providers import RobokassaProvider, SelfworkProvider, YookassaProvider, get_checkout_provider
from scripts import feature_flags
from scripts.pricing import get_plan
from scripts.security import get_current_user, new_trace_id, verify_browser_origin

logger = logging.getLogger(__name__)

router = APIRouter()


class CheckoutRequest(BaseModel):
    plan_id: str


def _iso(dt):
    return dt.isoformat() if dt else None


# One key per purchase intent (auth.js keeps it across retries of the same
# click). Same alphabet as the generation-jobs Idempotency-Key.
_IDEMPOTENCY_KEY_RE = re.compile(r"^[A-Za-z0-9_\-.:]{1,255}$")


def _checkout_idempotency_key(raw: Optional[str], trace_id: str) -> Optional[str]:
    if raw is None:
        return None
    key = raw.strip()
    if not _IDEMPOTENCY_KEY_RE.match(key):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail={"error": "invalid_idempotency_key", "trace_id": trace_id})
    return key


def _key_hash(key: Optional[str]) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:12] if key else "-"


def _paid_checkout_response(payment: dict) -> dict:
    invoice_id = payment["invoice_id"]
    return {"invoice_id": invoice_id, "status": "paid",
            "redirect_url": f"/payment/success.html?invoice_id={invoice_id}"}


_MAX_CHECKOUT_ROUNDS = 8

# Providers with payment attempts (migration 018): the module that asks the
# provider what happened to an attempt (settle_candidate / sync).
_ATTEMPT_BILLING = {SelfworkProvider.name: selfwork_billing, YookassaProvider.name: yookassa_billing}


def _provider_attempt(user: dict, plan, is_test: bool, idempotency_key: Optional[str], trace_id: str,
                      provider_name: str):
    """The attempt this checkout continues or starts. Returns
    (payment, outcome); outcome "paid" = this intent is already paid.

    One open attempt per user + plan + provider
    (payment_store.get_or_create_checkout, under the user's row lock). An
    existing attempt is a candidate only: the provider's status API decides
    whether it is continued (same provider order/payment), was already
    paid, or is closed -- then the next round starts a fresh attempt. Never
    a second provider order for one attempt."""
    billing = _ATTEMPT_BILLING[provider_name]
    key_hash = _key_hash(idempotency_key)
    for _round in range(_MAX_CHECKOUT_ROUNDS):
        try:
            payment, outcome = payment_store.get_or_create_checkout(
                user_id=str(user["id"]), plan_id=plan.plan_id, amount=plan.amount, is_test=is_test,
                plan_type=plan.plan_type, duration_days=plan.duration_days,
                plan_mode=plan.plan_mode, ops_limit=plan.ops_limit, max_source_sec=plan.max_source_sec,
                provider=provider_name, idempotency_key=idempotency_key,
            )
        except Exception as exc:
            plog("checkout_failed", logging.ERROR, trace_id=trace_id, user_id=user["id"], plan_id=plan.plan_id,
                 provider=provider_name, error_code=f"db_{type(exc).__name__}", key=key_hash)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail={"error": "internal_error", "trace_id": trace_id})
        if outcome == "existing_paid":
            plog("checkout_existing_paid", trace_id=trace_id, user_id=user["id"],
                 **selfwork_billing._base(payment), key=key_hash)
            return payment, "paid"
        if outcome == "created":
            return payment, outcome
        try:
            verdict = billing.settle_candidate(payment, trace_id, "checkout",
                                               reusable=payment.get("_reusable", False))
        except Exception as exc:
            plog("checkout_failed", logging.ERROR, trace_id=trace_id, **selfwork_billing._base(payment),
                 error_code=f"reconcile_{type(exc).__name__}", key=key_hash)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail={"error": "internal_error", "trace_id": trace_id})
        if verdict == "paid":
            return payment, "paid"
        if verdict == "error":
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,
                                detail={"error": f"{provider_name}_rejected", "trace_id": trace_id})
        if verdict == "continue":
            return payment, outcome
    plog("checkout_failed", logging.ERROR, trace_id=trace_id, user_id=user["id"], plan_id=plan.plan_id,
         provider=provider_name, error_code="too_many_rounds", key=key_hash)
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail={"error": "internal_error", "trace_id": trace_id})


# ── POST /api/billing/checkout ───────────────────────────────────────────────

@router.post("/api/billing/checkout")
async def checkout(
    body: CheckoutRequest,
    request: Request,
    user: dict = Depends(get_current_user),
    _origin: None = Depends(verify_browser_origin),
    idempotency_key_header: Optional[str] = Header(None, alias="Idempotency-Key"),
):
    trace_id = new_trace_id()
    idempotency_key = _checkout_idempotency_key(idempotency_key_header, trace_id)

    plan = get_plan(body.plan_id)
    if plan is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "unknown_plan", "trace_id": trace_id},
        )

    # Feature flag: Trailer plans cannot be bought while Trailer is offline
    # (scripts/feature_flags.py). Only NEW checkouts stop here -- the plan
    # stays in PLAN_CATALOG, so a payment created earlier is still priced,
    # confirmed and activated by its own callback / reconcile path.
    if feature_flags.plan_disabled(plan):
        logger.info("[payment] checkout_refused feature_disabled plan_id=%s user_id=%s trace_id=%s",
                    plan.plan_id, user["id"], trace_id)
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail=feature_flags.disabled_detail(trace_id, plan))

    # «Тест-драйв» is sold once per account (any earlier test:* period, running or not, blocks a new one).
    if plan.plan_mode == "test":
        try:
            had_test = any(str(s.get("plan_mode") or "").startswith("test:")
                           for s in entitlements.get_user_subscriptions(str(user["id"])))
        except Exception as exc:
            logger.error("[payment] db_error operation=get_user_subscriptions user_id=%s trace_id=%s error_type=%s",
                         user["id"], trace_id, type(exc).__name__)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail={"error": "internal_error", "trace_id": trace_id})
        if had_test:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail={
                "error": "test_plan_used", "trace_id": trace_id,
                "message": "Тест-драйв уже был на этом аккаунте. Выберите обычный тариф."})

    # A running period for the same mode that still has operations left
    # would be closed by the new purchase (see
    # payment_store._activate_subscription) -- refuse instead of silently
    # burning what the user already paid for. Exhausted/expired periods
    # don't block buying the next one.
    try:
        current = next(
            (s for s in entitlements.active_subscriptions(entitlements.get_user_subscriptions(str(user["id"])))
             if s["plan_mode"] == plan.plan_mode and s["ops_used"] < s["ops_limit"]),
            None,
        )
    except Exception as exc:
        logger.error(
            "[payment] db_error operation=get_user_subscriptions user_id=%s trace_id=%s error_type=%s",
            user["id"], trace_id, type(exc).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "internal_error", "trace_id": trace_id},
        )
    if current is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"error": "subscription_active", "trace_id": trace_id,
                    "subscription": entitlements.serialize_subscription(current)},
        )

    provider = get_checkout_provider()
    if provider is None:
        # No acquiring connected yet (Самозанятые.рф integration pending,
        # Robokassa retired) -- refuse before creating a payment row the
        # user could never pay.
        logger.warning("[payment] checkout_unavailable no_provider user_id=%s plan_id=%s trace_id=%s",
                       user["id"], plan.plan_id, trace_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "payment_unavailable", "trace_id": trace_id},
        )

    if not provider.is_configured():
        # Credentials / mode / receipt settings missing or invalid -- refuse
        # before an attempt exists (the reason is in config, never echoed).
        plog("checkout_failed", logging.ERROR, trace_id=trace_id, user_id=user["id"], plan_id=plan.plan_id,
             provider=provider.name, error_code="provider_not_configured")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "payment_unavailable", "trace_id": trace_id},
        )

    is_test = provider.is_test_mode()
    is_selfwork = provider.name == SelfworkProvider.name
    is_yookassa = provider.name == YookassaProvider.name
    key_hash = _key_hash(idempotency_key)

    if provider.name not in _ATTEMPT_BILLING:
        # Legacy providers (Robokassa) keep their pre-v2 behaviour: one row
        # per checkout, no attempt state machine.
        try:
            payment = payment_store.create_pending_payment(
                user_id=str(user["id"]), plan_id=plan.plan_id, amount=plan.amount, is_test=is_test,
                plan_type=plan.plan_type, duration_days=plan.duration_days,
                plan_mode=plan.plan_mode, ops_limit=plan.ops_limit, max_source_sec=plan.max_source_sec,
                provider=provider.name,
            )
        except Exception as exc:
            plog("checkout_failed", logging.ERROR, trace_id=trace_id, user_id=user["id"], plan_id=plan.plan_id,
                 provider=provider.name, error_code=f"db_{type(exc).__name__}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": "internal_error", "trace_id": trace_id},
            )
        outcome = "created"
    else:
        # Selfwork only accepts /init posted from the verified shop domain
        # (it matches the browser's Origin; from e.g. www. its page fails).
        # Refuse before an attempt exists and tell the client where to go.
        shop_origin = selfwork_billing.shop_origin() if is_selfwork else None
        origin = (request.headers.get("origin") or "").rstrip("/")
        if shop_origin and origin and origin != shop_origin:
            plog("checkout_refused", logging.WARNING, trace_id=trace_id, user_id=user["id"],
                 plan_id=plan.plan_id, provider=provider.name, error_code="wrong_origin", key=key_hash)
            raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                                detail={"error": "wrong_origin", "canonical_url": shop_origin, "trace_id": trace_id})
        # Blocking DB + provider calls: off the event loop.
        payment, outcome = await run_in_threadpool(
            _provider_attempt, user, plan, is_test, idempotency_key, trace_id, provider.name)
        if outcome == "paid":
            return _paid_checkout_response(payment)

    form = None
    try:
        if is_yookassa:
            # The payer's account email goes only into the 54-FZ receipt
            # (when YOOKASSA_RECEIPT_ENABLED) -- never into logs.
            redirect_url = await run_in_threadpool(
                provider.create_checkout, payment, plan, trace_id=trace_id,
                customer_email=user.get("email"), key=key_hash, user_id=user["id"])
        else:
            redirect_url = provider.create_checkout(payment, plan)
        if is_selfwork:
            form = selfwork_billing.handoff(payment, trace_id, "checkout", outcome=outcome, key=key_hash,
                                            user_id=user["id"])
    except yookassa.YookassaError as exc:
        # YooKassa refused / unreachable / its answer did not match: the
        # attempt stays open (a retry gets the SAME payment back via its
        # Idempotence-Key) or is closed if YooKassa canceled it.
        plog("checkout_failed", logging.ERROR, trace_id=trace_id, **selfwork_billing._base(payment),
             error_code="yookassa_auth" if exc.auth_failed else "yookassa_create_failed",
             provider_http=exc.http_status, provider_code=exc.provider_code, key=key_hash)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"error": "payment_provider_unavailable", "trace_id": trace_id},
        )
    except (RuntimeError, ValueError) as exc:
        # Missing/misconfigured provider credentials -- never leak which
        # env var, just fail safe with a trace_id for server-side logs.
        plog("checkout_failed", logging.ERROR, trace_id=trace_id, **selfwork_billing._base(payment),
             error_code="provider_config", key=key_hash)
        logger.error("[payment] checkout_config_error trace_id=%s: %s", trace_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "payment_unavailable", "trace_id": trace_id},
        )

    if provider.name not in _ATTEMPT_BILLING:
        plog("checkout_created", trace_id=trace_id, user_id=user["id"], **selfwork_billing._base(payment),
             key=key_hash)
    return {"invoice_id": payment["invoice_id"], "status": "pending", "reused": outcome != "created",
            "redirect_url": redirect_url, "form": form}


# ── POST /api/billing/robokassa/result (LEGACY Robokassa ResultURL) ─────────
# Kept only so a payment created through Robokassa can still be confirmed.
# It refuses any payment whose provider isn't "robokassa".

@router.post("/api/billing/robokassa/result")
async def robokassa_result(
    request: Request,
    OutSum: str = Form(...),
    InvId: int = Form(...),
    SignatureValue: str = Form(...),
):
    """
    Order matters here and is deliberate (see the review notes this fixes):
      1. Verify SignatureValue first, against BOTH production and test
         Password#2 -- no database call at all yet. A request with no
         valid signature for either mode is rejected immediately, so
         garbage/scanning traffic against this public, unauthenticated
         endpoint never reaches Postgres.
      2. Only once a signature validates, look up the payment by InvId and
         confirm its stored is_test flag matches whichever password
         actually matched (a signature valid under the TEST password must
         never activate a payment created in production mode, or vice
         versa).
      3. Only then compare Decimal(OutSum) against the stored amount and
         atomically process the payment.
    """
    trace_id = new_trace_id()

    # Imported here, not at module level: only this legacy route needs
    # Robokassa -- checkout and the tariff system never import it.
    from scripts.robokassa import verify_result_signature_for_either_mode

    matched_is_test = verify_result_signature_for_either_mode(OutSum, InvId, SignatureValue)
    if matched_is_test is None:
        logger.warning("[payment] result_bad_signature invoice_id=%s trace_id=%s", InvId, trace_id)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid signature")

    try:
        out_sum_decimal = Decimal(OutSum)
    except InvalidOperation:
        logger.warning("[payment] result_bad_out_sum invoice_id=%s out_sum=%r trace_id=%s", InvId, OutSum, trace_id)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid amount")

    try:
        payment = payment_store.get_payment_by_invoice_id(InvId)
    except Exception as exc:
        logger.error(
            "[payment] db_error operation=get_payment_by_invoice_id invoice_id=%s trace_id=%s error_type=%s",
            InvId, trace_id, type(exc).__name__,
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                             detail={"error": "internal_error", "trace_id": trace_id})

    if payment is None:
        logger.warning("[payment] result_unknown_invoice invoice_id=%s trace_id=%s", InvId, trace_id)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="unknown invoice")

    if (payment.get("provider") or RobokassaProvider.name) != RobokassaProvider.name:
        # invoice_id is a shared order number -- a Robokassa-signed callback
        # must never confirm another provider's payment.
        logger.warning("[payment] result_wrong_provider invoice_id=%s provider=%s trace_id=%s",
                       InvId, payment.get("provider"), trace_id)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="unknown invoice")

    if payment["is_test"] != matched_is_test:
        logger.warning(
            "[payment] result_mode_mismatch invoice_id=%s trace_id=%s expected_is_test=%s matched_is_test=%s",
            InvId, trace_id, payment["is_test"], matched_is_test,
        )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid signature")

    raw_params = dict((await request.form()))
    try:
        outcome = payment_store.process_successful_payment(InvId, out_sum_decimal, raw_params)
    except Exception as exc:
        logger.error(
            "[payment] db_error operation=process_successful_payment invoice_id=%s trace_id=%s error_type=%s",
            InvId, trace_id, type(exc).__name__,
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                             detail={"error": "internal_error", "trace_id": trace_id})

    if outcome["result"] in ("activated", "already_processed"):
        if outcome["result"] == "activated":
            logger.info("[payment] result_activated invoice_id=%s trace_id=%s", InvId, trace_id)
        return PlainTextResponse(f"OK{InvId}")

    logger.error(
        "[payment] result_rejected invoice_id=%s outcome=%s trace_id=%s",
        InvId, outcome["result"], trace_id,
    )
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=outcome["result"])


# ── Самозанятые.рф (Selfwork) ────────────────────────────────────────────────
# Smart payment per docs.selfwork.ru: Selfwork's /init takes a browser form
# post from the shop's own origin and answers with its own HTML payment
# page. Checkout returns the server-signed form (the API key never reaches
# the browser) and auth.js posts it straight from the SONYA page -- no
# extra hop. /selfwork/pay/{id} serves the same form as a self-submitting
# page: the fallback for an older cached auth.js and for "continue this
# payment" links. Neither ever creates an attempt. All provider decisions
# live in scripts/selfwork_billing.py.

_SELFWORK_PAY_PAGE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="referrer" content="origin">
<title>Переход к оплате — SONYA</title>
<style>body{{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;
font-family:system-ui,-apple-system,sans-serif;background:#0b0b10;color:#e8e8f0;text-align:center}}
button{{margin-top:16px;padding:12px 28px;border:0;border-radius:999px;font-size:16px;cursor:pointer}}
.js #selfwork-continue{{visibility:hidden}}</style>
<script>document.documentElement.className='js';</script>
</head><body>
<form id="selfwork-pay" action="{action}" method="POST">
{inputs}
<p>Переходим на страницу оплаты Самозанятые.рф…</p>
<button id="selfwork-continue" type="submit">Перейти к оплате</button>
</form>
<script>
document.getElementById('selfwork-pay').submit();
// Only if the automatic post did not navigate (blocked / stalled) does
// the user see the button -- it posts the SAME order again.
setTimeout(function () {{ document.getElementById('selfwork-continue').style.visibility = 'visible'; }}, 4000);
</script>
</body></html>"""

@router.get("/api/billing/selfwork/pay/{invoice_id}")
async def selfwork_pay(invoice_id: int, user: dict = Depends(get_current_user)):
    """Fallback hand-off of an EXISTING attempt. Never creates one."""
    trace_id = new_trace_id()
    payment = payment_store.get_payment_by_invoice_id(invoice_id)
    if (payment is None or payment["user_id"] != user["id"]
            or payment.get("provider") != SelfworkProvider.name):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail={"error": "not_found", "trace_id": trace_id})
    done = RedirectResponse(f"/payment/success.html?invoice_id={invoice_id}", status_code=303)
    if payment["status"] != "pending":
        return done

    reusable = (payment.get("attempt_state") in payment_store.ATTEMPT_OPEN and payment_store.is_fresh(payment))
    try:
        verdict = await run_in_threadpool(
            selfwork_billing.settle_candidate, payment, trace_id, "pay_page", reusable=reusable)
    except Exception as exc:
        plog("pay_page_failed", logging.ERROR, trace_id=trace_id, **selfwork_billing._base(payment),
             error_code=f"reconcile_{type(exc).__name__}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail={"error": "internal_error", "trace_id": trace_id})
    if verdict in ("paid", "closed"):
        return done
    if verdict == "error":
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,
                            detail={"error": "selfwork_rejected", "trace_id": trace_id})

    try:
        form = selfwork_billing.handoff(payment, trace_id, "pay_page", user_id=user["id"])
    except (RuntimeError, ValueError):
        plog("pay_page_failed", logging.ERROR, trace_id=trace_id, **selfwork_billing._base(payment),
             error_code="provider_config")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail={"error": "payment_unavailable", "trace_id": trace_id})

    inputs = "\n".join(
        f'<input type="hidden" name="{html.escape(k)}" value="{html.escape(v)}">' for k, v in form["fields"]
    )
    return HTMLResponse(
        _SELFWORK_PAY_PAGE.format(action=html.escape(form["action"]), inputs=inputs),
        headers={"Cache-Control": "no-store", "Referrer-Policy": "origin"},
    )


@router.post("/api/billing/selfwork/callback")
async def selfwork_callback(request: Request):
    """payment.succeeded notification. Its body only names the order: the
    decision is Selfwork's own GET /status (selfwork_billing.sync), so a
    forged, duplicated or out-of-order callback cannot activate anything
    twice or wrongly."""
    trace_id = new_trace_id()
    ip = request.client.host if request.client else "unknown"

    try:
        body = await request.json()
    except Exception:
        body = None
    invoice_id = selfwork.invoice_id_from_order_id(body.get("order_id")) if isinstance(body, dict) else None
    if invoice_id is None:
        plog("selfwork_callback", logging.WARNING, trace_id=trace_id, source="callback", ip=ip,
             error_code="bad_body")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid notification")
    try:
        signature_ok = selfwork.notification_signature_ok(body)
    except RuntimeError:
        signature_ok = None  # keys not configured -- GET /status below fails too

    try:
        payment = payment_store.get_payment_by_invoice_id(invoice_id)
    except Exception as exc:
        plog("selfwork_callback", logging.ERROR, trace_id=trace_id, source="callback", invoice_id=invoice_id,
             error_code=f"db_{type(exc).__name__}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail={"error": "internal_error", "trace_id": trace_id})
    received = dict(trace_id=trace_id, source="callback", ip=ip, ip_known=ip in selfwork.NOTIFY_IPS,
                    signature_ok=signature_ok, provider_status=str(body.get("status"))[:20])
    if payment is None or payment.get("provider") != SelfworkProvider.name:
        plog("selfwork_callback", logging.WARNING, invoice_id=invoice_id, error_code="unknown_order", **received)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="unknown order")

    if payment["status"] == "paid":
        state = "paid"
    else:
        try:
            state = await run_in_threadpool(selfwork_billing.sync, payment, trace_id, "callback",
                                            timeout=selfwork_billing.CALLBACK_STATUS_TIMEOUT)
        except Exception as exc:
            plog("selfwork_callback", logging.ERROR, **selfwork_billing._base(payment),
                 error_code=f"confirm_{type(exc).__name__}", **received)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail={"error": "internal_error", "trace_id": trace_id})
    plog("selfwork_callback", **selfwork_billing._base(payment), outcome=state, **received)

    if state in ("paid", "pending", "closed", "not_found"):
        return JSONResponse({"ok": True})
    if state == "unavailable":
        # Can't verify right now -- a non-2xx lets Selfwork retry, and
        # payment-status reconciles on the next poll anyway.
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail={"error": "status_unavailable", "trace_id": trace_id})
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=state)


# ── ЮKassa ───────────────────────────────────────────────────────────────────
# HTTP notifications (cabinet: Интеграция -> HTTP-уведомления, events
# payment.succeeded + payment.canceled). The body only NAMES a payment: the
# decision is YooKassa's own GET /payments/{id}, checked against our row
# (yookassa_billing.sync) -- a forged, duplicated, late or out-of-order
# notification cannot activate anything twice or wrongly. 200 = handled
# (also: ignored / refused for good); non-2xx only when a retry can help,
# YooKassa then redelivers for 24 h.

_YOOKASSA_PAYMENT_EVENTS = frozenset({"payment.succeeded", "payment.canceled", "payment.waiting_for_capture"})


@router.post("/api/billing/yookassa/webhook")
async def yookassa_webhook(request: Request):
    trace_id = new_trace_id()
    ip = request.client.host if request.client else "unknown"

    try:
        body = await request.json()
    except Exception:
        body = None
    event = body.get("event") if isinstance(body, dict) else None
    obj = body.get("object") if isinstance(body, dict) else None
    if not isinstance(event, str) or not isinstance(obj, dict) or body.get("type") != "notification":
        plog("yookassa_webhook", logging.WARNING, trace_id=trace_id, source="webhook", ip=ip, error_code="bad_body")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid notification")
    received = dict(trace_id=trace_id, source="webhook", ip=ip, ip_known=yookassa.ip_is_notify_source(ip),
                    notification=event[:40])
    if event not in _YOOKASSA_PAYMENT_EVENTS:
        # e.g. refund.succeeded -- not acted on (refunds are handled by hand).
        plog("yookassa_webhook", outcome="ignored", **received)
        return JSONResponse({"ok": True})
    provider_payment_id = obj.get("id")
    if not yookassa.valid_payment_id(provider_payment_id):
        plog("yookassa_webhook", logging.WARNING, error_code="bad_payment_id", **received)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid notification")

    try:
        payment = payment_store.get_payment_by_provider_payment_id(YookassaProvider.name, provider_payment_id)
    except Exception as exc:
        plog("yookassa_webhook", logging.ERROR, provider_payment_id=provider_payment_id,
             error_code=f"db_{type(exc).__name__}", **received)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail={"error": "internal_error", "trace_id": trace_id})
    if payment is None:
        # Not one of our attempts (another shop / forged / test vs live URL
        # mix-up). Nothing to do and nothing a redelivery would change.
        plog("yookassa_security", logging.WARNING, provider_payment_id=provider_payment_id,
             error_code="unknown_payment", **received)
        return JSONResponse({"ok": True})

    if payment["status"] == "paid":
        state = "paid"
    else:
        try:
            state = await run_in_threadpool(yookassa_billing.sync, payment, trace_id, "webhook",
                                            timeout=yookassa_billing.CALLBACK_STATUS_TIMEOUT)
        except Exception as exc:
            plog("yookassa_webhook", logging.ERROR, **selfwork_billing._base(payment),
                 provider_payment_id=provider_payment_id, error_code=f"confirm_{type(exc).__name__}", **received)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail={"error": "internal_error", "trace_id": trace_id})
    plog("yookassa_webhook", **selfwork_billing._base(payment), provider_payment_id=provider_payment_id,
         outcome=state, **received)

    if state == "unavailable":
        # Can't verify right now -- a non-2xx makes YooKassa redeliver, and
        # payment-status reconciles on the next poll anyway.
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,
                            detail={"error": "status_unavailable", "trace_id": trace_id})
    # paid / pending / closed, or "error" (refused + security event logged:
    # a redelivery would be refused the same way).
    return JSONResponse({"ok": True})


# ── GET /api/billing/payment-status ──────────────────────────────────────────

# ── GET /api/billing/checkout-availability ───────────────────────────────────
# Public (guests browse plans too). Lets the plan picker show its pay CTA as
# "Оплата скоро будет доступна" while no provider is configured, and enable it
# by itself once PAYMENT_PROVIDER is set -- no frontend release needed.

@router.get("/api/billing/checkout-availability")
async def checkout_availability():
    provider = get_checkout_provider()
    return {"available": provider is not None and provider.is_configured()}


@router.get("/api/billing/payment-status")
async def payment_status(
    invoice_id: int,
    provider_return: Optional[str] = None,
    user: dict = Depends(get_current_user),
):
    trace_id = new_trace_id()
    payment = payment_store.get_payment_by_invoice_id(invoice_id)
    if payment is None or payment["user_id"] != user["id"]:
        # Same 404-not-403 pattern as security.py::assert_job_owner -- never
        # confirm/deny existence of another user's payment.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail={"error": "not_found", "trace_id": trace_id})

    # The callback may lag or get lost: while the attempt is open, ask the
    # provider's authenticated status API ourselves (same decision as the
    # callback/webhook; never creates an order, never trusts the return URL).
    # 'expired' is our own guess (abandoned attempt), not the provider's
    # word -- a payment made on a stale page must still show up as paid here.
    billing = _ATTEMPT_BILLING.get(payment.get("provider"))
    if payment["status"] in ("pending", "expired") and billing is not None:
        try:
            if await run_in_threadpool(billing.sync, payment, trace_id, "payment_status") in ("paid", "closed"):
                payment = payment_store.get_payment_by_invoice_id(invoice_id)
        except Exception as exc:
            plog("payment_status_failed", logging.WARNING, trace_id=trace_id, **selfwork_billing._base(payment),
                 error_code=type(exc).__name__)
    if provider_return in ("success", "error"):
        plog("provider_return", trace_id=trace_id, **selfwork_billing._base(payment),
             provider_return=provider_return, state_after=payment["status"])

    # The subscription this payment opened (if paid and it's a new-style
    # plan) -- lets success.html show what was actually activated.
    subscription = None
    if payment["status"] == "paid" and payment.get("plan_mode"):
        try:
            subscription = next(
                (entitlements.serialize_subscription(s)
                 for s in entitlements.get_user_subscriptions(str(user["id"]))
                 if str(s.get("payment_id")) == str(payment["id"])),
                None,
            )
        except Exception as exc:
            logger.warning("[payment] payment_status_subscription_lookup_failed trace_id=%s error_type=%s",
                           trace_id, type(exc).__name__)

    return {
        "status": payment["status"],
        "plan_id": payment["plan_id"],
        "plan_type": user["plan_type"],
        "plan_active_until": subscription["period_end"] if subscription else _iso(user.get("plan_active_until")),
        "subscription": subscription,
    }
