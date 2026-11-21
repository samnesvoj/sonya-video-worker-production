"""
Signed internal API the EXISTING accounting bot polls (same pattern as /api/internal/fiscal/*: outbound-only bot, HMAC, nonce).
  POST /api/internal/accounting/vast/claim            -> {"reports":[{report_key, kind, period, text, attempt}]}
  POST /api/internal/accounting/vast/summary          -> {"text": "...", "parse_mode": "HTML"}   (bot's /vast command)
  POST /api/internal/accounting/vast/status           -> {"text", "parse_mode"}  (/status: self-check, scripts/ops/health.py)
  POST /api/internal/accounting/vast/daily            body {"day": "YYYY-MM-DD"|"DD.MM"|""} -> {"text", "parse_mode"}  (/vast_day; read-only)
  POST /api/internal/accounting/vast/{key}/sent       body {"message_id": "..."}
  POST /api/internal/accounting/vast/{key}/failed     body {"error": "..."}
X-Accounting-Timestamp / -Nonce / -Signature = hex HMAC-SHA256(secret, "<ts>\n<nonce>\n<METHOD>\n<path>\n<sha256(body)>").
The bot's Telegram token never leaves the bot VPS; this API never carries Vast keys, GitHub keys or user data.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import re
import time
from typing import Callable, Dict, Optional, Tuple

SKEW = 120
# 2026-11-19: + the ops alerts of accounting/balance.ops_alert (vast-balance:critical:N, yt-wait:…, fleet-budget:…);
# before, they were claimed and never delivered (the key did not match on either side).
KEY_RE = re.compile(r"^(vast-(daily:\d{4}-\d{2}-\d{2}|monthly:\d{4}-\d{2}|alert:\d{4}-\d{2}-\d{2})"
                    r"|[a-z][a-z0-9-]{1,40}(:[A-Za-z0-9._-]{1,64}){1,4})$")


def sign(secret: bytes, ts: str, nonce: str, method: str, path: str, body: bytes) -> str:
    msg = "\n".join((ts, nonce, method.upper(), path, hashlib.sha256(body).hexdigest())).encode()
    return hmac.new(secret, msg, hashlib.sha256).hexdigest()


class AccountingApi:
    def __init__(self, service, secret: str, clock: Callable[[], float] = time.time, summary: Optional[Callable[[float], str]] = None,
                 parse_mode: Optional[str] = "HTML", status: Optional[Callable[[float], str]] = None,
                 on_claim: Optional[Callable[[float], None]] = None):
        self.svc, self._secret, self.clock, self._summary, self.parse_mode = service, secret.encode(), clock, summary, parse_mode
        self._status, self._on_claim = status, on_claim
        self._nonces: Dict[str, float] = {}

    def _verify(self, headers: Dict[str, str], method: str, path: str, body: bytes) -> bool:
        h = {k.lower(): v for k, v in headers.items()}          # Starlette lower-cases header names
        ts, nonce, sig = h.get("x-accounting-timestamp", ""), h.get("x-accounting-nonce", ""), h.get("x-accounting-signature", "")
        now = self.clock()
        if not (ts.isdigit() and re.fullmatch(r"[0-9a-f]{16,64}", nonce) and sig):
            return False
        if abs(now - int(ts)) > SKEW or nonce in self._nonces:
            return False
        if not hmac.compare_digest(sign(self._secret, ts, nonce, method, path, body), sig):
            return False
        self._nonces = {n: t for n, t in self._nonces.items() if now - t < SKEW * 2}
        self._nonces[nonce] = now
        return True

    def _msg(self, text: str) -> dict:
        return {"text": text, "parse_mode": self.parse_mode} if self.parse_mode else {"text": text}

    def handle(self, method: str, path: str, headers: Dict[str, str], body: bytes) -> Tuple[int, dict]:
        if not self._verify(headers, method, path, body):
            return 401, {"error": "unauthorized"}
        now = self.clock()
        base = "/api/internal/accounting/vast/"
        if method == "POST" and path == base + "claim":
            if self._on_claim:
                try:
                    self._on_claim(now)        # the self-check watches that the bot keeps polling
                except Exception:  # noqa: BLE001
                    pass
            return 200, {"reports": self.svc.claim(now, limit=1)}
        if method == "POST" and path == base + "status" and self._status:
            try:
                return 200, self._msg(self._status(now))
            except Exception:  # noqa: BLE001
                return 502, {"error": "status_failed"}
        if method == "POST" and path == base + "summary" and self._summary:
            try:
                return 200, self._msg(self._summary(now))
            except Exception:  # noqa: BLE001 - never leak internals
                return 502, {"error": "summary_failed"}
        if method == "POST" and path == base + "daily" and hasattr(self.svc, "render_daily"):
            try:
                day = parse_day((json.loads(body or b"{}") or {}).get("day"), now)
            except (ValueError, TypeError, AttributeError):
                return 400, {"error": "bad_day"}
            try:
                text = self.svc.render_daily(day, now)
            except Exception:  # noqa: BLE001
                return 502, {"error": "daily_failed"}
            if text is None:
                return 404, {"error": "no_report", "day": day.isoformat()}
            return 200, self._msg(text)
        m = re.fullmatch(re.escape(base) + r"(.+)/(sent|failed)", path)
        if method == "POST" and m and KEY_RE.match(m.group(1)):
            try:
                data = json.loads(body or b"{}")
            except ValueError:
                return 400, {"error": "bad_json"}
            if m.group(2) == "sent":
                self.svc.ack_sent(m.group(1), str(data.get("message_id", "")), now)
            else:
                self.svc.ack_failed(m.group(1), str(data.get("error", "")), now)
            return 200, {"ok": True}
        return 404, {"error": "not_found"}


def parse_day(v, now: float):
    """'' / None -> yesterday (MSK); 'YYYY-MM-DD'; 'DD.MM' (current year); 'DD.MM.YYYY'. Never in the future."""
    from datetime import datetime, timedelta
    from accounting.money import MSK
    today = datetime.fromtimestamp(now, MSK).date()
    s = str(v or "").strip()[:10]
    if not s:
        return today - timedelta(days=1)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        d = datetime.strptime(s, "%Y-%m-%d").date()
    elif re.fullmatch(r"\d{1,2}\.\d{1,2}", s):
        d = datetime.strptime(f"{s}.{today.year}", "%d.%m.%Y").date()
    elif re.fullmatch(r"\d{1,2}\.\d{1,2}\.\d{4}", s):
        d = datetime.strptime(s, "%d.%m.%Y").date()
    else:
        raise ValueError("bad day")
    if d >= today:
        raise ValueError("day not closed yet")
    return d


def fastapi_router(api: AccountingApi):  # pragma: no cover - thin adapter for prod_generation_api (see integration/)
    from fastapi import APIRouter, Request
    from fastapi.responses import JSONResponse

    router = APIRouter()

    @router.post("/api/internal/accounting/vast/{rest:path}")
    async def _any(rest: str, request: Request):
        body = await request.body()
        status, data = api.handle("POST", request.url.path, dict(request.headers), body)
        return JSONResponse(data, status_code=status)

    return router
