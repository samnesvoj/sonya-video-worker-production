"""
Signed internal API the EXISTING accounting bot polls (same pattern as /api/internal/fiscal/*: outbound-only bot, HMAC, nonce).
  POST /api/internal/accounting/vast/claim            -> {"reports":[{report_key, kind, period, text, attempt}]}
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
KEY_RE = re.compile(r"^vast-(daily:\d{4}-\d{2}-\d{2}|monthly:\d{4}-\d{2})$")


def sign(secret: bytes, ts: str, nonce: str, method: str, path: str, body: bytes) -> str:
    msg = "\n".join((ts, nonce, method.upper(), path, hashlib.sha256(body).hexdigest())).encode()
    return hmac.new(secret, msg, hashlib.sha256).hexdigest()


class AccountingApi:
    def __init__(self, service, secret: str, clock: Callable[[], float] = time.time):
        self.svc, self._secret, self.clock = service, secret.encode(), clock
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

    def handle(self, method: str, path: str, headers: Dict[str, str], body: bytes) -> Tuple[int, dict]:
        if not self._verify(headers, method, path, body):
            return 401, {"error": "unauthorized"}
        now = self.clock()
        base = "/api/internal/accounting/vast/"
        if method == "POST" and path == base + "claim":
            return 200, {"reports": self.svc.claim(now, limit=1)}
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
