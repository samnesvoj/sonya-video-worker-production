"""
Vast.ai cost reports for the EXISTING accounting bot (@sonya_accounting_bot). No new bot, no new token.

The bot stays outbound-only. It polls SONYA's signed internal API (accounting/internal_api.py), sends the ready-made text
with ITS OWN Telegram client to its OWN notify chats, then acknowledges. The Telegram token never leaves the bot VPS and
this module never sees Vast / GitHub credentials. Python 3.9 compatible, standard library only.

Delivery is at-least-once: a sent report key is persisted BEFORE the ack, so a restart or a lost ack never re-sends it.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import secrets as _secrets
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit

KEY_RE = re.compile(r"^vast-(daily:\d{4}-\d{2}-\d{2}|monthly:\d{4}-\d{2})$")
BASE = "/api/internal/accounting/vast/"
MAX_TEXT = 3900                        # Telegram limit is 4096


class VastReportError(RuntimeError):
    pass


def sign(secret: bytes, ts: str, nonce: str, method: str, path: str, body: bytes) -> str:
    msg = "\n".join((ts, nonce, method.upper(), path, hashlib.sha256(body).hexdigest())).encode()
    return hmac.new(secret, msg, hashlib.sha256).hexdigest()


class VastReportPuller:
    def __init__(self, base_url: str, secret: str, api: Any, notify_ids: Sequence[int], state_path: Path, *,
                 interval: float = 60.0, clock: Callable[[], float] = time.time, opener: Optional[Callable] = None,
                 timeout: float = 20.0):
        if not base_url.startswith("https://") and not base_url.startswith("http://127.0.0.1"):
            raise VastReportError("base_url must be https")
        self.base, self._secret, self.api = base_url.rstrip("/"), secret.encode(), api
        self.notify_ids, self.state_path, self.interval, self.clock = list(notify_ids), Path(state_path), interval, clock
        self._open = opener or urllib.request.urlopen
        self.timeout, self._next = timeout, 0.0
        self.log = logging.getLogger("receipt_worker.vast_reports")

    # ---------------------------------------------------------------- state
    def _load(self) -> Dict[str, Any]:
        try:
            s = json.loads(self.state_path.read_text())
            if isinstance(s, dict):
                s.setdefault("sent", {}); s.setdefault("partial", {})
                return s
        except (OSError, ValueError):
            pass
        return {"sent": {}, "partial": {}}

    def _save(self, state: Dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        tmp = self.state_path.with_suffix(".tmp")
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            json.dump(state, fh)
        os.replace(str(tmp), str(self.state_path))

    # ----------------------------------------------------------------- http
    def _post(self, path: str, body: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        raw = json.dumps(body).encode()
        ts, nonce = str(int(self.clock())), _secrets.token_hex(16)
        url = self.base + path
        headers = {"X-Accounting-Timestamp": ts, "X-Accounting-Nonce": nonce,
                   "X-Accounting-Signature": sign(self._secret, ts, nonce, "POST", urlsplit(url).path, raw),
                   "Content-Type": "application/json", "User-Agent": "sonya-accounting-bot"}
        req = urllib.request.Request(url, data=raw, headers=headers, method="POST")
        try:
            with self._open(req, timeout=self.timeout) as r:
                status, text = r.status, r.read()
        except urllib.error.HTTPError as exc:
            status, text = exc.code, exc.read()
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, OSError) as exc:
            raise VastReportError("SONYA unreachable (%s)" % type(exc).__name__)
        try:
            data = json.loads(text) if text else {}
        except ValueError:
            raise VastReportError("SONYA answered non-JSON (HTTP %s)" % status)
        return status, data if isinstance(data, dict) else {}

    # ----------------------------------------------------------------- work
    def poll_safe(self) -> None:
        """Called from the bot's idle loop; never raises, rate-limited to `interval`."""
        now = self.clock()
        if now < self._next:
            return
        self._next = now + self.interval
        try:
            self.poll()
        except Exception as exc:  # noqa: BLE001 - reports must never stop the receipt loop
            self.log.warning("vast_reports_error %s", type(exc).__name__)

    def poll(self) -> int:
        status, data = self._post(BASE + "claim", {})
        if status != 200:
            raise VastReportError("claim HTTP %s" % status)
        n = 0
        for rep in data.get("reports", []):
            n += 1 if self._deliver(rep) else 0
        return n

    def _deliver(self, rep: Dict[str, Any]) -> bool:
        key, text = str(rep.get("report_key", "")), str(rep.get("text", ""))
        if not KEY_RE.match(key) or not text or len(text) > MAX_TEXT:
            self._post(BASE + key + "/failed", {"error": "malformed report"}) if KEY_RE.match(key) else None
            return False
        state = self._load()
        if key in state["sent"]:                                   # already delivered, ack was lost
            self._post(BASE + key + "/sent", {"message_id": str(state["sent"][key])})
            return True
        done = set(state["partial"].get(key, []))
        first_id: Optional[int] = None
        errors = 0
        for chat in self.notify_ids:
            if chat in done:
                continue
            try:
                mid = self.api.send(chat, text)
                done.add(chat)
                first_id = first_id or mid
                state["partial"][key] = sorted(done)
                self._save(state)
            except Exception:  # noqa: BLE001
                errors += 1
        if errors or not done:
            self._post(BASE + key + "/failed", {"error": "telegram send failed for %d chat(s)" % errors})
            return False
        state["sent"][key] = first_id or 0
        state["partial"].pop(key, None)
        self._save(state)                                          # persisted BEFORE the ack
        self._post(BASE + key + "/sent", {"message_id": str(first_id or 0)})
        return True
