"""«Ждёт денег Vast» (owner 2026-11-19, the card was blocked): when the fleet cannot get a server because the Vast
balance is empty, queued jobs do NOT fail - they simply stay 'queued' (nothing in the pool fails a queued job by age).
This module only makes it visible:
  - /var/lib/sonya-fleet/capacity.json {"state": "no_funds"|"ok", "since", "ts", "pending", "balance"} - the site API
    (same host) adds notice «Высокая нагрузка, клипы будут чуть позже» to the user's queued jobs while it is fresh;
  - an admin alert through the accounting bot: right away, then every 2 h while it lasts, and «снова работает».
No-funds = jobs wait AND no server is up/coming AND (balance <= SONYA_VAST_NOFUNDS_USD (0.30) OR Vast refused our
create/resume calls in the last 20 min while the balance is < $1).
"""
from __future__ import annotations

import json
import os
import time
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

NOTICE = "Высокая нагрузка, клипы будут чуть позже"
UP_STATES = ("creating", "running", "starting", "stop_pending")


def no_funds(pending: int, servers_up: int, balance: Optional[Decimal], refusals: int,
             threshold: Decimal = Decimal("0.30")) -> bool:
    """Pure decision (see the module doc)."""
    if pending <= 0 or servers_up > 0 or balance is None:
        return False
    return balance <= threshold or (refusals > 0 and balance < Decimal("1"))


def refusals(events: Iterable[Dict[str, Any]]) -> int:
    """Vast create/resume refusals among fleet events (HTTP 4xx on create/resume)."""
    n = 0
    for e in events:
        d = e.get("detail") or {}
        try:
            st = int(d.get("status") or 0)
        except (TypeError, ValueError):
            st = 0
        if e.get("event") in ("create_failed_next", "resume_failed") and 400 <= st < 500:
            n += 1
        elif e.get("event") == "create_failed" and ("HTTP 4" in str(d.get("reason") or "")):
            n += 1
    return n


def write_state(path: Path, state: str, now: float, prev: Optional[Dict[str, Any]], **extra) -> Dict[str, Any]:
    since = prev.get("since") if prev and prev.get("state") == state else now
    data = {"state": state, "since": since, "ts": now, **extra}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, default=str))
    os.chmod(tmp, 0o644)                   # the API reads it (root too, but keep it plain)
    tmp.replace(path)
    return data


def read_state(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return None


def alerts(ledger, cur: Dict[str, Any], prev: Optional[Dict[str, Any]], now: float) -> List[str]:
    from accounting import tgformat as tf
    from accounting.balance import ops_alert
    out: List[str] = []
    if cur["state"] == "no_funds":
        since = float(cur.get("since") or now)
        n = int(cur.get("pending") or 0)
        k = ops_alert(ledger, f"vast-nofunds:{int(since)}:{int((now - since) // 7200)}", [
            tf.title("💳 Vast: нет денег — новые серверы не создаются"),
            tf.esc(f"Баланс Vast: ${cur.get('balance')}. В очереди ждут {n} заданий, ни одного сервера нет. Задания НЕ падают — "
                   "ждут; пользователи видят «Высокая нагрузка, клипы будут чуть позже». Пополни баланс Vast (или смени карту) — "
                   "серверы поднимутся сами.")], now)
        if k:
            out.append(k)
    elif prev and prev.get("state") == "no_funds":
        k = ops_alert(ledger, f"vast-nofunds-ok:{int(float(prev.get('since') or now))}", [
            tf.title("✅ Vast: серверы снова создаются"),
            tf.esc(f"Ожидание денег длилось {int((now - float(prev.get('since') or now)) // 60)} мин. Очередь пошла.")], now)
        if k:
            out.append(k)
    return out


def tick(path: Path, pending: int, servers_up: int, balance: Optional[Decimal], events: Iterable[Dict[str, Any]],
         now: Optional[float] = None, ledger=None, threshold: Decimal = Decimal("0.30")) -> Dict[str, Any]:
    now = now or time.time()
    prev = read_state(path)
    st = "no_funds" if no_funds(pending, servers_up, balance, refusals(events), threshold) else "ok"
    cur = write_state(path, st, now, prev, pending=pending, balance=str(balance) if balance is not None else None)
    if ledger is not None:
        cur["alerts"] = alerts(ledger, cur, prev, now)
    return cur
