"""
Confirmed Vast charges (GET /api/v0/charges/) -> ledger, and reconciliation against our calculated numbers.
!! The exact row schema of /api/v0/charges/ could NOT be verified here (no Vast API key on this machine). The parser is
!! deliberately defensive and every row keeps its raw form in `detail`. Verify against the first real response before relying
!! on 'confirmed' figures (tests use a documented ASSUMED shape).
Charges are per UTC day. Corrections are DELTA adjustments (confirmed - calculated - already adjusted), so re-running an
import never double-counts.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from decimal import Decimal
from typing import Dict, Iterable, List

from accounting.money import UTC, D

TOLERANCE_ABS = Decimal("0.05")
TOLERANCE_REL = Decimal("0.03")


def _utc_day(v) -> date:
    return datetime.fromtimestamp(float(v), UTC).date() if not isinstance(v, str) or v.replace(".", "").isdigit() else date.fromisoformat(v[:10])


def parse_row(row: dict) -> List[tuple]:
    """-> [(utc_day, category, amount, detail)] ; unknown splits go to 'gpu' (instance) / 'disk' (volume) flagged unsplit."""
    day = _utc_day(row.get("day") if row.get("day") is not None else row.get("when"))
    typ = str(row.get("type") or "instance")
    total = D(row.get("amount", row.get("cost", row.get("total", 0))))
    parts = {"gpu": D(row.get("gpu") or row.get("compute") or 0), "disk": D(row.get("storage") or row.get("disk") or 0),
             "other": D(row.get("bandwidth") or row.get("network") or row.get("other") or 0)}
    if typ == "volume":
        return [(day, "disk", total, {"raw": row})]
    if sum(parts.values()) > 0:
        rest = total - sum(parts.values())
        if rest:
            parts["other"] += rest
        return [(day, c, a, {"raw": row}) for c, a in parts.items() if a]
    return [(day, "gpu", total, {"raw": row, "unsplit": True})]


def import_charges(ledger, rows: Iterable[dict], now: float = 0.0) -> int:
    n = 0
    for row in rows:
        h = hashlib.sha256(json.dumps(row, sort_keys=True, default=str).encode()).hexdigest()[:16]
        for day, cat, amount, detail in parse_row(row):
            key = f"conf:{row.get('id') or h}:{day}:{cat}"
            ledger.upsert_cost(key, "confirmed", day, cat, amount, None, detail, now)
            n += 1
    return n


def reconcile_day(ledger, utc_day: date, calculated: Dict[str, Decimal], now: float = 0.0) -> dict:
    """Compare confirmed vs calculated for one COMPLETE UTC day; record delta adjustments when they differ materially."""
    conf = ledger.sum_costs("confirmed", utc_day)
    if not any(conf.values()):
        return {"day": utc_day, "status": "no_confirmed_data"}
    out = {"day": utc_day, "status": "ok", "confirmed": conf, "calculated": calculated, "adjusted": {}}
    for cat in ("gpu", "disk", "other"):
        want = conf[cat] - calculated.get(cat, Decimal(0))
        applied = sum((D(r["amount_usd"]) for r in ledger.c.execute("SELECT amount_usd FROM adjustments WHERE day=? AND category=?", (utc_day.isoformat(), cat))), Decimal(0))
        delta = want - applied
        if abs(delta) > max(TOLERANCE_ABS, TOLERANCE_REL * max(abs(conf[cat]), abs(calculated.get(cat, 0)))):
            n = ledger.c.execute("SELECT COUNT(*) FROM adjustments WHERE day=? AND category=?", (utc_day.isoformat(), cat)).fetchone()[0]
            if ledger.add_adjustment(f"adj:{utc_day}:{cat}:{n + 1}", utc_day, cat, delta, "reconciliation with Vast charges", now):
                out["adjusted"][cat] = delta
                out["status"] = "adjusted"
    return out
