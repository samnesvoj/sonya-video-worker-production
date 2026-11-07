"""
Accounting ledger (SQLite, dispatcher host; separate from the production PostgreSQL).
Kinds are never mixed:
  confirmed   - charges returned by Vast billing (/api/v0/charges) - per UTC day
  calculated  - derived by us from the fleet lifecycle journal x price snapshot - per MSK day
  estimate    - model-based (e.g. traffic) - shown separately, NEVER added to totals
All amounts are Decimal strings. Entries are idempotent upserts keyed by entry_key.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import date
from decimal import Decimal
from typing import Any, Dict, List, Optional

from accounting.money import D

SCHEMA = """
CREATE TABLE IF NOT EXISTS fx_rates (day TEXT PRIMARY KEY, value TEXT NOT NULL, cbr_date TEXT NOT NULL, fallback INTEGER NOT NULL, saved_at REAL);
CREATE TABLE IF NOT EXISTS cost_entries (
  entry_key TEXT PRIMARY KEY, kind TEXT NOT NULL CHECK (kind IN ('confirmed','calculated','estimate')),
  day TEXT NOT NULL,                       -- MSK date for calculated/estimate, UTC date for confirmed
  category TEXT NOT NULL CHECK (category IN ('gpu','disk','other')),
  instance_id INTEGER, amount_usd TEXT NOT NULL, detail TEXT, updated_at REAL);
CREATE INDEX IF NOT EXISTS ix_cost_day ON cost_entries(kind, day);
CREATE TABLE IF NOT EXISTS day_facts (day TEXT PRIMARY KEY, videos INTEGER NOT NULL, gpu_hours TEXT NOT NULL, frozen INTEGER NOT NULL DEFAULT 0, detail TEXT);
CREATE TABLE IF NOT EXISTS adjustments (adj_key TEXT PRIMARY KEY, day TEXT NOT NULL, applied_on TEXT, category TEXT NOT NULL, amount_usd TEXT NOT NULL, reason TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS report_outbox (
  report_key TEXT PRIMARY KEY, kind TEXT NOT NULL, period TEXT NOT NULL, text TEXT NOT NULL, payload TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0, lease_until REAL, next_attempt_at REAL NOT NULL DEFAULT 0,
  created_at REAL, sent_at REAL, telegram_message_id TEXT, last_error TEXT);
"""


class Ledger:
    def __init__(self, path: str = ":memory:"):
        self.c = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        self.c.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self.c.executescript(SCHEMA)

    # ---- fx
    def get_rate(self, day: date) -> Optional[dict]:
        r = self.c.execute("SELECT * FROM fx_rates WHERE day=?", (day.isoformat(),)).fetchone()
        return {"value": D(r["value"]), "cbr_date": r["cbr_date"], "fallback": bool(r["fallback"])} if r else None

    def put_rate(self, day: date, value: Decimal, cbr_date: str, fallback: bool, now: float = 0.0) -> None:
        self.c.execute("INSERT OR REPLACE INTO fx_rates VALUES (?,?,?,?,?)", (day.isoformat(), str(value), cbr_date, int(fallback), now))

    def last_rate_before(self, day: date) -> Optional[dict]:
        r = self.c.execute("SELECT * FROM fx_rates WHERE day<=? AND fallback=0 ORDER BY day DESC LIMIT 1", (day.isoformat(),)).fetchone()
        return {"value": D(r["value"]), "cbr_date": r["cbr_date"]} if r else None

    # ---- cost entries
    def upsert_cost(self, key: str, kind: str, day: date, category: str, amount: Decimal, instance_id: Optional[int] = None,
                    detail: Any = None, now: float = 0.0) -> None:
        self.c.execute("INSERT INTO cost_entries(entry_key, kind, day, category, amount_usd, instance_id, detail, updated_at) VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(entry_key) DO UPDATE SET amount_usd=excluded.amount_usd, "
                       "detail=excluded.detail, updated_at=excluded.updated_at",
                       (key, kind, day.isoformat(), category, str(amount), instance_id, json.dumps(detail, default=str), now))

    def sum_costs(self, kind: str, day: date) -> Dict[str, Decimal]:
        out = {"gpu": Decimal(0), "disk": Decimal(0), "other": Decimal(0)}
        for r in self.c.execute("SELECT category, amount_usd FROM cost_entries WHERE kind=? AND day=?", (kind, day.isoformat())):
            out[r["category"]] += D(r["amount_usd"])
        return out

    def days_with(self, kind: str) -> List[str]:
        return [r[0] for r in self.c.execute("SELECT DISTINCT day FROM cost_entries WHERE kind=? ORDER BY day", (kind,))]

    # ---- facts
    def put_facts(self, day: date, videos: int, gpu_hours: Decimal, detail: Any = None) -> None:
        r = self.c.execute("SELECT frozen FROM day_facts WHERE day=?", (day.isoformat(),)).fetchone()
        if r and r["frozen"]:
            return
        self.c.execute("INSERT OR REPLACE INTO day_facts VALUES (?,?,?,0,?)", (day.isoformat(), videos, str(gpu_hours), json.dumps(detail, default=str)))

    def freeze_facts(self, day: date) -> None:
        self.c.execute("UPDATE day_facts SET frozen=1 WHERE day=?", (day.isoformat(),))

    def get_facts(self, day: date) -> Optional[dict]:
        r = self.c.execute("SELECT * FROM day_facts WHERE day=?", (day.isoformat(),)).fetchone()
        return {"videos": r["videos"], "gpu_hours": D(r["gpu_hours"]), "frozen": bool(r["frozen"])} if r else None

    # ---- adjustments (corrections of already reported periods)
    def add_adjustment(self, key: str, day: date, category: str, amount: Decimal, reason: str, now: float = 0.0) -> bool:
        cur = self.c.execute("INSERT OR IGNORE INTO adjustments(adj_key, day, category, amount_usd, reason, created_at) VALUES (?,?,?,?,?,?)",
                             (key, day.isoformat(), category, str(amount), reason, now))
        return cur.rowcount == 1

    def unapplied_adjustments(self) -> List[dict]:
        return [dict(r) for r in self.c.execute("SELECT * FROM adjustments WHERE applied_on IS NULL ORDER BY day, adj_key")]

    def mark_adjustments_applied(self, keys: List[str], applied_on: date) -> None:
        for k in keys:
            self.c.execute("UPDATE adjustments SET applied_on=? WHERE adj_key=? AND applied_on IS NULL", (applied_on.isoformat(), k))

    def adjustments_between(self, a: date, b: date) -> List[dict]:
        return [dict(r) for r in self.c.execute("SELECT * FROM adjustments WHERE applied_on BETWEEN ? AND ? ORDER BY applied_on, adj_key", (a.isoformat(), b.isoformat()))]
