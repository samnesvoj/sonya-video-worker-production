"""
Report outbox + scheduler. Reports are generated centrally and DELIVERED by the existing accounting bot (which owns the
Telegram token). At-least-once delivery with leases; idempotent by report_key; missed days are rebuilt in order.
  daily   for MSK day D : due at D+1 00:05 MSK
  monthly for month M   : due on the 1st of M+1 at 00:10 MSK
"""
from __future__ import annotations

import json
from datetime import date, datetime, time as dtime, timedelta
from typing import Callable, List, Optional

from accounting import reports, usage
from accounting.money import MSK, msk_day_bounds

DAILY_AT = dtime(0, 5)
MONTHLY_AT = dtime(0, 10)
MAX_BACKFILL_DAYS = 45
LEASE_SECONDS = 300
BACKOFF = (60, 300, 900, 3600)


def _due_ts(day: date, at: dtime) -> float:
    return datetime.combine(day, at, tzinfo=MSK).timestamp()


def daily_due(now: float, first_day: date) -> List[date]:
    today = datetime.fromtimestamp(now, MSK).date()
    out, d = [], max(first_day, today - timedelta(days=MAX_BACKFILL_DAYS))
    while d < today:
        if now >= _due_ts(d + timedelta(days=1), DAILY_AT):
            out.append(d)
        d += timedelta(days=1)
    return out


def monthly_due(now: float, first_day: date) -> List[tuple]:
    today = datetime.fromtimestamp(now, MSK).date()
    out, y, m = [], first_day.year, first_day.month
    while (y, m) < (today.year, today.month):
        ny, nm = (y + (m == 12), 1 if m == 12 else m + 1)
        if now >= _due_ts(date(ny, nm, 1), MONTHLY_AT):
            out.append((y, m))
        y, m = ny, nm
    return out


class ReportService:
    def __init__(self, ledger, store, rates, jobs_source, utc_cost: Callable[[date, float], dict], first_day: date):
        self.ledger, self.store, self.rates, self.jobs, self.utc_cost, self.first_day = ledger, store, rates, jobs_source, utc_cost, first_day

    def run_due(self, now: float) -> List[str]:
        built: List[str] = []
        for day in daily_due(now, self.first_day):
            key = f"vast-daily:{day.isoformat()}"
            if self.ledger.c.execute("SELECT 1 FROM report_outbox WHERE report_key=?", (key,)).fetchone():
                continue                                   # already built (sent or pending): never rebuilt -> never double counted
            usage.compute_day(self.ledger, self.store, day, self.jobs, now)
            recon = self._reconcile_previous_utc_day(day, now)
            rep = reports.build_daily(self.ledger, day, self.rates.get(day), recon)
            self._enqueue(rep, now)
            self.ledger.mark_adjustments_applied(rep["adj_keys"], day)
            built.append(key)
        for (y, m) in monthly_due(now, self.first_day):
            key = f"vast-monthly:{y:04d}-{m:02d}"
            if self.ledger.c.execute("SELECT 1 FROM report_outbox WHERE report_key=?", (key,)).fetchone():
                continue
            self._enqueue(reports.build_monthly(self.ledger, y, m, self.rates), now)
            built.append(key)
        return built

    def _reconcile_previous_utc_day(self, msk_day: date, now: float) -> Optional[dict]:
        from accounting import vast_billing
        utc_day = msk_day - timedelta(days=1)           # the last fully closed UTC day at 00:05 MSK
        calc = self.utc_cost(utc_day, now)
        return vast_billing.reconcile_day(self.ledger, utc_day, calc, now)

    def _enqueue(self, rep: dict, now: float) -> None:
        self.ledger.c.execute("INSERT OR IGNORE INTO report_outbox(report_key, kind, period, text, payload, created_at) VALUES (?,?,?,?,?,?)",
                              (rep["key"], rep["kind"], rep["period"], rep["text"], json.dumps(rep["payload"], default=str), now))

    # --- delivery side (called by the signed internal API)
    def claim(self, now: float, limit: int = 1) -> List[dict]:
        with self.ledger.lock:
            rows = self.ledger.c.execute(
                "SELECT * FROM report_outbox WHERE status='pending' AND next_attempt_at<=? AND (lease_until IS NULL OR lease_until<?) "
                "ORDER BY period, report_key LIMIT ?", (now, now, limit)).fetchall()
            for r in rows:
                self.ledger.c.execute("UPDATE report_outbox SET lease_until=?, attempts=attempts+1 WHERE report_key=?", (now + LEASE_SECONDS, r["report_key"]))
            return [{"report_key": r["report_key"], "kind": r["kind"], "period": r["period"], "text": r["text"], "attempt": r["attempts"] + 1} for r in rows]

    def ack_sent(self, key: str, message_id: str, now: float) -> bool:
        cur = self.ledger.c.execute("UPDATE report_outbox SET status='sent', sent_at=?, telegram_message_id=?, lease_until=NULL WHERE report_key=? AND status<>'sent'",
                                    (now, str(message_id), key))
        return cur.rowcount == 1

    def ack_failed(self, key: str, error: str, now: float) -> None:
        r = self.ledger.c.execute("SELECT attempts FROM report_outbox WHERE report_key=? AND status='pending'", (key,)).fetchone()
        if not r:
            return
        wait = BACKOFF[min(r["attempts"] - 1, len(BACKOFF) - 1)] if r["attempts"] else BACKOFF[0]
        self.ledger.c.execute("UPDATE report_outbox SET lease_until=NULL, next_attempt_at=?, last_error=? WHERE report_key=?", (now + wait, error[:200], key))
