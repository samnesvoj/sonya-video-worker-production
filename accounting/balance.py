"""Vast balance watch (2026-11-17: the balance went to -$0.33 in the evening, Vast stopped the servers and nobody was
warned -- the only alert was "spent more than N RUB today").

  check_balance_alert()  enqueues an alert into the report outbox (the accounting bot delivers it):
                         balance < SONYA_VAST_BALANCE_LOW_USD (3)      -> "пополни" once per 12 h
                         balance < SONYA_VAST_BALANCE_CRITICAL_USD (1) -> "СРОЧНО" once per 3 h
  balance_line()         "Баланс Vast: $X — хватит примерно на N ч" for the daily report.
Runway = balance / the average spend of the last 6 h (all servers incl. stopped disks, the same calculation as reports).
"""
from __future__ import annotations

import json
from decimal import Decimal
from typing import Optional

from accounting import tgformat as tf

KEY_PREFIX = "vast-balance"


def burn_usd_per_hour(store, now: float, hours: float = 6.0) -> Decimal:
    from accounting.usage import cost_between
    r = cost_between(store, now - hours * 3600, now, now)
    return (r["gpu"] + r["disk"] + r["other"]) / Decimal(str(hours))


def runway_hours(balance: Decimal, burn: Decimal) -> Optional[Decimal]:
    if burn <= 0:
        return None
    return max(Decimal(0), balance / burn)


def runway_text(balance: Decimal, burn: Decimal) -> str:
    h = runway_hours(balance, burn)
    if balance <= 0:
        return "баланс исчерпан — серверы Vast остановлены"
    if h is None:
        return "сейчас расхода нет"
    if h < 1:
        return f"хватит меньше чем на час (≈{int(h * 60)} мин)"
    if h < 48:
        return f"хватит примерно на {int(h)} ч"
    return f"хватит примерно на {int(h // 24)} дн"


def balance_line(balance: Optional[Decimal], burn: Optional[Decimal]) -> Optional[str]:
    if balance is None:
        return None
    s = f"Баланс Vast: ${balance:.2f}"
    if burn is not None:
        s += f" — {runway_text(balance, burn)} (расход ≈ ${burn:.2f}/ч)"
    return tf.esc(s)


def check_balance_alert(ledger, balance: Optional[Decimal], burn: Optional[Decimal], now: float,
                        low: Decimal = Decimal("3"), critical: Decimal = Decimal("1")) -> Optional[str]:
    """Enqueue at most one alert per level per period. Returns the key if enqueued."""
    if balance is None or balance >= low:
        return None
    if balance < critical:
        level, period, head = "critical", 3 * 3600, "🚨 СРОЧНО: деньги на Vast почти кончились"
    else:
        level, period, head = "low", 12 * 3600, "⚠️ Vast: баланс ниже $" + f"{low:.0f}"
    key = f"{KEY_PREFIX}:{level}:{int(now // period)}"
    if ledger.c.execute("SELECT 1 FROM report_outbox WHERE report_key=?", (key,)).fetchone():
        return None
    if level == "low" and ledger.c.execute(          # already screamed "critical" recently: no softer echo
            "SELECT 1 FROM report_outbox WHERE report_key LIKE ? AND created_at > ?", (f"{KEY_PREFIX}:critical:%", now - 3 * 3600)).fetchone():
        return None
    lines = [tf.title(head), tf.esc(f"На балансе ${balance:.2f}" + (f" — {runway_text(balance, burn)}" if burn is not None else "") + ".")]
    if burn is not None and burn > 0:
        lines.append(tf.esc(f"Расход сейчас ≈ ${burn:.2f}/ч (≈ ${burn * 24:.2f} в сутки)."))
    lines.append(tf.esc("Пополни баланс на vast.ai (Billing). Когда деньги кончаются, Vast останавливает серверы "
                        "и нарезка видео на сайте встаёт у всех."))
    ledger.c.execute("INSERT OR IGNORE INTO report_outbox(report_key, kind, period, text, payload, created_at) VALUES (?,?,?,?,?,?)",
                     (key, "alert", key, "\n".join(lines), json.dumps({"fmt": "html", "balance_usd": str(balance)}), now))
    return key


def ops_alert(ledger, key: str, text_lines, now: float) -> Optional[str]:
    """Generic once-per-key alert through the same outbox (YouTube relay / cookies monitoring)."""
    if ledger.c.execute("SELECT 1 FROM report_outbox WHERE report_key=?", (key,)).fetchone():
        return None
    ledger.c.execute("INSERT OR IGNORE INTO report_outbox(report_key, kind, period, text, payload, created_at) VALUES (?,?,?,?,?,?)",
                     (key, "alert", key, "\n".join(text_lines), json.dumps({"fmt": "html"}), now))
    return key
