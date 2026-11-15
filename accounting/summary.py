"""Short live summary for the bot's /vast command and the daily-overspend alert. READ-ONLY on the fleet journal;
numbers are CALCULATED (journal x price snapshot), same basis as the daily report. Balance comes from Vast (optional)."""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import date, timedelta
from decimal import Decimal
from typing import Callable, Dict, Optional

from accounting.money import D, MSK, both, month_bounds_msk, msk_date_of, msk_day_bounds, rub, usd
from accounting.usage import cost_between

ALERT_KEY_PREFIX = "vast-alert:"


def day_cost(store, day: date, now: float) -> dict:
    lo, hi = msk_day_bounds(day)
    return cost_between(store, lo, hi, now)


def _total(r: dict) -> Decimal:
    return r["gpu"] + r["disk"] + r["other"]


def gpu_hours_by_class(store, r: dict) -> Dict[str, Decimal]:
    names = {i.id: (i.gpu_name or "GPU") for i in store.all()}
    out: Dict[str, Decimal] = defaultdict(Decimal)
    for iid, d in r["per_instance"].items():
        out[names.get(iid, "GPU")] += D(d["gpu_hours"])
    return dict(out)


def build_summary(store, rates, now: float, balance_usd: Optional[Decimal] = None) -> str:
    today = msk_date_of(now)
    rate = rates.get(today).value
    t = day_cost(store, today, now)
    y = day_cost(store, today - timedelta(days=1), now)
    first, _ = month_bounds_msk(today.year, today.month)
    m_total, m_hours, d = Decimal(0), Decimal(0), first
    while d <= today:
        r = t if d == today else y if d == today - timedelta(days=1) else day_cost(store, d, now)
        m_total, m_hours, d = m_total + _total(r), m_hours + r["gpu_hours"], d + timedelta(days=1)
    lines = ["SONYA — Vast.ai, коротко (расчёт по журналу, без оценки трафика)", "",
             f"Сегодня ({today.strftime('%d.%m')}, на этот момент): {both(_total(t), rate)}",
             f"Вчера: {both(_total(y), rate)}",
             f"С начала месяца: {both(m_total, rate)}", "",
             "Баланс Vast: " + (f"{usd(balance_usd)} / {rub(balance_usd * rate)}" if balance_usd is not None else "не удалось получить"), "",
             f"GPU сегодня работал: {t['gpu_hours'].quantize(Decimal('0.01'))} ч" +
             ("".join(f"\n  • {n}: {h.quantize(Decimal('0.01'))} ч" for n, h in sorted(gpu_hours_by_class(store, t).items()))),
             f"GPU вчера: {y['gpu_hours'].quantize(Decimal('0.01'))} ч, с начала месяца: {m_hours.quantize(Decimal('0.01'))} ч",
             f"Курс ЦБ: {rate.quantize(Decimal('0.0001'))} ₽/$"]
    return "\n".join(lines)


def check_alert(ledger, store, rates, now: float, threshold_rub: Decimal) -> Optional[str]:
    """Enqueue ONE alert per MSK day when today's calculated spend exceeds the threshold. Returns the key if enqueued."""
    today = msk_date_of(now)
    key = f"{ALERT_KEY_PREFIX}{today.isoformat()}"
    if ledger.c.execute("SELECT 1 FROM report_outbox WHERE report_key=?", (key,)).fetchone():
        return None
    rate = rates.get(today).value
    total = _total(day_cost(store, today, now))
    if total * rate <= threshold_rub:
        return None
    text = (f"⚠️ SONYA — Vast.ai: расход за сегодня ({today.strftime('%d.%m.%Y')}) уже {both(total, rate)} — выше порога "
            f"{rub(threshold_rub)}/день.\nПроверь, не завис ли кто-то из серверов GPU (команда /vast — подробности).")
    ledger.c.execute("INSERT OR IGNORE INTO report_outbox(report_key, kind, period, text, payload, created_at) VALUES (?,?,?,?,?,?)",
                     (key, "alert", today.isoformat(), text, json.dumps({"total_usd": str(total), "threshold_rub": str(threshold_rub)}), now))
    return key
