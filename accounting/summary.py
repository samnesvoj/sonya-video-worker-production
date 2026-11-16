"""Live summary for the bot's /vast command and the daily-overspend alert. READ-ONLY on the fleet journal and the ledger.
РАСХОДЫ are CALCULATED gross figures (journal x price snapshot), the same basis as the daily report's «Расход за сутки».
Reconciliation adjustments and confirmed Vast charges are shown in their own rows, never mixed into those figures."""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Dict, Optional

from accounting import reports
from accounting import tgformat as tf
from accounting.money import D, UTC, month_bounds_msk, msk_date_of, msk_day_bounds
from accounting.usage import cost_between

ALERT_KEY_PREFIX = "vast-alert:"


def day_cost(store, day: date, now: float) -> dict:
    lo, hi = msk_day_bounds(day)
    return cost_between(store, lo, hi, now)


def utc_day_cost(store, day: date, now: float) -> dict:
    lo = datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp()
    return cost_between(store, lo, lo + 86400, now)


def _total(r: dict) -> Decimal:
    return r["gpu"] + r["disk"] + r["other"]


def gpu_hours_by_class(store, r: dict) -> Dict[str, Decimal]:
    names = {i.id: (i.gpu_name or "GPU") for i in store.all()}
    out: Dict[str, Decimal] = defaultdict(Decimal)
    for iid, d in r["per_instance"].items():
        h = D(d["gpu_hours"])
        if h:
            out[names.get(iid, "GPU")] += h
    return dict(out)


def summary_numbers(store, rates, now: float, balance_usd: Optional[Decimal] = None, ledger=None) -> dict:
    today = msk_date_of(now)
    yday = today - timedelta(days=1)
    try:
        rate = rates.get(today)
    except Exception:  # noqa: BLE001 - no rate at all: show dollars, say rubles are missing
        rate = None
    t, y = day_cost(store, today, now), day_cost(store, yday, now)
    first, _ = month_bounds_msk(today.year, today.month)
    m_total, m_hours, d = Decimal(0), Decimal(0), first
    while d <= today:
        r = t if d == today else y if d == yday else day_cost(store, d, now)
        m_total, m_hours, d = m_total + _total(r), m_hours + r["gpu_hours"], d + timedelta(days=1)
    n = {"today": today, "yday": yday, "first": first, "rate": rate, "r": rate.value if rate else None, "t": _total(t), "y": _total(y),
         "m": m_total, "t_hours": t["gpu_hours"], "y_hours": y["gpu_hours"], "m_hours": m_hours, "by_class": gpu_hours_by_class(store, t),
         "balance": balance_usd, "recon": None, "y_adj": [], "m_adj": Decimal(0), "conf_month": None}
    if ledger is not None:
        utc_y = datetime.fromtimestamp(now, UTC).date() - timedelta(days=1)       # last fully closed UTC day
        n["recon"] = reports.compare(ledger, utc_y, {k: v for k, v in utc_day_cost(store, utc_y, now).items() if k in reports.CATS})
        n["y_adj"] = [dict(a) for a in ledger.c.execute("SELECT * FROM adjustments WHERE applied_on=?", (yday.isoformat(),))]
        n["m_adj"] = sum((D(a["amount_usd"]) for a in ledger.adjustments_between(first, today)), Decimal(0))
        conf = Decimal(0)
        for k in ledger.c.execute("SELECT amount_usd FROM cost_entries WHERE kind='confirmed' AND day BETWEEN ? AND ?",
                                  (first.isoformat(), datetime.fromtimestamp(now, UTC).date().isoformat())):
            conf += D(k[0])
        n["conf_month"] = conf
    return n


def render_summary(n: dict, now: float) -> str:
    r = n["r"]
    R = lambda v: tf.to_rub(v, r)                                    # noqa: E731
    head = tf.title("SONYA · Vast.ai — сводка") + "\nРасчёт по журналу инстансов, без трафика."
    spend = tf.block("РАСХОДЫ", tf.table([
        tf.Row("Сегодня", tf.num(n["t"]), tf.num(R(n["t"]))),
        tf.Row("Вчера", tf.num(n["y"]), tf.num(R(n["y"]))),
        tf.Row("С начала месяца", tf.num(n["m"]), tf.num(R(n["m"])))], ("USD", "RUB")),
        f"Сегодня — на {tf.esc(datetime.fromtimestamp(now, reports.MSK).strftime('%H:%M'))} МСК.")
    adj_lines = []
    if n["y_adj"]:
        ya = sum((D(a["amount_usd"]) for a in n["y_adj"]), Decimal(0))
        days = ", ".join(sorted({reports._dm(date.fromisoformat(a["day"])) for a in n["y_adj"]}))
        adj_lines.append(f"В отчёте за {tf.esc(reports._dm(n['yday']))} к расчёту {tf.esc(tf.usd(n['y']))} добавлена корректировка "
                         f"{tf.esc(tf.usd(ya))} за UTC {tf.esc(days)} — итог отчёта {tf.esc(tf.pair(n['y'] + ya, R(n['y'] + ya)))}.")
    if n["m_adj"]:
        adj_lines.append(f"Корректировки с начала месяца (в таблицу не входят): {tf.esc(tf.pair(n['m_adj'], R(n['m_adj'])))}.")
    adj_b = "\n".join(adj_lines)

    if n["by_class"]:
        rows = [tf.Row(name, tf.num(h)) for name, h in sorted(n["by_class"].items())]
        rows.append(tf.Row("Итого", tf.num(n["t_hours"]), total=True))
        gpu_tab = tf.table(rows, ("ч",))
    else:
        gpu_tab = "Сегодня GPU не работал."
    gpu = tf.block("ИСПОЛЬЗОВАНИЕ GPU", gpu_tab, f"Вчера: {tf.esc(tf.hours(n['y_hours']))} · с начала месяца: {tf.esc(tf.hours(n['m_hours']))}",
                   sub="сегодня")

    blocks = [(head, False), (spend, False)]
    if adj_b:
        blocks.append((adj_b, False))
    blocks.append((gpu, False))
    rc = n["recon"]
    if rc is not None:
        lines = []
        if rc["has_confirmed"]:
            conf, calc = sum(rc["confirmed"].values(), Decimal(0)), sum(rc["calculated"].values(), Decimal(0))
            adj = sum(rc["adjustments"].values(), Decimal(0))
            u = reports._dm(rc["day"])
            rows = [tf.Row(f"Vast, UTC {u}", tf.num(conf), tf.num(R(conf))), tf.Row(f"Журнал, UTC {u}", tf.num(calc), tf.num(R(calc))),
                    tf.Row("Разница", tf.num(conf - calc), tf.num(R(conf - calc)), rule_before=True)]
            if adj:
                rows.append(tf.Row(f"Корр. за {u}", tf.num(adj), tf.num(R(adj))))
            if n["conf_month"] is not None:
                rows.append(tf.Row(f"Vast с {reports._dm(n['first'])} UTC", tf.num(n["conf_month"]), tf.num(R(n["conf_month"])), rule_before=True))
            lines.append(tf.table(rows, ("USD", "RUB")))
            lines.append("Vast — подтверждённые списания (UTC-сутки), журнал — наш расчёт за те же сутки.")
            if rc["status"] == "ok":
                lines.append(tf.status_line(True, "сходится в пределах допуска"))
            else:
                lines.append(tf.status_line(False, ""))
                lines.append(tf.bullets(rc["issues"]))
        else:
            lines.append(f"Подтверждённых списаний Vast за UTC {tf.esc(reports._fmt_date(rc['day']))} пока нет.")
        blocks.append((tf.block("СВЕРКА С VAST", *lines), True))
    bal = n["balance"]
    blocks.append((tf.title("БАЛАНС VAST") + ": " + (tf.esc(tf.pair(bal, R(bal))) if bal is not None else "не удалось получить"), False))
    blocks.append((reports.rate_line(n["rate"], now), False))
    return tf.assemble(blocks)


def build_summary(store, rates, now: float, balance_usd: Optional[Decimal] = None, ledger=None) -> str:
    return render_summary(summary_numbers(store, rates, now, balance_usd, ledger), now)


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
    text = "\n".join([
        "⚠️ " + tf.title("Vast.ai: расход выше порога"),
        tf.table([tf.Row(f"Сегодня, {today.strftime('%d.%m')}", tf.num(total), tf.num(total * rate)),
                  tf.Row("Порог в день", tf.DASH, tf.num(threshold_rub))], ("USD", "RUB")),
        "Проверь, не завис ли сервер GPU. Подробности — /vast",
        f"Данные на {tf.esc(reports._stamp(now))}"])
    ledger.c.execute("INSERT OR IGNORE INTO report_outbox(report_key, kind, period, text, payload, created_at) VALUES (?,?,?,?,?,?)",
                     (key, "alert", today.isoformat(), text, json.dumps({"fmt": "html", "total_usd": str(total), "threshold_rub": str(threshold_rub)}), now))
    return key
