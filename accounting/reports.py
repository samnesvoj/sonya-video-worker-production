"""Daily and monthly report builders + Russian renderers. confirmed / calculated / estimate are shown separately;
ONLY calculated (+ applied reconciliation adjustments) enter the ИТОГО line; estimates never do."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from typing import Dict, Optional

from accounting.money import D, HOURS_PER_MONTH, both, month_bounds_msk, rub, usd, usd4
from accounting.cbr import Rate

def S(v) -> str:
    """Canonical money string for payloads: 6 decimals, never a float."""
    return str(D(v).quantize(Decimal('0.000001')))


MONTHS = ["", "январь", "февраль", "март", "апрель", "май", "июнь", "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь"]


def _fmt_date(d: date) -> str:
    return d.strftime("%d.%m.%Y")


def day_figures(ledger, day: date) -> dict:
    calc = ledger.sum_costs("calculated", day)
    est = ledger.sum_costs("estimate", day)
    facts = ledger.get_facts(day) or {"videos": 0, "gpu_hours": Decimal(0), "frozen": False}
    return {"calc": calc, "est_other": est["other"], "videos": facts["videos"], "gpu_hours": facts["gpu_hours"]}


def build_daily(ledger, day: date, rate: Rate, utc_recon: Optional[dict] = None) -> dict:
    f = day_figures(ledger, day)
    adj = ledger.unapplied_adjustments()
    adj_total = sum((D(a["amount_usd"]) for a in adj), Decimal(0))
    c = f["calc"]
    total = c["gpu"] + c["disk"] + c["other"]
    grand = total + adj_total
    per_video = (grand / f["videos"]) if f["videos"] else None
    r = rate.value
    lines = [f"SONYA — Vast.ai, расходы за {_fmt_date(day)}", "",
             f"GPU: {both(c['gpu'], r)}", f"Диски: {both(c['disk'], r)}", f"Прочие расходы: {both(c['other'], r)}", "",
             f"ИТОГО: {both(grand, r)}", "",
             f"Обработано видео: {f['videos']}", f"GPU работал: {f['gpu_hours'].quantize(Decimal('0.01'))} ч",
             "Стоимость одного видео: " + (both(per_video, r) if per_video is not None else "— (видео не обработано)"), "",
             "Источники (не смешиваются):"]
    if utc_recon and utc_recon.get("status") not in (None, "no_confirmed_data"):
        k, cc = utc_recon["confirmed"], utc_recon["calculated"]
        lines.append(f"• Подтверждено Vast (UTC-сутки {_fmt_date(utc_recon['day'])}, закрыты): {usd(sum(k.values(), Decimal(0)))}; "
                     f"наш расчёт за те же сутки: {usd(sum(cc.values(), Decimal(0)))}")
    else:
        lines.append("• Подтверждено Vast: данных пока нет (сверка появится после закрытия UTC-суток)")
    lines.append(f"• Рассчитано по журналу инстансов: {usd(total)} (входит в ИТОГО)")
    lines.append(f"• Оценка трафика (НЕ входит в ИТОГО): ≈ {usd4(f['est_other'])}")
    if adj:
        lines.append(f"• Корректировки прошлых периодов (входят в ИТОГО): {usd(adj_total)}")
        for a in adj:
            lines.append(f"    {a['day']} {a['category']}: {usd(D(a['amount_usd']))} — {a['reason']}")
    lines.append(f"Курс ЦБ РФ на {_fmt_date(day)}: {r.quantize(Decimal('0.0001'))} ₽/$" + (" (взят последний известный: ЦБ недоступен)" if rate.fallback else ""))
    payload = {"kind": "daily", "day": day.isoformat(), "rate": str(r), "rate_cbr_date": rate.cbr_date, "rate_fallback": rate.fallback,
               "calculated": {k: S(v) for k, v in c.items()}, "estimate_other": S(f["est_other"]), "adjustments": [dict(a) for a in adj],
               "adjustments_total": S(adj_total), "total_usd": S(grand), "total_rub": S(grand * r), "videos": f["videos"], "gpu_hours": S(f["gpu_hours"])}
    return {"key": f"vast-daily:{day.isoformat()}", "kind": "daily", "period": day.isoformat(), "text": "\n".join(lines), "payload": payload,
            "adj_keys": [a["adj_key"] for a in adj]}


def build_monthly(ledger, year: int, month: int, rates, default_dph: Decimal = Decimal("0.2")) -> dict:
    first, last = month_bounds_msk(year, month)
    days = [first + timedelta(days=i) for i in range((last - first).days + 1)]
    gpu = disk = other = est = total_rub = Decimal(0)
    videos = 0
    hours = Decimal(0)
    dphs = []
    for d in days:
        f = day_figures(ledger, d)
        rate = rates.get(d)
        day_total = f["calc"]["gpu"] + f["calc"]["disk"] + f["calc"]["other"]
        gpu += f["calc"]["gpu"]; disk += f["calc"]["disk"]; other += f["calc"]["other"]; est += f["est_other"]
        total_rub += day_total * rate.value
        videos += f["videos"]; hours += f["gpu_hours"]
    adjs = ledger.adjustments_between(first, last)
    adj_total = sum((D(a["amount_usd"]) for a in adjs), Decimal(0))
    month_rate = rates.get(last)
    total_rub += adj_total * month_rate.value
    grand = gpu + disk + other + adj_total
    dph = _avg_dph(ledger, first, last) or default_dph
    always_on = dph * Decimal(24) * Decimal(len(days))
    saving = max(Decimal(0), always_on - gpu)
    per_video = grand / videos if videos else None
    conf = {c: Decimal(0) for c in ("gpu", "disk", "other")}
    conf_days = 0
    for d in days:
        k = ledger.sum_costs("confirmed", d)
        if any(k.values()):
            conf_days += 1
            for c in conf:
                conf[c] += k[c]
    lines = [f"SONYA — итог Vast.ai за {MONTHS[month]} {year}", "",
             f"Общие расходы: {usd(grand)} / {rub(total_rub)}", f"GPU: {usd(gpu)}", f"Диски и прочее: {usd(disk + other + adj_total)}",
             f"Обработано видео: {videos}",
             "Средняя стоимость видео: " + (f"{usd(per_video)} / {rub(total_rub / videos)}" if per_video is not None else "—"),
             f"Расчётная экономия от автоматической остановки: {usd(saving)} / {rub(saving * month_rate.value)} (ОЦЕНКА: сравнение с постоянной работой 24/7 одной карты ≈ {usd4(dph)}/ч)",
             "", "Источники (не смешиваются):",
             f"• Рассчитано по журналу (входит в общие расходы): {usd(gpu + disk + other)}",
             f"• Корректировки по сверке с Vast (входят): {usd(adj_total)}",
             (f"• Подтверждено Vast: {usd(sum(conf.values(), Decimal(0)))} за {conf_days} UTC-суток из {len(days)}" if conf_days else "• Подтверждено Vast: данных нет"),
             f"• Оценка трафика (НЕ входит): ≈ {usd4(est)}",
             "Рубли: сумма по дням по официальному курсу ЦБ РФ на каждый день" + (" (есть дни с последним известным курсом)" if any(rates.get(d).fallback for d in days) else "")]
    payload = {"kind": "monthly", "year": year, "month": month, "total_usd": S(grand), "total_rub": S(total_rub), "gpu_usd": S(gpu),
               "disk_other_usd": S(disk + other + adj_total), "videos": videos, "saving_usd": S(saving), "saving_is_estimate": True,
               "estimate_traffic_usd": S(est), "confirmed_usd": S(sum(conf.values(), Decimal(0))), "confirmed_days": conf_days}
    return {"key": f"vast-monthly:{year:04d}-{month:02d}", "kind": "monthly", "period": f"{year:04d}-{month:02d}", "text": "\n".join(lines), "payload": payload, "adj_keys": []}


def _avg_dph(ledger, first: date, last: date) -> Optional[Decimal]:
    vals = []
    for r in ledger.c.execute("SELECT detail FROM cost_entries WHERE kind='calculated' AND category='gpu' AND day BETWEEN ? AND ?", (first.isoformat(), last.isoformat())):
        d = json.loads(r[0] or "{}") or {}
        if d.get("dph_base"):
            vals.append(D(d["dph_base"]))
    return sum(vals, Decimal(0)) / len(vals) if vals else None
