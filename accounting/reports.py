"""Daily and monthly report builders. Numbers first (exact Decimals, no rounding), Telegram HTML second (accounting/tgformat).
Sources are never mixed:
  calculated   our journal x price snapshot, MSK days                -> gross spend (валовой расход)
  adjustments  reconciliation deltas for earlier UTC days             -> booked into the report they are applied in
  confirmed    Vast /charges, UTC days                                -> shown only in СВЕРКА, never added to ИТОГО
  estimate     traffic model                                          -> shown only in СВЕРКА, never added to ИТОГО
ИТОГО = gross + adjustments (unchanged business rule). Rubles are computed from the EXACT dollar amount, rounding only at display."""
from __future__ import annotations

import json
import time as _time
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Sequence

from accounting import gemini as gm
from accounting import tgformat as tf
from accounting.money import D, MSK, month_bounds_msk
from accounting.cbr import Rate

CATS = ("gpu", "disk", "other")
CAT_RU = {"gpu": "GPU", "disk": "Диски", "other": "Прочее"}
REASON_RU = {"reconciliation with Vast charges": "сверка со списаниями Vast"}
TOLERANCE_ABS = Decimal("0.05")          # same thresholds as vast_billing.reconcile_day
TOLERANCE_REL = Decimal("0.03")


def S(v) -> str:
    """Canonical money string for payloads: 6 decimals, never a float."""
    return str(D(v).quantize(Decimal('0.000001')))


MONTHS = ["", "январь", "февраль", "март", "апрель", "май", "июнь", "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь"]


def _fmt_date(d: date) -> str:
    return d.strftime("%d.%m.%Y")


def _dm(d: date) -> str:
    return d.strftime("%d.%m")


def _stamp(now: float) -> str:
    return datetime.fromtimestamp(now, MSK).strftime("%d.%m.%Y %H:%M МСК")


def rate_line(rate: Optional[Rate], now: float) -> str:
    if rate is None:
        r = "Курс ЦБ РФ: <b>нет данных</b> — суммы в рублях не рассчитаны"
    else:
        r = f"Курс ЦБ РФ на {tf.esc(rate.cbr_date)}: {tf.esc(tf.num(rate.value, 4))} ₽/$"
        if rate.fallback:
            r += " (ЦБ недоступен — взят последний известный)"
    return f"{r}\nДанные на {tf.esc(_stamp(now))}"


def _within_tolerance(diff: Decimal, a: Decimal, b: Decimal) -> bool:
    return abs(diff) <= max(TOLERANCE_ABS, TOLERANCE_REL * max(abs(a), abs(b)))


def _sum(xs) -> Decimal:
    return sum((D(x) for x in xs), Decimal(0))


def adjustments_for_day(ledger, utc_day: date) -> Dict[str, Decimal]:
    out = {c: Decimal(0) for c in CATS}
    for r in ledger.c.execute("SELECT category, amount_usd FROM adjustments WHERE day=?", (utc_day.isoformat(),)):
        out[r["category"]] = out.get(r["category"], Decimal(0)) + D(r["amount_usd"])
    return out


def compare(ledger, utc_day: date, calculated: Optional[Dict[str, Decimal]]) -> dict:
    """READ-ONLY comparison of one UTC day: confirmed vs calculated vs adjustments already booked for that day."""
    conf = ledger.sum_costs("confirmed", utc_day)
    has_conf = any(conf.values())
    calc = {c: D((calculated or {}).get(c, 0)) for c in CATS} if calculated is not None else None
    adj = adjustments_for_day(ledger, utc_day)
    out = {"day": utc_day, "has_confirmed": has_conf, "confirmed": conf, "calculated": calc, "adjustments": adj, "issues": []}
    if not has_conf or calc is None:
        out["status"] = "no_confirmed_data" if not has_conf else "no_calculation"
        return out
    out["diff"] = {c: conf[c] - calc[c] for c in CATS}
    out["residual"] = {c: conf[c] - calc[c] - adj[c] for c in CATS}
    bad = [c for c in CATS if not _within_tolerance(out["residual"][c], conf[c], calc[c] + adj[c])]
    out["status"] = "ok" if not bad else "mismatch"
    for c in bad:
        out["issues"].append(f"{CAT_RU[c]}: Vast {tf.usd(conf[c])}, журнал {tf.usd(calc[c])}, корректировка {tf.usd(adj[c])} — "
                             f"не сверено {tf.usd(out['residual'][c])}")
    return out


def day_figures(ledger, day: date) -> dict:
    calc = ledger.sum_costs("calculated", day)
    est = ledger.sum_costs("estimate", day)
    facts = ledger.get_facts(day) or {"videos": 0, "gpu_hours": Decimal(0), "frozen": False}
    return {"calc": calc, "est_other": est["other"], "videos": facts["videos"], "gpu_hours": facts["gpu_hours"], "has_facts": bool(ledger.get_facts(day))}


# ------------------------------------------------------------------------------------------------ gemini
def _g_usd(rub_v, r):
    return None if r is None or not r else tf.dec(rub_v) / r


def gemini_block(g: Dict[str, dict], r: Optional[Decimal], labels: Sequence[str], prices: gm.Prices,
                 balance: Optional[str] = None) -> str:
    """GEMINI: three periods in USD / RUB, requests, videos, RUB per video. Gemini is billed in RUB: USD = RUB / CBR rate."""
    rows = [tf.Row(lab, tf.num(_g_usd(g[k]["rub"], r)), tf.num(g[k]["rub"])) for lab, k in zip(labels, ("day", "prev", "month"))]
    m = g["month"]
    pv = m["per_video"]
    facts = [tf.Row(f"Запросов, {labels[0].lower()}", str(g["day"]["requests"])),
             tf.Row("Запросов, месяц", str(m["requests"])),
             tf.Row("Видео с Gemini, месяц", str(m["jobs"])),
             tf.Row("RUB на видео, месяц", tf.num(pv) if pv is not None else tf.DASH)]
    if prices.estimate:
        note = (f"Тарифы — <b>оценка</b>: {tf.esc(tf.num(prices.rub_in_per_m, 0))} ₽ вход / {tf.esc(tf.num(prices.rub_out_per_m, 0))} ₽ "
                "выход за 1 млн токенов (Timeweb не публикует цену модели).")
    else:
        note = (f"Тарифы Timeweb: {tf.esc(tf.num(prices.rub_in_per_m, 2))} ₽ вход / {tf.esc(tf.num(prices.rub_out_per_m, 2))} ₽ "
                "выход за 1 млн токенов.")
    lines = [tf.table(rows, ("USD", "RUB")), tf.table(facts), note]
    if not m["requests"]:
        lines.append("Запросов к Gemini в этом месяце пока нет.")
    if m["no_tokens"]:
        lines.append(f"Без счётчика токенов: {m['no_tokens']} запрос(ов) — в сумму вошли как 0.")
    if balance is not None:
        lines.append("Баланс Timeweb: " + balance)
    return tf.block("GEMINI", *lines, sub="Timeweb")


def grand_total_block(items: Sequence[tuple], r: Optional[Decimal]) -> str:
    """items: [(label, vast_usd, gemini_rub)] -> «ВСЕГО РАСХОДЫ = VAST + GEMINI», one row per period."""
    rows = []
    for label, vast_usd, g_rub in items:
        v_rub = tf.to_rub(vast_usd, r)
        rub_total = None if v_rub is None else v_rub + tf.dec(g_rub)
        usd_total = None if r is None else tf.dec(vast_usd) + tf.dec(g_rub) / r
        rows.append(tf.Row(label, tf.num(usd_total), tf.num(rub_total)))
    return tf.block("ВСЕГО РАСХОДЫ = VAST + GEMINI", tf.table(rows, ("USD", "RUB")))


# ------------------------------------------------------------------------------------------------ daily
def daily_numbers(ledger, day: date, rate: Optional[Rate], recon: Optional[dict], adjustments: Sequence[dict]) -> dict:
    f = day_figures(ledger, day)
    c = f["calc"]
    gross = c["gpu"] + c["disk"] + c["other"]
    adj_total = _sum(a["amount_usd"] for a in adjustments)
    grand = gross + adj_total
    r = rate.value if rate is not None else None
    return {"day": day, "rate": rate, "r": r, "calc": c, "gross": gross, "adjustments": list(adjustments), "adj_total": adj_total,
            "gemini": gm.periods(ledger, day, gm.Prices.from_env(), r), "g_prices": gm.Prices.from_env(),
            "grand": grand, "videos": f["videos"], "gpu_hours": f["gpu_hours"], "est_other": f["est_other"], "has_facts": f["has_facts"],
            "per_video": (grand / f["videos"]) if f["videos"] else None, "recon": recon}


def render_daily(n: dict, now: float, first_day: Optional[date] = None) -> str:
    day, r = n["day"], n["r"]
    R = lambda v: tf.to_rub(v, r)                                   # noqa: E731 - exact, rounded later by the table
    head = tf.title(f"SONYA · Vast.ai — расходы за {_fmt_date(day)}") + "\nСутки по Москве. Суммы в USD и рублях по курсу ЦБ."
    rows = [tf.Row(CAT_RU[k], tf.num(n["calc"][k]), tf.num(R(n["calc"][k]))) for k in CATS]
    rows.append(tf.Row("Расход за сутки", tf.num(n["gross"]), tf.num(R(n["gross"])), rule_before=True))
    rows.append(tf.Row("Корректировки", tf.num(n["adj_total"]), tf.num(R(n["adj_total"]))))
    rows.append(tf.Row("Итого", tf.num(n["grand"]), tf.num(R(n["grand"])), total=True))
    notes: List[str] = []
    if not n["has_facts"]:
        notes.append("<b>Нет данных журнала за эти сутки</b> — суммы могут быть неполными.")
    if n["adjustments"]:
        notes.append("Корректировки — по сверке прошлых суток (UTC), подробно ниже.")
    parts_usd = [n["calc"][k] for k in CATS]
    if r is not None and (tf.rounding_mismatch(parts_usd, n["gross"]) or tf.rounding_mismatch([n["gross"], n["adj_total"]], n["grand"])
                          or tf.rounding_mismatch([R(x) for x in parts_usd], R(n["gross"]))
                          or tf.rounding_mismatch([R(n["gross"]), R(n["adj_total"])], R(n["grand"]))):
        notes.append("<i>Считается по точным суммам, округление — только при выводе, поэтому строки могут не сходиться с итогом на 0.01.</i>")
    structure = tf.block("СТРУКТУРА РАСХОДОВ", tf.table(rows, ("USD", "RUB")), "\n".join(notes))

    pv = n["per_video"]
    eff_rows = [tf.Row("Обработано видео", str(n["videos"])), tf.Row("GPU-время, ч", tf.num(n["gpu_hours"])),
                tf.Row("Себестоимость, USD", tf.num(pv) if pv is not None else tf.DASH),
                tf.Row("Себестоимость, RUB", tf.num(R(pv)) if pv is not None else tf.DASH)]
    eff = tf.block("ЭФФЕКТИВНОСТЬ", tf.table(eff_rows), "" if pv is not None else "Видео не обработано — себестоимость не считается.")

    recon = tf.block("СВЕРКА С VAST", *reconciliation_lines(n, first_day))
    blocks = [(head, False), (structure, False), (eff, False), (recon, False)]
    if n.get("gemini") is not None:
        d = n["day"]
        blocks.append((gemini_block(n["gemini"], r, (f"За {_dm(d)}", f"За {_dm(d - timedelta(days=1))}", f"С {_dm(d.replace(day=1))}"),
                                    n["g_prices"]), False))
        blocks.append((grand_total_block([(f"За {_dm(d)}", n["grand"], n["gemini"]["day"]["rub"])], r), False))
    if n["adjustments"]:
        blocks.append((adjustments_block(n["adjustments"], r), True))
    blocks.append((rate_line(n["rate"], now), False))
    return tf.assemble(blocks)


def reconciliation_lines(n: dict, first_day: Optional[date] = None) -> List[str]:
    rc, r = n["recon"], n["r"]
    R = lambda v: tf.to_rub(v, r)                                   # noqa: E731
    issues: List[str] = []
    rows: List[tf.Row] = []
    lines: List[str] = []
    if rc and rc.get("has_confirmed") and rc.get("calculated") is not None:
        u = rc["day"]
        conf, calc, adj = _sum(rc["confirmed"].values()), _sum(rc["calculated"].values()), _sum(rc["adjustments"].values())
        rows += [tf.Row(f"Vast, UTC {_dm(u)}", tf.num(conf), tf.num(R(conf))),
                 tf.Row(f"Журнал, UTC {_dm(u)}", tf.num(calc), tf.num(R(calc))),
                 tf.Row("Разница", tf.num(conf - calc), tf.num(R(conf - calc)), rule_before=True),
                 tf.Row(f"Корр. за {_dm(u)}", tf.num(adj), tf.num(R(adj))),
                 tf.Row("Не сверено", tf.num(conf - calc - adj), tf.num(R(conf - calc - adj)))]
        lines.append(f"Сверяются закрытые UTC-сутки {tf.esc(_fmt_date(u))} (03:00 {tf.esc(_dm(u))} – 03:00 {tf.esc(_dm(u + timedelta(days=1)))} МСК).")
        issues += rc.get("issues", [])
    else:
        u = rc["day"] if rc else None
        lines.append("Подтверждённых списаний Vast" + (f" за UTC-сутки {tf.esc(_fmt_date(u))}" if u else "") + " пока нет — сверка появится позже.")
    rows.append(tf.Row("Трафик (оценка)", tf.num(n["est_other"]), tf.num(R(n["est_other"])), rule_before=bool(rows)))
    out = [tf.table(rows, ("USD", "RUB"))] + lines
    out.append("Vast — подтверждённые списания; журнал — наш расчёт по журналу инстансов. "
               f"Корректировки уже входят в ИТОГО. Оценка трафика за {tf.esc(_dm(n['day']))} в ИТОГО не входит.")
    for a in n["adjustments"]:
        if first_day and date.fromisoformat(a["day"]) < first_day:
            issues.append(f"Корректировка за UTC {_dm(date.fromisoformat(a['day']))} относится к суткам, которых нет в отчётах "
                          f"(учёт ведётся с {_dm(first_day)}), но вошла в ИТОГО за {_dm(n['day'])}")
            break
    if rc and rc.get("has_confirmed") and rc.get("calculated") is not None and not issues:
        out.append(tf.status_line(True, "сходится в пределах допуска ($0.05 или 3 %)"))
    elif issues:
        out.append(tf.status_line(False, ""))
        out.append(tf.bullets(issues))
    return out


def adjustments_block(adjustments: Sequence[dict], r: Optional[Decimal]) -> str:
    rows = [tf.Row(f"{_dm(date.fromisoformat(a['day']))} {CAT_RU.get(a['category'], a['category'])}", tf.num(a["amount_usd"]),
                   tf.num(tf.to_rub(a["amount_usd"], r))) for a in adjustments]
    reasons = sorted({REASON_RU.get(a.get("reason") or "", a.get("reason") or "—") for a in adjustments})
    return tf.block("КОРРЕКТИРОВКИ", tf.table(rows, ("USD", "RUB")), "Причина: " + tf.esc(", ".join(reasons)) + ". Дата — UTC-сутки, к которым относится.")


def daily_payload(n: dict) -> dict:
    r = n["r"]
    return {"kind": "daily", "fmt": "html", "day": n["day"].isoformat(), "rate": str(r) if r is not None else None,
            "rate_cbr_date": n["rate"].cbr_date if n["rate"] else None, "rate_fallback": bool(n["rate"] and n["rate"].fallback),
            "calculated": {k: S(v) for k, v in n["calc"].items()}, "gross_usd": S(n["gross"]), "estimate_other": S(n["est_other"]),
            "adjustments": [dict(a) for a in n["adjustments"]], "adjustments_total": S(n["adj_total"]), "total_usd": S(n["grand"]),
            "total_rub": S(n["grand"] * r) if r is not None else None, "videos": n["videos"], "gpu_hours": S(n["gpu_hours"]),
            "recon_status": (n["recon"] or {}).get("status"),
            **({"gemini_rub": S(n["gemini"]["day"]["rub"]), "gemini_requests": n["gemini"]["day"]["requests"],
                "gemini_price_estimate": n["g_prices"].estimate,
                "all_total_rub": S(n["grand"] * r + n["gemini"]["day"]["rub"]) if r is not None else None}
               if n.get("gemini") is not None else {})}


def build_daily(ledger, day: date, rate: Optional[Rate], utc_recon: Optional[dict] = None, adjustments: Optional[Sequence[dict]] = None,
                now: Optional[float] = None, first_day: Optional[date] = None) -> dict:
    """utc_recon: result of compare() (read-only). adjustments: default = not yet applied ones (scheduler); a re-render passes
    the ones applied on `day`."""
    adj = ledger.unapplied_adjustments() if adjustments is None else list(adjustments)
    n = daily_numbers(ledger, day, rate, utc_recon, adj)
    text = render_daily(n, _time.time() if now is None else now, first_day)
    return {"key": f"vast-daily:{day.isoformat()}", "kind": "daily", "period": day.isoformat(), "text": text, "payload": daily_payload(n),
            "adj_keys": [a["adj_key"] for a in adj]}


# ------------------------------------------------------------------------------------------------ monthly
def build_monthly(ledger, year: int, month: int, rates, default_dph: Decimal = Decimal("0.2"), now: Optional[float] = None) -> dict:
    now = _time.time() if now is None else now
    first, last = month_bounds_msk(year, month)
    days = [first + timedelta(days=i) for i in range((last - first).days + 1)]
    usd_ = {c: Decimal(0) for c in CATS}
    rub_ = {c: Decimal(0) for c in CATS}
    est = Decimal(0)
    videos, hours = 0, Decimal(0)
    for d in days:
        f = day_figures(ledger, d)
        rv = rates.get(d).value
        for c in CATS:
            usd_[c] += f["calc"][c]
            rub_[c] += f["calc"][c] * rv                 # rubles: each day at that day's official rate
        est += f["est_other"]
        videos += f["videos"]; hours += f["gpu_hours"]
    adjs = ledger.adjustments_between(first, last)
    adj_total = _sum(a["amount_usd"] for a in adjs)
    month_rate = rates.get(last)
    adj_rub = adj_total * month_rate.value
    gross, gross_rub = _sum(usd_.values()), _sum(rub_.values())
    grand, total_rub = gross + adj_total, gross_rub + adj_rub
    dph = _avg_dph(ledger, first, last) or default_dph
    always_on = dph * Decimal(24) * Decimal(len(days))
    saving = max(Decimal(0), always_on - usd_["gpu"])
    per_video = grand / videos if videos else None
    conf = {c: Decimal(0) for c in CATS}
    conf_days = 0
    for d in days:
        k = ledger.sum_costs("confirmed", d)
        if any(k.values()):
            conf_days += 1
            for c in conf:
                conf[c] += k[c]
    conf_total = _sum(conf.values())

    head = tf.title(f"SONYA · Vast.ai — итог за {MONTHS[month]} {year}") + "\nРубли — по курсу ЦБ на каждый день."
    rows = [tf.Row(CAT_RU[c], tf.num(usd_[c]), tf.num(rub_[c])) for c in CATS]
    rows += [tf.Row("Расход по журналу", tf.num(gross), tf.num(gross_rub), rule_before=True),
             tf.Row("Корректировки", tf.num(adj_total), tf.num(adj_rub)),
             tf.Row("Итого", tf.num(grand), tf.num(total_rub), total=True)]
    note = []
    if tf.rounding_mismatch(list(usd_.values()) + [adj_total], grand) or tf.rounding_mismatch(list(rub_.values()) + [adj_rub], total_rub):
        note.append("<i>Считается по точным суммам, округление — только при выводе (возможна разница 0.01).</i>")
    if any(rates.get(d).fallback for d in days):
        note.append("Есть дни, где ЦБ был недоступен — взят последний известный курс.")
    structure = tf.block("СТРУКТУРА РАСХОДОВ", tf.table(rows, ("USD", "RUB")), "\n".join(note))
    eff = tf.block("ЭФФЕКТИВНОСТЬ", tf.table([
        tf.Row("Обработано видео", str(videos)), tf.Row("GPU-время, ч", tf.num(hours)),
        tf.Row("Себестоимость, USD", tf.num(per_video) if per_video is not None else tf.DASH),
        tf.Row("Себестоимость, RUB", tf.num(total_rub / videos) if videos else tf.DASH)]))
    saving_b = tf.block("ЭКОНОМИЯ ОТ АВТООСТАНОВКИ", f"≈ {tf.esc(tf.pair(saving, saving * month_rate.value))}",
                        f"<i>Оценка: сравнение с работой одной карты 24/7 по ≈ {tf.esc(tf.usd(dph, 4))}/ч.</i>")
    rc_rows = [tf.Row("Расход по журналу", tf.num(gross), tf.num(gross_rub)),
               tf.Row(f"Vast, {conf_days} UTC-сут.", tf.num(conf_total) if conf_days else tf.DASH,
                      tf.num(conf_total * month_rate.value) if conf_days else tf.DASH),
               tf.Row("Корректировки", tf.num(adj_total), tf.num(adj_rub)),
               tf.Row("Трафик, оценка", tf.num(est), tf.num(est * month_rate.value), rule_before=True)]
    recon = tf.block("СВЕРКА С VAST", tf.table(rc_rows, ("USD", "RUB")),
                     f"Подтверждённые списания Vast есть за {conf_days} UTC-суток из {len(days)}; в ИТОГО они не входят. "
                     "Оценка трафика в ИТОГО не входит.")
    text = tf.assemble([(head, False), (structure, False), (eff, False), (recon, False), (saving_b, True), (rate_line(month_rate, now), False)])
    payload = {"kind": "monthly", "fmt": "html", "year": year, "month": month, "total_usd": S(grand), "total_rub": S(total_rub), "gpu_usd": S(usd_["gpu"]),
               "disk_other_usd": S(usd_["disk"] + usd_["other"] + adj_total), "videos": videos, "saving_usd": S(saving), "saving_is_estimate": True,
               "estimate_traffic_usd": S(est), "confirmed_usd": S(conf_total), "confirmed_days": conf_days}
    return {"key": f"vast-monthly:{year:04d}-{month:02d}", "kind": "monthly", "period": f"{year:04d}-{month:02d}", "text": text, "payload": payload, "adj_keys": []}


def _avg_dph(ledger, first: date, last: date) -> Optional[Decimal]:
    vals = []
    for r in ledger.c.execute("SELECT detail FROM cost_entries WHERE kind='calculated' AND category='gpu' AND day BETWEEN ? AND ?", (first.isoformat(), last.isoformat())):
        d = json.loads(r[0] or "{}") or {}
        if d.get("dph_base"):
            vals.append(D(d["dph_base"]))
    return sum(vals, Decimal(0)) / len(vals) if vals else None
