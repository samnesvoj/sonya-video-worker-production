"""Gemini spend in the accountant (owner 2026-11-16): import, prices, formatter, rounding, empty data."""
import json
from datetime import date
from decimal import Decimal

from accounting import gemini as gm
from accounting import reports
from accounting.ledger import Ledger
from accounting.summary import build_summary
from tests.acct.helpers import FixedRates, fixture_store, msk

P = gm.Prices()                                   # 40 / 240 RUB per 1M, estimate


def _doc(job, mode, *calls):
    return json.dumps({"v": 1, "job_id": job, "mode": mode, "calls": list(calls)}).encode()


def _call(ts, i=10000, o=1000, **kw):
    return {"ts": ts, "provider": "timeweb", "model": "gemini-3.1-flash-lite", "prompt_version": "cut-v1",
            "input_tokens": i, "output_tokens": o, "images": 0, **kw}


def _ledger_with_calls():
    led = Ledger()
    objs = {
        "worker-v2/llm-usage/2026-11-06/j1-a.json": _doc("j1", "virality", _call(msk(2026, 11, 6, 12)), _call(msk(2026, 11, 6, 12, 1))),
        "worker-v2/llm-usage/2026-11-05/j0-b.json": _doc("j0", "story", _call(msk(2026, 11, 5, 9), i=1_000_000, o=0)),
        "worker-v2/llm-usage/2026-11-01/jx-c.json": _doc("jx", "hooks", _call(msk(2026, 11, 1, 9), cost_usd=0.01, provider="openrouter")),
    }
    n = gm.import_objects(led, objs, lambda k: objs[k], now=msk(2026, 11, 6, 13))
    return led, objs, n


def test_import_is_idempotent_and_prices_calls():
    led, objs, n = _ledger_with_calls()
    assert n == 4
    assert gm.import_objects(led, objs, lambda k: objs[k]) == 0                      # same objects again -> nothing
    p = gm.periods(led, date(2026, 11, 6), P, Decimal("80"))
    # one call: 10k in * 40/1M + 1k out * 240/1M = 0.40 + 0.24 = 0.64 RUB
    assert p["day"]["rub"] == Decimal("1.28") and p["day"]["requests"] == 2 and p["day"]["jobs"] == 1
    assert p["prev"]["rub"] == Decimal("40") and p["prev"]["jobs"] == 1
    # OpenRouter reported $0.01 -> 0.80 RUB at 80
    assert p["month"]["rub"] == Decimal("42.08") and p["month"]["requests"] == 4 and p["month"]["jobs"] == 3
    assert p["month"]["per_video"].quantize(Decimal("0.01")) == Decimal("14.03")
    assert p["day"]["by_mode"] == {"virality": Decimal("1.28")}


def test_bad_objects_and_calls_without_tokens():
    led = Ledger()
    objs = {"a.json": b"not json", "b.json": _doc("j", "beep", _call(msk(2026, 11, 6, 10), i=None, o=None), "junk", {"ts": 0})}
    assert gm.import_objects(led, objs, lambda k: objs[k]) == 1
    s = gm.stats(led, date(2026, 11, 6), date(2026, 11, 6), P)
    assert s["requests"] == 1 and s["rub"] == 0 and s["no_tokens"] == 1


def test_prices_from_env():
    assert gm.Prices.from_env({}) == gm.Prices(Decimal("40"), Decimal("240"), True)
    p = gm.Prices.from_env({"SONYA_GEMINI_RUB_PER_M_IN": "12.5", "SONYA_GEMINI_RUB_PER_M_OUT": "50", "SONYA_GEMINI_PRICE_OFFICIAL": "1"})
    assert p == gm.Prices(Decimal("12.5"), Decimal("50"), False)


def test_s3_day_prefixes_cover_the_msk_day():
    assert gm.s3_day_prefixes("p/", msk(2026, 11, 6, 1), 2) == ["p/2026-11-05/", "p/2026-11-04/"]


def test_gemini_block_rounding_and_estimate_mark():
    led, _, _ = _ledger_with_calls()
    g = gm.periods(led, date(2026, 11, 6), P, Decimal("80"))
    txt = reports.gemini_block(g, Decimal("80"), ("Сегодня", "Вчера", "С начала месяца"), P, "1 234.56 ₽")
    assert "<b>GEMINI</b>" in txt and "1.28" in txt and "0.02" in txt          # 1.28 RUB = $0.016 -> 0.02
    assert "42.08" in txt and "14.03" in txt and "оценка" in txt and "Баланс Timeweb: 1 234.56 ₽" in txt
    official = reports.gemini_block(g, Decimal("80"), ("a", "b", "c"), gm.Prices(Decimal("40"), Decimal("240"), False))
    assert "оценка" not in official and "Тарифы Timeweb" in official


def test_gemini_block_empty_month_and_no_rate():
    g = gm.periods(Ledger(), date(2026, 11, 6), P, None)
    txt = reports.gemini_block(g, None, ("Сегодня", "Вчера", "С начала месяца"), P)
    assert "пока нет" in txt and "0.00" in txt and "—" in txt                    # no rate -> USD dash, no per-video
    tot = reports.grand_total_block([("Сегодня", Decimal("1"), Decimal(0))], None)
    assert "ВСЕГО РАСХОДЫ = VAST + GEMINI" in tot and "—" in tot


def test_grand_total_adds_vast_and_gemini_only_there():
    tot = reports.grand_total_block([("Сегодня", Decimal("1.50"), Decimal("20"))], Decimal("80"))
    assert "1.75" in tot and "140.00" in tot                                    # $1.50 + 20/80 ; 120 + 20 RUB


def test_summary_has_separate_gemini_block_and_total():
    led, _, _ = _ledger_with_calls()
    txt = build_summary(fixture_store(), FixedRates("80"), msk(2026, 11, 6, 13), Decimal("12.5"), led)
    assert txt.index("<b>GEMINI</b>") < txt.index("<b>ВСЕГО РАСХОДЫ = VAST + GEMINI</b>") < txt.index("<b>БАЛАНС VAST</b>")
    assert "по API не подключён" in txt and len(txt) <= 3900
    assert "<b>GEMINI</b>" not in build_summary(fixture_store(), FixedRates("80"), msk(2026, 11, 6, 13), Decimal("12.5"))


def test_daily_report_has_gemini_and_payload():
    led, _, _ = _ledger_with_calls()
    rep = reports.build_daily(led, date(2026, 11, 6), FixedRates("80").get(date(2026, 11, 6)), now=msk(2026, 11, 7, 9))
    assert "<b>GEMINI</b>" in rep["text"] and "ВСЕГО РАСХОДЫ = VAST + GEMINI" in rep["text"]
    pl = rep["payload"]
    assert pl["gemini_rub"] == "1.280000" and pl["gemini_requests"] == 2 and pl["gemini_price_estimate"] is True


def test_timeweb_balance_text():
    assert gm.timeweb_balance_text("t", get=lambda: {"finances": {"balance": 1234.5}}) == "1 234.50 ₽"
    assert gm.timeweb_balance_text("t", get=lambda: {}) is None
    assert gm.timeweb_balance_text("", get=lambda: 1 / 0) is None
