from datetime import date
from decimal import Decimal

import pytest

from accounting import cbr, reports, usage, vast_billing
from accounting.ledger import Ledger
from accounting.money import both, rub, usd
from tests.acct.helpers import FixedRates, Jobs, fixture_store, msk

DAY = date(2026, 11, 6)


def test_calculated_cost_is_exact_decimal():
    led, st = Ledger(), fixture_store()
    r = usage.compute_day(led, st, DAY, Jobs({DAY: 4}), now=msk(2026, 11, 7, 3))
    # GPU 1.5 h x 0.20 = 0.30 ; disk 80 GB x 0.36/720 x (1.5 run + 1.0 stopped) = 0.10
    assert r["gpu"] == Decimal("0.30") and r["disk"] == Decimal("0.10")
    assert r["gpu_hours"] == Decimal("1.5") and r["videos"] == 4
    assert led.sum_costs("calculated", DAY)["gpu"] == Decimal("0.30")


def test_recompute_is_idempotent_no_double_count():
    led, st = Ledger(), fixture_store()
    for _ in range(3):
        usage.compute_day(led, st, DAY, Jobs({DAY: 4}), now=msk(2026, 11, 7, 3))
    assert led.sum_costs("calculated", DAY) == {"gpu": Decimal("0.30"), "disk": Decimal("0.10"), "other": Decimal(0)}


def test_other_days_get_nothing_and_facts_freeze():
    led, st = Ledger(), fixture_store()
    usage.compute_day(led, st, date(2026, 11, 5), Jobs(), now=msk(2026, 11, 7, 3))
    assert led.sum_costs("calculated", date(2026, 11, 5))["gpu"] == 0
    usage.compute_day(led, st, DAY, Jobs({DAY: 4}), now=msk(2026, 11, 7, 3))
    assert led.get_facts(DAY)["frozen"] is True
    led.put_facts(DAY, 99, Decimal(9))                 # frozen facts cannot be rewritten
    assert led.get_facts(DAY)["videos"] == 4


def test_running_instance_open_interval_clipped_to_day():
    from dispatcher.store import FleetStore
    st = FleetStore()
    t0 = msk(2026, 11, 6, 22)
    st.add(t0, id=2, label="sonya-pool-bb", state="running", dph_base=Decimal("0.40"), storage_gb_month=Decimal("0.36"), disk_gb=Decimal("80"))
    st.event(t0, 2, "created"); st.event(t0, 2, "running")
    led = Ledger()
    r = usage.compute_day(led, st, DAY, Jobs(), now=msk(2026, 11, 7, 6))
    assert r["gpu_hours"] == Decimal("2")              # 22:00-24:00 only
    r2 = usage.compute_day(led, st, date(2026, 11, 7), Jobs(), now=msk(2026, 11, 7, 6))
    assert r2["gpu_hours"] == Decimal("6")


def test_daily_report_text_and_amounts():
    led, st = Ledger(), fixture_store()
    usage.compute_day(led, st, DAY, Jobs({DAY: 4}), now=msk(2026, 11, 7, 3))
    rep = reports.build_daily(led, DAY, FixedRates("80").get(DAY))
    t = rep["text"]
    assert t.startswith("SONYA — Vast.ai, расходы за 06.11.2026")
    assert "GPU: $0.30 / 24.00 ₽" in t and "Диски: $0.10 / 8.00 ₽" in t and "Прочие расходы: $0.00 / 0.00 ₽" in t
    assert "ИТОГО: $0.40 / 32.00 ₽" in t and "Обработано видео: 4" in t and "GPU работал: 1.50 ч" in t
    assert "Стоимость одного видео: $0.10 / 8.00 ₽" in t
    assert "Подтверждено Vast: данных пока нет" in t and "Оценка трафика (НЕ входит в ИТОГО)" in t
    assert "Курс ЦБ РФ на 06.11.2026: 80.0000" in t
    assert Decimal(rep["payload"]["total_usd"]) == Decimal("0.40")


def test_estimates_never_enter_the_total():
    led, st = Ledger(), fixture_store()
    usage.compute_day(led, st, DAY, Jobs({DAY: 100}), now=msk(2026, 11, 7, 3))
    assert led.sum_costs("estimate", DAY)["other"] > 0
    assert Decimal(reports.build_daily(led, DAY, FixedRates().get(DAY))["payload"]["total_usd"]) == Decimal("0.40")


def test_zero_videos_has_no_division_by_zero():
    led, st = Ledger(), fixture_store()
    usage.compute_day(led, st, DAY, Jobs(), now=msk(2026, 11, 7, 3))
    assert "— (видео не обработано)" in reports.build_daily(led, DAY, FixedRates().get(DAY))["text"]


def test_money_formatting():
    assert usd(Decimal("1234.567")) == "$1,234.57" and rub(Decimal("1234.5")) == "1 234.50 ₽"
    assert both(Decimal("0.4"), Decimal("82.1386")) == "$0.40 / 32.86 ₽"


# ---- confirmed charges + reconciliation
ROWS = [{"id": "c1", "day": 1_793_836_800, "type": "instance", "amount": "0.62"},        # 2026-11-05 UTC
        {"id": "c2", "day": 1_793_836_800, "type": "volume", "amount": "0.08"}]


def test_confirmed_import_is_idempotent_and_separate():
    led = Ledger()
    assert vast_billing.import_charges(led, ROWS) == 2
    vast_billing.import_charges(led, ROWS)
    k = led.sum_costs("confirmed", date(2026, 11, 5))
    assert k["gpu"] == Decimal("0.62") and k["disk"] == Decimal("0.08")
    assert led.sum_costs("calculated", date(2026, 11, 5))["gpu"] == 0           # never mixed


def test_reconciliation_adds_delta_once_and_corrections_are_incremental():
    led = Ledger()
    vast_billing.import_charges(led, ROWS)
    calc = {"gpu": Decimal("0.50"), "disk": Decimal("0.08"), "other": Decimal(0)}
    r1 = vast_billing.reconcile_day(led, date(2026, 11, 5), calc)
    assert r1["status"] == "adjusted" and r1["adjusted"] == {"gpu": Decimal("0.12")}
    r2 = vast_billing.reconcile_day(led, date(2026, 11, 5), calc)             # re-run: nothing new
    assert r2["status"] == "ok" and r2["adjusted"] == {}
    vast_billing.import_charges(led, [{"id": "c1", "day": 1_793_836_800, "type": "instance", "amount": "0.90"}])   # Vast revised the charge
    r3 = vast_billing.reconcile_day(led, date(2026, 11, 5), calc)
    assert r3["adjusted"] == {"gpu": Decimal("0.28")}                          # only the extra delta
    total = sum(Decimal(a["amount_usd"]) for a in led.unapplied_adjustments() if a["category"] == "gpu")
    assert total == Decimal("0.40")                                            # 0.90 - 0.50, counted exactly once


def test_small_differences_are_tolerated():
    led = Ledger()
    vast_billing.import_charges(led, [{"id": "c", "day": 1_793_836_800, "type": "instance", "amount": "0.52"}])
    r = vast_billing.reconcile_day(led, date(2026, 11, 5), {"gpu": Decimal("0.50"), "disk": Decimal(0), "other": Decimal(0)})
    assert r["adjusted"] == {} and led.unapplied_adjustments() == []


def test_adjustment_shows_in_next_daily_report_once():
    led, st = Ledger(), fixture_store()
    led.add_adjustment("adj:x", date(2026, 11, 5), "gpu", Decimal("0.12"), "reconciliation with Vast charges")
    usage.compute_day(led, st, DAY, Jobs({DAY: 4}), now=msk(2026, 11, 7, 3))
    rep = reports.build_daily(led, DAY, FixedRates("80").get(DAY))
    assert "Корректировки прошлых периодов" in rep["text"] and Decimal(rep["payload"]["total_usd"]) == Decimal("0.52")
    led.mark_adjustments_applied(rep["adj_keys"], DAY)
    assert Decimal(reports.build_daily(led, DAY, FixedRates("80").get(DAY))["payload"]["total_usd"]) == Decimal("0.40")


# ---- CBR
XML = ('<?xml version="1.0" encoding="windows-1251"?><ValCurs Date="07.11.2026" name="Foreign Currency Market"><Valute ID="R01235">'
       '<NumCode>840</NumCode><CharCode>USD</CharCode><Nominal>1</Nominal><Name>US</Name><Value>82,1386</Value></Valute></ValCurs>')


def test_cbr_parse_and_store():
    v, d = cbr.parse_xml(XML.encode("cp1251"))
    assert v == Decimal("82.1386") and d == "07.11.2026"
    led = Ledger()
    c = cbr.CbrRates(led, fetch=lambda day: XML.encode("cp1251"))
    assert c.get(DAY).value == Decimal("82.1386") and led.get_rate(DAY)["value"] == Decimal("82.1386")


def test_cbr_outage_uses_last_known_and_says_so_never_invents():
    led = Ledger()
    led.put_rate(date(2026, 11, 5), Decimal("81.5"), "05.11.2026", False)
    def boom(day): raise OSError("down")
    r = cbr.CbrRates(led, fetch=boom).get(DAY)
    assert r.value == Decimal("81.5") and r.fallback is True
    assert "последний известный" in reports.build_daily(led, DAY, r)["text"]
    with pytest.raises(cbr.RateUnavailable):
        cbr.CbrRates(Ledger(), fetch=boom).get(DAY)
