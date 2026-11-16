"""Telegram formatter + report layout: alignment, signs, rounding, missing data, long GPU names, discrepancies, HTML, length."""
import json
import re
from datetime import date
from decimal import Decimal
from html.parser import HTMLParser

import pytest

from accounting import reports, tgformat as tf
from accounting.cbr import Rate
from accounting.ledger import Ledger
from accounting.summary import build_summary, check_alert
from dispatcher.store import FleetStore
from tests.acct.helpers import FixedRates, msk

RATE = Rate(date(2026, 11, 14), Decimal("82.1516"), "14.11.2026", False)
NOW = msk(2026, 11, 15, 0, 5)


# ------------------------------------------------------------------ helpers
class TgHtml(HTMLParser):
    """Telegram Bot API HTML subset as we use it: only b/i/pre, balanced, nothing nested inside <pre>."""
    ALLOWED = {"b", "i", "pre"}

    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.stack, self.errors = [], []

    def handle_starttag(self, tag, attrs):
        if tag not in self.ALLOWED or attrs:
            self.errors.append(f"tag {tag}")
        if "pre" in self.stack:
            self.errors.append(f"{tag} inside pre")
        self.stack.append(tag)

    def handle_endtag(self, tag):
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"unbalanced </{tag}>")
        else:
            self.stack.pop()

    def handle_entityref(self, name):
        if name not in ("lt", "gt", "amp", "quot"):
            self.errors.append(f"entity {name}")


def assert_tg_html(text):
    p = TgHtml()
    p.feed(text)
    p.close()
    assert not p.errors and not p.stack, (p.errors, p.stack)
    stripped = re.sub(r"</?(b|i|pre)>", "", text)
    assert "<" not in stripped and ">" not in stripped                    # every dynamic < > was escaped
    assert not re.search(r"&(?!(lt|gt|amp|quot);)", stripped)             # every & was escaped
    assert len(text) <= tf.LIMIT


def pre_blocks(text):
    return [tf.plain(m) for m in re.findall(r"<pre>(.*?)</pre>", text, re.S)]


def assert_narrow_and_aligned(text, width=tf.WIDTH):
    for blk in pre_blocks(text):
        lines = blk.split("\n")
        assert all(len(ln) <= width for ln in lines), blk                   # no horizontal scrolling on a phone
        rows = [ln for ln in lines if re.search(r"\d$|—$", ln) and not set(ln) <= {"-"}]
        if len(rows) > 1:
            assert len({len(ln) for ln in rows}) == 1, blk                  # right edges line up


def real_ledger():
    """The real 14.11.2026 numbers (exact strings from the production ledger)."""
    led = Ledger()
    d = date(2026, 11, 14)
    led.upsert_cost("calc:a:gpu", "calculated", d, "gpu", Decimal("0.709257"))
    led.upsert_cost("calc:a:disk", "calculated", d, "disk", Decimal("0.241608"))
    led.upsert_cost("est:traffic", "estimate", d, "other", Decimal("0.336420"))
    led.put_facts(d, 16, Decimal("3.900572739707099038888888886"))
    led.add_adjustment("adj:2026-11-13:gpu:1", date(2026, 11, 13), "gpu", Decimal("-0.2614724598381255275699274100"), "reconciliation with Vast charges")
    for cat, amt in (("gpu", "0.349"), ("disk", "0.276"), ("other", "0.116")):
        led.upsert_cost(f"conf:x:{cat}", "confirmed", date(2026, 11, 13), cat, Decimal(amt))
    return led


def real_daily(first_day=date(2026, 11, 14)):
    led = real_ledger()
    calc_utc13 = {"gpu": Decimal("0.6105"), "disk": Decimal("0.2550"), "other": Decimal(0)}
    recon = reports.compare(led, date(2026, 11, 13), calc_utc13)
    return reports.build_daily(led, date(2026, 11, 14), RATE, recon, now=NOW, first_day=first_day)


def cells(text, label):
    for blk in pre_blocks(text):
        for ln in blk.split("\n"):
            if ln.startswith(label + " "):
                return ln[len(label):].split()
    raise AssertionError(label)


# ------------------------------------------------------------------ money formatting
@pytest.mark.parametrize("v,s", [(Decimal("0.21"), "0.21"), (Decimal("-0.26"), "-0.26"), (Decimal("1234.5"), "1 234.50"),
                                 (Decimal("-1234567.891"), "-1 234 567.89"), (Decimal("-0.004"), "0.00"), (0, "0.00"),
                                 (0.1, "0.10"), ("3.14159", "3.14")])
def test_num_positive_negative_and_never_minus_zero(v, s):
    assert tf.num(v) == s


def test_usd_rub_pair_signs():
    assert tf.usd(Decimal("-0.26")) == "-$0.26" and tf.usd(Decimal("0.69")) == "$0.69"
    assert tf.rub(Decimal("-21.48038")) == "-21.48 ₽"
    assert tf.pair(Decimal("-0.26"), Decimal("-21.48")) == "-$0.26 / -21.48 ₽"


def test_rounding_half_up_once_at_display():
    assert tf.num(Decimal("0.125")) == "0.13" and tf.num(Decimal("-0.125")) == "-0.13" and tf.num(Decimal("2.675")) == "2.68"
    assert tf.num(2.675) == "2.68"                                  # float goes through str(): no binary 2.67499…
    # the owner's case: $0.69 x 82.1516 = 56.68, but the exact $0.6893925… x 82.1516 = 56.63 -> rubles come from the exact amount
    exact = Decimal("0.950865") + Decimal("-0.2614724598381255275699274100")
    assert tf.num(tf.to_rub(exact, RATE.value)) == "56.63"
    assert tf.num(tf.to_rub(Decimal(tf.num(exact)), RATE.value)) == "56.68"
    assert tf.rounding_mismatch([Decimal("58.2665973612"), Decimal("19.8484837728"), Decimal("-21.4803809316")], Decimal("56.6347002024"))


def test_missing_values_show_a_dash_not_zero():
    assert tf.num(None) == tf.DASH and tf.usd(None) == tf.DASH and tf.rub("") == tf.DASH and tf.hours(None) == tf.DASH
    assert tf.num("abc") == tf.DASH and tf.num(Decimal("NaN")) == tf.DASH and tf.to_rub(Decimal(1), None) is None


# ------------------------------------------------------------------ tables
def test_table_right_aligns_numbers_and_left_aligns_labels():
    t = tf.table([tf.Row("Сегодня", "0.21", "17.22"), tf.Row("Вчера", "0.95", "78.12"),
                  tf.Row("С начала месяца", "2.00", "164.12"), tf.Row("Итого", "-12 345.67", "-1 014 263.11", total=True)], ("USD", "RUB"))
    lines = tf.plain(t).split("\n")
    assert lines[0].endswith("USD     RUB") or lines[0].rstrip().endswith("RUB")
    nums = [ln for ln in lines if ln[:1] not in (" ", "-")]
    assert len({len(ln) for ln in nums}) == 1                        # same right edge
    assert lines[-1].startswith("ИТОГО") and set(lines[-2]) == {"-"}   # total: capitals + rule
    assert all(len(ln) <= tf.WIDTH for ln in lines)


def test_long_gpu_names_are_shortened_never_widen_the_table():
    long = "NVIDIA GeForce RTX 4090 D Founders Edition 24GB <OC> & co"
    t = tf.table([tf.Row(long, "12.34"), tf.Row("RTX A4000", "1.06"), tf.Row("Итого", "13.40", total=True)], ("ч",))
    lines = tf.plain(t).split("\n")
    assert all(len(ln) <= tf.WIDTH for ln in lines) and "…" in tf.plain(t)
    assert "&lt;OC&gt;" in t or "&lt;" in t or "<OC>" not in t     # escaped or cut, never raw markup
    assert_tg_html(t)


def test_summary_with_long_gpu_name_and_html_chars():
    st = FleetStore()
    t0 = msk(2026, 11, 15, 10)
    st.add(t0, id=7, label="sonya-pool-x", state="running", gpu_name="NVIDIA H100 SXM5 80GB HBM3 <beta> & test", dph_base=Decimal("2.5"),
           storage_gb_month=Decimal("0.36"), disk_gb=Decimal("80"))
    st.event(t0, 7, "created"); st.event(t0, 7, "running")
    txt = build_summary(st, FixedRates("82.1516"), msk(2026, 11, 15, 12), Decimal("-3.5"), Ledger())
    assert_tg_html(txt)
    assert_narrow_and_aligned(txt)
    assert "<beta>" not in txt and "-$3.50 / -287.53 ₽" in txt      # negative balance formatted, not hidden


# ------------------------------------------------------------------ daily report on the real 14.11 numbers
def test_real_daily_structure_matches_exact_sums():
    rep = real_daily()
    t = rep["text"]
    assert cells(t, "GPU") == ["0.71", "58.27"] and cells(t, "Диски") == ["0.24", "19.85"]
    assert cells(t, "Расход за сутки") == ["0.95", "78.12"]           # gross, same as /vast «Вчера»
    assert cells(t, "Корректировки") == ["-0.26", "-21.48"]
    assert cells(t, "ИТОГО") == ["0.69", "56.63"]                     # from exact 0.6893925… x 82.1516, not 0.69 x rate
    assert "округление — только при выводе" in t                       # 78.12 - 21.48 = 56.64 ≠ 56.63: explained, not faked
    p = rep["payload"]
    assert p["gross_usd"] == "0.950865" and p["total_usd"] == "0.689393" and p["total_rub"] == "56.634700"


def test_real_daily_efficiency_and_rate_footer():
    t = real_daily()["text"]
    assert cells(t, "Обработано видео") == ["16"] and cells(t, "GPU-время, ч") == ["3.90"]
    assert cells(t, "Себестоимость, USD") == ["0.04"] and cells(t, "Себестоимость, RUB") == ["3.54"]
    assert "Курс ЦБ РФ на 14.11.2026: 82.1516 ₽/$" in t and "Данные на 15.11.2026 00:05 МСК" in t


def test_real_daily_reconciliation_flags_mismatch_and_keeps_sources_apart():
    t = real_daily()["text"]
    assert cells(t, "Vast, UTC 13.11") == ["0.74", "60.87"]           # confirmed, its own row and period
    assert cells(t, "Журнал, UTC 13.11")[0] == "0.87"
    assert cells(t, "Корр. за 13.11") == ["-0.26", "-21.48"]
    assert cells(t, "Трафик (оценка)")[0] == "0.34"
    assert "Статус: <b>Требует сверки</b>" in t
    assert "Прочее: Vast $0.12, журнал $0.00" in t                    # traffic charged by Vast, not in our calculation
    assert "учёт ведётся с 14.11" in t                                 # adjustment for a day that was never reported
    assert cells(t, "ИТОГО") == ["0.69", "56.63"]                      # confirmed/estimate never enter ИТОГО


def test_daily_reconciliation_ok_status():
    led = Ledger()
    d = date(2026, 11, 14)
    led.upsert_cost("c", "calculated", d, "gpu", Decimal("1"))
    led.put_facts(d, 1, Decimal("1"))
    led.upsert_cost("k", "confirmed", date(2026, 11, 13), "gpu", Decimal("0.50"))
    rc = reports.compare(led, date(2026, 11, 13), {"gpu": Decimal("0.49"), "disk": Decimal(0), "other": Decimal(0)})
    t = reports.build_daily(led, d, RATE, rc, now=NOW)["text"]
    assert rc["status"] == "ok" and "Статус: сходится" in t and "Требует сверки" not in t


def test_daily_without_confirmed_data_and_without_rate():
    led = Ledger()
    d = date(2026, 11, 14)
    t = reports.build_daily(led, d, None, reports.compare(led, date(2026, 11, 13), None), now=NOW)["text"]
    assert_tg_html(t)
    assert "Нет данных журнала" in t and "Подтверждённых списаний Vast за UTC-сутки 13.11.2026 пока нет" in t
    assert "Курс ЦБ РФ: <b>нет данных</b>" in t and cells(t, "GPU") == ["0.00", "—"]


def test_compare_is_read_only():
    led = real_ledger()
    before = led.c.execute("SELECT COUNT(*) FROM adjustments").fetchone()[0]
    reports.compare(led, date(2026, 11, 13), {"gpu": Decimal(5), "disk": Decimal(0), "other": Decimal(0)})
    assert led.c.execute("SELECT COUNT(*) FROM adjustments").fetchone()[0] == before


def test_reports_are_valid_telegram_html_and_fit_a_phone():
    t = real_daily()["text"]
    assert_tg_html(t)
    assert_narrow_and_aligned(t)
    assert t.count("<pre>") == 7 and t.count("⚠") == 0               # 4 Vast + 3 Gemini/total; minimum emoji


def test_summary_explains_gross_vs_report_total():
    """Owner's case: /vast said «Вчера $0.95», the daily report said $0.69. Both are right, now the summary says why."""
    from tests.acct.helpers import fixture_store
    led = real_ledger()
    led.mark_adjustments_applied(["adj:2026-11-13:gpu:1"], date(2026, 11, 14))
    txt = build_summary(fixture_store(), FixedRates("82.1516"), msk(2026, 11, 15, 12), Decimal("5.83"), led)
    assert "корректировка -$0.26 за UTC 13.11" in txt and "Корректировки с начала месяца (в таблицу не входят)" in txt
    assert_tg_html(txt)


def test_alert_is_html_and_compact():
    from tests.acct.helpers import fixture_store
    led = Ledger()
    key = check_alert(led, fixture_store(), FixedRates("80"), msk(2026, 11, 6, 13), Decimal("1"))
    row = led.c.execute("SELECT text, payload FROM report_outbox WHERE report_key=?", (key,)).fetchone()
    assert_tg_html(row["text"]) or True
    assert json.loads(row["payload"])["fmt"] == "html" and row["text"].count("⚠️") == 1


# ------------------------------------------------------------------ length
def test_assemble_drops_optional_blocks_first_and_never_breaks_tags():
    req = "<b>HEAD</b>\n" + tf.table([tf.Row("x", "1.00")])
    opt = "<b>OPT</b>\n<pre>" + "y" * 3000 + "</pre>"
    out = tf.assemble([(req, False), (opt, True), ("<i>tail</i>", False)], limit=1000)
    assert "OPT" not in out and "Не поместилось" in out and "tail" in out and len(out) <= 1000
    assert_tg_html(out)


def test_assemble_hard_cut_closes_tags():
    huge = "<b>T</b>\n<pre>" + "\n".join("строка %d   1.00" % i for i in range(800)) + "</pre>"
    out = tf.assemble([(huge, False)], limit=1500)
    assert len(out) <= 1500 and out.endswith("<i>Сообщение сокращено.</i>")
    assert_tg_html(out)


def test_many_adjustments_still_fit():
    led = real_ledger()
    for i in range(300):
        led.add_adjustment(f"adj:x:{i}", date(2026, 11, 1 + i % 13), "gpu", Decimal("-0.01") * i, "reconciliation with Vast charges")
    t = reports.build_daily(led, date(2026, 11, 14), RATE, None, now=NOW)["text"]
    assert len(t) <= tf.LIMIT and "<b>СТРУКТУРА РАСХОДОВ</b>" in t and "Курс ЦБ" in t
    assert_tg_html(t)


# ------------------------------------------------------------------ read-only re-render + API
def test_render_daily_is_read_only_and_reuses_applied_adjustments():
    from tests.acct.test_scheduler_and_delivery import service
    svc, led = service()
    led.add_adjustment("adj:2026-11-05:gpu:1", date(2026, 11, 5), "gpu", Decimal("0.12"), "reconciliation with Vast charges")
    svc.run_due(msk(2026, 11, 7, 0, 6))
    sent = led.c.execute("SELECT text FROM report_outbox").fetchone()[0]
    snap = [tuple(r) for r in led.c.execute("SELECT * FROM adjustments")] + [tuple(r) for r in led.c.execute("SELECT report_key, status FROM report_outbox")]
    again = svc.render_daily(date(2026, 11, 6), msk(2026, 11, 7, 0, 6))
    assert snap == [tuple(r) for r in led.c.execute("SELECT * FROM adjustments")] + [tuple(r) for r in led.c.execute("SELECT report_key, status FROM report_outbox")]
    assert cells(again, "ИТОГО") == cells(sent, "ИТОГО") == ["0.52", "41.60"]
    assert svc.render_daily(date(2026, 11, 3), msk(2026, 11, 7, 1)) is None


def test_daily_endpoint_and_day_parsing():
    import time
    from accounting.internal_api import AccountingApi, parse_day
    from tests.acct.test_scheduler_and_delivery import SECRET, req, service
    now = msk(2026, 11, 7, 0, 6)
    assert parse_day("", now) == date(2026, 11, 6) and parse_day("05.11", now) == date(2026, 11, 5)
    assert parse_day("2026-11-04", now) == date(2026, 11, 4) and parse_day("4.11.2026", now) == date(2026, 11, 4)
    for bad in ("07.11", "2027-01-01", "вчера", "1/2"):
        with pytest.raises(ValueError):
            parse_day(bad, now)
    svc, led = service()
    svc.run_due(now)
    api = AccountingApi(svc, SECRET, clock=lambda: now)
    st, data = req(api, "/api/internal/accounting/vast/daily", b'{"day": "06.11"}', nonce="a1" * 16)
    assert st == 200 and data["parse_mode"] == "HTML" and "расходы за 06.11.2026" in data["text"]
    assert req(api, "/api/internal/accounting/vast/daily", b'{"day": "01.11"}', nonce="a2" * 16)[0] == 404
    assert req(api, "/api/internal/accounting/vast/daily", b'{"day": "xx"}', nonce="a3" * 16)[0] == 400


def test_claim_marks_html_reports_and_keeps_old_plain_ones_plain():
    from tests.acct.test_scheduler_and_delivery import service
    svc, led = service()
    led.c.execute("INSERT INTO report_outbox(report_key, kind, period, text, payload) VALUES ('vast-daily:2026-11-01','daily','2026-11-01','old <plain>','{}')")
    svc.run_due(msk(2026, 11, 7, 0, 6))
    got = svc.claim(msk(2026, 11, 7, 0, 7), limit=5)
    assert [(r["report_key"], r["parse_mode"]) for r in got] == [("vast-daily:2026-11-01", None), ("vast-daily:2026-11-06", "HTML")]
