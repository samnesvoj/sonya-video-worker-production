from datetime import date
from decimal import Decimal

from accounting.internal_api import AccountingApi
from accounting.summary import build_summary, check_alert
from tests.acct.helpers import FixedRates, fixture_store, msk
from tests.acct.test_scheduler_and_delivery import SECRET, service
from accounting.ledger import Ledger


def test_summary_text_and_alert_once():
    st, led = fixture_store(), Ledger()
    now = msk(2026, 11, 6, 13)
    txt = build_summary(st, FixedRates("80"), now, Decimal("12.5"))
    assert "Сегодня (06.11" in txt and "Баланс Vast: $12.50" in txt and "RTX 3090" in txt and "1.50 ч" in txt
    assert "не удалось" in build_summary(st, FixedRates("80"), now, None)
    assert check_alert(led, st, FixedRates("80"), now, Decimal("500")) is None          # ~$0.3 -> far below 500 ₽
    key = check_alert(led, st, FixedRates("80"), now, Decimal("1"))                      # threshold 1 ₽ -> fires
    assert key == "vast-alert:2026-11-06"
    assert check_alert(led, st, FixedRates("80"), now, Decimal("1")) is None             # once per day
    assert led.c.execute("SELECT kind FROM report_outbox WHERE report_key=?", (key,)).fetchone()[0] == "alert"


def test_summary_endpoint_signed_only():
    import json, time
    from accounting.internal_api import sign
    svc = service()[0] if isinstance(service(), tuple) else service()
    api = AccountingApi(svc, SECRET, summary=lambda now: "hello")
    path = "/api/internal/accounting/vast/summary"
    assert api.handle("POST", path, {}, b"{}")[0] == 401
    ts, nonce = str(int(time.time())), "ab" * 16
    h = {"X-Accounting-Timestamp": ts, "X-Accounting-Nonce": nonce, "X-Accounting-Signature": sign(SECRET.encode(), ts, nonce, "POST", path, b"{}")}
    assert api.handle("POST", path, h, b"{}") == (200, {"text": "hello"})


def test_real_charges_row_shape():
    from accounting.vast_billing import import_charges
    row = {"start": 1794614400, "end": 1794614400, "type": "instance", "source": "instance-1", "amount": 0.063,
           "items": [{"type": "gpu", "amount": 0.025}, {"type": "disk", "amount": 0.007}, {"type": "bwd", "amount": 0.03}, {"type": "bwu", "amount": 0.001}]}
    led = Ledger()
    assert import_charges(led, [row], 1.0) == 4 and import_charges(led, [row], 2.0) == 4      # idempotent upsert
    s = led.sum_costs("confirmed", date(2026, 11, 13))
    assert (s["gpu"], s["disk"], s["other"]) == (Decimal("0.025"), Decimal("0.007"), Decimal("0.031"))
