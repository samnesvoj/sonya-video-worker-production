import json
from datetime import date
from decimal import Decimal

from accounting import usage
from accounting.internal_api import AccountingApi, sign
from accounting.ledger import Ledger
from accounting.outbox import ReportService, daily_due, monthly_due
from tests.acct.helpers import FixedRates, Jobs, fixture_store, msk

SECRET = "acct-secret-LEAKCHECK"


def service(now_jobs=None, first=date(2026, 11, 6)):
    led, st = Ledger(), fixture_store()
    svc = ReportService(led, st, FixedRates("80"), Jobs(now_jobs or {date(2026, 11, 6): 4}), lambda d, now: {"gpu": Decimal(0), "disk": Decimal(0), "other": Decimal(0)}, first)
    return svc, led


def test_daily_not_due_before_00_05_msk_and_due_after():
    assert daily_due(msk(2026, 11, 7, 0, 4, 59), date(2026, 11, 6)) == []
    assert daily_due(msk(2026, 11, 7, 0, 5, 0), date(2026, 11, 6)) == [date(2026, 11, 6)]


def test_monthly_due_at_00_10_on_the_first():
    assert monthly_due(msk(2026, 12, 1, 0, 9, 59), date(2026, 11, 1)) == []
    assert monthly_due(msk(2026, 12, 1, 0, 10), date(2026, 11, 1)) == [(2026, 11)]
    assert monthly_due(msk(2027, 1, 1, 0, 10), date(2026, 11, 1)) == [(2026, 11), (2026, 12)]


def test_run_due_builds_once_and_recovers_missed_days_in_order():
    svc, led = service(first=date(2026, 11, 4))
    built = svc.run_due(msk(2026, 11, 7, 0, 6))
    assert built == ["vast-daily:2026-11-04", "vast-daily:2026-11-05", "vast-daily:2026-11-06"]
    assert svc.run_due(msk(2026, 11, 7, 0, 7)) == []                       # no duplicates
    rows = led.c.execute("SELECT report_key FROM report_outbox ORDER BY period").fetchall()
    assert len(rows) == 3


def test_month_report_after_the_first():
    svc, led = service(first=date(2026, 11, 1))
    svc.run_due(msk(2026, 12, 1, 0, 11))
    row = led.c.execute("SELECT text, payload FROM report_outbox WHERE report_key='vast-monthly:2026-11'").fetchone()
    assert row["text"].startswith("SONYA — итог Vast.ai за ноябрь 2026")
    p = json.loads(row["payload"])
    assert Decimal(p["total_usd"]) == Decimal("0.40") and p["videos"] == 4 and p["saving_is_estimate"] is True and Decimal(p["saving_usd"]) > 0
    assert "ОЦЕНКА" in row["text"] and "Источники (не смешиваются)" in row["text"]


def test_outbox_claim_lease_ack_and_retry():
    svc, led = service()
    svc.run_due(msk(2026, 11, 7, 0, 6))
    now = msk(2026, 11, 7, 0, 6)
    (r,) = svc.claim(now)
    assert svc.claim(now + 10) == []                                       # leased
    svc.ack_failed(r["report_key"], "telegram down", now + 20)
    assert svc.claim(now + 30) == []                                       # backoff
    (r2,) = svc.claim(now + 200)
    assert r2["report_key"] == r["report_key"] and r2["attempt"] == 2
    assert svc.ack_sent(r["report_key"], "42", now + 210) is True
    assert svc.ack_sent(r["report_key"], "42", now + 211) is False          # idempotent
    assert svc.claim(now + 10_000) == []


def test_lease_expiry_redelivers_after_bot_crash():
    svc, led = service()
    svc.run_due(msk(2026, 11, 7, 0, 6))
    now = msk(2026, 11, 7, 0, 6)
    svc.claim(now)
    assert len(svc.claim(now + 301)) == 1


def req(api, path, body=b"{}", ts=None, nonce="ab" * 16, secret=SECRET, method="POST"):
    ts = str(int(ts if ts is not None else api.clock()))
    h = {"X-Accounting-Timestamp": ts, "X-Accounting-Nonce": nonce, "X-Accounting-Signature": sign(secret.encode(), ts, nonce, method, path, body)}
    return api.handle(method, path, h, body)


def test_api_rejects_bad_signature_replay_skew_and_unknown_paths():
    svc, led = service()
    now = msk(2026, 11, 7, 0, 6)
    api = AccountingApi(svc, SECRET, clock=lambda: now)
    assert req(api, "/api/internal/accounting/vast/claim", secret="wrong")[0] == 401
    assert req(api, "/api/internal/accounting/vast/claim", ts=now - 500)[0] == 401
    assert req(api, "/api/internal/accounting/vast/claim", nonce="cd" * 16)[0] == 200
    assert req(api, "/api/internal/accounting/vast/claim", nonce="cd" * 16)[0] == 401      # replay
    assert req(api, "/api/internal/accounting/vast/nope", nonce="ef" * 16)[0] == 404
    assert api.handle("POST", "/api/internal/accounting/vast/claim", {}, b"{}")[0] == 401
    assert req(api, "/api/internal/accounting/vast/../../etc/sent", nonce="12" * 16)[0] == 404
