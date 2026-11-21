"""Self-check (scripts/ops/health.py): one alert per problem + «починилось», no streams."""
import json
import time

from scripts.ops import health as H

T0 = 1795200000.0


def _all_ok(now, **over):
    cs = [H.check_api({"status": "ok"}), H.check_public(200), H.check_postgres(12.0), H.check_fleet(True, 30),
          H.check_balance(25.0, None), H.check_create(0), H.check_queue({"queued": 0, "active": 0}),
          H.check_host("bot_host", {"ts": now, "services": {"sonya-accounting-bot": "active"}}, now, 1200, ("sonya-accounting-bot",)),
          H.check_host("mac_host", {"ts": now}, now, 2700),
          *H.check_relays({"bot": {"ts": now, "canary": {"ok": True}}, "mac": {"ts": now, "canary": {"ok": True}}}, now),
          H.check_cookies({"total": 2, "live": 2, "ts": now}, now), H.check_tg(True, 0), H.check_accountant(30, 0, True),
          H.check_disk("disk_vps", {"total_gb": 38, "free_gb": 24}), H.check_disk("disk_bot", {"total_gb": 38, "free_gb": 30}),
          H.check_disk("disk_mac", {"total_gb": 228, "free_gb": 19}, min_free_gb=8), H.check_ssl(34)]
    by = {c.name: c for c in cs}
    by.update(over)
    return list(by.values())


def _run(state, checks, now):
    return H.update(state, checks, now)


def test_all_ok_no_alerts_and_status_ok():
    st, al = _run({}, _all_ok(T0), T0)
    assert al == [] and st["incidents"] == {}
    txt = H.status_text(H.snapshot(_all_ok(T0), st, T0), T0 + 60)
    assert "Всё работает" in txt and "баланс Vast $25.00" in txt


def test_zero_balance_one_alert_for_a_day_then_one_recovery():
    st, sent = {}, []
    for i in range(96):                                    # 24 h of 15-min runs, balance 0, a job waiting, creates refused
        now = T0 + i * 900
        cs = _all_ok(now, vast_balance=H.check_balance(0.0, None, pending=1),
                     queue=H.check_queue({"queued": 1, "oldest_queued_s": 3600 + i * 900, "active": 0}),
                     vast_create=H.check_create(5, ["HTTP 402"]))
        st, al = _run(st, cs, now)
        sent += al
    assert [a.key.split(":")[1] for a in sent] == ["vast_balance"]
    assert "деньги кончились" in sent[0].lines[0] and "напоминать больше не буду" in sent[0].lines[1]
    st, al = _run(st, _all_ok(T0 + 97 * 900), T0 + 97 * 900)
    assert [a.key.split(":")[0] for a in al] == ["health-ok"] and "Починилось" in al[0].lines[0]
    for j in (98, 99, 100):                                  # create errors linger in the 1-h window after the top-up
        st, al = _run(st, _all_ok(T0 + j * 900, vast_create=H.check_create(4)) if j == 98 else _all_ok(T0 + j * 900), T0 + j * 900)
        assert al == []
    assert st["incidents"] == {}                             # queue / create closed silently (never announced)


def test_child_announced_if_it_outlives_its_cause():
    st, _ = _run({}, _all_ok(T0, vast_balance=H.check_balance(0.0, None),
                              queue=H.check_queue({"queued": 2, "oldest_queued_s": 7200})), T0)
    st, al = _run(st, _all_ok(T0 + 900, queue=H.check_queue({"queued": 2, "oldest_queued_s": 8100})), T0 + 900)
    assert [a.key.split(":")[1] for a in al] == ["vast_balance"]      # healed; the queue gets 90 min to drain
    st, al = _run(st, _all_ok(T0 + 7200, queue=H.check_queue({"queued": 2, "oldest_queued_s": 14000})), T0 + 7200)
    assert [a.key.split(":")[1] for a in al] == ["queue"]    # outlived its cause -> now it is news


def test_confirm_two_runs_for_flappy_and_flap_is_silent():
    bad = H.check_api(None, "ConnectionRefusedError")
    st, al = _run({}, _all_ok(T0, api=bad), T0)
    assert al == []
    st, al = _run(st, _all_ok(T0 + 900), T0 + 900)          # recovered before confirmation: nothing at all
    assert al == [] and st["incidents"] == {}
    st, _ = _run(st, _all_ok(T0 + 1800, api=bad), T0 + 1800)
    st, al = _run(st, _all_ok(T0 + 2700, api=bad), T0 + 2700)
    assert [a.key.split(":")[1] for a in al] == ["api"]
    # site_public fails too because of api: suppressed
    st, al = _run(st, _all_ok(T0 + 3600, api=bad, site_public=H.check_public(502)), T0 + 3600)
    st, al = _run(st, _all_ok(T0 + 4500, api=bad, site_public=H.check_public(502)), T0 + 4500)
    assert al == []


def test_one_escalation_never_repeat():
    st, al1 = _run({}, _all_ok(T0, vast_balance=H.check_balance(2.0, 0.5)), T0)
    st, al2 = _run(st, _all_ok(T0 + 900, vast_balance=H.check_balance(0.5, 0.5)), T0 + 900)
    st, al3 = _run(st, _all_ok(T0 + 1800, vast_balance=H.check_balance(0.0, 0.5)), T0 + 1800)
    st, al4 = _run(st, _all_ok(T0 + 2700, vast_balance=H.check_balance(2.0, 0.5)), T0 + 2700)   # back to warn: silent
    assert len(al1) == 1 and len(al2) == 1 and al2[0].key.endswith(":crit") and al3 == [] and al4 == []


def test_unknown_keeps_state():
    st, _ = _run({}, _all_ok(T0, vast_balance=H.check_balance(0.0, None)), T0)
    st, al = _run(st, _all_ok(T0 + 900, vast_balance=H.check_balance(None, None)), T0 + 900)
    assert al == [] and "vast_balance" in st["incidents"]


def test_host_down_suppresses_relay_and_accountant():
    now = T0
    down = H.check_host("bot_host", {"ts": now - 4000}, now, 1200)
    relays = H.check_relays({"bot": {"ts": now - 4000}, "mac": {"ts": now, "canary": {"ok": True}}}, now)
    over = {c.name: c for c in relays}
    st, al = {}, []
    for i in range(3):
        st, a = _run(st, _all_ok(now + i * 900, bot_host=down, accountant=H.check_accountant(4000, 0, None), **over), now + i * 900)
        al += a
    assert [a.key.split(":")[1] for a in al] == ["bot_host"]


def test_cookies_zero_is_one_alert():
    st, al = {}, []
    for i in range(40):
        st, a = _run(st, _all_ok(T0 + i * 900, yt_cookies=H.check_cookies({"total": 2, "live": 0, "ts": T0}, T0)), T0 + i * 900)
        al += a
    assert len(al) == 1 and "cookies" in al[0].lines[0]


def test_status_text_lists_problems():
    cs = _all_ok(T0, vast_balance=H.check_balance(0.0, None), yt_cookies=H.check_cookies({"total": 2, "live": 0}, T0))
    st, _ = _run({}, cs, T0)
    txt = H.status_text(json.loads(json.dumps(H.snapshot(cs, st, T0))), T0 + 600)
    assert "Сломано: 2" in txt and "Vast: деньги кончились" in txt and "10 мин назад" in txt
    assert H.status_text(None) .startswith("<b>Самопроверка")
    assert "давно не запускалась" in H.status_text(H.snapshot(cs, st, T0), T0 + 4000)


def test_disk_and_ssl_rules():
    assert H.check_disk("disk_mac", {"total_gb": 228, "free_gb": 19}, min_free_gb=8).level == H.OK
    assert H.check_disk("disk_vps", {"total_gb": 38, "free_gb": 1.5}).level == H.CRIT
    assert H.check_ssl(10).level == H.WARN and H.check_ssl(2).level == H.CRIT and H.check_ssl(None, "x").level == H.UNKNOWN


def test_queue_rules():
    assert H.check_queue({"queued": 3, "oldest_queued_s": 600}).level == H.OK
    assert H.check_queue({"queued": 3, "oldest_queued_s": 3000}).level == H.WARN
    assert H.check_queue({"stale_active": 1, "oldest_stale_s": 4000}).level == H.CRIT
    assert H.check_queue(None).level == H.UNKNOWN


def test_internal_api_status_and_claim_hook():
    from accounting.internal_api import AccountingApi, sign

    class Svc:
        def claim(self, now, limit=1):
            return []
    seen = []
    api = AccountingApi(Svc(), "s3cret", clock=lambda: 1000.0, status=lambda now: "<b>ok</b>", on_claim=seen.append)

    def call(path, nonce):
        ts = "1000"
        sig = sign(b"s3cret", ts, nonce, "POST", path, b"")
        return api.handle("POST", path, {"X-Accounting-Timestamp": ts, "X-Accounting-Nonce": nonce, "X-Accounting-Signature": sig}, b"")
    assert call("/api/internal/accounting/vast/status", "a" * 16) == (200, {"text": "<b>ok</b>", "parse_mode": "HTML"})
    assert call("/api/internal/accounting/vast/claim", "b" * 16)[0] == 200 and seen == [1000.0]


def test_bot_status_command():
    from integration.accounting_bot import vast_reports as vr

    class Api:
        def __init__(self):
            self.sent = []

        def send(self, chat, text, reply_to=None, parse_mode=None):
            self.sent.append((chat, text, parse_mode))
            return 1
    api = Api()
    p = vr.VastReportPuller("https://x.invalid", "k", api, [1], "/tmp/none.json")
    p._post = lambda path, body: (200, {"text": "<b>✅ Всё работает</b>", "parse_mode": "HTML"}) if path.endswith("/status") else (404, {})
    assert p.command(7, "/status@sonya_accounting_bot") is True
    assert api.sent == [(7, "<b>✅ Всё работает</b>", "HTML")]
    p._post = lambda path, body: (_ for _ in ()).throw(vr.VastReportError("down"))
    p.command(7, "/status")
    assert "не ответил на /status" in api.sent[-1][1]


def test_health_owns_alerts_needs_fresh_file(tmp_path, monkeypatch):
    import dispatcher.main as m
    f = tmp_path / "health.json"
    monkeypatch.setattr(m, "HEALTH_JSON", f)
    assert m.health_owns_alerts({"SONYA_HEALTH_ALERTS": "on"}) is False
    f.write_text("{}")
    assert m.health_owns_alerts({"SONYA_HEALTH_ALERTS": "on"}) is True
    assert m.health_owns_alerts({"SONYA_HEALTH_ALERTS": "off"}) is False
    assert m.health_owns_alerts({"SONYA_HEALTH_ALERTS": "on"}, now=time.time() + 3000) is False
