"""YouTube relay (scripts/yt_relay.py) + balance/YouTube alerts (accounting/balance.py, accounting/yt_watch.py)."""
import json
import sqlite3
from decimal import Decimal
from pathlib import Path

import pytest

from scripts import yt_relay


def hb(prio, fresh=True):
    return {"priority": prio, "fresh": fresh, "ts": 0}


def test_fast_relay_goes_first_home_relay_waits_its_turn():
    req = {"created": 1000.0}
    hbs = {"bot": hb(0), "mac": hb(1)}
    assert yt_relay.should_take("bot", 0, req, {}, hbs, 1001)
    assert not yt_relay.should_take("mac", 1, req, {}, hbs, 1010)               # bot has CLAIM_GRACE to claim
    assert yt_relay.should_take("mac", 1, req, {}, hbs, 1000 + yt_relay.CLAIM_GRACE + 1)   # it did not: take over


def test_home_relay_takes_over_after_a_failure_or_silence_but_not_while_working():
    req = {"created": 1000.0}
    hbs = {"bot": hb(0), "mac": hb(1)}
    assert yt_relay.should_take("mac", 1, req, {"bot": {"state": "failed", "ts": 1005}}, hbs, 1006)
    assert not yt_relay.should_take("mac", 1, req, {"bot": {"state": "downloading", "ts": 1100}}, hbs, 1120)
    assert yt_relay.should_take("mac", 1, req, {"bot": {"state": "downloading", "ts": 1100}}, hbs,
                                1100 + yt_relay.STATUS_STALE + 1)
    assert yt_relay.should_take("mac", 1, req, {}, {"bot": hb(0, fresh=False), "mac": hb(1)}, 1001)   # bot is down
    assert not yt_relay.should_take("mac", 1, req, {"bot": {"state": "done", "ts": 1}}, hbs, 2000)
    assert not yt_relay.should_take("bot", 0, req, {"bot": {"state": "failed", "ts": 1}}, hbs, 2000)  # never twice


class FakeS3:
    def __init__(self):
        self.o = {}


@pytest.fixture
def s3(monkeypatch):
    f = FakeS3()
    monkeypatch.setattr(yt_relay, "put_json", lambda k, d: f.o.__setitem__(k, json.loads(json.dumps(d))))
    monkeypatch.setattr(yt_relay, "get_json", lambda k: f.o.get(k))
    monkeypatch.setattr(yt_relay, "list_keys", lambda p: sorted(k for k in f.o if k.startswith(p)))

    class C:
        def delete_object(self, Bucket, Key):
            f.o.pop(Key, None)
    monkeypatch.setattr(yt_relay, "_s3", lambda: (C(), "b"))
    return f


def test_worker_waits_for_the_relay_then_reads_the_file(s3, monkeypatch, tmp_path):
    import time as _t
    pre = yt_relay.prefix()
    calls = {"n": 0}

    def fake_sleep(_):
        calls["n"] += 1
        now = _t.time()
        s3.o[pre + "hb/bot.json"] = {"ts": now, "priority": 0}
        if calls["n"] == 1:
            assert pre + "req/j1.json" in s3.o                 # the request is out
            s3.o[pre + "st/j1/bot.json"] = {"state": "downloading", "ts": now}
        elif calls["n"] == 2:
            s3.o[pre + "st/j1/bot.json"] = {"state": "done", "ts": now, "duration": 61.0}
    monkeypatch.setattr(yt_relay.time, "sleep", fake_sleep)
    got_key = {}
    import scripts.prod_s3_storage as st
    monkeypatch.setattr(st, "download_file", lambda k, p: (got_key.update(k=k), Path(p).write_bytes(b"x" * 10)))
    ticks = []
    r = yt_relay.fetch_via_relay("j1", {"url": "https://youtu.be/x"}, "virality", "users/u/in.mp4", tmp_path,
                                 tick=ticks.append, poll=0)
    assert r["relay"] == "bot" and r["duration"] == 61.0 and got_key["k"] == "users/u/in.mp4"
    assert pre + "req/j1.json" not in s3.o                     # request removed


def test_worker_gets_the_users_reason_from_a_relay(s3, monkeypatch, tmp_path):
    from scripts.source_fetch import SourceFetchError
    import time as _t
    pre = yt_relay.prefix()

    def fake_sleep(_):
        s3.o[pre + "hb/bot.json"] = {"ts": _t.time(), "priority": 0}
        s3.o[pre + "st/j2/bot.json"] = {"state": "failed", "ts": _t.time(), "code": "SOURCE_PRIVATE", "message": "priv"}
    monkeypatch.setattr(yt_relay.time, "sleep", fake_sleep)
    with pytest.raises(SourceFetchError) as e:
        yt_relay.fetch_via_relay("j2", {"url": "u"}, "virality", "k", tmp_path, poll=0)
    assert e.value.code == "SOURCE_PRIVATE"


def test_no_relay_alive_is_reported_not_waited_forever(s3, monkeypatch, tmp_path):
    clock = {"t": 1000.0}
    monkeypatch.setattr(yt_relay.time, "time", lambda: clock["t"])
    monkeypatch.setattr(yt_relay.time, "sleep", lambda _: clock.__setitem__("t", clock["t"] + 10))
    with pytest.raises(yt_relay.RelayUnavailable):
        yt_relay.fetch_via_relay("j3", {"url": "u"}, "virality", "k", tmp_path, wait_alive=60, poll=0)
    assert clock["t"] < 1200


def test_all_relays_failed_bot_check_is_unavailable(s3, monkeypatch, tmp_path):
    import time as _t
    pre = yt_relay.prefix()

    def fake_sleep(_):
        for n, p in (("bot", 0), ("mac", 1)):
            s3.o[pre + f"hb/{n}.json"] = {"ts": _t.time(), "priority": p}
            s3.o[pre + f"st/j4/{n}.json"] = {"state": "failed", "ts": _t.time(), "code": "SOURCE_BOT_CHECK"}
    monkeypatch.setattr(yt_relay.time, "sleep", fake_sleep)
    with pytest.raises(yt_relay.RelayUnavailable):
        yt_relay.fetch_via_relay("j4", {"url": "u"}, "virality", "k", tmp_path, poll=0)


# ── url_ingest YouTube options ────────────────────────────────────────────────
def test_youtube_opts_from_env(monkeypatch):
    from scripts import url_ingest
    monkeypatch.setenv("SONYA_YT_DASHY", "1")
    monkeypatch.setenv("SONYA_YT_FRAGMENTS", "16")
    monkeypatch.setenv("SONYA_YT_POT_URL", "http://127.0.0.1:4416")
    o = url_ingest._youtube_opts("youtube")
    assert o["concurrent_fragment_downloads"] == 16
    assert o["extractor_args"]["youtube"]["formats"] == ["dashy"]
    assert o["extractor_args"]["youtubepot-bgutilhttp"]["base_url"] == ["http://127.0.0.1:4416"]
    assert url_ingest._youtube_opts("rutube") == {}


def test_cookies_only_on_a_bot_check_retry_and_via_a_private_copy(monkeypatch, tmp_path):
    from scripts import url_ingest
    master = tmp_path / "cookies.txt"
    master.write_text("# Netscape HTTP Cookie File\n")
    monkeypatch.setenv("SONYA_YT_COOKIES", str(master))
    seen = []

    class YDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def extract_info(self, url, download=True):
            ck = self.opts.get("cookiefile")
            seen.append(ck)
            if not ck:
                raise Exception("ERROR: [youtube] x: Sign in to confirm you’re not a bot.")
            assert ck != str(master) and oct(Path(ck).stat().st_mode & 0o777) == "0o600"
            out = Path(self.opts["outtmpl"].replace("%(ext)s", "mp4"))
            out.write_bytes(b"v")
            return {"requested_downloads": [{"filepath": str(out)}]}
    import types
    monkeypatch.setattr(url_ingest, "yt_dlp", types.SimpleNamespace(YoutubeDL=YDL))
    monkeypatch.setattr(url_ingest, "YT_DLP_AVAILABLE", True)
    monkeypatch.setattr(url_ingest, "assert_safe_url", lambda u: None)
    path, _ = url_ingest.download_video("https://www.youtube.com/watch?v=abcdefghijk", "youtube")
    assert seen[0] is None and seen[1] and not Path(seen[1]).exists()      # temp copy removed
    url_ingest.cleanup(path)


# ── alerts ────────────────────────────────────────────────────────────────────
@pytest.fixture
def ledger():
    from accounting.ledger import Ledger
    return Ledger(":memory:")


def test_balance_alerts_low_then_critical_not_repeated(ledger):
    from accounting.balance import check_balance_alert
    now = 1_800_000_000.0
    assert check_balance_alert(ledger, Decimal("5"), Decimal("0.3"), now) is None
    k = check_balance_alert(ledger, Decimal("2.4"), Decimal("0.3"), now)
    assert k and k.startswith("vast-balance:low")
    assert check_balance_alert(ledger, Decimal("2.3"), Decimal("0.3"), now + 60) is None     # once per 12 h
    k2 = check_balance_alert(ledger, Decimal("0.6"), Decimal("0.3"), now + 120)
    assert k2 and "critical" in k2
    text = ledger.c.execute("SELECT text FROM report_outbox WHERE report_key=?", (k2,)).fetchone()[0]
    assert "СРОЧНО" in text and "меньше чем на час" not in text and "2 ч" in text
    assert check_balance_alert(ledger, Decimal("0.5"), Decimal("0.3"), now + 600) is None
    assert check_balance_alert(ledger, Decimal("0.4"), Decimal("0.3"), now + 3 * 3600 + 10)   # again after 3 h


def test_runway_text():
    from accounting.balance import runway_text
    assert runway_text(Decimal("6"), Decimal("0.25")) == "хватит примерно на 24 ч"
    assert runway_text(Decimal("30"), Decimal("0.25")) == "хватит примерно на 5 дн"
    assert "исчерпан" in runway_text(Decimal("-0.3"), Decimal("0.25"))


def test_youtube_watch_alerts(ledger):
    from accounting import yt_watch
    now = 1_800_000_000.0
    ok = {"ts": now, "canary": {"ok": True}, "cookies": "ok"}
    assert yt_watch.check(ledger, {"bot": ok, "mac": ok}, now) == []
    keys = yt_watch.check(ledger, {"bot": {**ok, "canary": {"ok": False, "bot_check": True}},
                                   "mac": {**ok, "ts": now - 3600, "cookies": "expired"}}, now)
    assert any(k.startswith("yt-all-down") for k in keys)
    assert any(k.startswith("yt-relay-down:mac") for k in keys) and any(k.startswith("yt-canary:bot") for k in keys)
    assert yt_watch.check(ledger, {"bot": {**ok, "canary": {"ok": False}}, "mac": {**ok, "ts": now - 3600}}, now + 60) == []
    assert any(k.startswith("yt-cookies") for k in yt_watch.check(ledger, {"bot": {**ok, "cookies": "expired"}}, now))


def test_restarted_relay_takes_its_own_abandoned_job_again():
    req = {"created": 1000.0}
    hbs = {"bot": hb(0)}
    assert not yt_relay.should_take("bot", 0, req, {"bot": {"state": "downloading", "ts": 1100}}, hbs, 1200)
    assert yt_relay.should_take("bot", 0, req, {"bot": {"state": "downloading", "ts": 1100}}, hbs, 1100 + yt_relay.STATUS_STALE + 1)
