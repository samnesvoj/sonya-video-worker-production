"""YouTube patience (scripts/yt_wait.py), relay round outcome, cookies pool checks (tools/yt_cookies.py), pool alerts,
«ждёт денег Vast» (dispatcher/capacity.py). Owner 2026-11-19: no «YouTube не отдал» in front of the users."""
import json
from decimal import Decimal
from pathlib import Path

import pytest

from scripts import yt_relay, yt_wait

NOW = 1_800_000_000.0


@pytest.fixture
def ledger():
    from accounting.ledger import Ledger
    return Ledger(":memory:")


def outbox(ledger):
    return {k: t for k, t in ledger.c.execute("SELECT report_key, text FROM report_outbox")}


# ── relay round outcome ────────────────────────────────────────────────────────
def test_outcome_done_user_retry_wait():
    hbs = {"bot": {"fresh": True, "priority": 0}, "mac": {"fresh": True, "priority": 1}}
    o = yt_relay.outcome
    assert o("j", {"mac": {"state": "done", "ts": NOW, "duration": 61}}, hbs, NOW - 50, NOW) == \
        {"kind": "done", "relay": "mac", "duration": 61}
    u = o("j", {"bot": {"state": "failed", "code": "SOURCE_PRIVATE", "message": "Видео приватное", "ts": NOW}}, hbs, NOW - 9, NOW)
    assert u["kind"] == "user" and u["code"] == "SOURCE_PRIVATE"
    both = {"bot": {"state": "failed", "code": "SOURCE_BOT_CHECK", "ts": NOW}, "mac": {"state": "failed", "code": "RELAY_ERROR", "ts": NOW}}
    assert o("j", both, hbs, NOW - 100, NOW)["kind"] == "retry"
    assert o("j", {"bot": {"state": "failed", "code": "SOURCE_BOT_CHECK", "ts": NOW}}, hbs, NOW - 30, NOW)["kind"] == "wait"  # mac not yet
    assert o("j", {}, {}, NOW - 200, NOW)["kind"] == "retry"                                   # no relay alive
    assert o("j", {}, {}, NOW - 60, NOW)["kind"] == "wait"
    assert o("j", {"mac": {"state": "downloading", "ts": NOW - 20}}, hbs, NOW - 3000, NOW)["kind"] == "wait"   # long download
    stale = {"mac": {"state": "downloading", "ts": NOW - 400}}
    assert o("j", stale, hbs, NOW - 3000, NOW)["kind"] == "retry"
    assert o("j", stale, {**hbs, "mac": {"fresh": True, "priority": 1, "active": ["j"]}}, NOW - 3000, NOW)["kind"] == "wait"


def test_submit_clears_the_previous_round(monkeypatch):
    o = {}
    pre = yt_relay.prefix()
    o[f"{pre}st/j1/bot.json"] = {"state": "failed"}
    o[f"{pre}st/j2/bot.json"] = {"state": "failed"}
    monkeypatch.setattr(yt_relay, "put_json", lambda k, d: o.__setitem__(k, d))
    monkeypatch.setattr(yt_relay, "list_keys", lambda p: sorted(k for k in o if k.startswith(p)))
    monkeypatch.setattr(yt_relay, "delete_keys", lambda ks: [o.pop(k) for k in ks])
    yt_relay.submit("j1", {"url": "https://youtu.be/x"}, "virality", "in/j1.mp4", NOW)
    assert f"{pre}st/j1/bot.json" not in o and f"{pre}st/j2/bot.json" in o
    req = o[f"{pre}req/j1.json"]
    assert req["created"] == NOW and req["s3_key"] == "in/j1.mp4" and req["waiter"]
    # a relay may take it again (its old terminal status is gone)
    assert yt_relay.should_take("bot", 0, req, {}, {"bot": {"priority": 0, "fresh": True}}, NOW + 1)


# ── the waiter's decision ──────────────────────────────────────────────────────
def test_schedule_grows_and_gives_up_after_two_hours():
    limit = 7200.0
    d = yt_wait.decide({}, None, NOW, limit)
    assert d["do"] == "init" and d["w"]["next_at"] == NOW + 120
    w, t, starts = d["w"], NOW, []
    for _ in range(40):
        t = max(t, float(w.get("next_at") or t))
        d = yt_wait.decide(w, None, t, limit)
        if d["do"] == "give_up":
            break
        assert d["do"] == "submit"
        starts.append(round((t - NOW) / 60))
        w = d["w"]
        t += 60                                                # the round takes a minute and fails
        d = yt_wait.decide(w, {"kind": "retry", "reason": "all relays failed"}, t, limit)
        w = d["w"]
        if d["do"] == "give_up":
            break
        assert d["do"] == "schedule"
    assert starts[:7] == [2, 7, 16, 32, 53, 84, 115]      # 2, then pauses 4, 8, 15, 20, 30 … after 1-min rounds
    assert len(starts) >= 6 and starts[-1] <= 119              # the last try is before the 2-h mark
    assert d["do"] == "give_up" and t - NOW <= limit + 120


def test_success_requeues_and_user_reason_fails_and_long_download_is_not_cut():
    w = {"since": NOW - 600, "attempt": 3, "req_at": NOW - 60, "next_at": NOW - 60}
    d = yt_wait.decide(w, {"kind": "done", "relay": "mac", "duration": 1234}, NOW, 7200)
    assert d["do"] == "requeue" and d["relay"] == "mac" and d["duration"] == 1234 and d["w"]["req_at"] is None
    d = yt_wait.decide(w, {"kind": "user", "code": "SOURCE_PRIVATE", "message": "приватное"}, NOW, 7200)
    assert d["do"] == "fail_user" and d["code"] == "SOURCE_PRIVATE"
    late = dict(w, since=NOW - 7300)
    assert yt_wait.decide(late, {"kind": "wait"}, NOW, 7200)["do"] == "none"     # a relay is downloading: let it finish
    assert yt_wait.decide(dict(w, since=NOW - 9100), {"kind": "wait"}, NOW, 7200)["do"] == "give_up"


# ── the waiter's tick with fakes ───────────────────────────────────────────────
class FakeDb:
    def __init__(self, jobs):
        self.jobs = {j["id"]: j for j in jobs}

    def waiting(self):
        return [dict(j) for j in self.jobs.values() if j["status"] == "source_wait"]

    def save_wait(self, jid, w):
        self.jobs[jid]["params"] = dict(self.jobs[jid]["params"], yt_wait=w)
        return True

    def requeue(self, jid, sf, w):
        j = self.jobs[jid]
        j["status"], j["params"] = "queued", dict(j["params"], source_fetch=sf, yt_wait=w)
        return True


class FakeRelay:
    def __init__(self):
        self.sts, self.subs, self.wd = {}, [], []
        self.hbs = {"bot": {"fresh": True, "priority": 0}, "mac": {"fresh": True, "priority": 1}}

    def submit(self, jid, sf, mode, key, now):
        self.subs.append((jid, now))
        self.sts[jid] = {}

    def withdraw(self, jid):
        self.wd.append(jid)

    def heartbeats(self, now):
        return self.hbs

    def statuses(self, jid):
        return self.sts.get(jid, {})

    outcome = staticmethod(yt_relay.outcome)


def _job(jid="j1"):
    return {"id": jid, "user_id": "u1", "mode": "virality", "status": "source_wait", "s3_input_key": f"in/{jid}.mp4",
            "params": {"source_fetch": {"url": "https://www.youtube.com/watch?v=abc", "platform": "youtube"}}}


def test_tick_parks_retries_then_requeues_with_prefetched(ledger):
    db, rl, fails = FakeDb([_job()]), FakeRelay(), []
    f = lambda j, c, m: fails.append((j, c, m)) or True
    acts = yt_wait.tick(db, rl, NOW, ledger=ledger, env={}, fail=f, pool={"total": 2, "live": 0})
    assert any("init" in a for a in acts)
    box = outbox(ledger)
    k = [k for k in box if k.startswith("yt-wait:")]
    assert k and "нужен новый файл cookies" in box[k[0]] and "Ждут скачивания: 1 задание" in box[k[0]]
    assert "живых 0 из 2" in box[k[0]]
    yt_wait.tick(db, rl, NOW + 60, ledger=ledger, env={}, fail=f)
    assert not rl.subs and len([k for k in outbox(ledger) if k.startswith("yt-wait:")]) == 1    # no spam within the hour
    yt_wait.tick(db, rl, NOW + 121, ledger=ledger, env={}, fail=f)
    assert rl.subs == [("j1", NOW + 121)]
    rl.sts["j1"] = {"bot": {"state": "failed", "code": "SOURCE_BOT_CHECK", "ts": NOW + 130},
                    "mac": {"state": "failed", "code": "SOURCE_BOT_CHECK", "ts": NOW + 150}}
    acts = yt_wait.tick(db, rl, NOW + 160, ledger=ledger, env={}, fail=f)
    assert any("schedule" in a for a in acts) and db.jobs["j1"]["status"] == "source_wait"
    w = db.jobs["j1"]["params"]["yt_wait"]
    assert w["attempt"] == 1 and w["next_at"] == NOW + 160 + 240
    yt_wait.tick(db, rl, NOW + 400, ledger=ledger, env={}, fail=f)
    assert len(rl.subs) == 2
    rl.sts["j1"] = {"mac": {"state": "done", "ts": NOW + 500, "duration": 600}}
    acts = yt_wait.tick(db, rl, NOW + 510, ledger=ledger, env={}, fail=f)
    j = db.jobs["j1"]
    assert j["status"] == "queued" and not fails
    sf = j["params"]["source_fetch"]
    assert sf["prefetched"] and sf["prefetched_duration"] == 600 and sf["prefetched_relay"] == "mac" and sf["url"]
    assert "j1" in rl.wd
    assert any(k.startswith("yt-wait-ok:") for k in outbox(ledger))


def test_tick_gives_up_politely_after_two_hours(ledger):
    job = _job()
    job["params"]["yt_wait"] = {"since": NOW - 7300, "attempt": 7, "next_at": NOW - 10, "req_at": None}
    db, rl, fails = FakeDb([job]), FakeRelay(), []
    yt_wait.tick(db, rl, NOW, ledger=ledger, env={}, fail=lambda j, c, m: fails.append((j, c, m)) or True)
    assert fails == [("j1", "SOURCE_YT_UNAVAILABLE", yt_wait.GIVEUP_TEXT)]
    assert "эта попытка не списана" in yt_wait.GIVEUP_TEXT and "VK/Rutube" in yt_wait.GIVEUP_TEXT
    assert "YouTube не отдал" not in yt_wait.GIVEUP_TEXT and "робот" not in yt_wait.GIVEUP_TEXT
    assert any(k.startswith("yt-wait-gaveup:j1") for k in outbox(ledger))


def test_tick_switch_off_and_user_reason():
    db, rl = FakeDb([_job()]), FakeRelay()
    assert yt_wait.tick(db, rl, NOW, env={"SONYA_YT_WAIT": "off"}) == []
    db.jobs["j1"]["params"]["yt_wait"] = {"since": NOW - 100, "attempt": 1, "req_at": NOW - 50, "next_at": NOW - 50}
    rl.sts["j1"] = {"bot": {"state": "failed", "code": "SOURCE_UNAVAILABLE", "message": "Видео удалено", "ts": NOW - 5}}
    fails = []
    yt_wait.tick(db, rl, NOW, env={}, fail=lambda j, c, m: fails.append((c, m)) or True)
    assert fails == [("SOURCE_UNAVAILABLE", "Видео удалено")]


# ── worker: never a «YouTube не отдал» to the user ────────────────────────────
def test_worker_parks_instead_of_failing(monkeypatch, tmp_path):
    from scripts import gpu_worker as gw
    from scripts.source_fetch import SourceFetchError
    calls = []
    monkeypatch.setattr(gw, "_fail", lambda j, c, m, retry: calls.append((c, m, retry)))
    monkeypatch.setattr(gw, "_update_status", lambda *a: None)
    monkeypatch.setattr(gw, "assert_mode_allowed", lambda m: None)
    monkeypatch.setattr(gw, "sweep_stale_tmp", lambda *a, **k: 0)

    def refused(*a, **k):
        raise SourceFetchError("SOURCE_BOT_CHECK", "not a bot")
    monkeypatch.setattr("scripts.source_fetch.fetch_source", refused)
    monkeypatch.setattr(gw, "_fetch_via_relay", lambda *a, **k: refused())
    monkeypatch.setattr(gw, "_YT_DIRECT", {"blocked_at": 0.0})
    job = {"id": "j1", "mode": "virality", "user_id": "u", "s3_input_key": "in/j1.mp4",
           "params": {"source_fetch": {"url": "https://www.youtube.com/watch?v=abc", "platform": "youtube"}}}
    gw.process_job(job, "w1") if hasattr(gw, "process_job") else gw._process_job(job, "w1")
    assert calls and calls[0][0] == "SOURCE_YT_WAIT" and calls[0][2] is True
    assert "Скачиваем видео" in calls[0][1]


def test_worker_reads_a_prefetched_source(monkeypatch, tmp_path):
    from scripts import gpu_worker as gw
    calls, got = [], {}
    monkeypatch.setattr(gw, "_fail", lambda j, c, m, retry: calls.append((c, m, retry)))
    monkeypatch.setattr(gw, "_update_status", lambda *a: None)
    monkeypatch.setattr(gw, "assert_mode_allowed", lambda m: None)
    monkeypatch.setattr(gw, "sweep_stale_tmp", lambda *a, **k: 0)
    monkeypatch.setattr("scripts.source_fetch.fetch_source", lambda *a, **k: pytest.fail("must not download again"))

    def dl(key, dest):
        got["key"] = key
        Path(dest).write_bytes(b"x" * 100)
    monkeypatch.setattr(gw, "download_file", dl)

    def stop(*a, **k):
        raise RuntimeError("stop after the download")
    monkeypatch.setattr(gw, "ensure_models_for_mode", stop)
    job = {"id": "j2", "mode": "virality", "user_id": "u", "s3_input_key": "in/j2.mp4",
           "params": {"source_fetch": {"url": "https://youtu.be/abc", "platform": "youtube", "prefetched": True,
                                       "prefetched_duration": 61}}}
    gw.process_job(job, "w1") if hasattr(gw, "process_job") else gw._process_job(job, "w1")
    assert got["key"] == "in/j2.mp4"
    assert not any(c[0].startswith("SOURCE") for c in calls)


# ── cookies pool ───────────────────────────────────────────────────────────────
def test_pool_checks_due_flip_and_rotation(monkeypatch):
    from tools import yt_cookies as yc
    st = {"entries": [{"name": "pool-1", "bad": False}, {"name": "pool-2", "bad": False}, {"name": "pool-3", "bad": True,
                                                                                          "checked": NOW - 60}],
          "active": "pool-2", "pushed": True}
    assert yc.due_checks(st, NOW) == ["pool-1", "pool-2"]                      # dead one: only every 6 h
    res = {"pool-1": "ok", "pool-2": "expired"}
    flipped = yc.check_pool(st, NOW, checker=lambda n: res[n])
    assert flipped == ["pool-2"] and st["pushed"] is False                     # active died -> activate() again
    assert yc.summary(st, NOW)["live"] == 1 and yc.summary(st, NOW)["total"] == 3
    assert yc.due_checks(st, NOW + 60) == []
    pushed = []
    monkeypatch.setattr(yc, "load", lambda n: b"data-" + n.encode())
    monkeypatch.setattr(yc, "_write_targets", lambda d: pushed.append(d) or True)
    yc.activate(st)
    assert st["active"] == "pool-1" and pushed == [b"data-pool-1"]
    assert not yc.apply_check(st, "pool-1", "error", NOW + 9999)               # a network error changes nothing
    assert yc.apply_check(st, "pool-3", "ok", NOW) and not st["entries"][2]["bad"]   # a false «expired» comes back


def test_pool_alerts(ledger):
    from accounting import yt_watch
    assert yt_watch.check_pool(ledger, None, NOW) == []
    assert yt_watch.check_pool(ledger, {"total": 3, "live": 2}, NOW) == []
    k1 = yt_watch.check_pool(ledger, {"total": 3, "live": 1}, NOW)
    assert k1 and "остался 1" in outbox(ledger)[k1[0]]
    assert yt_watch.check_pool(ledger, {"total": 3, "live": 1}, NOW + 60) == []
    k0 = yt_watch.check_pool(ledger, {"total": 3, "live": 0}, NOW)
    assert k0 and "нужен новый файл cookies" in outbox(ledger)[k0[0]]


# ── «ждёт денег Vast» ──────────────────────────────────────────────────────────
def test_no_funds_decision():
    from dispatcher import capacity as cp
    assert cp.no_funds(2, 0, Decimal("-0.5"), 0)
    assert cp.no_funds(1, 0, Decimal("0.8"), 2)
    assert not cp.no_funds(1, 0, Decimal("0.8"), 0)
    assert not cp.no_funds(0, 0, Decimal("-1"), 3)                             # nobody waits
    assert not cp.no_funds(3, 1, Decimal("-1"), 3)                             # a server is up: the queue moves
    assert not cp.no_funds(3, 0, None, 3)
    ev = [{"event": "create_failed_next", "detail": {"status": 400}}, {"event": "resume_failed", "detail": {"status": 500}},
          {"event": "create_failed", "detail": {"reason": "vast HTTP 402 on PUT /api/v0/asks/1/"}}]
    assert cp.refusals(ev) == 2


def test_capacity_state_file_and_alerts(tmp_path, ledger):
    from dispatcher import capacity as cp
    p = tmp_path / "capacity.json"
    c = cp.tick(p, 3, 0, Decimal("-0.2"), [], NOW, ledger=ledger)
    assert c["state"] == "no_funds" and json.loads(p.read_text())["since"] == NOW
    box = outbox(ledger)
    k = [k for k in box if k.startswith("vast-nofunds:")]
    assert k and "ждут 3 заданий" in box[k[0]] and "Высокая нагрузка" in box[k[0]]
    c = cp.tick(p, 3, 0, Decimal("-0.2"), [], NOW + 600, ledger=ledger)
    assert c["since"] == NOW and not c["alerts"]                               # same episode, no repeat for 2 h
    c = cp.tick(p, 3, 0, Decimal("-0.2"), [], NOW + 7300, ledger=ledger)
    assert c["alerts"]                                                         # reminder
    c = cp.tick(p, 0, 1, Decimal("20"), [], NOW + 8000, ledger=ledger)
    assert c["state"] == "ok" and any(a.startswith("vast-nofunds-ok") for a in c["alerts"])


def test_ops_alert_keys_are_deliverable():
    """2026-11-19: ops alerts (balance, YouTube, budget) were claimed by the bot and never delivered: key format."""
    from accounting.internal_api import KEY_RE as api_re
    import importlib.util
    spec = importlib.util.spec_from_file_location("vr", str(Path(__file__).resolve().parents[1] / "integration/accounting_bot/vast_reports.py"))
    vr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vr)
    for k in ("vast-daily:2026-11-19", "vast-monthly:2026-11", "vast-balance:critical:166208", "fleet-budget:day_limit:20775",
              "yt-wait:1795151096:0", "yt-wait-gaveup:551ea1ed-be5e-4d59-bbbe-000646214dca", "yt-pool-0:83108",
              "vast-nofunds:1795151096:1", "yt-cookies:bot:20776"):
        assert api_re.match(k) and vr.KEY_RE.match(k), k
    for k in ("../x", "a", "yt-wait", "yt wait:1", "x:" + "a" * 70, "vast-daily:2026-11-19/../sent"):
        assert not api_re.match(k) and not vr.KEY_RE.match(k), k
