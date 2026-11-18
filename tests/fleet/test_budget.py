"""Create only when there is no other way (owner 2026-11-18: 25 servers created in a day instead of 2-3)."""
import io
import json
import os
import stat
import subprocess
from decimal import Decimal
from pathlib import Path

from accounting import fleet_stats
from dispatcher import launcher
from dispatcher.config import FleetConfig
from scripts.ops import release_pointer
from tests.fleet.fakes import ENV, IMAGE, SHA, make, offer

PROD = {"warm_prime": False, "max_instances": 3, "jobs_per_instance": 2, "warm_instances": 1, "spare_instances": 2,
        "resume_retry_seconds": 1800, "max_resume_failures": 3}


def run_until(mgr, clock, secs, step=10):
    for _ in range(0, secs, step):
        mgr.tick()
        clock.advance(step)


def fleet(**over):
    return make({**PROD, **over})


def ev(store, name):
    return [e for e in store.events() if e["event"] == name]


def serve(mgr, queue, clock, jid, store):
    """job arrives -> server(s) up -> a running server takes it -> done."""
    queue.add(jid, ts=clock())
    mgr.tick()
    clock.advance(30)
    mgr.tick()
    lab = next(i.label for i in store.all(("running",)))
    queue.set(jid, "mode_running", lab, clock())
    mgr.tick()
    clock.advance(120)
    queue.set(jid, "completed", lab, clock())
    return lab


def _warm_plus_burst(mgr, vast, queue, store, clock):
    """3 jobs at once: two servers (2 slots each); both idle afterwards."""
    vast.offers = [offer(1, 7), offer(2, 8), offer(3, 9)]
    for j in ("a", "b", "c"):
        queue.add(j, ts=clock())
    mgr.tick()
    clock.advance(30)
    mgr.tick()
    labs = [i.label for i in store.all(("running",))]
    assert len(labs) == 2
    for j, lab in zip(("a", "b", "c"), labs + labs):
        queue.set(j, "mode_running", lab, clock())
    mgr.tick()
    clock.advance(200)
    for j, lab in zip(("a", "b", "c"), labs + labs):
        queue.set(j, "completed", lab, clock())
    run_until(mgr, clock, 200)


# ------------------------------------------------------------------ 1. stop + resume instead of destroy + create
def test_idle_burst_is_stopped_as_spare_and_resumed_before_any_create():
    mgr, vast, queue, store, clock = fleet()
    _warm_plus_burst(mgr, vast, queue, store, clock)
    assert vast.n("destroy") == 0 and vast.n("create") == 2
    roles = sorted(i.role for i in store.all(("stopped",)))
    assert roles == ["spare", "warm"]
    for j in ("d", "e", "f"):                        # the next rush: two resumes, zero creates
        queue.add(j, ts=clock())
    mgr.tick()
    assert vast.n("resume") == 2 and vast.n("create") == 2


def test_spare_disk_too_dear_or_limit_reached_is_destroyed_as_before():
    mgr, vast, queue, store, clock = fleet(spare_max_storage_gb_month=Decimal("0.10"))
    _warm_plus_burst(mgr, vast, queue, store, clock)
    assert [e["detail"]["reason"] for e in ev(store, "destroy_requested")] == ["burst_idle"]
    mgr, vast, queue, store, clock = fleet(spare_instances=0)
    _warm_plus_burst(mgr, vast, queue, store, clock)
    assert vast.n("destroy") == 1


def test_spare_is_destroyed_after_retention():
    mgr, vast, queue, store, clock = fleet(spare_retention_hours=24)
    _warm_plus_burst(mgr, vast, queue, store, clock)
    run_until(mgr, clock, 23 * 3600, step=600)
    assert vast.n("destroy") == 0
    run_until(mgr, clock, 2 * 3600, step=600)
    assert [e["detail"]["reason"] for e in ev(store, "destroy_requested")] == ["spare_retention"]
    assert [i.role for i in store.all(("stopped",))] == ["warm"]


def test_spare_becomes_the_warm_instead_of_a_prime():
    mgr, vast, queue, store, clock = fleet(warm_prime=True, warm_instances=1)
    _warm_plus_burst(mgr, vast, queue, store, clock)
    assert sorted(i.role for i in store.all(("stopped",))) == ["spare", "warm"]
    mgr.cfg = FleetConfig(**{**mgr.cfg.__dict__, "warm_instances": 2})       # owner: a 2nd warm
    run_until(mgr, clock, 600)
    assert vast.n("create") == 2                      # the 2nd warm is the paused spare: promoted, not created
    assert sorted(i.role for i in store.all(("stopped",))) == ["warm", "warm"] and ev(store, "warm_promoted")


# ------------------------------------------------------------------ 4. a busy GPU: the next stopped server, then later
def test_resume_timeout_tries_next_stopped_and_keeps_the_disk():
    mgr, vast, queue, store, clock = fleet()
    _warm_plus_burst(mgr, vast, queue, store, clock)
    vast.resume_gets_gpu = False
    queue.add("x", ts=clock())
    mgr.tick()
    first = [c[1] for c in vast.calls if c[0] == "resume"]
    assert len(first) == 1
    vast.resume_gets_gpu = True
    run_until(mgr, clock, 120)
    resumed = [c[1] for c in vast.calls if c[0] == "resume"]
    assert len(resumed) == 2 and resumed[1] != resumed[0]          # the other stopped server, same minute
    assert vast.n("create") == 2 and vast.n("destroy") == 0
    assert store.get(first[0]).state == "stopped" and store.get(first[0]).retry_after > clock()
    assert ev(store, "resume_retry_later")


def test_resume_failures_up_to_the_limit_then_destroy():
    mgr, vast, queue, store, clock = fleet(max_resume_failures=2, spare_instances=0)
    _warm_plus_burst(mgr, vast, queue, store, clock)
    (w,) = store.all(("stopped",))
    vast.resume_gets_gpu = False
    vast.offers = []                                   # nothing to create: only the warm
    queue.add("x", ts=clock())
    run_until(mgr, clock, 120)
    assert store.get(w.id).state == "stopped" and vast.n("destroy") == 1      # 1st failure: kept (the 1 destroy = burst)
    clock.advance(1800)
    run_until(mgr, clock, 120)
    assert store.get(w.id).state in ("destroying", "destroyed")


# ------------------------------------------------------------------ 2. a new release re-creates nothing
def test_new_release_keeps_self_updating_servers():
    mgr, vast, queue, store, clock = fleet(release_pointer=True)
    _warm_plus_burst(mgr, vast, queue, store, clock)
    payload = [c for c in vast.calls if c[0] == "create"][0][2]
    assert payload["env"]["SONYA_RELEASE_POINTER"] == "on" and payload["env"]["SONYA_BASE_IMAGE_REF"] == IMAGE
    mgr.cfg = FleetConfig(**{**mgr.cfg.__dict__, "release_sha": "c" * 40})
    run_until(mgr, clock, 3600, step=300)
    assert vast.n("destroy") == 0
    queue.add("x", ts=clock())
    mgr.tick()
    assert vast.n("resume") == 1 and vast.n("create") == 2
    assert {i.release_sha for i in store.all(("starting", "running"))} == {"c" * 40}


def test_new_release_without_pointer_or_new_image_is_stale():
    mgr, vast, queue, store, clock = fleet()
    _warm_plus_burst(mgr, vast, queue, store, clock)
    mgr.cfg = FleetConfig(**{**mgr.cfg.__dict__, "release_sha": "c" * 40})
    mgr.tick()
    assert {e["detail"]["reason"] for e in ev(store, "destroy_requested")} == {"stale_release"}
    mgr, vast, queue, store, clock = fleet(release_pointer=True)
    _warm_plus_burst(mgr, vast, queue, store, clock)
    mgr.cfg = FleetConfig(**{**mgr.cfg.__dict__, "image": IMAGE[:-1] + "c"})
    mgr.tick()
    assert vast.n("destroy") == 2


def test_compatible_old_release_is_kept():
    mgr, vast, queue, store, clock = fleet()
    _warm_plus_burst(mgr, vast, queue, store, clock)
    mgr.cfg = FleetConfig(**{**mgr.cfg.__dict__, "release_sha": "c" * 40, "release_compat": (SHA[:7],)})
    run_until(mgr, clock, 600, step=60)
    assert vast.n("destroy") == 0


# ------------------------------------------------------------------ 5. budget
def test_day_create_limit_blocks_creates_but_not_resumes():
    mgr, vast, queue, store, clock = fleet(max_creates_per_day=2, max_instances=4)
    _warm_plus_burst(mgr, vast, queue, store, clock)
    for j in "defghij":
        queue.add(j, ts=clock())
    mgr.tick()
    assert vast.n("resume") == 2 and vast.n("create") == 2
    (b,) = ev(store, "create_blocked")
    assert b["detail"]["reason"] == "day_limit"
    run_until(mgr, clock, 1200, step=60)
    assert len(ev(store, "create_blocked")) <= 3      # throttled, not every tick


def test_day_budget_usd_blocks_creates():
    mgr, vast, queue, store, clock = fleet(day_budget_usd=Decimal("0.05"), create_download_gb=10)
    vast.offers = [offer(1, 7, internet_down_cost_per_tb=20)]
    queue.add("a", ts=clock())
    mgr.tick()                                         # first create: spend 0 -> allowed (10 GB x $20/TB = $0.20)
    assert vast.n("create") == 1
    assert mgr.day_spend_usd(clock() + 61) >= Decimal("0.20")
    for j in "bcd":
        queue.add(j, ts=clock())
    clock.advance(120)
    mgr.tick()
    assert vast.n("create") == 1 and ev(store, "create_blocked")[0]["detail"]["reason"] == "budget"


def test_prime_at_most_once_per_interval_and_not_at_night():
    mgr, vast, queue, store, clock = fleet(warm_prime=True, warm_instances=2, spare_instances=0, prime_min_interval_hours=4)
    clock.t = 1_800_000_000 - (1_800_000_000 + 3 * 3600) % 86400 + 10 * 3600     # 10:00 MSK
    vast.offers = [offer(1, 7), offer(2, 8)]
    run_until(mgr, clock, 1200)
    assert vast.n("create") == 1                       # the 2nd warm waits for the interval
    run_until(mgr, clock, 4 * 3600, step=300)
    assert vast.n("create") == 2
    mgr, vast, queue, store, clock = fleet(warm_prime=True, warm_instances=2, spare_instances=0, night_warm_instances=1)
    clock.t = 1_800_000_000 - (1_800_000_000 + 3 * 3600) % 86400 + 1 * 3600      # 01:00 MSK
    vast.offers = [offer(1, 7), offer(2, 8)]
    run_until(mgr, clock, 3600, step=60)
    assert vast.n("create") == 1                       # one warm at night
    run_until(mgr, clock, 8 * 3600, step=600)
    assert vast.n("create") == 2                       # the 2nd after 08:00 MSK


# ------------------------------------------------------------------ 3. proven hosts only, probes limited, NVENC ban
def test_probe_limit_uses_proven_hosts_only():
    mgr, vast, queue, store, clock = fleet(probes_per_day=0, proven_first=True, spare_instances=0)
    store.mark_proven(9, None, clock())
    vast.offers = [offer(1, 7, dph=0.05), offer(2, 9, dph=0.15)]
    queue.add("a", ts=clock())
    mgr.tick()                                         # no server up at all: the best offer even if never proven
    assert vast.created_offers == [2]                  # ... but the proven one is ranked first anyway
    for j in "bcd":
        queue.add(j, ts=clock())
    vast.offers = [offer(1, 7, dph=0.05)]
    clock.advance(70)
    mgr.tick()
    assert vast.created_offers == [2]                  # a server is up, the probe limit holds
    assert any("probe limit" in e["detail"]["reason"] for e in ev(store, "create_failed"))


def test_nvenc_missing_is_banned_forever_and_unproven():
    mgr, vast, queue, store, clock = make({"nvenc_ban_forever": True, "warm_prime": False})
    store.ban_machine(5, clock() + 3600, "nvenc_missing")
    mgr2 = type(mgr)(mgr.cfg, store, vast, queue, lambda: "", ENV, clock)
    assert 5 in store.banned_machines(clock() + 365 * 86400)
    store.mark_proven(7, None, clock())
    flags = {}
    mgr2._nvenc_missing = lambda label: flags.get(label, True)
    queue.add("a", ts=clock())
    for _ in range(3):
        mgr2.tick()
        clock.advance(40)
    assert ev(store, "nvenc_missing") and 7 in store.banned_machines(clock() + 10 * 365 * 86400)
    assert 7 not in store.proven_machines()


def test_proven_mark_and_ranking():
    cfg = FleetConfig(proven_first=True, rank_traffic=True)
    cheap_far = offer(1, 7, dph=0.06, internet_down_cost_per_tb=40)     # $0.40 per pull
    dear_near = offer(2, 8, dph=0.10, internet_down_cost_per_tb=3)
    assert [o["id"] for o in launcher.rank_offers([cheap_far, dear_near], cfg, set())] == [2, 1]
    assert [o["id"] for o in launcher.rank_offers([dear_near, cheap_far], cfg, set(), proven={7})] == [1, 2]
    cap = FleetConfig(max_inet_down_usd_tb=Decimal(25))
    assert [o["id"] for o in launcher.rank_offers([cheap_far, dear_near], cap, set())] == [2]


def test_worker_that_took_a_job_marks_its_host_proven():
    mgr, vast, queue, store, clock = fleet()
    serve(mgr, queue, clock, "a", store)
    assert store.proven_machines() == {7}


# ------------------------------------------------------------------ futile wakes
def test_futile_wakes_back_off():
    mgr, vast, queue, store, clock = fleet(futile_wakes_per_hour=2, futile_backoff_seconds=1200, spare_instances=0,
                                           max_instances=1)
    vast.offers = [offer(1, 7), offer(2, 8), offer(3, 9), offer(4, 10)]
    queue.add("ghost", ts=clock())                    # queued, but the API never hands it out
    run_until(mgr, clock, 2 * 3600, step=30)
    assert ev(store, "futile_wake") and any(e["detail"]["reason"] == "futile_wakes" for e in ev(store, "create_blocked"))
    assert vast.n("create") <= 3                       # without the guard: a new server every ~25 min


def test_active_job_of_a_foreign_or_dead_worker_creates_nothing():
    mgr, vast, queue, store, clock = fleet(demand_live_only=True)
    queue.add("mac", status="mode_running", worker_id="mac-mini-local", ts=clock())
    queue.add("dead", status="mode_running", worker_id="sonya-pool-deadbeef", ts=clock())
    run_until(mgr, clock, 600)
    assert vast.n("create") == 0
    mgr, vast, queue, store, clock = fleet()
    queue.add("mac", status="mode_running", worker_id="mac-mini-local", ts=clock())
    mgr.tick()
    assert vast.n("create") == 1                       # the old rule (off): a server for somebody else's job


# ------------------------------------------------------------------ 6. accounting lines + alerts
def test_fleet_lines_and_budget_alert():
    mgr, vast, queue, store, clock = fleet(max_creates_per_day=2, max_instances=4)
    _warm_plus_burst(mgr, vast, queue, store, clock)
    for j in "defghij":
        queue.add(j, ts=clock())
    mgr.tick()
    text = "\n".join(fleet_stats.lines(store, clock(), cfg=mgr.cfg))
    assert "Серверов создано за сутки: 2 (из них под задания 2)" in text and "resume): 2" in text
    assert "новых серверов сегодня (МСК) 2 из 2" in text

    class L:
        def __init__(self):
            import sqlite3
            self.c = sqlite3.connect(":memory:")
            self.c.execute("CREATE TABLE report_outbox(report_key TEXT PRIMARY KEY, kind TEXT, period TEXT, text TEXT, "
                           "payload TEXT, created_at REAL)")
    led = L()
    assert len(fleet_stats.check_budget_alerts(led, store, clock())) == 1
    assert fleet_stats.check_budget_alerts(led, store, clock()) == []       # once per day
    store.event(clock(), None, "create_blocked", reason="budget", kind="reserve")
    assert fleet_stats.check_budget_alerts(led, store, clock()) == []       # background reserve: no alert


# ------------------------------------------------------------------ release pointer (worker side)
class _S3:
    def __init__(self):
        self.obj = {}

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.obj:
            raise KeyError(Key)
        return {"Body": io.BytesIO(self.obj[(Bucket, Key)])}

    def put_object(self, Bucket, Key, Body, ContentType):
        self.obj[(Bucket, Key)] = Body


def test_release_pointer_publish_and_decide():
    s3, env = _S3(), {"S3_BUCKET_NAME": "b"}
    assert release_pointer.read(s3, env) is None
    assert release_pointer.publish(SHA, IMAGE, s3, env) and not release_pointer.publish(SHA, IMAGE, s3, env)
    p = release_pointer.read(s3, env)
    assert release_pointer.decide(p, "c" * 40, IMAGE) == SHA
    assert release_pointer.decide(p, SHA, IMAGE) == ""
    assert release_pointer.decide(p, "c" * 40, "other-image") == ""         # needs another base image: stay
    assert release_pointer.decide({"sha": "main"}, "c" * 40, IMAGE) == ""


def _exe(p: Path, body: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body)
    p.chmod(p.stat().st_mode | stat.S_IXUSR)


def test_run_worker_switches_to_the_pointer_release(tmp_path):
    root = Path(__file__).resolve().parents[2]
    new = "d" * 40
    rel = tmp_path / "rel"
    _exe(rel / "scripts" / "ops" / "release_pointer.py", f"print({new!r})\n")
    (rel / "scripts" / "__init__.py").write_text("")
    (rel / "scripts" / "ops" / "__init__.py").write_text("")
    newdir = tmp_path / "releases" / new
    _exe(newdir / "deploy" / "vast" / "run_worker.sh", "#!/usr/bin/env bash\necho HOPPED $SONYA_RELEASE_SHA $SONYA_RELEASE_DIR\n")
    boot = tmp_path / "boot"
    _exe(boot / "sonya_bootstrap.py", f"import os\nassert os.environ['SONYA_RELEASE_SHA'] == {new!r}\nprint({str(newdir)!r})\n")
    env = {**os.environ, "SONYA_RELEASE_DIR": str(rel), "SONYA_RELEASE_SHA": SHA, "SONYA_RELEASE_POINTER": "on",
           "SONYA_BOOTSTRAP_DIR": str(boot), "SONYA_STATE_DIR": str(tmp_path / "st"), "TMPDIR": str(tmp_path / "tmp"),
           "MODELS_LOCAL_DIR": str(tmp_path / "m"), "SONYA_WORKER_CONCURRENCY": "bad"}
    out = subprocess.run(["bash", str(root / "deploy/vast/run_worker.sh")], env=env, capture_output=True, text=True, timeout=60)
    assert out.stdout.strip() == f"HOPPED {new} {newdir}", out.stderr
    # pointer off -> the old path (here: it stops at the concurrency check, exit 3)
    out = subprocess.run(["bash", str(root / "deploy/vast/run_worker.sh")], env={**env, "SONYA_RELEASE_POINTER": "off"},
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 3 and "HOPPED" not in out.stdout
    # bootstrap failure -> stays on its release
    _exe(boot / "sonya_bootstrap.py", "raise SystemExit(2)\n")
    out = subprocess.run(["bash", str(root / "deploy/vast/run_worker.sh")], env=env, capture_output=True, text=True, timeout=60)
    assert out.returncode == 3 and "switch failed" in out.stderr
