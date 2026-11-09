"""Multi-GPU POWER class in the fleet: off by default (the 1 warm RTX 3060 + burst scheme is unchanged), on -> a rush gets one
multi-GPU server (e.g. 2x A40 Belgium) counted with its own video slots; never the warm; no power offer -> default class."""
import sqlite3
from decimal import Decimal

import pytest

from dispatcher import launcher
from dispatcher.config import FleetConfig
from dispatcher.store import FleetStore
from tests.fleet.fakes import ENV, GH_TOKEN, IMAGE, SHA, make, offer


def a40x2(oid=50, machine=500, **kw):
    """2x A40 (48 GB each) + EPYC 32 threads + 128 GB, Belgium, $0.72/h."""
    base = dict(gpu="A40", cpu="AMD EPYC 7543 32-Core Processor", cores=32.0, dph=0.72, geo="Belgium, BE", storage=0.15,
                num_gpus=2, gpu_ram=46068, cpu_ram=128 * 1024)
    return offer(oid, machine, **{**base, **kw})


def run_until(mgr, clock, secs, step=10):
    for _ in range(0, secs, step):
        mgr.tick()
        clock.advance(step)


def rush(queue, clock, n):
    for k in range(n):
        queue.add(f"j{k}", ts=clock())


def created(vast):
    return [c[2] for c in vast.calls if c[0] == "create"]


# ── config ────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_power_is_off_by_default_and_env_parsing():
    c = FleetConfig.from_env({})
    assert c.power_enabled is False and c.power_gap() == 3 and c.power_gpus[0] == "A40"
    c = FleetConfig.from_env({"SONYA_POWER_ENABLED": "true", "SONYA_POWER_GPUS": "A40, L40S", "SONYA_POWER_MIN_GPUS": "2",
                              "SONYA_POWER_MAX_GPUS": "8", "SONYA_POWER_MAX_DPH": "0.9", "SONYA_POWER_MIN_GAP": "5",
                              "SONYA_POWER_JOBS_PER_INSTANCE": "6"})
    assert (c.power_enabled, c.power_gpus, c.power_max_gpus, c.power_max_dph, c.power_gap(), c.power_jobs_per_instance) == \
        (True, ("A40", "L40S"), 8, Decimal("0.9"), 5, 6)
    with pytest.raises(ValueError):
        FleetConfig.from_env({"SONYA_POWER_MIN_GPUS": "4", "SONYA_POWER_MAX_GPUS": "2"})
    with pytest.raises(ValueError):
        FleetConfig.from_env({"SONYA_POWER_JOBS_PER_INSTANCE": "17"})


# ── offer rules ───────────────────────────────────────────────────────────────────────────────────────────────────
def test_offer_slots_one_formula_with_the_worker():
    cfg = FleetConfig()
    assert launcher.offer_slots(offer(1, 7), cfg) == 2                               # default class: jobs_per_instance
    assert launcher.offer_slots(a40x2(), cfg, power=True) == 8                       # CPU: 32 threads / 4
    assert launcher.offer_slots(a40x2(cores=64.0), cfg, power=True) == 16            # VRAM 8/card cap, max 16 processes
    assert launcher.offer_slots(a40x2(cpu_ram=24 * 1024), cfg, power=True) == 5      # RAM: (24 - 4) / 4
    assert launcher.offer_slots(a40x2(), cfg, power=True, environ={"SONYA_CPU_THREADS_PER_JOB": "8"}) == 4
    assert launcher.offer_slots(a40x2(), FleetConfig(power_jobs_per_instance=6), power=True) == 6
    assert launcher.total_vram_mb(a40x2()) == 92136 and launcher.total_vram_mb({"gpu_total_ram": 1000, "gpu_ram": 1}) == 1000


def test_power_ok_rules():
    cfg = FleetConfig(power_enabled=True)
    assert launcher.power_ok(a40x2(), cfg)
    assert not launcher.power_ok(a40x2(num_gpus=1), cfg)                            # one card = not a power server
    assert not launcher.power_ok(a40x2(num_gpus=8), cfg)                            # above power_max_gpus (4)
    assert not launcher.power_ok(a40x2(gpu="RTX 3060", gpu_ram=12288), cfg)
    assert not launcher.power_ok(a40x2(cores=8.0), cfg)
    assert not launcher.power_ok(a40x2(dph=1.5), cfg)
    assert not launcher.power_ok(a40x2(gpu="L40S", gpu_ram=20000), cfg)              # 40 GB total < 48
    assert launcher.power_ok(a40x2(gpu="RTX A6000", gpu_ram=49140, cpu="Intel Xeon Gold 6338"), cfg)   # EPYC/Xeon fine here


def test_power_query_and_pick_prefers_europe_then_price_per_video():
    cfg = FleetConfig(power_enabled=True)
    q = launcher.power_search_query(cfg)
    assert q["num_gpus"] == {"gte": 2, "lte": 4} and "A40" in q["gpu_name"]["in"] and q["dph_total"] == {"lte": 1.0}
    assert launcher.search_query(cfg)["num_gpus"] == {"eq": 1}                       # the default search is untouched
    us = a40x2(51, 501, geo="Texas, US", dph=0.40)
    cheap_eu = a40x2(52, 502, dph=0.60)
    big_eu = a40x2(53, 503, num_gpus=4, cores=64.0, cpu_ram=256 * 1024, dph=0.95)   # 16 videos for 0.95 -> per video cheaper
    assert launcher.pick_power_offer([us, cheap_eu], cfg, set())["id"] == 52
    assert launcher.pick_power_offer([us, cheap_eu, big_eu], cfg, set())["id"] == 53
    assert launcher.pick_power_offer([us, cheap_eu, big_eu], cfg, {503, 502})["id"] == 51
    assert launcher.pick_offer([cheap_eu], cfg, set()) is None                       # never mistaken for the default class


def test_power_payload_lets_the_worker_size_itself_within_the_fleet_count():
    cfg = FleetConfig(image=IMAGE, release_sha=SHA)
    p = launcher.create_payload(cfg, "sonya-pool-x", GH_TOKEN, ENV, power_slots=8)
    assert p["env"]["SONYA_WORKER_CONCURRENCY"] == "auto" and p["env"]["SONYA_WORKER_CONCURRENCY_MAX"] == "8"
    d = launcher.create_payload(cfg, "sonya-pool-y", GH_TOKEN, ENV)
    assert d["env"]["SONYA_WORKER_CONCURRENCY"] == "2" and "SONYA_WORKER_CONCURRENCY_MAX" not in d["env"]


# ── fleet behaviour ───────────────────────────────────────────────────────────────────────────────────────────────
def fleet(power=True, **over):
    base = {"warm_prime": True, "max_instances": 3, "jobs_per_instance": 2, "power_enabled": power}
    mgr, vast, queue, store, clock = make({**base, **over})
    vast.offers = [offer(1, 7), offer(2, 8, geo="Germany, DE", dph=0.09), offer(3, 9, geo="Spain, ES", dph=0.095), a40x2()]
    return mgr, vast, queue, store, clock


def prime(mgr, clock):
    run_until(mgr, clock, 700)
    (w,) = mgr.store.all()
    assert w.role == "warm" and w.state == "stopped"
    return w


def test_power_off_rush_is_exactly_the_old_scheme():
    mgr, vast, queue, store, clock = fleet(power=False)
    w = prime(mgr, clock)
    rush(queue, clock, 12)
    mgr.tick()
    assert vast.n("resume") == 1 and len(created(vast)) == 3                         # prime + 2 bursts (max 3 servers)
    assert all(p["env"]["SONYA_WORKER_CONCURRENCY"] == "2" for p in created(vast))
    assert {i.gpu_class for i in store.all()} == {"standard"} and all(i.slots == 0 for i in store.all())
    assert store.get(w.id).state == "starting"


def test_rush_gets_one_power_server_counted_with_its_slots():
    mgr, vast, queue, store, clock = fleet()
    w = prime(mgr, clock)
    rush(queue, clock, 10)
    mgr.tick()                                         # warm resumes (2 slots); gap 8 >= 3 -> ONE 2x A40 (8 slots) covers it
    assert vast.n("resume") == 1 and len(created(vast)) == 2
    (pw,) = [i for i in store.all() if i.gpu_class == "power"]
    assert pw.slots == 8 and pw.num_gpus == 2 and pw.role == "burst" and pw.note == "" and mgr.slots(pw) == 8
    assert created(vast)[-1]["env"]["SONYA_WORKER_CONCURRENCY"] == "auto"
    ev = next(e for e in store.events(pw.id) if e["event"] == "created")
    assert ev["detail"]["gpu_class"] == "power" and ev["detail"]["slots"] == 8 and ev["detail"]["num_gpus"] == 2
    mgr.tick()
    assert len(created(vast)) == 2                    # 2 + 8 = 10 covered: nothing more
    assert store.get(w.id).role == "warm"


def test_small_rush_stays_on_the_default_class():
    mgr, vast, queue, store, clock = fleet()
    prime(mgr, clock)
    rush(queue, clock, 4)                              # warm 2 + gap 2 < power_gap 3 -> an RTX 3060 burst
    mgr.tick()
    assert [i.gpu_class for i in store.all()] == ["standard", "standard"]


def test_no_power_offer_falls_back_to_default_bursts():
    mgr, vast, queue, store, clock = fleet()
    prime(mgr, clock)
    vast.offers = [o for o in vast.offers if o.get("num_gpus", 1) == 1]
    rush(queue, clock, 10)
    mgr.tick()
    assert len(created(vast)) == 3 and {i.gpu_class for i in store.all()} == {"standard"}


def test_power_server_is_destroyed_when_idle_and_never_becomes_the_warm():
    mgr, vast, queue, store, clock = fleet(warm_prime=False)
    rush(queue, clock, 10)
    mgr.tick()                                         # no warm yet: first create goes to the warm-eligible 3060, then power
    classes = sorted(i.gpu_class for i in store.all())
    assert classes == ["power", "standard"]
    pw = next(i for i in store.all() if i.gpu_class == "power")
    std = next(i for i in store.all() if i.gpu_class == "standard")
    for k in range(10):
        queue.set(f"j{k}", "completed", pw.label if k < 8 else std.label, clock())
    run_until(mgr, clock, 200)
    assert store.get(pw.id).state in ("destroying", "destroyed") and store.get(pw.id).role == "burst"
    assert store.get(std.id).state == "stopped" and store.get(std.id).role == "warm"


def test_dry_run_logs_the_power_choice_once():
    base = {"warm_prime": False, "max_instances": 3, "jobs_per_instance": 2, "power_enabled": True, "warm_instances": 0}
    mgr, vast, queue, store, clock = make(base, dry=True)
    vast.offers = [offer(1, 7), a40x2()]
    rush(queue, clock, 10)
    mgr.tick()
    mgr.tick()
    ev = [e for e in store.events() if e["event"].startswith("dry_run_create")]
    assert vast.n("create") == 0 and [e["event"] for e in ev] == ["dry_run_create_power"]
    assert ev[0]["detail"]["slots"] == 8 and ev[0]["detail"]["num_gpus"] == 2


def test_old_fleet_db_gets_the_new_columns(tmp_path):
    db = tmp_path / "fleet.db"
    c = sqlite3.connect(db)
    c.execute("CREATE TABLE fleet_instances (id INTEGER PRIMARY KEY, label TEXT NOT NULL UNIQUE, state TEXT NOT NULL, "
              "generation INTEGER NOT NULL DEFAULT 0, gpu_name TEXT, dph_base TEXT, storage_gb_month TEXT, inet_up_tb TEXT, "
              "inet_down_tb TEXT, disk_gb TEXT, release_sha TEXT, machine_id INTEGER, created_at REAL, running_since REAL, "
              "last_work_at REAL, stop_requested_at REAL, stopped_at REAL, resume_requested_at REAL, destroyed_at REAL, "
              "resume_failures INTEGER NOT NULL DEFAULT 0, note TEXT, role TEXT NOT NULL DEFAULT 'burst')")
    c.execute("INSERT INTO fleet_instances(id, label, state, role) VALUES (1, 'sonya-pool-old', 'stopped', 'warm')")
    c.commit()
    c.close()
    (i,) = FleetStore(str(db)).all()
    assert (i.slots, i.num_gpus, i.gpu_class, i.role) == (0, 1, "standard", "warm")
