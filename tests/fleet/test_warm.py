"""Warm / burst fleet (owner decision 2026-11-08) and the server-choice rule (RTX 3060 + Ryzen 7 5800X class)."""
import sqlite3

from dispatcher import launcher
from dispatcher.config import FleetConfig
from dispatcher.store import FleetStore
from tests.fleet.fakes import ENV, IMAGE, SHA, make, offer


def run_until(mgr, clock, secs, step=10):
    for _ in range(0, secs, step):
        mgr.tick()
        clock.advance(step)


def fleet(**over):
    base = {"warm_prime": True, "max_instances": 3, "jobs_per_instance": 2}
    return make({**base, **over})


def live(vast, store, state=None):
    return [i for i in store.all() if i.id in vast.inst and (state is None or i.state == state)]


def finish_all(queue, label_of, clock):
    for jid, lab in label_of.items():
        queue.set(jid, "completed", lab, clock())


# ------------------------------------------------------------------ warm priming
def test_no_warm_and_no_load_primes_one_and_stops_it_after_boot():
    mgr, vast, queue, store, clock = fleet()
    mgr.tick()
    assert vast.n("create") == 1
    (w,) = store.all()
    assert w.role == "warm"
    run_until(mgr, clock, 590)
    assert vast.n("stop") == 0                       # boot grace: image + release + models land on its disk first
    run_until(mgr, clock, 40)
    assert store.get(w.id).state == "stopped" and vast.n("create") == 1 and vast.n("destroy") == 0
    run_until(mgr, clock, 3 * 3600, step=600)        # stays paused, nothing else is created
    assert vast.n("create") == 1 and store.get(w.id).state == "stopped"


def test_warm_is_only_primed_on_the_default_class_with_cheap_storage():
    mgr, vast, queue, store, clock = fleet()
    vast.offers = [offer(1, 7, gpu="RTX 4060 Ti", cpu="AMD Ryzen 9 5950X 16-Core Processor", cores=16, dph=0.161),
                   offer(2, 8, storage=0.9)]          # 3060 but storage $0.9/GB-month: the pause would cost ~$36/month
    mgr.tick()
    assert vast.n("create") == 0 and any(e["event"] == "prime_failed" for e in store.events())
    run_until(mgr, clock, 300)
    assert vast.n("create") == 0                     # 10 min back-off, no create storm


def test_failed_prime_does_not_delay_a_real_job():
    """Production 2026-11-12: prime_failed (no EU warm offer) set the shared 10 min back-off and a user's job
    waited 5.5 min for a server although a burst offer was available."""
    mgr, vast, queue, store, clock = fleet()
    vast.offers = [offer(2, 8, storage=0.9)]          # burst-capable, not warm-eligible (pause too expensive)
    mgr.tick()
    assert vast.n("create") == 0 and any(e["event"] == "prime_failed" for e in store.events())
    clock.advance(20)
    queue.add("j1", ts=clock())
    mgr.tick()
    assert vast.n("create") == 1                     # the job gets a server at once
    run_until(mgr, clock, 300)
    assert sum(1 for e in store.events() if e["event"] == "prime_failed") == 1   # prime itself still backs off


def test_prime_off_and_warm_off_create_nothing_without_load():
    for over in ({"warm_prime": False}, {"warm_instances": 0}):
        mgr, vast, queue, store, clock = fleet(**over)
        run_until(mgr, clock, 120)
        assert vast.n("create") == 0


# ------------------------------------------------------------------ burst
def _stopped_warm(mgr, vast, queue, store, clock):
    mgr.tick()
    run_until(mgr, clock, 630)
    (w,) = store.all()
    assert store.get(w.id).state == "stopped"
    return w


def test_six_jobs_wake_the_warm_and_create_two_bursts_in_one_tick_then_bursts_are_destroyed():
    mgr, vast, queue, store, clock = fleet()
    vast.offers += [offer(2, 8), offer(3, 9, gpu="RTX 4060 Ti", cpu="AMD Ryzen 7 5700X3D 8-Core Processor", dph=0.169, geo="Ireland, IE")]
    w = _stopped_warm(mgr, vast, queue, store, clock)
    for k in range(6):
        queue.add(f"j{k}", ts=clock())
    mgr.tick()
    assert vast.n("resume") == 1 and vast.n("create") == 3       # 1 prime earlier + 2 bursts now: ceil(6/2)=3 servers
    bursts = [i for i in store.all() if i.id != w.id]
    assert all(b.role == "burst" for b in bursts)
    assert len({b.machine_id for b in bursts}) == 2              # never two creates on the same offer in one tick
    mgr.tick()
    labels = {i.label for i in store.all()}
    owner = {f"j{k}": sorted(labels)[k % 3] for k in range(6)}
    for jid, lab in owner.items():
        queue.set(jid, "mode_running", lab, clock())
    run_until(mgr, clock, 300)
    assert vast.n("stop") == 1 and vast.n("destroy") == 0       # nobody idles while working
    finish_all(queue, owner, clock)
    run_until(mgr, clock, 90)
    assert store.get(w.id).state == "stopped" and store.get(w.id).role == "warm"
    assert all(store.get(b.id).state == "destroyed" for b in bursts) and vast.n("destroy") == 2
    assert vast.n("stop") == 2                                    # the warm only: prime stop + this one
    assert set(vast.inst) == {w.id}


def test_burst_cap_is_max_instances():
    mgr, vast, queue, store, clock = fleet(max_instances=2)
    vast.offers += [offer(k, 100 + k) for k in range(2, 6)]
    for k in range(9):
        queue.add(f"j{k}", ts=clock())
    run_until(mgr, clock, 60)
    assert len(vast.inst) == 2


def test_first_instance_under_load_becomes_the_warm_but_a_fallback_host_never_does():
    mgr, vast, queue, store, clock = fleet(warm_prime=False)
    vast.offers = [offer(1, 7, gpu="RTX 4060 Ti", cpu="AMD Ryzen 9 5950X 16-Core Processor", cores=16, dph=0.161, geo="Germany, DE")]
    queue.add("j1", ts=clock())
    mgr.tick(); mgr.tick()
    (b,) = store.all()
    assert b.role == "burst" and b.note == ""                    # no 3060 offered: fallback class serves the job
    queue.set("j1", "completed", b.label, clock())
    run_until(mgr, clock, 80)
    assert store.get(b.id).state == "destroyed" and vast.n("stop") == 0
    vast.offers = [offer(2, 8)]
    queue.add("j2", ts=clock())
    mgr.tick(); mgr.tick()
    w = [i for i in store.all() if i.id != b.id][0]
    assert w.note == "warm_eligible"
    queue.set("j2", "completed", w.label, clock())
    run_until(mgr, clock, 80)
    assert store.get(w.id).state == "stopped" and store.get(w.id).role == "warm"


# ------------------------------------------------------------------ warm cannot wake up
def test_warm_that_cannot_resume_is_replaced_and_the_new_one_becomes_warm():
    mgr, vast, queue, store, clock = fleet()
    vast.offers.append(offer(2, 8, dph=0.09))
    w = _stopped_warm(mgr, vast, queue, store, clock)
    vast.resume_gets_gpu = False                                  # host rented by somebody else
    queue.add("j1", ts=clock())
    mgr.tick()
    assert store.get(w.id).state == "starting"
    run_until(mgr, clock, 110)                                    # resume_timeout 90 s, then ONE failure is enough
    assert store.get(w.id).state in ("destroying", "destroyed") and w.id not in vast.inst
    assert 7 in store.banned_machines(clock())
    assert any(e["event"] == "warm_replaced" for e in store.events(w.id))
    (n,) = [i for i in store.all() if i.id in vast.inst]
    assert n.machine_id == 8 and n.note == "warm_eligible"
    queue.set("j1", "completed", n.label, clock())
    run_until(mgr, clock, 80)
    assert store.get(n.id).state == "stopped" and store.get(n.id).role == "warm"
    assert vast.n("create") == 2 and set(vast.inst) == {n.id}  # exactly one paused server again


def test_stale_release_replaces_the_warm():
    mgr, vast, queue, store, clock = fleet()
    w = _stopped_warm(mgr, vast, queue, store, clock)
    mgr.cfg = FleetConfig(enabled=True, dry_run=False, image=IMAGE, release_sha="c" * 40, max_instances=3, jobs_per_instance=2)
    mgr.tick(); mgr.tick()
    assert store.get(w.id).state == "destroyed"
    assert vast.n("create") == 2 and [i for i in store.all() if i.id in vast.inst][0].release_sha == "c" * 40


def test_dry_run_prime_sends_nothing():
    mgr, vast, queue, store, clock = make({"warm_prime": True}, dry=True)
    run_until(mgr, clock, 60)
    assert vast.n("create") == 0 and any(e["event"] == "dry_run_prime" for e in store.events())


# ------------------------------------------------------------------ server choice
CFG = FleetConfig(image=IMAGE, release_sha=SHA)
A_UK = offer(1, 1)                                                                          # 3060 + 5800X UK $0.082
XEON = offer(2, 2, cpu="Xeon® E5-2680 v4", cores=14, dph=0.069, geo="Japan, JP")           # cheaper, old CPU
WEAK = offer(3, 3, cpu="AMD Ryzen 5 5600X 6-Core Processor", cores=6, dph=0.075, geo="Poland, PL")
B_IE = offer(4, 4, gpu="RTX 4060 Ti", cpu="AMD Ryzen 7 5700X3D 8-Core Processor", dph=0.169, geo="Ireland, IE", down=2317)
C_DE = offer(5, 5, gpu="RTX 4060 Ti", cpu="AMD Ryzen 9 5950X 16-Core Processor", cores=16, dph=0.161, geo="Germany, DE", down=892)
US_3060 = offer(6, 6, cpu="12th Gen Core™ i7-12700", cores=20, dph=0.135, geo="Maryland, US", down=1490)


def test_rule_default_class_first_fallback_only_without_it():
    assert launcher.pick_offer([XEON, WEAK, B_IE, C_DE, A_UK], CFG, set())["id"] == 1
    assert launcher.pick_offer([XEON, WEAK, B_IE, C_DE], CFG, set())["id"] == 5             # cheaper of B/C
    assert launcher.pick_offer([XEON, WEAK], CFG, set()) is None                            # old / 6-core CPUs never
    assert launcher.pick_offer([B_IE, US_3060], CFG, set())["id"] == 4                      # Europe first (2.5x faster than Asia/US to RU S3)
    assert launcher.pick_offer([US_3060], CFG, set())["id"] == 6                            # outside Europe only when nothing in Europe
    assert not launcher.warm_eligible(US_3060, CFG) and launcher.warm_eligible(A_UK, CFG)  # the warm must be a European 3060


def test_rule_europe_cached_host_internet_price_reliability():
    a_us = offer(7, 7, dph=0.080, geo="Texas, US")
    a_de = offer(12, 12, dph=0.090, geo="Germany, DE")
    assert launcher.pick_offer([a_us, A_UK], CFG, set())["id"] == 1                         # Europe over $0.002/h
    assert launcher.pick_offer([a_us, A_UK], CFG, set(), known_machines={7})["id"] == 1     # a cache outside Europe does not win
    assert launcher.pick_offer([a_de, A_UK], CFG, set(), known_machines={12})["id"] == 12   # image already cached there
    assert launcher.pick_offer([offer(8, 8, down=500)], CFG, set()) is None                 # < 800 Mbit/s
    assert launcher.pick_offer([offer(9, 9, rel=0.95)], CFG, set()) is None
    assert launcher.pick_offer([offer(10, 10, dph=0.25)], CFG, set()) is None               # over the price cap
    assert launcher.pick_offer([A_UK], CFG, {1}) is None                                     # banned machine
    assert launcher.pick_offer([A_UK, offer(11, 11, storage=0.6)], CFG, set(), exclude={1}, warm=True) is None


def test_cpu_families():
    for name in ("AMD Ryzen 7 5800X 8-Core Processor", "AMD Ryzen 9 7950X3D 16-Core Processor", "AMD Ryzen 7 5700X3D 8-Core Processor",
                 "12th Gen Core™ i5-12400F", "Core™ i7-14700", "AMD Ryzen Threadripper PRO 5975WX 32-Cores", "AMD Ryzen 9 9900X 12-Core Processor"):
        assert launcher.cpu_modern({"cpu_name": name}), name
    for name in ("Xeon® E5-2680 v4", "AMD Ryzen 9 3900X 12-Core Processor", "AMD Ryzen Threadripper PRO 3945WX 12-Cores",
                 "AMD EPYC 7452 32-Core Processor", "Core™ i7-4930K", "11th Gen Core™ i7-11700KF", ""):
        assert not launcher.cpu_modern({"cpu_name": name}), name


def test_search_query_and_payload():
    q = launcher.search_query(CFG)
    assert "RTX 3060" in q["gpu_name"]["in"] and q["inet_down"]["gte"] == 800 and q["gpu_ram"]["gte"] <= 12288
    assert q["disk_space"]["gte"] == 30 and q["dph_total"]["lte"] == 0.18 and q["cpu_ram"]["gte"] == 16000
    p = launcher.create_payload(CFG, "sonya-pool-x", "tok", ENV)
    assert p["disk"] == 30
    assert p["env"]["NVIDIA_DRIVER_CAPABILITIES"] == "compute,utility,video"            # NVENC inside the container
    assert p["env"]["SONYA_WORKER_CONCURRENCY"] == str(CFG.jobs_per_instance)


def test_production_defaults():
    c = FleetConfig.from_env({})
    assert (c.warm_instances, c.warm_prime, c.max_instances, c.jobs_per_instance, c.max_resume_failures, c.disk_gb) == (1, True, 3, 2, 1, 30)
    c = FleetConfig.from_env({"SONYA_WARM_INSTANCES": "0", "SONYA_JOBS_PER_INSTANCE": "3", "SONYA_DISK_GB": "35"})
    assert (c.warm_instances, c.jobs_per_instance, c.disk_gb) == (0, 3, 35)


def test_old_fleet_db_without_role_column_is_migrated(tmp_path):
    db = tmp_path / "fleet.db"
    c = sqlite3.connect(db)
    c.executescript("""CREATE TABLE fleet_instances (id INTEGER PRIMARY KEY, label TEXT NOT NULL UNIQUE, state TEXT NOT NULL,
        generation INTEGER NOT NULL DEFAULT 0, gpu_name TEXT, dph_base TEXT, storage_gb_month TEXT, inet_up_tb TEXT, inet_down_tb TEXT,
        disk_gb TEXT, release_sha TEXT, machine_id INTEGER, created_at REAL, running_since REAL, last_work_at REAL,
        stop_requested_at REAL, stopped_at REAL, resume_requested_at REAL, destroyed_at REAL,
        resume_failures INTEGER NOT NULL DEFAULT 0, note TEXT);
        INSERT INTO fleet_instances(id, label, state) VALUES (1, 'sonya-pool-old', 'stopped');""")
    c.commit(); c.close()
    s = FleetStore(str(db))
    assert s.get(1).role == "burst" and s.get(1).state == "stopped"


def test_burst_that_got_no_job_is_destroyed_after_the_idle_timeout_not_the_boot_grace():
    mgr, vast, queue, store, clock = fleet()
    vast.offers += [offer(2, 8), offer(3, 9)]
    w = _stopped_warm(mgr, vast, queue, store, clock)
    for k in range(6):
        queue.add(f"j{k}", ts=clock())
    mgr.tick(); mgr.tick()
    wl = store.get(w.id).label
    for k in range(6):                                  # the warm wakes first and takes everything
        queue.set(f"j{k}", "completed", wl, clock())
    run_until(mgr, clock, 80)
    bursts = [i for i in store.all() if i.id != w.id]
    assert all(store.get(b.id).state == "destroyed" for b in bursts) and store.get(w.id).state == "stopped"


def test_warm_fallback_switch():
    assert not launcher.warm_eligible(C_DE, CFG)
    on = FleetConfig(image=IMAGE, release_sha=SHA, warm_fallback=True)
    assert launcher.warm_eligible(C_DE, on) and not launcher.warm_eligible(US_3060, on)
    assert launcher.pick_offer([C_DE, A_UK], on, set(), warm=True)["id"] == 1          # a European 3060 still wins
    assert FleetConfig.from_env({"SONYA_WARM_FALLBACK": "true"}).warm_fallback


def test_create_payload_public_repo_carries_no_github_credential():
    p = launcher.create_payload(CFG, "sonya-pool-x", "", ENV)
    assert "GITHUB_RELEASE_TOKEN" not in p["env"] and p["env"]["SONYA_RELEASE_SHA"] == CFG.release_sha
    assert launcher.create_payload(CFG, "sonya-pool-x", "tok", ENV)["env"]["GITHUB_RELEASE_TOKEN"] == "tok"
