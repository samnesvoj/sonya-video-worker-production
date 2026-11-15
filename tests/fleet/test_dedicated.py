"""Dedicated servers (CEO 2026-11-16): a paying user (cut_pro / cut_studio / streamer_start), an admin or a legacy Pro gets a
server of their own. The API claim enforces who may join a server; the fleet must count each such user as ONE WHOLE server.
Off by default = exactly the old slot count."""
from dispatcher import launcher
from dispatcher.config import FleetConfig
from dispatcher.queue import QueueSnapshot, dedicated_params, fold_dedicated
from tests.fleet.fakes import make, offer
from tests.fleet.test_power import a40x2


def created(vast):
    return [c for c in vast.calls if c[0] == "create"]


def fleet(**over):
    base = {"warm_prime": False, "max_instances": 6, "jobs_per_instance": 2, "dedicated": True}
    mgr, vast, queue, store, clock = make({**base, **over})
    vast.offers = [offer(1, 7), offer(2, 8, dph=0.09), offer(3, 9, dph=0.095), offer(4, 10, dph=0.1), offer(5, 11, dph=0.11)]
    return mgr, vast, queue, store, clock


def test_env_parsing_and_off_by_default():
    assert FleetConfig().dedicated is False and FleetConfig.from_env({}).dedicated is False
    cfg = FleetConfig.from_env({"SONYA_DEDICATED": "on", "SONYA_DEDICATED_MAX_INSTANCES": "3"})
    assert cfg.dedicated and cfg.dedicated_max_instances == 3
    p = dedicated_params({})
    assert p["ded_plans"] == ["cut_pro", "cut_studio", "streamer_start"] and p["ded_admin"] and p["ded_legacy"]
    assert "sonya_gen" not in p["ded_modes"]


def test_fold_counts_users_and_their_servers():
    s = QueueSnapshot()
    fold_dedicated(s, [("u1", "queued", None), ("u1", "mode_running", "w1"), ("u2", "queued", None)])
    assert (s.dedicated_users, s.dedicated_jobs, s.dedicated_workers) == (2, 3, {"w1"})


def test_off_is_the_old_slot_count():
    mgr, vast, queue, store, clock = fleet(dedicated=False)
    queue.dedicated_user_ids = {"pro"}
    for k in range(3):
        queue.add(f"p{k}", user_id="pro")
    mgr.tick()
    assert len(created(vast)) == 2                      # 3 jobs / 2 slots


def test_three_paying_users_get_three_servers_and_shared_jobs_their_own():
    mgr, vast, queue, store, clock = fleet()
    queue.dedicated_user_ids = {"pro1", "pro2", "studio"}
    queue.add("a", user_id="pro1")
    queue.add("b", user_id="pro2")
    queue.add("c", user_id="studio")
    queue.add("d", user_id="studio")                    # the same user: on the same own server
    queue.add("f1", user_id="free1")
    queue.add("f2", user_id="free2")
    mgr.tick()
    assert len(created(vast)) == 4                      # 3 own servers + 1 shared (2 free jobs, 2 slots)
    mgr.tick()
    assert len(created(vast)) == 4                      # idempotent


def test_paying_user_never_gets_a_power_server():
    mgr, vast, queue, store, clock = fleet(power_enabled=True)
    vast.offers.append(a40x2())
    queue.dedicated_user_ids = {"pro"}
    queue.add("p", user_id="pro")
    for k in range(9):
        queue.add(f"f{k}", user_id=f"free{k}")
    mgr.tick()
    classes = [i.gpu_class for i in store.all()]
    assert classes[0] == "standard"                     # the paying user's server first, default class
    assert "power" in classes                           # the rush of 9 shared jobs still gets the power server


def test_a_big_server_holding_a_paying_user_counts_as_one_server():
    mgr, vast, queue, store, clock = fleet(power_enabled=True)
    vast.offers = [a40x2()] + vast.offers
    queue.add("f0", user_id="free0")
    for k in range(1, 5):
        queue.add(f"f{k}", user_id=f"free{k}")
    mgr.tick()
    (pw,) = [i for i in store.all() if i.gpu_class == "power"]
    vast.set_running(pw.id) if hasattr(vast, "set_running") else None
    # the power server took the paying user's job; 5 shared jobs are now waiting
    queue.dedicated_user_ids = {"pro"}
    queue.add("p", status="mode_running", worker_id=pw.label, user_id="pro")
    n = len(created(vast))
    mgr.tick()
    assert len(created(vast)) > n                       # its 8 slots no longer count for the shared jobs


def test_cap_on_own_servers():
    mgr, vast, queue, store, clock = fleet(dedicated_max_instances=2)
    queue.dedicated_user_ids = {"u1", "u2", "u3", "u4"}
    for u in ("u1", "u2", "u3", "u4"):
        queue.add(u, user_id=u)
    mgr.tick()
    assert len(created(vast)) == 2                      # the 3rd / 4th wait for a free server, first in line


def test_power_ceiling_never_above_the_hardware():
    a6000 = a40x2(gpu="RTX A6000", cores=30.72)
    assert launcher.offer_slots(a6000, FleetConfig(power_jobs_per_instance=32), power=True) == 7
    assert launcher.offer_slots(a40x2(cores=123.0), FleetConfig(power_jobs_per_instance=32), power=True) == 16


def test_warm_may_be_an_old_cpu_european_host_only_when_allowed():
    xeon = offer(9, 99, cpu="Xeon® E5-2697A v4", cores=16.0, geo="Norway, NO", storage=0.133)
    small = offer(8, 98, cpu="AMD Ryzen 7 3700X 8-Core Processor", cores=8.0, geo="Spain, ES", storage=0.2)
    assert launcher.offer_class(xeon, FleetConfig()) == 2
    assert not launcher.warm_eligible(xeon, FleetConfig())                         # default: unchanged
    on = FleetConfig(warm_old_cpu=True)
    assert launcher.warm_eligible(xeon, on) and not launcher.warm_eligible(small, on)   # 8 cores < 12
    assert not launcher.warm_eligible(offer(7, 97, cpu="Xeon® E5-2697A v4", cores=16.0, geo="Vietnam, VN"), on)
    assert FleetConfig.from_env({"SONYA_WARM_OLD_CPU": "true"}).warm_old_cpu
