"""Spare offers per class + the paused POWER reserve (owner decisions 2026-11-09). Both OFF by default: then the fleet does
exactly what it did before (one offer per create, no extra searches, no reserve)."""
from decimal import Decimal

import pytest

from dispatcher import launcher
from dispatcher.config import FleetConfig
from dispatcher.status import status_text
from tests.fleet.fakes import make, offer
from tests.fleet.test_power import a40x2


def run_until(mgr, clock, secs, step=10):
    for _ in range(0, secs, step):
        mgr.tick()
        clock.advance(step)


def rush(queue, clock, n, prefix="j"):
    for k in range(n):
        queue.add(f"{prefix}{k}", ts=clock())


def events(store, name):
    return [e for e in store.events() if e["event"] == name]


OFFERS = [offer(1, 7), offer(2, 8, geo="Germany, DE", dph=0.09), offer(3, 9, geo="Spain, ES", dph=0.095),
          offer(4, 10, geo="Texas, US", dph=0.06),
          offer(5, 11, gpu="RTX 4060 Ti", cpu="AMD Ryzen 9 5950X 16-Core Processor", cores=16, dph=0.10, geo="Poland, PL")]
POWER = [a40x2(50, 500), a40x2(51, 501, dph=0.80), a40x2(52, 502, dph=0.60, storage=0.30)]   # 52: pause 30x0.30x82 = 738 RUB


def fleet(**over):
    base = {"warm_prime": True, "max_instances": 3, "jobs_per_instance": 2}
    mgr, vast, queue, store, clock = make({**base, **over})
    vast.offers = OFFERS + POWER
    return mgr, vast, queue, store, clock


# ── config ────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_both_options_are_off_by_default_and_env_parsing():
    c = FleetConfig.from_env({})
    assert (c.spares_enabled, c.power_reserve, c.spares_count, c.spares_refresh_seconds) == (False, False, 5, 300)
    assert c.reserve_wake_at() == 3 * 2 + 1 and c.power_reserve_max_storage_rub == 500 and c.power_jobs_per_instance == 8
    c = FleetConfig.from_env({"SONYA_SPARES_ENABLED": "true", "SONYA_SPARES_COUNT": "3", "SONYA_SPARES_REFRESH_SECONDS": "120",
                              "SONYA_POWER_RESERVE": "1", "SONYA_POWER_RESERVE_WAKE_AT": "5",
                              "SONYA_POWER_RESERVE_MAX_STORAGE_RUB": "300", "SONYA_USD_RUB": "90"})
    assert (c.spares_enabled, c.spares_count, c.spares_refresh_seconds, c.power_reserve, c.reserve_wake_at(),
            c.power_reserve_max_storage_rub, c.usd_rub) == (True, 3, 120, True, 5, 300, Decimal("90"))
    with pytest.raises(ValueError):
        FleetConfig.from_env({"SONYA_POWER_RESERVE": "true", "SONYA_POWER_GPUS": " "})
    with pytest.raises(ValueError):
        FleetConfig.from_env({"SONYA_SPARES_COUNT": "0"})


# ── ranking ───────────────────────────────────────────────────────────────────────────────────────────────────────
def test_ranking_europe_class_cache_price_and_no_banned_hosts():
    cfg = FleetConfig()
    ranked = launcher.rank_offers(OFFERS, cfg, set())
    assert [o["id"] for o in ranked] == [1, 2, 3, 5, 4]           # EU 3060s by price, EU 4060 Ti, then the US 3060
    assert launcher.pick_offer(OFFERS, cfg, set())["id"] == ranked[0]["id"]
    assert [o["id"] for o in launcher.rank_offers(OFFERS, cfg, {7, 9})] == [2, 5, 4]          # banned hosts left out
    assert [o["id"] for o in launcher.rank_offers(OFFERS, cfg, set(), {9})][:3] == [3, 1, 2]  # image cached on 9: first
    assert [o["id"] for o in launcher.rank_offers(OFFERS, cfg, set(), {10})][-1] == 4         # cache never beats Europe
    assert [o["id"] for o in launcher.rank_offers(OFFERS, cfg, set(), warm=True)] == [1, 2, 3]  # warm: EU 3060 only


def test_power_ranking_and_reserve_storage_cap():
    cfg = FleetConfig(power_reserve=True)
    assert [o["id"] for o in launcher.rank_power_offers(POWER, cfg, set())] == [52, 50, 51]   # price per video slot
    assert [o["id"] for o in launcher.rank_power_offers(POWER, cfg, set(), reserve=True)] == [50, 51]   # 52: 738 RUB > 500
    assert launcher.pause_rub_month(POWER[0], cfg) == Decimal("369.00")                       # 30 GB x $0.15 x 82
    assert [o["id"] for o in launcher.rank_power_offers(POWER, cfg, set(), {501}, reserve=True)] == [51, 50]   # cached first
    assert launcher.pick_power_offer(POWER, cfg, {502})["id"] == 50


# ── the spares list ───────────────────────────────────────────────────────────────────────────────────────────────
def test_spares_list_is_refreshed_every_5_min_and_logged_when_it_changes():
    mgr, vast, queue, store, clock = fleet(spares_enabled=True, spares_count=3, warm_prime=False, power_enabled=True)
    mgr.tick()
    sp = store.spares()
    assert [o["offer"] for o in sp["warm"]] == [1, 2, 3] and [o["offer"] for o in sp["burst"]] == [1, 2, 3]
    assert [o["offer"] for o in sp["power"]] == [52, 50, 51] and "reserve" not in sp
    assert {e["detail"]["cls"] for e in events(store, "spares")} == {"warm", "burst", "power"}
    searches = vast.n("search")
    run_until(mgr, clock, 290)
    assert vast.n("search") == searches                               # no new search inside the 5 minutes
    store.ban_machine(7, clock() + 3600, "test")
    vast.offers = vast.offers + [offer(9, 99, geo="Germany, DE", dph=0.085)]
    run_until(mgr, clock, 20)
    sp = store.spares()
    assert [o["offer"] for o in sp["burst"]] == [9, 2, 3]            # banned host gone, a new cheaper one in
    assert len(events(store, "spares")) == 5                          # warm + burst changed, power did not
    assert all("LEAKCHECK" not in str(e) for e in store.events())


def test_status_shows_servers_and_spares():
    mgr, vast, queue, store, clock = fleet(spares_enabled=True, spares_count=2, power_reserve=True)
    run_until(mgr, clock, 30)
    txt = status_text(store, clock())
    assert "Servers:" in txt and "warm" in txt and "Spare offers" in txt
    assert "1. offer 1 machine 7: 1x RTX 3060, GB, $0.082/h" in txt
    assert "power reserve" in txt and "offer 50 machine 500: 2x A40, BE" in txt
    vast.offers = OFFERS
    clock.advance(400)
    mgr.tick()
    assert "(no acceptable offer right now)" in status_text(store, clock())
    empty = make()[3]
    assert "no list yet" in status_text(empty, 0)


# ── next offer when Vast refuses ──────────────────────────────────────────────────────────────────────────────────
def test_refused_create_goes_straight_to_the_next_offer():
    mgr, vast, queue, store, clock = fleet(spares_enabled=True, warm_prime=False)
    vast.fail_offers = {1, 2}
    rush(queue, clock, 1)
    mgr.tick()
    assert vast.created_offers == [3]                                 # same tick: 1 and 2 refused, 3 created
    assert [e["detail"]["offer"] for e in events(store, "create_failed_next")] == [1, 2]
    (c,) = events(store, "created")
    assert c["detail"]["attempt"] == 3 and not events(store, "create_failed")


def test_refused_create_gives_up_after_spares_count_tries():
    mgr, vast, queue, store, clock = fleet(spares_enabled=True, spares_count=2, warm_prime=False)
    vast.fail_offers = {1, 2, 3}
    rush(queue, clock, 1)
    mgr.tick()
    assert vast.created_offers == [] and vast.n("create_refused") == 2 and len(events(store, "create_failed")) == 1


def test_spares_off_one_try_only_as_before():
    mgr, vast, queue, store, clock = fleet(warm_prime=False)
    vast.fail_offers = {1}
    rush(queue, clock, 1)
    mgr.tick()
    assert vast.created_offers == [] and vast.n("create_refused") == 1
    assert len(events(store, "create_failed")) == 1 and not events(store, "create_failed_next")
    clock.advance(61)
    mgr.tick()
    assert vast.n("create_refused") == 2                              # retried after the old 60 s back-off


def test_host_with_our_image_cached_is_preferred():
    mgr, vast, queue, store, clock = fleet(spares_enabled=True, warm_prime=False, warm_instances=0)
    vast.offers = [offer(2, 8, geo="Germany, DE", dph=0.09)]
    rush(queue, clock, 1)
    run_until(mgr, clock, 20)                                         # booted on machine 8 -> remembered in fleet.db
    assert 8 in store.good_machines()
    first = store.all()[0]
    queue.set("j0", "completed", first.label, clock())
    vast.destroy(first.id)                                            # gone (e.g. destroyed by hand): the next one is a create
    run_until(mgr, clock, 30)
    vast.offers = OFFERS                                              # 1 (machine 7) is cheaper, 8 has the image
    rush(queue, clock, 1, prefix="k")
    mgr.tick()
    assert vast.created_offers[-1] == 2
    assert events(store, "created")[-1]["detail"]["cached"] is True


def test_warm_resume_refused_is_replaced_in_the_same_tick():
    for spares, created_now in ((True, 2), (False, 1)):
        mgr, vast, queue, store, clock = fleet(spares_enabled=spares)
        run_until(mgr, clock, 700)
        (w,) = store.all()
        assert w.state == "stopped" and w.role == "warm"
        vast.fail_resume = True
        rush(queue, clock, 1)
        mgr.tick()
        assert len(vast.created_offers) == created_now
        if spares:
            assert store.get(w.id).state in ("destroying", "destroyed") and 7 in store.banned_machines(clock())
            assert vast.created_offers[-1] == 2                       # the next European 3060 (7 is banned now)
            assert events(store, "warm_replaced")
        else:
            assert store.get(w.id).state == "stopped"                 # old behaviour: counted, waits for the next tick


# ── the paused POWER reserve ──────────────────────────────────────────────────────────────────────────────────────
def reserve_fleet(**over):
    mgr, vast, queue, store, clock = fleet(power_reserve=True, **over)
    run_until(mgr, clock, 800)
    roles = {i.role: i for i in store.all()}
    assert set(roles) == {"warm", "reserve"} and all(i.state == "stopped" for i in roles.values())
    return mgr, vast, queue, store, clock, roles["warm"], roles["reserve"]


def test_reserve_is_primed_after_the_warm_and_paused_on_a_cheap_storage_host():
    mgr, vast, queue, store, clock, w, r = reserve_fleet()
    assert r.gpu_class == "power" and r.machine_id == 500 and r.slots == 8 and r.num_gpus == 2   # 502 pauses too dear
    assert vast.created_offers == [1, 50] and vast.n("destroy") == 0
    p = [c for c in vast.calls if c[0] == "create"][1][2]
    assert p["env"]["SONYA_WORKER_CONCURRENCY"] == "auto" and p["env"]["SONYA_WORKER_CONCURRENCY_MAX"] == "8"
    run_until(mgr, clock, 3 * 3600, step=600)                         # kept paused: retention does not remove it
    assert store.get(r.id).state == "stopped" and vast.n("create") == 2


def test_small_load_does_not_wake_the_reserve():
    mgr, vast, queue, store, clock, w, r = reserve_fleet()
    rush(queue, clock, 6)                                             # warm + 2 bursts carry 6
    mgr.tick()
    assert store.get(r.id).state == "stopped" and store.get(w.id).state == "starting"
    assert len(vast.created_offers) == 4


def test_rush_wakes_the_reserve_and_it_sleeps_again_after():
    mgr, vast, queue, store, clock, w, r = reserve_fleet()
    rush(queue, clock, 10)                                            # >= 7 = 3 x 2 + 1
    mgr.tick()
    assert ("resume", r.id) in vast.calls and ("resume", w.id) in vast.calls
    assert len(vast.created_offers) == 2                              # 8 + 2 slots cover 10: no burst
    assert events(store, "reserve_wake")
    for k in range(10):
        queue.set(f"j{k}", "completed", store.get(r.id).label if k < 8 else store.get(w.id).label, clock())
    run_until(mgr, clock, 200)
    assert store.get(r.id).state == "stopped" and store.get(r.id).role == "reserve"
    assert store.get(w.id).state == "stopped" and vast.n("destroy") == 0


def test_reserve_that_does_not_wake_in_90s_is_destroyed_and_the_next_one_created():
    mgr, vast, queue, store, clock, w, r = reserve_fleet()
    real_resume = vast.resume

    def resume(iid):
        real_resume(iid)
        if iid == r.id:
            vast.inst[iid]["status"] = "scheduling"                  # its GPUs were rented by someone else
    vast.resume = resume
    rush(queue, clock, 10)
    run_until(mgr, clock, 90)
    assert store.get(r.id).state == "starting" and len(vast.created_offers) == 2
    run_until(mgr, clock, 20)
    assert store.get(r.id).state in ("destroying", "destroyed") and 500 in store.banned_machines(clock())
    new = [i for i in store.all() if i.role == "reserve" and i.id != r.id]
    assert len(new) == 1 and new[0].machine_id == 501 and vast.created_offers[-1] == 51
    assert events(store, "reserve_replaced")


def test_reserve_resume_refused_is_replaced_at_once():
    mgr, vast, queue, store, clock, w, r = reserve_fleet()
    real_resume = vast.resume

    def resume(iid):
        if iid == r.id:
            from dispatcher.vast_api import VastError
            raise VastError("resume refused", 400)
        real_resume(iid)
    vast.resume = resume
    rush(queue, clock, 10)
    mgr.tick()
    assert store.get(r.id).state in ("destroying", "destroyed") and vast.created_offers[-1] == 51
    assert [i.machine_id for i in store.all() if i.role == "reserve" and i.state == "creating"] == [501]


def test_reserve_switched_off_later_goes_away():
    mgr, vast, queue, store, clock, w, r = reserve_fleet()
    mgr.cfg = FleetConfig(**{**mgr.cfg.__dict__, "power_reserve": False})
    run_until(mgr, clock, 1900, step=100)
    assert store.get(r.id).state in ("destroying", "destroyed") and store.get(w.id).state == "stopped"


def test_dry_run_logs_the_reserve_choice_once():
    mgr, vast, queue, store, clock = make({"warm_prime": False, "power_reserve": True, "max_instances": 3}, dry=True)
    vast.offers = OFFERS + POWER
    run_until(mgr, clock, 60)
    ev = events(store, "dry_run_create_reserve")
    assert vast.n("create") == 0 and len(ev) == 1 and ev[0]["detail"]["offer"] == 50


# ── nothing changes when both are off ─────────────────────────────────────────────────────────────────────────────
def test_options_off_nothing_new_happens():
    mgr, vast, queue, store, clock = fleet()
    run_until(mgr, clock, 800)
    rush(queue, clock, 10)
    run_until(mgr, clock, 30)
    names = {e["event"] for e in store.events()}
    assert not names & {"spares", "spares_failed", "create_failed_next", "reserve_wake", "reserve_failed", "reserve_replaced"}
    assert store.spares() == {} and {i.role for i in store.all()} <= {"warm", "burst"}
    assert {i.gpu_class for i in store.all()} == {"standard"}
    assert vast.n("search") == vast.n("create")                       # exactly one offer search per create, as before
