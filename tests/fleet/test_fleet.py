import json

import pytest

from dispatcher import launcher
from dispatcher.config import FleetConfig
from tests.fleet.fakes import ENV, GH_TOKEN, IMAGE, SHA, make, offer


def run_until(mgr, clock, secs, step=10):
    for _ in range(0, secs, step):
        mgr.tick()
        clock.advance(step)


def boot_one(mgr, vast, queue, clock):
    """job arrives -> instance created, running."""
    queue.add("j1", ts=clock())
    mgr.tick()
    clock.advance(1)
    mgr.tick()
    (iid,) = vast.inst
    return iid, vast.inst[iid]["label"]


def test_creates_instance_on_demand_without_secret_leaks(tmp_path):
    mgr, vast, queue, store, clock = make()
    queue.add("j1", ts=clock())
    mgr.tick()
    assert vast.n("create") == 1
    payload = [c for c in vast.calls if c[0] == "create"][0][2]
    assert payload["env"]["SONYA_RELEASE_SHA"] == SHA and payload["image"] == IMAGE
    assert payload["env"]["ENABLED_MODES"] == "virality,beep" and "trailer" not in json.dumps(payload["env"]).lower()
    blob = json.dumps(store.events()) + json.dumps(launcher.redacted(payload))
    for leak in (GH_TOKEN, ENV["WORKER_SECRET"], ENV["S3_SECRET_ACCESS_KEY"], ENV["S3_ACCESS_KEY_ID"]):
        assert leak not in blob


def test_idle_stop_exactly_after_60s_from_last_job(monkeypatch):
    mgr, vast, queue, store, clock = make()
    iid, label = boot_one(mgr, vast, queue, clock)
    queue.set("j1", "mode_running", label, clock())
    mgr.tick()
    clock.advance(300)
    queue.set("j1", "completed", label, clock())
    done = clock()
    clock.advance(59)
    mgr.tick()
    assert vast.n("stop") == 0                      # 59 s: not yet
    clock.advance(2)
    mgr.tick()
    assert vast.n("stop") == 1 and store.get(iid).state == "stopping"
    assert clock() - done >= 60
    mgr.tick()
    assert store.get(iid).state == "stopped"


def test_active_upload_blocks_stop_and_timer_starts_after_confirmation():
    mgr, vast, queue, store, clock = make()
    iid, label = boot_one(mgr, vast, queue, clock)
    queue.set("j1", "uploading_result", label, clock())
    clock.advance(500)
    mgr.tick()
    assert vast.n("stop") == 0                      # still uploading, however long it takes
    queue.set("j1", "completed", label, clock())
    mgr.tick()
    clock.advance(30)
    mgr.tick()
    assert vast.n("stop") == 0
    clock.advance(31)
    mgr.tick()
    assert vast.n("stop") == 1


def test_queued_job_prevents_stop_and_back_to_back_jobs_need_no_restart():
    mgr, vast, queue, store, clock = make()
    iid, label = boot_one(mgr, vast, queue, clock)
    queue.set("j1", "completed", label, clock())
    queue.add("j2", ts=clock())                     # next job already waiting
    clock.advance(120)
    mgr.tick()
    assert vast.n("stop") == 0 and vast.n("create") == 1
    queue.set("j2", "mode_running", label, clock())
    queue.set("j2", "completed", label, clock())
    clock.advance(61)
    mgr.tick()
    assert vast.n("stop") == 1 and vast.n("create") == 1 and vast.n("resume") == 0


def test_race_job_arrives_between_decision_and_stop_cancels_stop():
    mgr, vast, queue, store, clock = make()
    iid, label = boot_one(mgr, vast, queue, clock)
    queue.set("j1", "completed", label, clock())
    clock.advance(61)
    real = queue.snapshot
    calls = {"n": 0}

    def snap(modes, prefix):
        calls["n"] += 1
        if calls["n"] == 2:                          # the fenced re-check sees a brand new job
            queue.add("j2", ts=clock())
        return real(modes, prefix)
    queue.snapshot = snap
    mgr.tick()
    assert vast.n("stop") == 0 and store.get(iid).state == "running"
    assert any(e["event"] == "stop_cancelled" for e in store.events(iid))


def test_stop_failure_reverts_to_running_and_is_retried_next_tick():
    mgr, vast, queue, store, clock = make()
    iid, label = boot_one(mgr, vast, queue, clock)
    queue.set("j1", "completed", label, clock())
    clock.advance(61)
    vast.fail_stop = True
    mgr.tick()
    assert store.get(iid).state == "running"
    vast.fail_stop = False
    mgr.tick()
    assert store.get(iid).state == "stopping"


def test_repeated_ticks_are_idempotent():
    mgr, vast, queue, store, clock = make()
    iid, label = boot_one(mgr, vast, queue, clock)
    queue.set("j1", "completed", label, clock())
    clock.advance(61)
    for _ in range(5):
        mgr.tick()
    assert vast.n("stop") == 1 and vast.n("create") == 1


def test_new_job_while_stopped_resumes_same_instance():
    mgr, vast, queue, store, clock = make()
    iid, label = boot_one(mgr, vast, queue, clock)
    queue.set("j1", "completed", label, clock())
    clock.advance(61); mgr.tick(); mgr.tick()
    assert store.get(iid).state == "stopped"
    queue.add("j2", ts=clock())
    mgr.tick(); mgr.tick()
    assert vast.n("resume") == 1 and vast.n("create") == 1 and store.get(iid).state == "running"


def test_job_during_stopping_waits_then_resumes():
    mgr, vast, queue, store, clock = make()
    iid, label = boot_one(mgr, vast, queue, clock)
    queue.set("j1", "completed", label, clock())
    clock.advance(61)
    vast.stop = lambda i: vast.calls.append(("stop", i))        # Vast is slow to report 'exited'
    mgr.tick()
    queue.add("j2", ts=clock())
    mgr.tick()
    assert vast.n("resume") == 0 and vast.n("create") == 1       # no resume/create while stopping
    vast.inst[iid].update(status="exited")
    mgr.tick(); mgr.tick()
    assert vast.n("resume") == 1


def test_resume_stuck_in_scheduling_falls_back_to_create_and_cleans_disk():
    mgr, vast, queue, store, clock = make({"max_resume_failures": 1})
    vast.offers.append(dict(vast.offers[0], id=2, machine_id=8, dph_total=0.12))   # the failed machine gets banned
    iid, label = boot_one(mgr, vast, queue, clock)
    queue.set("j1", "completed", label, clock())
    clock.advance(61); mgr.tick(); mgr.tick()
    vast.resume_gets_gpu = False                                 # GPU taken by someone else
    queue.add("j2", ts=clock())
    mgr.tick()
    assert store.get(iid).state == "starting"
    clock.advance(95)
    mgr.tick()
    assert vast.n("destroy") == 1 and iid not in vast.inst       # old disk destroyed (no storage leak)
    mgr.tick(); mgr.tick()
    assert vast.n("create") == 2                                  # a new instance was created


def test_retention_destroys_old_stopped_instance_when_idle():
    # warm_instances=0: the pre-warm behaviour (every stopped instance is destroyed after the retention)
    mgr, vast, queue, store, clock = make({"stopped_retention_seconds": 3600, "warm_instances": 0})
    iid, label = boot_one(mgr, vast, queue, clock)
    queue.set("j1", "completed", label, clock())
    clock.advance(61); mgr.tick(); mgr.tick()
    clock.advance(3601)
    mgr.tick(); mgr.tick()
    assert vast.n("destroy") == 1 and store.get(iid).state == "destroyed"


def test_retention_never_destroys_the_warm_server():
    mgr, vast, queue, store, clock = make({"stopped_retention_seconds": 3600})
    iid, label = boot_one(mgr, vast, queue, clock)
    queue.set("j1", "completed", label, clock())
    clock.advance(61); mgr.tick(); mgr.tick()
    assert store.get(iid).state == "stopped" and store.get(iid).role == "warm"
    clock.advance(30 * 86400)
    mgr.tick(); mgr.tick()
    assert vast.n("destroy") == 0 and store.get(iid).state == "stopped"


def test_stale_release_is_never_resumed():
    mgr, vast, queue, store, clock = make()
    iid, label = boot_one(mgr, vast, queue, clock)
    queue.set("j1", "completed", label, clock())
    clock.advance(61); mgr.tick(); mgr.tick()
    mgr.cfg = FleetConfig(enabled=True, dry_run=False, image=IMAGE, release_sha="c" * 40)    # new release pinned
    queue.add("j2", ts=clock())
    mgr.tick()
    assert vast.n("resume") == 0 and vast.n("create") == 2


def test_fresh_instance_gets_boot_grace_before_idle_stop():
    mgr, vast, queue, store, clock = make()
    iid, label = boot_one(mgr, vast, queue, clock)               # worker still booting, has not claimed
    queue.set("j1", "cancelled", None, clock())                  # job vanished
    clock.advance(120)
    mgr.tick()
    assert vast.n("stop") == 0                                    # inside boot grace
    clock.advance(600)
    mgr.tick()
    assert vast.n("stop") == 1


def test_unassigned_active_job_blocks_stop():
    mgr, vast, queue, store, clock = make()
    iid, label = boot_one(mgr, vast, queue, clock)
    queue.set("j1", "completed", label, clock())
    queue.add("j9", status="downloading", worker_id=None, ts=clock())
    clock.advance(300); mgr.tick()
    assert vast.n("stop") == 0


def test_dry_run_sends_no_write_calls():
    mgr, vast, queue, store, clock = make(dry=True)
    queue.add("j1", ts=clock())
    mgr.tick(); mgr.tick()
    assert vast.n("create") == 0 and vast.n("stop") == 0 and vast.n("destroy") == 0
    assert any(e["event"] == "dry_run_create" for e in store.events())


def test_disabled_manager_does_nothing():
    mgr, vast, queue, store, clock = make()
    mgr.cfg = FleetConfig(enabled=False, image=IMAGE, release_sha=SHA)
    queue.add("j1", ts=clock())
    assert mgr.tick() == [] and vast.calls == []


def test_create_rate_limit_and_failure_backoff():
    mgr, vast, queue, store, clock = make({"max_creates_per_hour": 1})
    vast.fail_create = True
    queue.add("j1", ts=clock())
    mgr.tick(); mgr.tick()
    assert vast.n("create") == 0                                  # backoff after failure, no hammering
    assert any(e["event"] == "create_failed" for e in store.events())


def test_max_instances_cap():
    mgr, vast, queue, store, clock = make({"max_instances": 1})
    for i in range(5):
        queue.add(f"j{i}", ts=clock())
    for _ in range(6):
        mgr.tick(); clock.advance(5)
    assert vast.n("create") == 1


def test_orphan_labelled_instance_is_adopted_and_eventually_removed():
    mgr, vast, queue, store, clock = make()
    vast.inst[777] = {"label": "sonya-pool-orphan", "status": "running", "intended": "running"}
    mgr.tick()
    assert store.get(777).state == "running"
    clock.advance(700)
    mgr.tick()                                     # unknown host -> burst: destroyed, never paused (its disk would bill)
    assert vast.n("destroy") == 1 and vast.n("stop") == 0
    mgr2, vast2, _, store2, clock2 = make({"warm_instances": 0})
    vast2.inst[777] = {"label": "sonya-pool-orphan", "status": "running", "intended": "running"}
    mgr2.tick(); clock2.advance(700); mgr2.tick()
    assert vast2.n("stop") == 1


def test_foreign_instances_are_never_touched():
    mgr, vast, queue, store, clock = make()
    vast.inst[5] = {"label": "someone-else", "status": "running", "intended": "running"}
    clock.advance(10_000)
    mgr.tick(); mgr.tick()
    assert vast.calls == [] and store.get(5) is None


def test_release_and_image_must_be_pinned():
    with pytest.raises(launcher.LaunchError):
        launcher.validate_release(FleetConfig(image="ghcr.io/x/base:latest", release_sha=SHA))
    with pytest.raises(launcher.LaunchError):
        launcher.validate_release(FleetConfig(image=IMAGE, release_sha="main"))


def test_config_refuses_trailer_in_pool_modes_and_clamps_idle():
    with pytest.raises(ValueError):
        FleetConfig.from_env({"SONYA_POOL_MODES": "virality,trailer_film_breaker"})
    assert FleetConfig.from_env({}).idle_timeout_seconds == 60 and FleetConfig.from_env({}).dry_run is True
    with pytest.raises(ValueError):
        FleetConfig.from_env({"IDLE_TIMEOUT_SECONDS": "1"})


def test_crash_between_cas_and_vast_call_recovers():
    mgr, vast, queue, store, clock = make()
    iid, label = boot_one(mgr, vast, queue, clock)
    inst = store.get(iid)
    store.cas(iid, "running", inst.generation, "stop_pending", stop_requested_at=clock())   # process died here
    queue.set("j1", "completed", label, clock())
    mgr.tick()
    assert vast.n("stop") == 1 and store.get(iid).state == "stopping"


def test_offer_choice_ranks_by_full_cost_including_disk():
    cfg = FleetConfig(image=IMAGE, release_sha=SHA, disk_gb=80)
    cheap_gpu_dear_disk = offer(1, 1, dph=0.090, storage=1.5)        # 0.087 + 1.5*80/720 = 0.254 $/h
    dearer_gpu_cheap_disk = offer(2, 2, dph=0.110, storage=0.1)      # 0.107 + 0.011 = 0.118 $/h
    assert launcher.pick_offer([cheap_gpu_dear_disk, dearer_gpu_cheap_disk], cfg, set())["id"] == 2
    assert launcher.pick_offer([cheap_gpu_dear_disk], cfg, {1}) is None
    over = {**cheap_gpu_dear_disk, "dph_total": 0.99}
    assert launcher.pick_offer([over], cfg, set()) is None


# ---------------------------------------------------------------- hardening: concurrency, hangs, network faults
def test_concurrent_jobs_single_instance_no_stop_until_last_done_plus_60s():
    mgr, vast, queue, store, clock = make()
    for i in range(3):
        queue.add(f"j{i}", ts=clock())
    mgr.tick(); clock.advance(1); mgr.tick()
    (iid,) = vast.inst
    label = vast.inst[iid]["label"]
    done = []
    for j in ("j0", "j1", "j2"):
        queue.set(j, "mode_running", label, clock())
        clock.advance(300); mgr.tick()
        assert vast.n("stop") == 0                       # others still queued / this one active
        queue.set(j, "completed", label, clock())
        done.append(clock())
        clock.advance(20); mgr.tick()
        if j != "j2":
            assert vast.n("stop") == 0
    clock.advance(45); mgr.tick()                        # 65 s after the last completion
    assert vast.n("stop") == 1 and vast.n("create") == 1


def test_two_dispatcher_processes_on_same_state_stop_once():
    mgr, vast, queue, store, clock = make()
    iid, label = boot_one(mgr, vast, queue, clock)
    queue.set("j1", "completed", label, clock())
    clock.advance(61)
    from dispatcher.fleet import FleetManager
    other = FleetManager(mgr.cfg, store, vast, queue, lambda: GH_TOKEN, ENV, clock)
    mgr.tick(); other.tick(); mgr.tick(); other.tick()
    assert vast.n("stop") == 1


def test_hung_worker_idle_while_jobs_wait_is_replaced_but_active_jobs_are_never_killed():
    mgr, vast, queue, store, clock = make()
    vast.offers.append({**vast.offers[0], "id": 2, "machine_id": 8, "dph_total": 0.12, "dph_base": 0.12})   # the stalled host gets banned
    iid, label = boot_one(mgr, vast, queue, clock)         # j1 queued, worker never claims it
    clock.advance(600 + 400)                               # boot grace + part of stall window
    mgr.tick()
    assert vast.n("destroy") == 0
    clock.advance(600)                                     # > grace + stall_seconds(900)
    mgr.tick()
    assert vast.n("destroy") == 1 and any(e["event"] == "worker_stalled" for e in store.events(iid))
    mgr.tick(); mgr.tick()
    assert vast.n("create") == 2                           # replacement created (the waiting job is not lost)

    mgr2, vast2, queue2, store2, clock2 = make()
    iid2, label2 = boot_one(mgr2, vast2, queue2, clock2)
    queue2.set("j1", "mode_running", label2, clock2())     # actively running, possibly for hours
    queue2.add("j2", ts=clock2())                          # and another job waits
    for _ in range(40):
        clock2.advance(600); mgr2.tick()                   # ~6.7 hours
    assert vast2.n("destroy") == 0 and vast2.n("stop") == 0


def test_vast_outage_aborts_tick_without_any_action_or_state_change():
    mgr, vast, queue, store, clock = make()
    iid, label = boot_one(mgr, vast, queue, clock)
    queue.set("j1", "completed", label, clock())
    clock.advance(61)
    before = (store.get(iid).state, len(vast.calls))
    from dispatcher.vast_api import VastError
    def down():
        raise VastError("network", retryable=True)
    vast.list_instances = down
    with pytest.raises(VastError):
        mgr.tick()
    assert (store.get(iid).state, len(vast.calls)) == before


def test_database_outage_never_causes_a_stop():
    mgr, vast, queue, store, clock = make()
    iid, label = boot_one(mgr, vast, queue, clock)
    queue.set("j1", "completed", label, clock())
    clock.advance(61)
    def db_down(modes, prefix):
        raise RuntimeError("db unreachable")
    queue.snapshot = db_down
    for _ in range(3):
        with pytest.raises(RuntimeError):
            mgr.tick()
    assert vast.n("stop") == 0 and store.get(iid).state == "running"


def test_db_failure_during_the_fenced_recheck_leaves_stop_pending_and_never_stops():
    mgr, vast, queue, store, clock = make()
    iid, label = boot_one(mgr, vast, queue, clock)
    queue.set("j1", "completed", label, clock())
    clock.advance(61)
    real, calls = queue.snapshot, {"n": 0}
    def flaky(modes, prefix):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("db blip")
        return real(modes, prefix)
    queue.snapshot = flaky
    with pytest.raises(RuntimeError):
        mgr.tick()
    assert vast.n("stop") == 0 and store.get(iid).state == "stop_pending"
    mgr.tick()                                               # recovers: re-validates, then stops
    assert vast.n("stop") == 1


def test_config_stall_bounds():
    assert FleetConfig.from_env({}).stall_seconds == 900
    with pytest.raises(ValueError):
        FleetConfig.from_env({"SONYA_STALL_SECONDS": "10"})


CDI_ERROR = ("Error response from daemon: failed to create task for container: failed to create shim task: OCI runtime "
             "create failed: could not apply required modification to OCI specification: error modifying OCI spec: "
             "failed to inject CDI devices: unresolvable CDI devices D.98855df9/gpu=4: unknown")


def test_host_that_cannot_start_the_container_is_dropped_in_a_minute_not_15():
    # run 5 (2026-11-09): Vast kept a broken-GPU contract in "created"; only create_timeout (900 s) saved it
    mgr, vast, queue, store, clock = make()
    vast.offers.append(dict(vast.offers[0], id=2, machine_id=8, dph_total=0.12))   # pricier, other machine
    vast.host_error = CDI_ERROR
    queue.add("j1", ts=clock())
    mgr.tick()
    (bad,) = vast.inst
    run_until(mgr, clock, 50)
    assert bad in vast.inst and vast.n("destroy") == 0          # a short grace: Vast messages can be transient
    run_until(mgr, clock, 40)
    assert bad not in vast.inst and vast.n("destroy") == 1
    assert any(e["event"] == "host_error" for e in store.events())
    assert vast.n("create") == 2                                  # a fresh instance elsewhere, within ~90 s
    assert [c for c in vast.calls if c[0] == "create"][-1][1] != bad
    (good,) = vast.inst
    assert vast.inst[good]["status"] == "running"


def test_host_error_message_on_a_running_instance_is_ignored():
    from dispatcher.vast_api import is_host_error
    assert is_host_error(CDI_ERROR) and not is_host_error("") and not is_host_error("success, running sonya")
    mgr, vast, queue, store, clock = make()
    iid, _ = boot_one(mgr, vast, queue, clock)
    vast.inst[iid]["msg"] = CDI_ERROR                              # stale message, container is up
    run_until(mgr, clock, 120)
    assert iid in vast.inst and vast.n("destroy") == 0


def test_running_then_created_with_host_error_is_dropped_and_banned_at_once():
    # the exact run 5 sequence: running for seconds, then "created" + CDI error
    mgr, vast, queue, store, clock = make()
    vast.offers.append(dict(vast.offers[0], id=2, machine_id=8, dph_total=0.12))
    iid, _ = boot_one(mgr, vast, queue, clock)
    vast.inst[iid].update(status="created", msg=CDI_ERROR)
    mgr.tick()
    assert iid not in vast.inst and vast.n("resume") == 0
    assert 7 in store.banned_machines(clock())
    run_until(mgr, clock, 30)
    assert vast.n("create") == 2


def test_machine_that_failed_to_resume_twice_is_banned():
    mgr, vast, queue, store, clock = make({"max_resume_failures": 1})
    iid, label = boot_one(mgr, vast, queue, clock)
    queue.set("j1", "completed", label, clock())
    clock.advance(61); mgr.tick(); mgr.tick()
    vast.resume_gets_gpu = False
    queue.add("j2", ts=clock())
    mgr.tick()
    clock.advance(95)
    mgr.tick()
    assert 7 in store.banned_machines(clock())
