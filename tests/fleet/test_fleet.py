import json

import pytest

from dispatcher import launcher
from dispatcher.config import FleetConfig
from tests.fleet.fakes import ENV, GH_TOKEN, IMAGE, SHA, make


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
    mgr, vast, queue, store, clock = make({"stopped_retention_seconds": 3600})
    iid, label = boot_one(mgr, vast, queue, clock)
    queue.set("j1", "completed", label, clock())
    clock.advance(61); mgr.tick(); mgr.tick()
    clock.advance(3601)
    mgr.tick(); mgr.tick()
    assert vast.n("destroy") == 1 and store.get(iid).state == "destroyed"


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


def test_orphan_labelled_instance_is_adopted_and_eventually_stopped():
    mgr, vast, queue, store, clock = make()
    vast.inst[777] = {"label": "sonya-pool-orphan", "status": "running", "intended": "running"}
    mgr.tick()
    assert store.get(777).state == "running"
    clock.advance(700)
    mgr.tick()
    assert vast.n("stop") == 1


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
    cheap_gpu_dear_disk = {"id": 1, "gpu_name": "RTX 3090", "dph_total": 0.189, "dph_base": 0.18667, "storage_cost": 0.36667, "machine_id": 1}
    dearer_gpu_cheap_disk = {"id": 2, "gpu_name": "RTX 3090", "dph_total": 0.201, "dph_base": 0.2, "storage_cost": 0.2, "machine_id": 2}
    assert launcher.pick_offer([cheap_gpu_dear_disk, dearer_gpu_cheap_disk], cfg, set())["id"] == 2
    assert launcher.pick_offer([cheap_gpu_dear_disk], cfg, {1}) is None
    over = {**cheap_gpu_dear_disk, "dph_total": 0.99}
    assert launcher.pick_offer([over], cfg, set()) is None
