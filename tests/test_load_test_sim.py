"""tools/load_test.py scenario on a simulated Vast + emulated workers: proves the phases run end to end and leave nothing behind
(a harness bug must not burn a paid run)."""
import sqlite3  # noqa: F401
import tempfile
from pathlib import Path

from dispatcher.config import FleetConfig
from dispatcher.fleet import FleetManager
from dispatcher.store import FleetStore
from tests.fleet.fakes import ENV, GH_TOKEN, IMAGE, SHA, Clock, FakeVast, offer
from tools.load_test import Run, scenario
from tools.stub_worker_api import StubState

BOOT_S, RESUME_S, JOB_S = 240, 20, 130


class SimVast(FakeVast):
    def __init__(self, clock):
        super().__init__()
        self.clock = clock
        self.meta = {}
        self.offers = [offer(1, 7, geo="Maryland, US", cpu="12th Gen Core™ i7-12700", cores=20, dph=0.135),
                       offer(2, 8, dph=0.09), offer(3, 9, dph=0.10),
                       offer(4, 10, gpu="RTX 4060 Ti", cpu="AMD Ryzen 9 5950X 16-Core Processor", cores=16, dph=0.161, geo="Germany, DE")]

    def create(self, offer_id, payload):
        iid = super().create(offer_id, payload)
        self.meta[iid] = {"conc": int(payload["env"]["SONYA_WORKER_CONCURRENCY"]), "ready_at": self.clock() + BOOT_S}
        self.offers = [o for o in self.offers if o["id"] != offer_id]          # rented now
        return iid

    def resume(self, iid):
        super().resume(iid)
        self.meta[iid]["ready_at"] = self.clock() + RESUME_S


def workers(vast: SimVast, stub: StubState, clock: Clock):
    running = {}                                                   # jid -> (label, done_at)

    def step():
        for jid, (lab, done) in list(running.items()):
            if clock() >= done:
                stub.status(jid, "uploading_result")
                stub.complete(jid, {"clip_count": 3})
                running.pop(jid)
        for iid, d in vast.inst.items():
            m = vast.meta.get(iid)
            if not m or d["status"] != "running" or clock() < m["ready_at"]:
                continue
            busy = sum(1 for lab, _ in running.values() if lab == d["label"])
            for _ in range(m["conc"] - busy):
                j = stub.claim(d["label"], ["virality", "beep"])
                if not j:
                    break
                stub.status(j["id"], "downloading"); stub.status(j["id"], "model_downloading"); stub.status(j["id"], "mode_running")
                running[j["id"]] = (d["label"], clock() + JOB_S * (1 + 0.3 * busy))
                busy += 1
        for iid in [i for i in vast.meta if i not in vast.inst]:
            vast.meta.pop(iid)
    return step


def test_all_phases_finish_and_leave_only_what_teardown_removes():
    clock = Clock()
    vast = SimVast(clock)
    stub = StubState(clock)
    run = Run(1.3, 300, clock=clock, sleep=clock.advance, step=workers(vast, stub, clock))
    base = dict(enabled=True, dry_run=False, image=IMAGE, release_sha=SHA, max_creates_per_hour=6, warm_prime=False)

    def manager(prefix, **over):
        cfg = FleetConfig(label_prefix=prefix, **{**base, **over})
        m = FleetManager(cfg, FleetStore(str(Path(tempfile.mkdtemp()) / "f.db")), vast, stub, lambda: GH_TOKEN, ENV, clock)
        run.managers.append(m)
        return m

    scenario(run, stub, manager, vast, "fast-start-test/x/input.mp4", {"A", "B", "C"}, 2, "fast-start-test/x/beep.mp4")
    assert run.aborted is None
    A, B = run.phases["A"], run.phases["B"]
    assert [A[k]["n"] for k in ("A1", "A2", "A3", "A4", "A5")] == [1, 2, 3, 4, 1]
    assert all(A[k]["failed"] == 0 for k in ("A1", "A2", "A3", "A4", "A5"))
    assert A["A1"]["instance"]["gpu"] == "RTX 3060"
    assert A["A6"]["failed"] == 0 and stub.jobs[A["A6"]["jobs"][0]["id"]]["mode"] == "beep"
    assert A["A1"]["jobs"][0]["wait_s"] >= BOOT_S and A["A5"]["jobs"][0]["wait_s"] < 60     # cold vs warm
    assert B["failed"] == 0 and len(B["jobs"]) == 6
    roles = sorted((i["role"], i["state"]) for i in B["instances"])
    assert roles == [("burst", "destroyed"), ("burst", "destroyed"), ("warm", "stopped")]
    assert run.phases["C"]["C3"]["failed"] == 0
    # what the finally-block of main() destroys: only the warm (stopped) + the C server
    alive = [i for m in run.managers for i in m.store.all() if i.state not in ("destroyed", "lost", "destroying")]
    assert sorted(i.role for i in alive) == ["burst", "warm"]          # the warm (stopped) + the C reference server
    b_ev = [e for e in B["events"] if e["event"] in ("destroy_requested",)]
    last_done = max(j["total_s"] for j in B["jobs"]) + B["start"] - run.t0
    assert all(e["t"] - last_done <= 75 for e in b_ev), (b_ev, last_done)    # bursts go ~60 s after THEIR last job


def test_phase_g_levels_on_one_pinned_host_each_level_new_server_destroyed():
    """Phase G: one named host (L40S + EPYC, class rule bypassed), a NEW server per level with concurrency = level,
    destroyed after each level; the extra level runs only when the last one was not saturated and the budget allows."""
    from tools.load_test import single_gpu
    clock = Clock()
    vast = SimVast(clock)
    vast.offers.append(offer(9, 28581, gpu="L40S", cpu="AMD EPYC 9334 32-Core Processor", cores=16, dph=0.80, geo="Finland, FI"))
    stub = StubState(clock)
    run = Run(1.7, 300, clock=clock, sleep=clock.advance, step=workers(vast, stub, clock))
    base = dict(enabled=True, dry_run=False, image=IMAGE, release_sha=SHA, max_creates_per_hour=6, warm_prime=False)
    created = []

    def manager(prefix, **over):
        cfg = FleetConfig(label_prefix=prefix, **{**base, **over})
        m = FleetManager(cfg, FleetStore(str(Path(tempfile.mkdtemp()) / "f.db")), vast, stub, lambda: GH_TOKEN, ENV, clock)
        run.managers.append(m)
        return m

    orig = vast.create

    def create(offer_id, payload):
        created.append((offer_id, payload["env"]["SONYA_WORKER_CONCURRENCY"]))
        iid = orig(offer_id, payload)
        if offer_id == 9:                       # the host is offered again after our destroy
            vast.offers.append(offer(9, 28581, gpu="L40S", cpu="AMD EPYC 9334 32-Core Processor", cores=16, dph=0.80, geo="Finland, FI"))
        return iid
    vast.create = create

    res = lambda label: [{"samples": [{"t": clock(), "cg_cpu_cores": 6.0, "vram_used_mb": 9000, "vram_total_mb": 46068}]}]
    single_gpu(run, stub, manager, vast, "fast-start-test/x/input.mp4", gpu="L40S", machine_id=28581, levels=[1, 2, 4],
               extra_level=6, max_dph=__import__("decimal").Decimal("1.0"), fetch_res=res, eff_cores=16.0)
    G = run.phases["G"]
    assert list(G["levels"]) == ["1", "2", "4", "6"] and "extra_skipped" not in G
    assert created == [(9, "1"), (9, "2"), (9, "4"), (9, "6")]               # only the pinned host, concurrency = level
    for n, lv in G["levels"].items():
        assert lv["failed"] == 0 and len(lv["jobs"]) == int(n) and lv["wall_s"] > 0 and lv["cold_start_s"] >= BOOT_S
        assert lv["instance"]["gpu"] == "L40S" and lv["limit"] == []
    assert not vast.list_instances()                                            # every level's server is gone
    assert all(i.state in ("destroyed", "destroying") for m in run.managers for i in m.store.all())


def test_phase_g_aborts_at_once_when_the_host_stops_our_server():
    import pytest
    from tools.load_test import single_gpu
    clock = Clock()
    vast = SimVast(clock)
    vast.offers.append(offer(9, 28581, gpu="L40S", cpu="AMD EPYC 9334 32-Core Processor", cores=16, dph=0.80, geo="Finland, FI"))
    stub = StubState(clock)
    sim = workers(vast, stub, clock)

    def step():
        sim()
        for d in vast.inst.values():          # the host stops the container once it is up
            if d["status"] == "running" and clock() - run.t0 > 300:
                d["status"] = "exited"
    run = Run(1.7, 300, clock=clock, sleep=clock.advance, step=step)
    base = dict(enabled=True, dry_run=False, image=IMAGE, release_sha=SHA, max_creates_per_hour=6, warm_prime=False)

    def manager(prefix, **over):
        m = FleetManager(FleetConfig(label_prefix=prefix, **{**base, **over}), FleetStore(str(Path(tempfile.mkdtemp()) / "f.db")),
                         vast, stub, lambda: GH_TOKEN, ENV, clock)
        run.managers.append(m)
        return m
    with pytest.raises(TimeoutError, match="host unstable"):
        single_gpu(run, stub, manager, vast, "k.mp4", gpu="L40S", machine_id=28581, levels=[1], fetch_res=lambda l: [])
    assert clock() - run.t0 < 600


def test_stub_level_jobs_only_go_to_that_levels_server():
    """2026-11-09: the G1 container still ran seconds after Vast stopped listing it and claimed a G2 job, which then hung."""
    stub = StubState(Clock())
    jid = stub.enqueue("virality", "k.mp4", only_worker_prefix="sonya-g2-")
    assert stub.claim("sonya-g1-abcd", ["virality"]) is None
    assert stub.claim("sonya-g12-abcd", ["virality"]) is None
    assert stub.claim("sonya-g2-abcd", ["virality"])["id"] == jid


def test_phase_g_hung_job_ends_the_level_and_skips_bigger_levels_clip_encoders_recorded():
    """L40S 2026-11-09: one job hung forever. Now the level ends once the rest is done and the hung job ran hang_factor x longer
    than the slowest finished one; with > 25 % not completed no bigger level is rented. Per clip encoders come from
    selected_segments.json (render_encoder: nvenc / nvenc->cpu)."""
    from decimal import Decimal
    from tools.load_test import single_gpu
    clock = Clock()
    vast = SimVast(clock)
    host = lambda: offer(9, 113477, gpu="RTX 4090", cpu="AMD EPYC 7742 64-Core Processor", cores=128, dph=0.91, geo="Hungary, HU")
    vast.offers.append(host())
    stub = StubState(clock)
    sim = workers(vast, stub, clock)
    first = []

    def step():
        for jid, j in stub.jobs.items():
            if j["status"] == "mode_running" and not first:
                first.append(jid)                                # this one never finishes (hung ffmpeg)
            if j["status"] == "mode_running" and jid not in first and not j["files"]:
                stub.add_file(jid, {"file_type": "enrichment_json", "s3_key": f"out/{jid}/selected_segments.json",
                                    "filename": "selected_segments.json"})
        sim()
        if first and stub.jobs[first[0]]["status"] == "completed":
            stub.jobs[first[0]]["status"] = "mode_running"       # (the sim worker "finished" it: undo, it is hung)
            stub.jobs[first[0]]["history"] = [h for h in stub.jobs[first[0]]["history"] if h[0] not in ("uploading_result", "completed")]
    run = Run(2.0, 300, clock=clock, sleep=clock.advance, step=step)
    base = dict(enabled=True, dry_run=False, image=IMAGE, release_sha=SHA, max_creates_per_hour=6, warm_prime=False)

    def manager(prefix, **over):
        m = FleetManager(FleetConfig(label_prefix=prefix, **{**base, **over}), FleetStore(str(Path(tempfile.mkdtemp()) / "f.db")),
                         vast, stub, lambda: GH_TOKEN, ENV, clock)
        run.managers.append(m)
        return m
    orig = vast.create
    vast.create = lambda oid, p: (vast.offers.append(host()), orig(oid, p))[1]
    segs = {"segments": [{"file": "viral_clip_01.mp4", "render_encoder": "nvenc"}, {"file": "viral_clip_02.mp4", "render_encoder": "nvenc->cpu"},
                         {"file": None, "render_encoder": None}]}
    single_gpu(run, stub, manager, vast, "k.mp4", gpu="RTX 4090", machine_id=113477, levels=[2, 4], max_dph=Decimal("1.0"),
               fetch_res=lambda l: [], fetch_json=lambda k: segs)
    G = run.phases["G"]
    lv = G["levels"]["2"]
    assert list(G["levels"]) == ["2"] and "bigger levels skipped" in G["stopped"]
    assert lv["hung"] == first and lv["failed"] == 1 and lv["videos_per_hour"] and lv["wall_done_s"] < lv["wall_s"]
    assert lv["encoders"] == {"None": 1, "nvenc": 1, "nvenc->cpu": 1} and lv["clips_missing"] == 1
    assert clock() - run.t0 < 40 * 60 and not vast.list_instances()


def test_pinned_search_allows_a_multi_gpu_host_production_query_stays_one_card():
    from dispatcher import launcher
    base = dict(enabled=True, dry_run=False, image=IMAGE, release_sha=SHA)
    assert launcher.search_query(FleetConfig(**base))["num_gpus"] == {"eq": 1}
    q = launcher.search_query(FleetConfig(**base, pin_machine_id=51337))
    assert q["num_gpus"] == {"gte": 1} and q["machine_id"] == {"eq": 51337}


def test_peaks_keep_every_card_of_a_multi_gpu_server():
    from tools.load_test import peaks
    docs = [{"samples": [{"t": 10, "vram_used_mb": 30000, "gpus": [{"i": 0, "vram_used_mb": 20000, "gpu_util": 90, "procs": 4},
                                                                    {"i": 1, "vram_used_mb": 10000, "gpu_util": 40, "procs": 4}]},
                         {"t": 20, "vram_used_mb": 31000, "gpus": [{"i": 0, "vram_used_mb": 15000, "gpu_util": 99},
                                                                    {"i": 1, "vram_used_mb": 16000, "gpu_util": 30}]},
                         {"t": 99, "gpus": [{"i": 0, "vram_used_mb": 47000}]}]}]          # outside the level: ignored
    p = peaks(docs, 0, 50)
    assert p["vram_used_mb"] == 31000 and p["samples"] == 2
    assert p["per_gpu"] == {"0": {"vram_used_mb": 20000, "gpu_util": 99, "procs": 4},
                            "1": {"vram_used_mb": 16000, "gpu_util": 40, "procs": 4}}
