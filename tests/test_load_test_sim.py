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

    scenario(run, stub, manager, vast, "fast-start-test/x/input.mp4", {"A", "B", "C"}, 2)
    assert run.aborted is None
    A, B = run.phases["A"], run.phases["B"]
    assert [A[k]["n"] for k in ("A1", "A2", "A3", "A4", "A5")] == [1, 2, 3, 4, 1]
    assert all(A[k]["failed"] == 0 for k in ("A1", "A2", "A3", "A4", "A5"))
    assert A["A1"]["instance"]["gpu"] == "RTX 3060"
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
