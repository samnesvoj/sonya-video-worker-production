import itertools
import sqlite3
from decimal import Decimal

from dispatcher.config import FleetConfig
from dispatcher.fleet import FleetManager
from dispatcher.queue import SqliteQueueView
from dispatcher.store import FleetStore
from dispatcher.vast_api import VastError, VastInstance

SHA = "a" * 40
IMAGE = "ghcr.io/x/base@sha256:" + "b" * 64
ENV = {"BACKEND_API_URL": "https://api.example", "WORKER_SECRET": "WSECRET-LEAKCHECK", "S3_ENDPOINT_URL": "https://s3.example",
       "S3_ACCESS_KEY_ID": "AKIA-LEAKCHECK", "S3_SECRET_ACCESS_KEY": "S3SECRET-LEAKCHECK", "S3_BUCKET_NAME": "b", "S3_REGION": "r",
       "MODELS_S3_BUCKET": "b"}
GH_TOKEN = "ghs_FAKE_INSTALLATION_TOKEN_LEAKCHECK"


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


class FakeVast:
    """In-memory Vast: create/stop/resume/destroy with controllable scheduling behaviour."""
    dry_run = False

    def __init__(self):
        self.inst = {}
        self.calls = []
        self._ids = itertools.count(1000)
        self.offers = [{"id": 1, "gpu_name": "RTX 3090", "dph_total": 0.19, "dph_base": 0.187, "machine_id": 7, "storage_cost": 0.2,
                        "internet_up_cost_per_tb": 4, "internet_down_cost_per_tb": 2.6}]
        self.resume_gets_gpu = True
        self.fail_stop = False
        self.fail_create = False

    def list_instances(self):
        return [VastInstance(id=i, label=d["label"], actual_status=d["status"], intended_status=d["intended"], gpu_name="RTX 3090",
                             dph_total=Decimal("0.19"), disk_gb=Decimal("80")) for i, d in self.inst.items()]

    def search_offers(self, q):
        self.calls.append(("search",))
        return list(self.offers)

    def create(self, offer_id, payload):
        if self.fail_create:
            raise VastError("boom", 500)
        iid = next(self._ids)
        self.calls.append(("create", iid, payload))
        self.inst[iid] = {"label": payload["label"], "status": "running", "intended": "running"}
        return iid

    def stop(self, iid):
        self.calls.append(("stop", iid))
        if self.fail_stop:
            raise VastError("nope", 500)
        self.inst[iid].update(status="exited", intended="stopped")

    def resume(self, iid):
        self.calls.append(("resume", iid))
        self.inst[iid].update(intended="running", status="running" if self.resume_gets_gpu else "scheduling")

    def destroy(self, iid):
        self.calls.append(("destroy", iid))
        self.inst.pop(iid, None)

    def n(self, name):
        return sum(1 for c in self.calls if c[0] == name)


def make(cfg_over=None, dry=False):
    cfg = FleetConfig(enabled=True, dry_run=dry, image=IMAGE, release_sha=SHA, **(cfg_over or {}))
    clock, vast = Clock(), FakeVast()
    vast.dry_run = dry
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    queue = SqliteQueueView(conn)
    store = FleetStore()
    mgr = FleetManager(cfg, store, vast, queue, lambda: GH_TOKEN, ENV, clock)
    return mgr, vast, queue, store, clock
