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


def offer(oid, machine, gpu="RTX 3060", cpu="AMD Ryzen 7 5800X 8-Core Processor", cores=8.0, dph=0.082, storage=0.2,
          geo="United Kingdom, GB", down=5900.0, rel=0.995, **kw):
    """The owner's default class (2026-11-08): RTX 3060 12 GB + Ryzen 7 5800X, UK, 5.9 Gbit/s, $0.082/h."""
    return {"id": oid, "machine_id": machine, "gpu_name": gpu, "cpu_name": cpu, "cpu_cores_effective": cores, "dph_total": dph,
            "dph_base": round(dph - 0.003, 4), "storage_cost": storage, "geolocation": geo, "inet_down": down, "reliability2": rel,
            "internet_up_cost_per_tb": 4, "internet_down_cost_per_tb": 2.6, **kw}


class FakeVast:
    """In-memory Vast: create/stop/resume/destroy with controllable scheduling behaviour."""
    dry_run = False

    def __init__(self):
        self.inst = {}
        self.calls = []
        self._ids = itertools.count(1000)
        self.offers = [offer(1, 7)]
        self.resume_gets_gpu = True
        self.fail_stop = False
        self.fail_create = False
        self.host_error = ""            # next create lands on a host that cannot start the container
        self.fail_offers = set()        # create on these offer ids -> VastError (offer rented a second ago)
        self.fail_resume = False        # resume call -> VastError
        self.created_offers = []        # offer id of every successful create, in order
        self.instance_offer = {}        # instance id -> offer id

    def list_instances(self):
        return [VastInstance(id=i, label=d["label"], actual_status=d["status"], intended_status=d["intended"], gpu_name="RTX 3060",
                             dph_total=Decimal("0.082"), disk_gb=Decimal("40"), status_msg=d.get("msg", ""))
                for i, d in self.inst.items()]

    def search_offers(self, q):
        self.calls.append(("search",))
        return list(self.offers)

    def create(self, offer_id, payload):
        if self.fail_create:
            raise VastError("boom", 500)
        if offer_id in self.fail_offers:
            self.calls.append(("create_refused", offer_id))
            raise VastError("no_such_ask", 400)
        iid = next(self._ids)
        self.created_offers.append(offer_id)
        self.instance_offer[iid] = offer_id
        self.calls.append(("create", iid, payload))
        self.inst[iid] = {"label": payload["label"], "status": "created" if self.host_error else "running",
                          "intended": "running", "msg": self.host_error}
        self.host_error = ""
        return iid

    def stop(self, iid):
        self.calls.append(("stop", iid))
        if self.fail_stop:
            raise VastError("nope", 500)
        self.inst[iid].update(status="exited", intended="stopped")

    def resume(self, iid):
        self.calls.append(("resume", iid))
        if self.fail_resume:
            raise VastError("resume refused", 400)
        self.inst[iid].update(intended="running", status="running" if self.resume_gets_gpu else "scheduling")

    def destroy(self, iid):
        self.calls.append(("destroy", iid))
        self.inst.pop(iid, None)

    def n(self, name):
        return sum(1 for c in self.calls if c[0] == name)


def make(cfg_over=None, dry=False):
    # the lifecycle tests below pre-date the warm/burst split: one instance, no priming (tests/fleet/test_warm.py covers those)
    base = {"warm_prime": False, "max_instances": 1, "jobs_per_instance": 1}
    cfg = FleetConfig(enabled=True, dry_run=dry, image=IMAGE, release_sha=SHA, **{**base, **(cfg_over or {})})
    clock, vast = Clock(), FakeVast()
    vast.dry_run = dry
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    queue = SqliteQueueView(conn)
    store = FleetStore()
    mgr = FleetManager(cfg, store, vast, queue, lambda: GH_TOKEN, ENV, clock)
    return mgr, vast, queue, store, clock
