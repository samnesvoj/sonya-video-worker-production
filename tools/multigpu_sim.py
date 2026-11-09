#!/usr/bin/env python3
"""
multigpu_sim.py - DRY-RUN simulation of a rush of videos: the REAL dispatcher FleetManager + the REAL worker plan
(scripts/ops/gpu_plan.py) on a simulated Vast and simulated workers, in virtual time. No network, no money, no secrets.

Shows, for the current default scheme (1 warm RTX 3060 + bursts) and with the POWER class on (e.g. 2x A40 Belgium):
which servers are created, how many videos each counts / runs, how the workers of a multi-GPU server spread over its cards,
when the last video is done and what the servers cost.

  python tools/multigpu_sim.py --jobs 12                 # default scheme vs power class, 12 videos at once
  python tools/multigpu_sim.py --jobs 12 --broken-card   # the power server's GPU 1 fails readiness: works on the other card

Job time model (orchestration, not a benchmark): measured ~110 s per kishlak_15m video in Europe (SPEED_REPORT.md), slower when
the CPU is oversubscribed: t = 110 s x max(1, busy videos x 4 threads / server threads).
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from decimal import Decimal
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dispatcher.config import FleetConfig  # noqa: E402
from dispatcher.fleet import FleetManager  # noqa: E402
from dispatcher.store import FleetStore  # noqa: E402
from dispatcher.vast_api import VastInstance  # noqa: E402
from scripts.ops import gpu_plan  # noqa: E402
from tools.stub_worker_api import StubState  # noqa: E402

BOOT_S, RESUME_S, JOB_S, THREADS_PER_JOB = 160, 20, 110.0, 4.0
SHA = "f" * 40
IMAGE = "ghcr.io/sim/base@sha256:" + "0" * 64
SIM_ENV = {k: "sim" for k in ("BACKEND_API_URL", "WORKER_SECRET", "S3_ENDPOINT_URL", "S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY",
                              "S3_BUCKET_NAME", "S3_REGION", "MODELS_S3_BUCKET")}


def offer(oid, machine, gpu, n_gpus, vram_mb, cpu, cores, ram_gb, dph, geo, storage=0.10):
    return {"id": oid, "machine_id": machine, "gpu_name": gpu, "num_gpus": n_gpus, "gpu_ram": vram_mb, "cpu_name": cpu,
            "cpu_cores_effective": cores, "cpu_ram": ram_gb * 1024, "dph_total": dph, "dph_base": round(dph - 0.004, 4),
            "storage_cost": storage, "geolocation": geo, "inet_down": 2000.0, "reliability2": 0.995,
            "internet_up_cost_per_tb": 4, "internet_down_cost_per_tb": 2}


def market() -> List[dict]:
    """A plausible European market (prices like the 2026-11-09 load test)."""
    return [offer(1, 101, "RTX 3060", 1, 12288, "AMD Ryzen 7 5800X 8-Core Processor", 12, 32, 0.085, "United Kingdom, GB"),
            offer(2, 102, "RTX 3060", 1, 12288, "AMD Ryzen 5 5600X 6-Core Processor", 12, 32, 0.090, "Germany, DE", storage=0.25),
            offer(3, 103, "RTX A4000", 1, 16376, "Core™ i7-14700", 14, 32, 0.120, "Spain, ES", storage=0.30),
            offer(4, 104, "RTX 4060 Ti", 1, 16380, "AMD Ryzen 9 5950X 16-Core Processor", 16, 64, 0.161, "Germany, DE"),
            offer(5, 105, "A40", 2, 46068, "AMD EPYC 7543 32-Core Processor", 32, 128, 0.72, "Belgium, BE"),
            offer(6, 106, "L40S", 4, 46068, "AMD EPYC 9354 32-Core Processor", 64, 256, 1.60, "Netherlands, NL")]


class Clock:
    def __init__(self):
        self.t = 1_700_000_000.0

    def __call__(self):
        return self.t


class SimVast:
    dry_run = False

    def __init__(self, clock: Clock, offers: List[dict], broken_card: bool = False):
        self.clock, self.offers, self.broken_card = clock, offers, broken_card
        self.inst: Dict[int, dict] = {}
        self.gone: List[dict] = []                 # destroyed servers (kept for the report)
        self._ids = itertools.count(5000)

    def list_instances(self):
        return [VastInstance(id=i, label=d["label"], actual_status=d["status"], intended_status=d["intended"],
                             gpu_name=d["offer"]["gpu_name"], dph_total=Decimal(str(d["offer"]["dph_total"])), disk_gb=Decimal("30"))
                for i, d in self.inst.items()]

    def search_offers(self, q):
        rented = {d["offer"]["id"] for d in self.inst.values()}
        return [o for o in self.offers if o["id"] not in rented]

    def create(self, offer_id, payload):
        iid = next(self._ids)
        o = next(x for x in self.offers if x["id"] == offer_id)
        self.inst[iid] = {"label": payload["label"], "status": "running", "intended": "running", "offer": o, "env": payload["env"],
                          "ready_at": self.clock() + BOOT_S, "plan": self._plan(o, payload["env"]), "run_s": 0.0, "since": self.clock()}
        return iid

    def _plan(self, o: dict, env: dict) -> dict:
        """What readiness would decide on this server (the real gpu_plan on the offer's hardware)."""
        gpus = gpu_plan._fake_gpus(f"{o.get('num_gpus', 1)}x{o['gpu_name']}:{o['gpu_ram']}")
        if self.broken_card and len(gpus) > 1:
            gpus[1].cuda_ok, gpus[1].error = False, "CUDA probe failed (simulated)"
        return gpu_plan.build_plan(gpus, env, threads=o["cpu_cores_effective"], ram=o["cpu_ram"])

    def stop(self, iid):
        d = self.inst[iid]
        d["run_s"] += self.clock() - d["since"]
        d.update(status="exited", intended="stopped")

    def resume(self, iid):
        self.inst[iid].update(status="running", intended="running", ready_at=self.clock() + RESUME_S, since=self.clock())

    def destroy(self, iid):
        d = self.inst.pop(iid)
        if d["status"] == "running":
            d["run_s"] += self.clock() - d["since"]
        self.gone.append(d)


def simulate(n_jobs: int, power: bool, broken_card: bool = False, max_instances: int = 3) -> dict:
    clock = Clock()
    vast = SimVast(clock, market(), broken_card)
    stub = StubState(clock)
    cfg = FleetConfig(enabled=True, dry_run=False, image=IMAGE, release_sha=SHA, max_instances=max_instances,
                      power_enabled=power, label_prefix="sonya-pool-")
    fleet = FleetManager(cfg, FleetStore(), vast, stub, lambda: "sim-token", SIM_ENV, clock)
    running: Dict[str, dict] = {}          # jid -> {label, gpu, done}
    log: List[str] = []

    def step():
        for jid, r in list(running.items()):
            if clock() >= r["done"]:
                stub.status(jid, "uploading_result")
                stub.complete(jid, {"clip_count": 3})
                vast_d = next((d for d in list(vast.inst.values()) + vast.gone if d["label"] == r["label"]), None)
                if vast_d is not None:
                    vast_d.setdefault("done_by_gpu", {}).setdefault(str(r["gpu"]), 0)
                    vast_d["done_by_gpu"][str(r["gpu"])] += 1
                running.pop(jid)
        for d in vast.inst.values():
            if d["status"] != "running" or clock() < d["ready_at"]:
                continue
            threads = float(d["offer"]["cpu_cores_effective"])
            for w in d["plan"]["workers"]:
                key = (d["label"], w["slot"])
                if any((r["label"], r["slot"]) == key for r in running.values()):
                    continue
                j = stub.claim(d["label"], ["virality", "beep"])
                if not j:
                    break
                busy = 1 + sum(1 for r in running.values() if r["label"] == d["label"])
                stub.status(j["id"], "mode_running")
                running[j["id"]] = {"label": d["label"], "slot": w["slot"], "gpu": w.get("gpu_index", 0),
                                    "done": clock() + JOB_S * max(1.0, busy * THREADS_PER_JOB / threads)}

    def run_until(pred, limit_s):
        end = clock() + limit_s
        while clock() < end:
            step()
            for a in fleet.tick():
                log.append(f"+{int(clock() - t0):5d}s {a}")
            if pred():
                return True
            clock.t += 5
        return False

    t0 = clock()
    # 1) the dispatcher primes the warm (created, booted, stopped) - the state production is in before a rush
    run_until(lambda: any(i.role == "warm" and i.state == "stopped" for i in fleet.store.all()), 3600)
    t_rush = clock()
    log.append(f"+{int(t_rush - t0):5d}s --- rush: {n_jobs} videos at once ---")
    jids = [stub.enqueue("virality", "sim/input.mp4") for _ in range(n_jobs)]
    run_until(lambda: all(stub.jobs[j]["status"] == "completed" for j in jids), 4 * 3600)
    t_done = clock()
    run_until(lambda: not fleet.store.all(("creating", "running", "starting", "stop_pending", "stopping", "destroying")), 3600)
    servers = []
    for d in list(vast.inst.values()) + vast.gone:
        inst = next((i for i in fleet.store.all() if i.label == d["label"]), None)
        o = d["offer"]
        servers.append({"label": d["label"], "class": inst.gpu_class if inst else "?", "role": inst.role if inst else "?",
                        "gpu": f"{o.get('num_gpus', 1)}x {o['gpu_name']}", "geo": o["geolocation"],
                        "slots_counted": fleet.slots(inst) if inst else None, "workers": d["plan"]["concurrency"],
                        "workers_per_gpu": d["plan"]["per_gpu_workers"], "videos_per_gpu": d.get("done_by_gpu", {}),
                        "warnings": d["plan"]["warnings"], "state": inst.state if inst else "?",
                        "usd": round(d["run_s"] / 3600 * o["dph_total"], 4)})
    waits = [stub.jobs[j]["history"][-1][1] - t_rush for j in jids]
    return {"power": power, "jobs": n_jobs, "all_done_s": round(t_done - t_rush), "last_video_s": round(max(waits)),
            "median_video_s": round(sorted(waits)[len(waits) // 2]), "servers": servers,
            "usd_total": round(sum(s["usd"] for s in servers), 4), "log": log}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--max-instances", type=int, default=3)
    ap.add_argument("--broken-card", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    out = [simulate(a.jobs, power, a.broken_card, a.max_instances) for power in (False, True)]
    if a.json:
        print(json.dumps(out, indent=1))
        return 0
    for r in out:
        print(f"\n=== {'POWER class ON' if r['power'] else 'default scheme (1 warm RTX 3060 + bursts)'}: {r['jobs']} videos at once ===")
        print(f"all done after {r['all_done_s']} s (median video {r['median_video_s']} s), servers cost ${r['usd_total']}")
        for s in r["servers"]:
            print(f"  {s['class']:8s} {s['role']:5s} {s['gpu']:12s} {s['geo']:20s} slots {s['slots_counted']}, workers {s['workers']} "
                  f"per card {s['workers_per_gpu'] or '-'}, videos per card {s['videos_per_gpu']}, ${s['usd']} -> {s['state']}")
            for w in s["warnings"]:
                print(f"           WARNING {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
