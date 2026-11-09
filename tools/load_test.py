#!/usr/bin/env python3
"""
load_test.py - the SPEED + LOAD test on Vast (2026-11-08 owner tasks): start time, NVENC, parallel jobs, warm server, autoscaling.

Same building blocks as tools/fast_start_test.py (stub worker API behind a tunnel, real Vast, the REAL dispatcher FleetManager).
Never touches the production queue. Everything is destroyed at the end (teardown is in `finally`).

Phases (one paid run):
  A  default class (RTX 3060 + modern CPU), worker concurrency 4, max 1 server:
     A1 one virality job on a COLD server (create -> boot -> ready -> job)   = cold start + job time with NVENC
     A2..A4 2 / 3 / 4 virality jobs at once on the same server               = parallel jobs, resource peaks (res_sampler)
     A5 idle -> the server becomes the WARM one and is STOPPED; one job -> RESUME = warm wake-up
  B  autoscaling: jobs_per_instance=J, max_instances=3, 6 jobs at once -> warm resume + burst creates; bursts DESTROYED after,
     warm STOPPED; wait of the last user
  C  (reference only, not for production) one "powerful" server (>=16 CPU cores, <= $0.30/h), 1 job then 3 at once
  G  ONE named host (--machine-id, e.g. an L40S), levels --levels 1,2,4,6,8 (+ --extra-level 12 if 8 is not saturated and the
     budget allows): per level a NEW server with jobs_per_instance = SONYA_WORKER_CONCURRENCY = level, N virality jobs at once,
     NVENC required (readiness refuses to start without a working h264_nvenc), resource peaks from res_sampler, then destroyed.

  python tools/load_test.py --release-sha SHA --image IMG@sha256:.. --backend-url http://bore.pub:NNNNN --control-video k15.mp4 \
      --max-usd 1.3 --apply --confirm-spend 1.3 [--phases A,B,C] [--jobs-per-instance 2]
  phase G:  ... --phases G --gpu L40S --machine-id 28581 --levels 1,2,4,6,8 --extra-level 12 --max-dph 1.0 --max-usd 1.7
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import secrets as _secrets
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from accounting import usage                                   # noqa: E402
from dispatcher import startup_metrics                         # noqa: E402
from dispatcher.config import FleetConfig                      # noqa: E402
from dispatcher.fleet import FleetManager                      # noqa: E402
from dispatcher.store import FleetStore                        # noqa: E402
from dispatcher.vast_api import VastError                      # noqa: E402
from tools.fast_start_test import SITE_BEEP_PARAMS, SITE_VIRALITY_PARAMS, _tunnel_url_ok   # noqa: E402
from tools.stub_worker_api import StubState                    # noqa: E402

TERMINAL = ("completed", "failed")


def job_times(stub: StubState, jid: str) -> dict:
    j = stub.jobs[jid]
    h = {}
    for st, t in j["history"]:
        h.setdefault(st, t)
    q = j["queued_at"]
    end = next((t for st, t in reversed(j["history"]) if st in TERMINAL), None)
    run = h.get("mode_running")
    dl0, dl1 = h.get("downloading"), h.get("model_downloading")
    return {"id": jid, "status": j["status"], "worker": j["worker_id"],
            "wait_s": round(run - q, 1) if run else None,                       # what the user feels before processing starts
            "process_s": round(end - h.get("claimed", q), 1) if end else None,  # claim -> done (download + cut + upload)
            "total_s": round(end - q, 1) if end else None,
            "s3_download_s": round(dl1 - dl0, 1) if dl0 and dl1 else None}


class Run:
    def __init__(self, max_usd: float, max_minutes: float, clock=time.time, sleep=time.sleep, step=lambda: None):
        self.max_usd, self.max_minutes = max_usd, max_minutes
        self.clock, self.sleep, self.step = clock, sleep, step
        self.t0 = clock()
        self.log: List[str] = []
        self.phases: Dict[str, dict] = {}
        self.managers: List[FleetManager] = []
        self.aborted: Optional[str] = None

    def say(self, m: str) -> None:
        line = f"[+{self.clock() - self.t0:6.0f}s] {m}"
        self.log.append(line)
        print(line, flush=True)

    # ------------------------------------------------------------------ guards
    def spent(self) -> Decimal:
        now = self.clock()
        return sum((sum((c["gpu"], c["disk"]), Decimal(0)) for c in
                    (usage.cost_between(m.store, self.t0 - 1, now, now) for m in self.managers)), Decimal(0))

    def tick(self, mgrs: List[FleetManager]) -> None:
        for m in mgrs:
            try:
                acts = m.tick()
                for a in acts:
                    self.say(f"{m.cfg.label_prefix} {a}")
            except VastError as exc:
                if not exc.retryable:
                    raise
                self.say(f"vast api blip ({exc}) - tick skipped")
        self.step()
        usd = self.spent()
        if usd > Decimal(str(self.max_usd)) or self.clock() - self.t0 > self.max_minutes * 60:
            raise TimeoutError(f"budget/time guard (usd={usd:.3f}, minutes={(self.clock() - self.t0) / 60:.0f})")
        self.sleep(5)

    def wait(self, mgrs, pred, timeout_s, what) -> None:
        end = self.clock() + timeout_s
        while not pred():
            if self.clock() > end:
                raise TimeoutError(f"timeout waiting for {what}")
            self.tick(mgrs)
        self.say(what)


def batch(run: Run, mgrs, stub: StubState, key: str, n: int, name: str, timeout_s: int = 40 * 60, mode: str = "virality") -> dict:
    t = run.clock()
    params = SITE_BEEP_PARAMS if mode == "beep" else SITE_VIRALITY_PARAMS
    ids = [stub.enqueue(mode, key, params) for _ in range(n)]
    run.say(f"{name}: {n} {mode} job(s) queued")
    run.wait(mgrs, lambda: all(stub.jobs[j]["status"] in TERMINAL for j in ids), timeout_s, f"{name}: all {n} finished")
    jobs = [job_times(stub, j) for j in ids]
    return {"name": name, "n": n, "start": t, "end": run.clock(), "wall_s": round(run.clock() - t, 1), "jobs": jobs,
            "failed": sum(1 for j in jobs if j["status"] != "completed")}


SLOWDOWN_OK = 1.6     # parallel jobs: each user may wait at most 60 % longer than alone


def optimum(ph: dict) -> int:
    """Largest n (1..4) whose batch finished with no failure within SLOWDOWN_OK x the single-job wall time on the warm server."""
    one = (ph.get("A5") or ph.get("A1") or {}).get("jobs", [{}])[0].get("process_s")
    best = 1
    for n in (2, 3, 4):
        r = ph.get(f"A{n}")
        if one and r and not r["failed"] and r["wall_s"] <= SLOWDOWN_OK * one * 1.0 + 30:
            best = n
    return best


POWER_GPUS = ("RTX 4060 Ti", "RTX 4070", "RTX A4000", "RTX 3090", "RTX 4070 Ti", "RTX 3080", "RTX 3060")


def scenario(run: Run, stub: StubState, manager, vast, key: str, phases: set, jpi: int, beep_key: Optional[str] = None) -> None:
    """The phases A/B/C (module doc). `manager(prefix, **cfg)` makes a FleetManager on the shared stub queue."""
    if "A" in phases or "B" in phases:
        A = manager("sonya-pool-", max_instances=1, jobs_per_instance=4)
        mg = [A]
        ph: dict = {}
        r = batch(run, mg, stub, key, 1, "A1 cold")
        ph["A1"] = r
        insts = A.store.all()
        inst = insts[0]
        ev = A.store.events(inst.id)
        r["instance"] = {"id": inst.id, "gpu": inst.gpu_name, "machine": inst.machine_id,
                         "created": next((e["detail"] for e in ev if e["event"] == "created"), {})}
        ph["A1"]["_ev"] = (inst.id, 0, None, r["jobs"][0]["id"])
        if "A" in phases:
            for n in (2, 3, 4):
                ph[f"A{n}"] = batch(run, mg, stub, key, n, f"A{n} parallel")
        run.wait(mg, lambda: A.store.get(inst.id).state == "stopped", 6 * 60, "A: idle -> warm server stopped")
        ph["warm_stop_at"] = run.clock()
        r5 = batch(run, mg, stub, key, 1, "A5 warm resume")
        r5["_ev"] = (inst.id, 1, ph["warm_stop_at"], r5["jobs"][0]["id"])
        ph["A5"] = r5
        if beep_key:
            ph["A6"] = batch(run, mg, stub, beep_key, 1, "A6 beep", mode="beep")
        run.phases["A"] = ph
        if "B" in phases:
            run.wait(mg, lambda: A.store.get(inst.id).state == "stopped", 6 * 60, "B: warm stopped again")
            jpi = jpi or optimum(ph)
            ph["optimum_jobs_per_instance"] = jpi
            A.cfg = dataclasses.replace(A.cfg, max_instances=3, jobs_per_instance=jpi)
            rb = batch(run, mg, stub, key, 6, f"B autoscale (6 jobs, max 3, {jpi}/server)")
            run.wait(mg, lambda: not [i for i in A.store.all(("creating", "running", "starting", "stop_pending", "stopping", "destroying"))],
                     8 * 60, "B: bursts destroyed, warm stopped")
            rb["instances"] = [{"id": i.id, "role": i.role, "state": i.state, "gpu": i.gpu_name, "machine": i.machine_id} for i in A.store.all()]
            rb["events"] = [{"t": round(e["ts"] - run.t0, 1), "id": e["instance_id"], "event": e["event"]} for e in A.store.events()
                            if e["ts"] >= rb["start"]]
            run.phases["B"] = rb
    if "C" in phases:
        C = manager("sonya-pwr-", max_instances=1, jobs_per_instance=4, warm_instances=0, min_cpu_cores=16.0, max_dph=Decimal("0.30"),
                    default_gpus=POWER_GPUS, fallback_gpus=())
        mg = [C]
        c1 = batch(run, mg, stub, key, 1, "C1 powerful cold")
        i = C.store.all()[0]
        c1["_ev"] = (i.id, 0, None, c1["jobs"][0]["id"])
        c1["instance"] = {"id": i.id, "gpu": i.gpu_name, "created": next((e["detail"] for e in C.store.events(i.id) if e["event"] == "created"), {})}
        c3 = batch(run, mg, stub, key, 3, "C3 powerful parallel")
        run.phases["C"] = {"C1": c1, "C3": c3}         # the reference server is destroyed by the teardown in main()


# ── phase G: how many videos at once on ONE (big) GPU ─────────────────────────────────────────────────────────────────
RES_KEYS = ("vram_used_mb", "vram_total_mb", "cg_mem_mb", "ram_used_mb", "cg_cpu_cores", "cpu_pct", "load1", "gpu_util", "enc_util",
            "ffmpeg", "ff_nvenc", "ff_x264", "tmp_gb", "disk_used_gb")


def peaks(docs: List[dict], t0: float, t1: float) -> dict:
    """Per-metric max of the res_sampler samples taken in [t0, t1] (one level's batch)."""
    out: dict = {}
    for d in docs:
        for x in d.get("samples", []):
            if t0 <= x.get("t", 0) <= t1:
                out["samples"] = out.get("samples", 0) + 1
                for k in RES_KEYS:
                    if k in x:
                        out[k] = max(out.get(k, x[k]), x[k])
    return out


def saturated(lv: dict, eff_cores: float) -> List[str]:
    """Why a level counts as 'hit the limit' (then no bigger level is tried)."""
    p, why = lv.get("peaks", {}), []
    if lv["failed"]:
        why.append(f"{lv['failed']} failed")
    if eff_cores and p.get("cg_cpu_cores", 0) >= 0.9 * eff_cores:
        why.append(f"CPU {p['cg_cpu_cores']:.1f}/{eff_cores:.0f} cores")
    if p.get("vram_total_mb") and p.get("vram_used_mb", 0) >= 0.9 * p["vram_total_mb"]:
        why.append("VRAM >= 90 %")
    return why


def single_gpu(run: Run, stub: StubState, manager, vast, key: str, *, gpu: str, machine_id: int, levels: List[int],
               extra_level: int = 0, max_dph: Decimal = Decimal("1.0"), fetch_res=lambda label: [], flush_s: int = 65,
               eff_cores: float = 0.0, fetch_json=lambda s3_key: None, hang_factor: float = 3.0, hang_min_s: float = 600,
               stop_fail_share: float = 0.25) -> dict:
    """Phase G (module doc). Each level gets its OWN fresh server (concurrency = level), so idle worker processes of a bigger
    level never inflate the peaks of a smaller one. Same host every time (pin_machine_id)."""
    out: dict = run.phases.setdefault("G", {})     # filled level by level: an abort keeps the finished levels in the report
    out.update(gpu=gpu, machine_id=machine_id, levels={})
    todo = list(levels)
    while todo:
        n = todo.pop(0)
        m = manager(f"sonya-g{n}-", max_instances=1, jobs_per_instance=n, warm_instances=0, default_gpus=(gpu,), fallback_gpus=(),
                    min_vram_gb=24, max_dph=max_dph, pin_machine_id=machine_id, idle_timeout_seconds=3600, max_creates_per_hour=3,
                    min_cpu_ram_gb=max(16, 4 * n + 4))
        mg = [m]
        t = run.clock()
        ids = [stub.enqueue("virality", key, SITE_VIRALITY_PARAMS, only_worker_prefix=f"sonya-g{n}-") for _ in range(n)]
        run.say(f"G{n}: {n} virality job(s) queued for a {gpu} with {n} worker process(es)")
        run.wait(mg, lambda: bool(m.store.all()), 5 * 60, f"G{n}: server created")
        inst = m.store.all()[0]
        def done() -> bool:
            # 2026-11-09 Finland L40S (machine 28581): the HOST stopped the container 20 s into the first job ("SIGTERM" from
            # outside, Vast status exited); the fleet then resumed it 3x and the claimed job hung. A host that stops / loses
            # our server during a level is unstable: abort at once instead of waiting 20+ min.
            bad = [e["event"] for e in m.store.events(inst.id) if e["event"] in ("unexpected_stop", "host_error", "lost", "worker_stalled")]
            if bad:
                raise TimeoutError(f"G{n}: host unstable ({', '.join(sorted(set(bad)))}) on machine {machine_id}")
            if all(stub.jobs[j]["status"] in TERMINAL for j in ids):
                return True
            # L40S 2026-11-09: one job hung forever (ffmpeg at CPU 0). Once some jobs are done, a job still running
            # hang_factor x longer than the slowest finished one counts as HUNG: the level ends instead of burning budget.
            fin = [job_times(stub, j)["process_s"] for j in ids if stub.jobs[j]["status"] in TERMINAL]
            if fin:
                lim, now = max(hang_min_s, hang_factor * max([x for x in fin if x] or [0])), run.clock()
                if all(stub.jobs[j]["status"] in TERMINAL or now - next((h for st, h in stub.jobs[j]["history"] if st == "claimed"), now) > lim
                       for j in ids):
                    hung[:] = [j for j in ids if stub.jobs[j]["status"] not in TERMINAL]
                    return True
            return False
        hung: List[str] = []
        run.wait(mg, done, 20 * 60 + 4 * 60 * n, f"G{n}: all {n} finished")
        t_done = run.clock()
        jobs = [job_times(stub, j) for j in ids]
        claims = [h_t for j in ids for st, h_t in stub.jobs[j]["history"] if st == "claimed"]
        ends = [next((h_t for st, h_t in reversed(stub.jobs[j]["history"]) if st in TERMINAL), t_done) for j in ids]
        lv = {"n": n, "instance": {"id": inst.id, "label": inst.label, "gpu": inst.gpu_name, "dph": str(inst.dph_base),
                                   "created": next((e["detail"] for e in m.store.events(inst.id) if e["event"] == "created"), {})},
              "jobs": jobs, "failed": sum(1 for j in jobs if j["status"] != "completed"),
              "cold_start_s": round(min(claims) - t, 1) if claims else None,          # queue -> first job claimed (boot of a new server)
              "wall_s": round(max(ends) - min(claims), 1) if claims else None,        # first claim -> last done = the batch itself
              "start": t, "end": t_done}
        lv["hung"] = hung
        done_n = sum(1 for j in jobs if j["status"] == "completed")
        ok_end = [e for j, e in zip(ids, ends) if stub.jobs[j]["status"] == "completed"]
        lv["wall_done_s"] = round(max(ok_end) - min(claims), 1) if ok_end and claims else None   # without the hang wait
        lv["videos_per_hour"] = round(done_n * 3600 / lv["wall_done_s"], 1) if lv["wall_done_s"] else None
        for j, jt in zip(ids, jobs):          # per clip: which encoder rendered it (selected_segments.json, read-only from S3)
            f = next((x for x in stub.jobs[j]["files"] if (x.get("filename") or "") == "selected_segments.json"), None)
            try:
                segs = (fetch_json(f["s3_key"]) or {}).get("segments", []) if f else []
            except Exception as exc:  # noqa: BLE001
                segs, jt["segments_error"] = [], type(exc).__name__
            jt["clips"] = [{"file": x.get("file"), "enc": x.get("render_encoder"), "render_s": x.get("render_s")} for x in segs]
        encs = [str(c["enc"]) for jt in jobs for c in jt["clips"]]
        lv["encoders"] = {e: encs.count(e) for e in sorted(set(encs))}
        lv["clips_missing"] = sum(1 for jt in jobs for c in jt["clips"] if not c["file"])
        run.wait(mg, lambda: run.clock() >= t_done + flush_s, flush_s + 30, f"G{n}: res_sampler flush")
        lv["peaks"] = peaks(fetch_res(inst.label), min(claims or [t]), max(ends))
        lv["cost_usd"] = float(sum(usage.cost_between(m.store, run.t0 - 1, run.clock(), run.clock())[k] for k in ("gpu", "disk")))
        m._destroy(m.store.get(inst.id), run.clock(), [], reason="level_done")
        run.wait(mg, lambda: inst.id not in {i.id for i in vast.list_instances()}, 4 * 60, f"G{n}: server destroyed")
        out["levels"][str(n)] = lv
        lv["limit"] = saturated(lv, eff_cores)
        run.say(f"G{n}: wall {lv['wall_s']} s, each {[j['process_s'] for j in jobs]}, {lv['videos_per_hour']} videos/h, "
                f"failed {lv['failed']} hung {len(hung)}, encoders {lv['encoders']}, clips missing {lv['clips_missing']}, "
                f"s3 download {[j['s3_download_s'] for j in jobs]}, peaks {lv['peaks']}, limit {lv['limit'] or 'no'}")
        bad_share = lv["failed"] / n                         # "failed" = not completed, hung jobs included
        if todo and bad_share > stop_fail_share:
            out["stopped"] = f"level {n}: {lv['failed']} of {n} not completed (hung {len(hung)}) - bigger levels skipped: {todo}"
            run.say(f"G: {out['stopped']}")
            todo = []
            continue
        if not todo and extra_level and n < extra_level:
            spent = float(run.spent())
            est = (lv["cold_start_s"] or 300) + (lv["wall_s"] or 0) * extra_level / n * 1.2 + flush_s + 120
            est_usd = est / 3600 * float(Decimal(str(inst.dph_base)) + Decimal("0.02")) * 1.3
            if lv["limit"]:
                out["extra_skipped"] = f"level {n} hit the limit: {', '.join(lv['limit'])}"
            elif spent + est_usd > run.max_usd * 0.95:
                out["extra_skipped"] = f"budget: spent {spent:.2f} + est {est_usd:.2f} > {run.max_usd * 0.95:.2f}"
            else:
                todo.append(extra_level)
            run.say(f"G: extra level {extra_level}: {out.get('extra_skipped') or 'yes'}")
    return out


def main(argv=None) -> int:  # pragma: no cover - needs real Vast / S3 / tunnel
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release-sha", required=True); ap.add_argument("--image", required=True); ap.add_argument("--backend-url", required=True)
    ap.add_argument("--control-video", required=True); ap.add_argument("--control-video-beep"); ap.add_argument("--stub-port", type=int, default=8787)
    ap.add_argument("--max-usd", type=float, default=1.3); ap.add_argument("--confirm-spend", type=float); ap.add_argument("--apply", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=150); ap.add_argument("--phases", default="A,B,C")
    ap.add_argument("--jobs-per-instance", type=int, default=0, help="phase B; 0 = the A-phase optimum")
    ap.add_argument("--gpu", default="L40S"); ap.add_argument("--machine-id", type=int, default=0, help="phase G: the Vast machine to rent")
    ap.add_argument("--levels", default="1,2,4,6,8"); ap.add_argument("--extra-level", type=int, default=0); ap.add_argument("--max-dph", default="1.0")
    ap.add_argument("--usd-rub", default="82.1386"); ap.add_argument("--out", type=pathlib.Path, default=ROOT / "results" / "load")
    a = ap.parse_args(argv)
    phases = set(a.phases.split(","))
    cap = 2.0 if phases == {"G"} else 1.5           # phase G: owner budget 2026-11-09 = $2 for the whole L40S task
    if not (a.apply and a.confirm_spend == a.max_usd and a.max_usd <= cap and _tunnel_url_ok(a.backend_url)
            and ("G" not in phases or a.machine_id)):
        print(f"needs --apply --confirm-spend == --max-usd (<= {cap}), a valid tunnel URL (+ --machine-id for G): NOTHING started, NO money spent")
        return 3

    import boto3
    from dispatcher.main import _vast_key
    from dispatcher.token_provider import static_token_provider
    from dispatcher.vast_api import VastClient
    from tools.stub_worker_api import serve_in_thread

    run = Run(a.max_usd, a.max_minutes)
    secret = _secrets.token_urlsafe(32)
    stub = StubState()
    srv, _ = serve_in_thread(stub, secret, port=a.stub_port)
    s3 = boto3.client("s3", endpoint_url=os.environ["S3_ENDPOINT_URL"], aws_access_key_id=os.environ["S3_ACCESS_KEY_ID"],
                      aws_secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"], region_name=os.environ.get("S3_REGION", "ru-1"))
    bucket = os.environ.get("S3_BUCKET_NAME") or os.environ["MODELS_S3_BUCKET"]
    run_id = _secrets.token_hex(4)
    key = f"fast-start-test/{run_id}/input{pathlib.Path(a.control_video).suffix}"
    s3.upload_file(a.control_video, bucket, key)
    beep_key = None
    if a.control_video_beep:
        beep_key = f"fast-start-test/{run_id}/beep{pathlib.Path(a.control_video_beep).suffix}"
        s3.upload_file(a.control_video_beep, bucket, beep_key)
    environ = {**os.environ, "BACKEND_API_URL": a.backend_url, "WORKER_SECRET": secret, "S3_BUCKET_NAME": bucket, "SONYA_RES_SAMPLER": "1",
               **({"SONYA_REQUIRE_NVENC": "1"} if "G" in phases else {})}
    vast = VastClient(_vast_key(os.environ), dry_run=False)
    token = static_token_provider(os.environ["GITHUB_RELEASE_TOKEN"])
    base = dict(enabled=True, dry_run=False, image=a.image, release_sha=a.release_sha, max_creates_per_hour=6, warm_prime=False)

    from dispatcher import launcher
    eff_cores = 0.0
    if "G" in phases:
        probe = FleetConfig(**base, default_gpus=(a.gpu,), fallback_gpus=(), min_vram_gb=24, max_dph=Decimal(a.max_dph), pin_machine_id=a.machine_id,
                            jobs_per_instance=max(int(x) for x in a.levels.split(",")))
        best = launcher.pick_offer(vast.search_offers(launcher.search_query(probe)), probe, set())
        if best is None:
            print(f"machine {a.machine_id} ({a.gpu}) is not offered right now: NOTHING started"); srv.shutdown(); return 3
        eff_cores = float(best.get("cpu_cores_effective") or 0)
    else:
        probe = FleetConfig(**base)
        best = launcher.pick_offer(vast.search_offers(launcher.search_query(probe)), probe, set(), warm=True)
    if best is None:
        print("no warm-eligible RTX 3060 + modern CPU offer right now: NOTHING started"); srv.shutdown(); return 3
    print(f"offer now: {best.get('gpu_name')} + {best.get('cpu_name')} {best.get('geolocation')} ${best.get('dph_total'):.3f}/h "
          f"inet {best.get('inet_down'):.0f} Mbit/s storage ${best.get('storage_cost'):.3f}/GB-month")

    def manager(prefix: str, **over) -> FleetManager:
        cfg = FleetConfig(label_prefix=prefix, **{**base, **over})
        m = FleetManager(cfg, FleetStore(str(pathlib.Path(tempfile.mkdtemp()) / "fleet.db")), vast, stub, token, environ)
        run.managers.append(m)
        return m

    def fetch_res(label: str) -> List[dict]:
        docs = []
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"fast-start-test/res/{label}/"):
            docs += [json.loads(s3.get_object(Bucket=bucket, Key=o["Key"])["Body"].read()) for o in page.get("Contents", [])]
        return docs

    try:
        if "G" in phases:
            single_gpu(run, stub, manager, vast, key, gpu=a.gpu, machine_id=a.machine_id,
                                         levels=[int(x) for x in a.levels.split(",")], extra_level=a.extra_level,
                                         max_dph=Decimal(a.max_dph), fetch_res=fetch_res, eff_cores=eff_cores,
                                         fetch_json=lambda k: json.loads(s3.get_object(Bucket=bucket, Key=k)["Body"].read()))
        else:
            scenario(run, stub, manager, vast, key, phases, a.jobs_per_instance, beep_key)
    except (TimeoutError, VastError, KeyboardInterrupt) as exc:      # Ctrl-C: teardown AND the report of the finished levels
        run.aborted = str(exc) or type(exc).__name__
        run.say(f"ABORT: {exc}")
    finally:
        for m in run.managers:                                # ALWAYS: nothing may be left (running or stopped)
            for i in m.store.all():
                if i.state not in ("destroyed", "lost"):
                    try:
                        vast.destroy(i.id)
                    except Exception as exc:  # noqa: BLE001
                        run.say(f"destroy failed for {i.id}: {type(exc).__name__}")
        for _ in range(24):
            left = [i for i in vast.list_instances() if i.label.startswith(("sonya-pool-", "sonya-pwr-", "sonya-g"))]
            if not left:
                break
            time.sleep(5)
        srv.shutdown()
    left = [i.id for i in vast.list_instances() if i.label.startswith(("sonya-pool-", "sonya-pwr-", "sonya-g"))]
    labels = sorted({i.label for m in run.managers for i in m.store.all()})
    res = {}
    for lab in labels:
        objs = []
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"fast-start-test/res/{lab}/"):
            objs += [o["Key"] for o in page.get("Contents", [])]
        res[lab] = [json.loads(s3.get_object(Bucket=bucket, Key=k)["Body"].read()) for k in sorted(objs)]
    def timelines(label):
        r = s3.list_objects_v2(Bucket=bucket, Prefix=f"fleet-metrics/{label}/")
        return [json.loads(s3.get_object(Bucket=bucket, Key=k)["Body"].read()).get("marks", [])
                for k in sorted(o["Key"] for o in r.get("Contents", []))]
    by_id = {i.id: (m, i) for m in run.managers for i in m.store.all()}
    for ph in run.phases.values():
        for r in ph.values() if isinstance(ph, dict) else []:
            if isinstance(r, dict) and r.get("_ev"):
                iid, boot, since, jid = r.pop("_ev")
                m, i = by_id[iid]
                ev = [e for e in m.store.events(iid) if since is None or e["ts"] >= since]
                if since is None:      # first boot only: up to the first stop
                    stop = next((e["ts"] for e in ev if e["event"] in ("stop_requested", "destroy_requested")), None)
                    ev = [e for e in ev if stop is None or e["ts"] < stop]
                tl = timelines(i.label)
                try:
                    r["startup"] = startup_metrics.breakdown(ev, tl[boot] if len(tl) > boot else [], stub.jobs[jid]["queued_at"])
                except Exception as exc:  # noqa: BLE001
                    r["startup"] = {"error": type(exc).__name__}
    cost = {"usd": float(run.spent()), "rub": float(run.spent() * Decimal(a.usd_rub))}
    report = {"started": datetime.fromtimestamp(run.t0, timezone.utc).isoformat(), "duration_s": round(time.time() - run.t0), "aborted": run.aborted,
              "release": a.release_sha, "image": a.image, "phases": run.phases, "cost": cost, "leftover_instances": left,
              "instances": [{"prefix": m.cfg.label_prefix, "id": i.id, "label": i.label, "role": i.role, "gpu": i.gpu_name, "machine": i.machine_id,
                             "dph": str(i.dph_base), "storage_gb_month": str(i.storage_gb_month), "events": m.store.events(i.id)}
                            for m in run.managers for i in m.store.all()],
              "resources": res, "log": run.log}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "load_report.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({"cost": cost, "left": left, "aborted": run.aborted}, default=str))
    return 0 if not run.aborted and not left else 1


if __name__ == "__main__":
    sys.exit(main())
