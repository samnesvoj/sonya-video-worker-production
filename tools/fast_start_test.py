#!/usr/bin/env python3
"""
fast_start_test.py - ONE test that measures the whole Fast Start path on Vast and checks the success criteria (docs/FAST_START_TEST.md).

Scenario (same code path for simulation and for the live run):
  1. enqueue job 1 (virality) in a STUB worker API (never the production queue)  -> fleet CREATEs an instance -> bootstrap -> models -> CUDA -> ready -> processing
  2. job 1 completes -> 60 s idle -> the fleet STOPs the instance                     (measures the stop latency)
  3. enqueue job 2 (beep)                                                            -> the fleet RESUMEs the same instance (measures resume path)
  4. job 2 completes -> teardown: DESTROY, verify no sonya-pool instance is left
Measured per phase from the dispatcher journal + the worker's uploaded timeline: instance creation + Docker pull, container start, code download,
models, CUDA init, readiness, prewarm, claim, first processing, USER WAIT; plus cost (USD/RUB) and the pass/fail verdict per criterion.

  python tools/fast_start_test.py simulate                      # no network, no money: proves the harness + criteria evaluator
  python tools/fast_start_test.py live --release-sha ... --image ...@sha256:... --backend-url https://<tunnel> --control-video clip.mp4 --max-usd 3
        (prints the preflight and the plan and STOPS unless --apply --confirm-spend 3 are both given)
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import secrets as _secrets
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Callable, Dict, List, Optional

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from accounting import usage                       # noqa: E402
from dispatcher import startup_metrics             # noqa: E402
from dispatcher.config import FleetConfig          # noqa: E402
from dispatcher.fleet import FleetManager          # noqa: E402
from dispatcher.store import FleetStore            # noqa: E402
from tools.stub_worker_api import StubState        # noqa: E402
from scripts.ops.models_prefix import s3_key       # noqa: E402

THRESHOLDS = {
    "create_request_to_ready_s": 480,        # new instance: request -> worker ready
    "create_user_wait_s": 600,               # new instance: job queued -> processing starts (what the user feels)
    "resume_request_to_ready_s": 120,
    "resume_user_wait_s": 150,
    "stop_after_last_job_s": 90,             # 60 s idle timeout + a tick or two
    "max_usd": 3.0,
    "max_failed_jobs": 0,
}


# ----------------------------------------------------------------------------- criteria
# What the production site really sends (auth.js collectFormData + apiCreateVideoJob): only the user's choices,
# never Cut settings -> the worker applies the Mac mini recipe itself (scripts/shared/cut_profile.py).
SITE_VIRALITY_PARAMS = {"source": {"mode": "upload", "url": "", "fileName": "control.mp4", "platform": "upload"},
                        "brand": "sonya", "clipType": "viral",
                        "settings": {"subtitleStyle": "viral", "voiceoverEnabled": False, "voiceType": "male-1",
                                     "translation": "none"},
                        "frontend": "legacy_static_sonya", "production_endpoint": "/api/generation/jobs"}
SITE_BEEP_PARAMS = {"source": {"mode": "upload", "url": "", "fileName": "control_beep.mp4", "platform": "upload"},
                    "brand": "sonya", "frontend": "legacy_static_sonya", "production_endpoint": "/api/generation/jobs"}

def evaluate(rep: dict, thr: dict = THRESHOLDS) -> List[dict]:
    c: List[dict] = []

    def add(name, ok, value, limit=""):
        c.append({"criterion": name, "ok": bool(ok), "value": value, "limit": limit})

    s = rep["phases"]
    cs, rs = s["create"]["stages"], s["resume"]["stages"]
    add("all jobs completed, none failed", rep["jobs_completed"] == rep["jobs_total"] and rep["failures"] <= thr["max_failed_jobs"], f'{rep["jobs_completed"]}/{rep["jobs_total"]} completed, {rep["failures"]} failures', "all, 0 failures")
    add("create: request -> ready", cs.get("request_to_ready") is not None and cs["request_to_ready"] <= thr["create_request_to_ready_s"], cs.get("request_to_ready"), f'<= {thr["create_request_to_ready_s"]} s')
    add("create: user wait (queued -> processing)", cs.get("user_wait") is not None and cs["user_wait"] <= thr["create_user_wait_s"], cs.get("user_wait"), f'<= {thr["create_user_wait_s"]} s')
    add("resume: request -> ready", rs.get("request_to_ready") is not None and rs["request_to_ready"] <= thr["resume_request_to_ready_s"], rs.get("request_to_ready"), f'<= {thr["resume_request_to_ready_s"]} s')
    add("resume: user wait", rs.get("user_wait") is not None and rs["user_wait"] <= thr["resume_user_wait_s"], rs.get("user_wait"), f'<= {thr["resume_user_wait_s"]} s')
    add("resume reused the cached release/models (no re-download)", s["resume"]["release_cached"] is True, s["resume"]["release_cached"], "release_cached=true")
    add("auto-stop after the last job", rep["stop_latency_s"] is not None and rep["stop_latency_s"] <= thr["stop_after_last_job_s"], rep["stop_latency_s"], f'<= {thr["stop_after_last_job_s"]} s')
    add("one instance only, created once", rep["creates"] == 1 and rep["resumes"] == 1, f'creates={rep["creates"]} resumes={rep["resumes"]}', "1 create, 1 resume")
    add("cost within the limit", rep["cost"]["usd"] <= thr["max_usd"], f'${rep["cost"]["usd"]:.3f} / {rep["cost"]["rub"]:.1f} RUB', f'<= ${thr["max_usd"]}')
    add("teardown: no instance left running or stopped", rep["leftover_instances"] == 0, rep["leftover_instances"], "0")
    q = rep.get("quality") or {}
    add("outputs valid (ffprobe: playable, duration > 0, video+audio)", q.get("outputs_valid") is True, q.get("outputs_valid"), "true")
    add("output vs Mac mini reference: no REVIEW flags", q.get("compare_verdict") == "NO_VISIBLE_REGRESSION_IN_THESE_CHECKS", q.get("compare_verdict"), "no regression flags (then a human watches it)")
    sec = rep.get("secret_scan") or {}
    add("no token / secret in worker logs and timelines", sec.get("clean") is True, sec.get("clean") if sec else "not checked", "clean")
    return c


# ----------------------------------------------------------------------------- environments
@dataclass
class Env:
    clock: Callable[[], float]
    sleep: Callable[[float], None]
    mgr: FleetManager
    store: FleetStore
    stub: StubState
    step: Callable[[], None]                                  # simulation hook (worker emulation); no-op for live
    timelines: Callable[[str], List[List[dict]]]              # label -> [timeline of boot 1, timeline of boot 2, ...]
    vast: object
    tick_s: float = 5.0
    log: List[str] = field(default_factory=list)
    collect_logs: Callable[[int], str] = lambda iid: ""      # live: container log of the instance, captured BEFORE it is destroyed
    logs: List[str] = field(default_factory=list)


from dispatcher.vast_api import VastError  # noqa: E402

def run_scenario(env: Env, max_usd: float, max_minutes: float, control_key: str, usd_rub: Decimal, beep_key: Optional[str] = None) -> dict:
    t0 = env.clock()
    say = lambda m: env.log.append(f"[+{env.clock()-t0:6.0f}s] {m}")
    aborted = {"reason": None}

    def tick() -> None:
        try:
            env.mgr.tick()
        except VastError as exc:          # a Vast/network blip (run 3 died on one URLError): skip this tick like the
            if not exc.retryable:         # production loop (dispatcher/main.py) does; the guards below still apply
                raise
            say(f"vast api blip ({exc}) - tick skipped")
        env.step()
        cost = usage.cost_between(env.store, t0, env.clock(), env.clock())
        usd = cost["gpu"] + cost["disk"]
        if usd > Decimal(str(max_usd)) or env.clock() - t0 > max_minutes * 60:
            aborted["reason"] = f"budget/time guard (usd={usd:.3f}, minutes={(env.clock()-t0)/60:.0f})"
            raise TimeoutError(aborted["reason"])
        env.sleep(env.tick_s)

    def wait(pred, timeout_s, what):
        end = env.clock() + timeout_s
        while not pred():
            if env.clock() > end:
                raise TimeoutError(f"timeout waiting for {what}")
            tick()
        say(what)

    j1 = j2 = None
    try:
        j1 = env.stub.enqueue("virality", control_key, SITE_VIRALITY_PARAMS); say("job 1 (virality) queued")
        done = lambda jid: (lambda: env.stub.jobs[jid]["status"] in ("completed", "failed"))
        wait(done(j1), 25 * 60, "job 1 finished")
        insts = env.store.all(); iid = insts[0].id if insts else None
        wait(lambda: env.store.get(iid).state in ("stopped", "stopping"), 5 * 60, "instance stop requested")
        wait(lambda: env.store.get(iid).state == "stopped", 3 * 60, "instance stopped")
        j2 = env.stub.enqueue("beep", beep_key or control_key, SITE_BEEP_PARAMS); say("job 2 (beep) queued")
        wait(done(j2), 25 * 60, "job 2 finished (via RESUME)")
        wait(lambda: env.store.get(iid).state in ("stopped", "stopping"), 5 * 60, "instance stopped again")
    except TimeoutError as exc:
        aborted["reason"] = aborted["reason"] or str(exc)
        say(f"ABORT: {exc}")
    finally:                                                   # ALWAYS tear down, whatever happened
        for i in env.store.all():
            if i.state not in ("destroyed", "lost"):
                try:
                    env.logs.append(env.collect_logs(i.id) or "")
                except Exception:  # noqa: BLE001
                    env.logs.append("")
                try:
                    env.vast.destroy(i.id)
                except Exception as exc:  # noqa: BLE001
                    say(f"destroy failed for {i.id}: {type(exc).__name__}")
        for _ in range(12):
            if not [i for i in env.vast.list_instances() if i.label.startswith(env.mgr.cfg.label_prefix)]:
                break
            env.sleep(env.tick_s)
    end = env.clock()
    left = [i for i in env.vast.list_instances() if i.label.startswith(env.mgr.cfg.label_prefix)]
    return build_report(env, t0, end, [j for j in (j1, j2) if j], usd_rub, len(left), aborted["reason"])


def build_report(env: Env, t0: float, end: float, job_ids: List[str], usd_rub: Decimal, leftover: int, aborted: Optional[str]) -> dict:
    insts = env.store.all()
    inst = insts[0] if insts else None
    ev = env.store.events(inst.id) if inst else []
    names = [e["event"] for e in ev]
    tls = env.timelines(inst.label) if inst else []
    resume_at = next((e["ts"] for e in ev if e["event"] == "resume_requested"), None)
    jobs = [env.stub.jobs[j] for j in job_ids]
    queued = {j["id"]: j["queued_at"] for j in jobs}
    ev_create = [e for e in ev if resume_at is None or e["ts"] < resume_at]
    ev_resume = [e for e in ev if resume_at is not None and e["ts"] >= resume_at]
    create = startup_metrics.breakdown(ev_create, tls[0] if tls else [], queued.get(job_ids[0]) if job_ids else None)
    resume = startup_metrics.breakdown(ev_resume, tls[1] if len(tls) > 1 else [], queued.get(job_ids[1]) if len(job_ids) > 1 else None)
    last_term = max((j["history"][-1][1] for j in jobs[:1] if j["status"] in ("completed", "failed")), default=None)
    stop_req = next((e["ts"] for e in ev if e["event"] == "stop_requested"), None)
    cost = usage.cost_between(env.store, t0 - 1, end, end)
    usd = cost["gpu"] + cost["disk"]
    return {"mode": "live", "started": datetime.fromtimestamp(t0, timezone.utc).isoformat(), "duration_s": round(end - t0), "aborted": aborted,
            "phases": {"create": create, "resume": resume},
            "stop_latency_s": round(stop_req - last_term, 1) if stop_req and last_term else None,
            "jobs_total": 2, "jobs_completed": sum(1 for j in jobs if j["status"] == "completed"), "failures": len(env.stub.failures) + sum(1 for j in jobs if j["status"] == "failed"),
            "creates": names.count("created"), "resumes": names.count("resume_requested"), "leftover_instances": leftover,
            "cost": {"usd": float(usd), "rub": float(usd * usd_rub), "gpu_hours": float(cost["gpu_hours"]), "usd_rub": str(usd_rub), "basis": "CALCULATED from the lifecycle journal x offer price (not a Vast charge)"},
            "events": [{"t": round(e["ts"] - t0, 1), "event": e["event"]} for e in ev], "log": env.log,
            "job_files": {j["id"]: j["files"] for j in jobs}}


# ----------------------------------------------------------------------------- simulation
def build_sim_env(worker_ok: bool = True, models_s: float = 40.0, pull_s: float = 90.0, secret_leak: bool = False):
    from tests.fleet.fakes import ENV, FakeVast, IMAGE, SHA
    now = [1_000_000.0]
    clock = lambda: now[0]

    class SimVast(FakeVast):
        def create(self, offer_id, payload):
            iid = super().create(offer_id, payload)
            self.inst[iid].update(status="loading", ready_at=clock() + pull_s, born=clock())
            return iid

        def resume(self, iid):
            super().resume(iid)
            self.inst[iid].update(ready_at=clock() + 3, status="running", resumed=clock())

        def list_instances(self):
            for d in self.inst.values():
                if d["status"] == "loading" and clock() >= d["ready_at"]:
                    d["status"] = "running"
            return super().list_instances()

    vast = SimVast(); vast.dry_run = False
    stub = StubState(clock)
    cfg = FleetConfig(enabled=True, dry_run=False, image=IMAGE, release_sha=SHA, stopped_retention_seconds=1800, max_instances=1)
    store = FleetStore()
    mgr = FleetManager(cfg, store, vast, stub, lambda: "ghs_SIM", ENV, clock)
    timelines: Dict[str, List[List[dict]]] = {}
    state = {"boots": {}}

    def step():
        for iid, d in list(vast.inst.items()):
            lab = d["label"]
            if d["status"] != "running":
                state["boots"].pop(iid, None)
                continue
            b = state["boots"].get(iid)
            inst = store.get(iid)
            if b is None and inst and inst.state == "running":
                cold = len(timelines.get(lab, [])) == 0
                t = clock(); m = []
                mk = lambda n, dt, **kw: m.append({"name": n, "t": round(t + dt, 3), **kw})
                mk("entrypoint_start", 4, release_cached="no" if cold else "yes"); mk("bootstrap_done", 4 + (3 if cold else 0.5)); mk("run_worker_start", 8); mk("preflight_done", 9)
                mk("models_start", 9.5); mk("models_done", 9.5 + (models_s if cold else 0.2), bytes_downloaded=160_000_000 if cold else 0)
                mk("cuda_init_start", 10 + (models_s if cold else 0.2)); mk("cuda_init_done", 14 + (models_s if cold else 0.2), ms=3900, gpu="SIM RTX 3090")
                mk("readiness_ok", 15 + (models_s if cold else 0.2)); mk("worker_started", 16 + (models_s if cold else 0.2)); mk("prewarm_done", 22 + (models_s if cold else 0.2))
                if secret_leak:
                    m[0]["token"] = "ghs_LEAKED_TOKEN"
                timelines.setdefault(lab, []).append(m)
                b = state["boots"][iid] = {"ready_at": t + 22 + (models_s if cold else 0.2), "job": None, "end": None, "marks": m}
            if b and clock() >= b["ready_at"]:
                if b["job"] is None:
                    j = stub.claim(lab, ["virality", "beep"]) if worker_ok else None
                    if j:
                        b["job"] = j["id"]; b["end"] = clock() + 70
                        b["marks"] += [{"name": "first_job_claimed", "t": clock()}, {"name": "first_job_processing", "t": clock() + 2}]
                        stub.status(j["id"], "mode_running")
                elif clock() >= b["end"]:
                    stub.add_file(b["job"], {"file_type": "output", "s3_key": f"outputs/{b['job']}.mp4", "filename": "clip.mp4", "size_bytes": 1})
                    stub.complete(b["job"], {"s3_output_key": "x", "clip_count": 1, "processing_ms": 70000}); b["job"] = None; b["end"] = None

    class _Sleep:
        def __call__(self, s): now[0] += s

    env = Env(clock, _Sleep(), mgr, store, stub, step, lambda lab: timelines.get(lab, []), vast)
    return env


def simulate(args) -> dict:
    env = build_sim_env()
    rep = run_scenario(env, args.max_usd, 90, "fast-start-test/sim/input.mp4", Decimal(args.usd_rub))
    rep["mode"] = "SIMULATION"
    rep["quality"] = {"outputs_valid": True, "compare_verdict": "NO_VISIBLE_REGRESSION_IN_THESE_CHECKS", "note": "SIMULATED"}
    secrets_blob = json.dumps(rep) + json.dumps(env.timelines(env.store.all()[0].label))
    rep["secret_scan"] = {"clean": not any(x in secrets_blob for x in ("ghs_", "LEAK", "WSECRET", "AKIA")), "note": "SIMULATED"}
    return rep


# ----------------------------------------------------------------------------- live (preflight + run); NOT executed in CI
def preflight(args) -> List[dict]:
    from dispatcher.launcher import DIGEST_RE, SHA_RE
    res: List[dict] = []
    add = lambda n, ok, d="": res.append({"check": n, "ok": bool(ok), "detail": d})
    add("release SHA is a full 40-hex commit", SHA_RE.match(args.release_sha or "") is not None, (args.release_sha or "")[:12])
    add("image is pinned by @sha256 digest", DIGEST_RE.search(args.image or "") is not None)
    lock = json.loads((ROOT / "release" / "models.lock.json").read_text()); man = json.loads((ROOT / "release" / "models.required.json").read_text())
    missing = [m["key"] for m in man["models"] if m.get("required_for_gpu_test", True) and m["key"] not in lock.get("models", {})]
    add("models.lock.json pins every required model", not missing, f"missing: {missing}" if missing else f'{len(lock["models"])} pinned')
    add("control video exists", args.control_video and pathlib.Path(args.control_video).is_file(), str(args.control_video))
    beep = getattr(args, "control_video_beep", None)
    if beep:
        add("BEEP control video exists", pathlib.Path(beep).is_file(), str(beep))
    add("spend limit given and confirmed", args.apply and args.confirm_spend == args.max_usd, f"max_usd={args.max_usd}")
    add("max-usd <= 3", args.max_usd <= 3)
    app = all(os.environ.get(n) for n in ("GH_APP_ID", "GH_APP_INSTALLATION_ID", "GH_APP_PRIVATE_KEY_FILE"))
    add("GitHub release access: GITHUB_RELEASE_TOKEN or GH_APP_* (value not shown)", bool(os.environ.get("GITHUB_RELEASE_TOKEN")) or app,
        "static token" if os.environ.get("GITHUB_RELEASE_TOKEN") else ("GitHub App" if app else "none"))
    for name in ("S3_ENDPOINT_URL", "S3_REGION", "S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY", "MODELS_S3_BUCKET"):
        add(f"env {name} present (value not shown)", bool(os.environ.get(name)))
    add("Vast API key file present (chmod 600)", bool(os.environ.get("VAST_API_KEY_FILE")) and (pathlib.Path(os.environ["VAST_API_KEY_FILE"]).stat().st_mode & 0o077) == 0 if os.environ.get("VAST_API_KEY_FILE") and pathlib.Path(os.environ["VAST_API_KEY_FILE"]).exists() else False)
    add("backend URL is https, or the bore.pub test tunnel (stub only, throwaway secret)", _tunnel_url_ok(args.backend_url), "")
    if getattr(args, "online_checks", False):
        res += online_checks(args, lock)
    return res


def _tunnel_url_ok(url: Optional[str]) -> bool:
    """https (any tunnel) or plain http to bore.pub: bore is a raw TCP tunnel that can re-open the SAME public port
    after a drop, so the worker's BACKEND_API_URL never changes mid-run (pinggy's free URL died after ~18 min in run 2).
    Plain http is acceptable ONLY for the stub API: the worker secret is throwaway, made for this run."""
    import re as _re
    url = url or ""
    return url.startswith("https://") or bool(_re.fullmatch(r"http://bore\.pub:\d{4,5}", url))


def _http(url: str, headers: Optional[dict] = None, method: str = "GET", timeout: int = 20):
    import urllib.error
    import urllib.request
    req = urllib.request.Request(url, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, b""
    except Exception as e:  # noqa: BLE001
        return None, type(e).__name__.encode()


def online_checks(args, lock: dict) -> List[dict]:
    """Free, read-only checks against GitHub / GHCR / S3 / the tunnel. No Vast call, nothing written. Values of secrets are never shown."""
    res: List[dict] = []
    add = lambda n, ok, d="": res.append({"check": n, "ok": bool(ok), "detail": d})
    repo = os.environ.get("SONYA_RELEASE_REPO", "samnesvoj/sonya-video-worker-production")
    tok = os.environ.get("GITHUB_RELEASE_TOKEN")
    if tok and args.release_sha:
        st, body = _http(f"https://api.github.com/repos/{repo}/commits/{args.release_sha}", {"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json"})
        add("token can read the release SHA on GitHub (what bootstrap fetches)", st == 200, f"HTTP {st}")
        if st == 200:
            st2, _ = _http(f"https://api.github.com/repos/{repo}/contents/release/models.lock.json?ref={args.release_sha}", {"Authorization": f"Bearer {tok}"})
            add("release SHA contains release/models.lock.json", st2 == 200, f"HTTP {st2}")
    img = args.image or ""
    if "@sha256:" in img and img.startswith("ghcr.io/"):
        name, digest = img[len("ghcr.io/"):].split("@", 1)
        st, body = _http(f"https://ghcr.io/token?scope=repository:{name}:pull")
        ok = False
        if st == 200:
            t = json.loads(body).get("token", "")
            st, _ = _http(f"https://ghcr.io/v2/{name}/manifests/{digest}", {"Authorization": f"Bearer {t}", "Accept": "application/vnd.oci.image.index.v1+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json"}, "HEAD")
            ok = st == 200
        add("image pullable WITHOUT login (Vast pulls anonymously)", ok, f"HTTP {st}" + ("" if ok else " - make the GHCR package public (base image has no SONYA code)"))
    if args.backend_url and _tunnel_url_ok(args.backend_url):
        from tools.stub_worker_api import serve_in_thread
        try:
            srv, _ = serve_in_thread(StubState(), _secrets.token_urlsafe(16), port=getattr(args, "stub_port", 8787))   # same port live() will use
        except OSError as e:
            srv = None; add("stub API port is free", False, f"{type(e).__name__} on port {getattr(args, 'stub_port', 8787)}")
        if srv:
            try:
                st, _ = _http(args.backend_url.rstrip("/") + "/healthz")
                add("tunnel reaches the stub API (GET /healthz through the tunnel)", st == 200, f"HTTP {st}")
            finally:
                srv.shutdown(); srv.server_close()
    if all(os.environ.get(n) for n in ("S3_ENDPOINT_URL", "S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY", "MODELS_S3_BUCKET")):
        try:
            import boto3
            s3 = boto3.client("s3", endpoint_url=os.environ["S3_ENDPOINT_URL"], aws_access_key_id=os.environ["S3_ACCESS_KEY_ID"],
                              aws_secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"], region_name=os.environ.get("S3_REGION", "ru-1"))
            bad = []
            for k, pin in lock.get("models", {}).items():
                try:
                    h = s3.head_object(Bucket=os.environ["MODELS_S3_BUCKET"], Key=s3_key(pin["s3_path"]))
                    if h["ContentLength"] != pin["size"] or h.get("Metadata", {}).get("sha256") != pin["sha256"]:
                        bad.append(f"{k}: differs")
                except Exception as e:  # noqa: BLE001
                    bad.append(f"{k}: {type(e).__name__}")
            add("every locked model is in S3 with the locked size + sha256", not bad, "; ".join(bad) if bad else f'{len(lock.get("models", {}))} ok')
        except Exception as e:  # noqa: BLE001
            add("every locked model is in S3 with the locked size + sha256", False, type(e).__name__)
    return res


def live(args) -> dict:                                       # pragma: no cover - needs real Vast/S3/GitHub; documented in docs/FAST_START_TEST.md
    pf = preflight(args)
    for r in pf:
        print(("PASS " if r["ok"] else "FAIL ") + r["check"] + (f"  [{r['detail']}]" if r["detail"] else ""))
    if not all(r["ok"] for r in pf):
        print("\nPreflight failed or --apply/--confirm-spend missing: NOTHING was started, NO money spent.")
        sys.exit(3)
    import boto3
    from dispatcher.main import _vast_key
    from dispatcher.token_provider import github_app_token_provider, static_token_provider
    from dispatcher.vast_api import VastClient
    from tools.stub_worker_api import serve_in_thread
    secret = _secrets.token_urlsafe(32)                       # throwaway worker secret for this run only
    stub = StubState(); srv, local = serve_in_thread(stub, secret, port=args.stub_port)   # start `cloudflared tunnel --url http://127.0.0.1:<port>` first
    run_id = _secrets.token_hex(4)
    s3 = boto3.client("s3", endpoint_url=os.environ["S3_ENDPOINT_URL"], aws_access_key_id=os.environ["S3_ACCESS_KEY_ID"], aws_secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"], region_name=os.environ.get("S3_REGION", "ru-1"))
    bucket = os.environ.get("S3_BUCKET_NAME") or os.environ["MODELS_S3_BUCKET"]
    key = f"fast-start-test/{run_id}/input{pathlib.Path(args.control_video).suffix}"
    s3.upload_file(args.control_video, bucket, key)
    beep_key = None
    if getattr(args, "control_video_beep", None):
        beep_key = f"fast-start-test/{run_id}/beep{pathlib.Path(args.control_video_beep).suffix}"
        s3.upload_file(args.control_video_beep, bucket, beep_key)
    environ = {**os.environ, "BACKEND_API_URL": args.backend_url, "WORKER_SECRET": secret, "S3_BUCKET_NAME": bucket}
    cfg = FleetConfig(enabled=True, dry_run=False, image=args.image, release_sha=args.release_sha, max_instances=1, stopped_retention_seconds=1800, max_creates_per_hour=2)
    vast = VastClient(_vast_key(os.environ), dry_run=False)
    store = FleetStore(str(pathlib.Path(tempfile.mkdtemp()) / "fleet.db"))
    mgr = FleetManager(cfg, store, vast, stub, (static_token_provider(os.environ["GITHUB_RELEASE_TOKEN"]) if os.environ.get("GITHUB_RELEASE_TOKEN")
                                                     else github_app_token_provider(cfg.release_repo.split("/")[-1], os.environ)), environ)

    def timelines(label):
        r = s3.list_objects_v2(Bucket=bucket, Prefix=f"fleet-metrics/{label}/")
        keys = sorted(o["Key"] for o in r.get("Contents", []))
        return [json.loads(s3.get_object(Bucket=bucket, Key=k)["Body"].read()).get("marks", []) for k in keys]

    import subprocess

    def collect_logs(iid: int) -> str:
        out = subprocess.run(["vastai", "logs", str(iid), "--tail", "3000"], capture_output=True, text=True, timeout=120)
        return out.stdout + out.stderr

    env = Env(time.time, time.sleep, mgr, store, stub, lambda: None, timelines, vast, tick_s=5.0, collect_logs=collect_logs)
    try:
        rep = run_scenario(env, args.max_usd, args.max_minutes, key, Decimal(args.usd_rub), beep_key)
    finally:
        srv.shutdown()
    rep["quality"] = {"outputs_valid": None, "compare_verdict": None, "note": "download the outputs listed in job_files, run tools/compare_runs.py vs the Mac mini reference, watch them; then set these two fields"}
    blob = "\n".join(env.logs) + json.dumps([env.timelines(i.label) for i in env.store.all()])
    needles = [secret, os.environ.get("S3_SECRET_ACCESS_KEY", ""), os.environ.get("S3_ACCESS_KEY_ID", ""), os.environ.get("GITHUB_RELEASE_TOKEN", ""),
               os.environ.get("TIMEWEB_GEMINI_API_KEY", ""), os.environ.get("OPENROUTER_API_KEY", ""), "ghs_", "github_pat_", "BEGIN PRIVATE KEY"]
    have_logs = any(l.strip() for l in env.logs)
    rep["secret_scan"] = {"clean": (not any(n and n in blob for n in needles)) if have_logs else None, "note": "scanned container log + timelines for the run secret, S3 keys, GitHub tokens" if have_logs else "could not capture logs: scan them by hand BEFORE trusting this"}
    (args.out).mkdir(parents=True, exist_ok=True)
    red = "\n".join(env.logs)
    for n in needles:
        if n:
            red = red.replace(n, "***")
    (args.out / "worker-log.redacted.txt").write_text(red)
    return rep


def render_md(rep: dict, crit: List[dict]) -> str:
    L = [f"# Fast Start test - {'PASS' if all(c['ok'] for c in crit) else 'FAIL'}" + (" (SIMULATION: synthetic timings, validates the harness only - NOT a measurement)" if rep.get("mode") == "SIMULATION" else ""), f"duration {rep['duration_s']} s; aborted: {rep['aborted']}", "", "| criterion | result | value | limit |", "|---|---|---|---|"]
    L += [f"| {c['criterion']} | {'PASS' if c['ok'] else '**FAIL**'} | {c['value']} | {c['limit']} |" for c in crit]
    for ph in ("create", "resume"):
        L += ["", f"### {ph} - stages (s)", "| stage | seconds |", "|---|---|"] + [f"| {k} | {v} |" for k, v in rep["phases"][ph]["stages"].items()]
    L += ["", f"Cost: **${rep['cost']['usd']:.3f} / {rep['cost']['rub']:.2f} RUB** ({rep['cost']['basis']}); GPU {rep['cost']['gpu_hours']:.3f} h", "", "Events:"] + [f"- +{e['t']}s {e['event']}" for e in rep["events"]]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["simulate", "live"])
    ap.add_argument("--release-sha"); ap.add_argument("--image"); ap.add_argument("--backend-url"); ap.add_argument("--control-video"); ap.add_argument("--control-video-beep", help="separate input for job 2 (beep); default: --control-video")
    ap.add_argument("--no-online-checks", dest="online_checks", action="store_false", help="live preflight: skip the free GitHub/GHCR/tunnel/S3 read-only checks")
    ap.add_argument("--stub-port", type=int, default=8787); ap.add_argument("--max-usd", type=float, default=3.0); ap.add_argument("--confirm-spend", type=float); ap.add_argument("--max-minutes", type=float, default=90)
    ap.add_argument("--usd-rub", default="82.1386"); ap.add_argument("--apply", action="store_true"); ap.add_argument("--out", type=pathlib.Path, default=ROOT / "results" / "fast-start")
    a = ap.parse_args(argv)
    rep = simulate(a) if a.mode == "simulate" else live(a)
    crit = evaluate(rep)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "report.json").write_text(json.dumps({**rep, "criteria": crit}, indent=1, default=str))
    (a.out / "report.md").write_text(render_md(rep, crit))
    print(render_md(rep, crit))
    return 0 if all(c["ok"] for c in crit) else 1


if __name__ == "__main__":
    sys.exit(main())
