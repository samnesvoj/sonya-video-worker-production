#!/usr/bin/env python3
"""ab_fill_test.py - owner A/B test 2026-11-15: re-render the moments of an earlier production job with
crop_mode="fill_9x16" on ONE throwaway Vast instance (own label prefix "sonya-abtest-": the production fleet
"sonya-pool-*" is never listed, touched or counted). Stub worker API on the Mac mini (tunnel), source copied
server-side inside the bucket to fast-start-test/<run>/, hard USD/time guard, instance destroyed at the end.

  python tools/ab_fill_test.py --release-sha <40hex> --image ...@sha256:... --backend-url http://bore.pub:N \
      --source-key users/.../input/original.mp4 --segments selected_segments.json --out DIR --max-usd 1 --apply
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import secrets as _secrets
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict
from decimal import Decimal

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from accounting import usage                        # noqa: E402
from dispatcher.config import FleetConfig           # noqa: E402
from dispatcher.fleet import FleetManager           # noqa: E402
from dispatcher.store import FleetStore             # noqa: E402
from dispatcher.vast_api import VastError           # noqa: E402
from tools.fast_start_test import SITE_VIRALITY_PARAMS  # noqa: E402

PREFIX = "sonya-abtest-"


def main() -> int:  # pragma: no cover - real Vast/S3
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-sha", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--backend-url", required=True)
    ap.add_argument("--source-key", required=True)
    ap.add_argument("--segments", required=True, type=pathlib.Path, help='selected_segments.json, or "full" (own selection)')
    ap.add_argument("--only", default="", help="comma list of 1-based clip indexes to render (default: all)")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--max-usd", type=float, default=1.0)
    ap.add_argument("--max-minutes", type=float, default=75)
    ap.add_argument("--stub-port", type=int, default=8787)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--extra", action="append", default=[],
                    help="more jobs on the same instance: NAME|SOURCE_KEY|SEGMENTS_JSON|ONLY[|PARAMS_JSON[|MODE]] (files go to OUT/NAME)")
    ap.add_argument("--mode", default="virality", help="worker mode of the first job (educational, beep, streamer ...)")
    ap.add_argument("--min-cpu-cores", type=float, default=8.0, help="effective cores of the test host (prod 'good server': 12+)")
    ap.add_argument("--jobs-per-instance", type=int, default=2, help="1 = one job at a time (clean per-stage timings)")
    ap.add_argument("--params", default="", help="JSON merged into the first job's params (e.g. {\"ad_banner\": \"mini\"})")
    a = ap.parse_args()

    def job_params(seg_path: pathlib.Path, only: str) -> Dict[str, Any]:
        if str(seg_path) == "full":            # the whole site path: own selection (editor), boundaries, render
            return dict(SITE_VIRALITY_PARAMS)
        segs = json.loads(seg_path.read_text())["segments"]
        if only:
            keep = {int(x) for x in only.split(",")}
            segs = [s for s in segs if int(s["index"]) in keep]
        fixed = [{k: s[k] for k in ("start", "end", "hook", "keywords", "editor_score", "selection_reason") if k in s}
                 for s in segs]
        # no crop_mode: since 2026-11-15 fill_9x16 is the worker default (this checks the default path)
        return dict(SITE_VIRALITY_PARAMS, fixed_segments=fixed)
    jobs_in = [("", a.source_key, dict(job_params(a.segments, a.only), **json.loads(a.params or "{}")))]
    modes = [a.mode]
    for e in a.extra:
        name, src, segp, only, extra, mode = (e.split("|") + ["", "", ""])[:6]
        jobs_in.append((name, src, dict(job_params(pathlib.Path(segp), only), **json.loads(extra or "{}"))))
        modes.append(mode or "virality")
    params = jobs_in[0][2]
    print(f"release {a.release_sha[:12]} | {[len(j[2].get('fixed_segments') or []) or 'full' for j in jobs_in]} fixed moments | "
          f"max ${a.max_usd} | {'APPLY' if a.apply else 'dry'}")
    if not a.apply:
        print(json.dumps(params, ensure_ascii=False)[:600])
        return 0
    import boto3
    from dispatcher.main import _vast_key
    from dispatcher.token_provider import static_token_provider
    from dispatcher.vast_api import VastClient
    from tools.stub_worker_api import StubState, serve_in_thread
    secret = _secrets.token_urlsafe(32)
    stub = StubState()
    srv, _ = serve_in_thread(stub, secret, port=a.stub_port)
    run_id = _secrets.token_hex(4)
    s3 = boto3.client("s3", endpoint_url=os.environ["S3_ENDPOINT_URL"], aws_access_key_id=os.environ["S3_ACCESS_KEY_ID"],
                      aws_secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"], region_name=os.environ.get("S3_REGION", "ru-1"))
    bucket = os.environ["S3_BUCKET_NAME"]
    keys = []
    for n, (name, src, _) in enumerate(jobs_in):
        key = f"fast-start-test/{run_id}/input{n}.mp4"
        s3.copy({"Bucket": bucket, "Key": src}, bucket, key)       # server side: never re-uploaded
        keys.append(key)
    print("sources copied:", len(keys))
    environ = {**os.environ, "BACKEND_API_URL": a.backend_url, "WORKER_SECRET": secret, "S3_BUCKET_NAME": bucket}
    cfg = FleetConfig(enabled=True, dry_run=False, image=a.image, release_sha=a.release_sha, max_instances=1, jobs_per_instance=a.jobs_per_instance, min_cpu_cores=a.min_cpu_cores,
                      stopped_retention_seconds=600, max_creates_per_hour=2, label_prefix=PREFIX)
    vast = VastClient(_vast_key(os.environ), dry_run=False)
    store = FleetStore(str(pathlib.Path(tempfile.mkdtemp()) / "fleet.db"))
    mgr = FleetManager(cfg, store, vast, stub, static_token_provider(os.environ["GITHUB_RELEASE_TOKEN"]), environ)
    t0 = time.time()
    jids = [stub.enqueue(m, k, j[2], user_id="ab-fill-test") for m, k, j in zip(modes, keys, jobs_in)]
    jid = jids[0]
    say = lambda m: print(f"[+{time.time() - t0:5.0f}s] {m}", flush=True)
    say(f"jobs {[x[:8] for x in jids]} queued")
    last, reason = None, None
    try:
        while True:
            try:
                mgr.tick()
            except VastError as exc:
                if not exc.retryable:
                    raise
                say(f"vast blip: {exc}")
            st = tuple(stub.jobs[x]["status"] for x in jids)
            inst = [(i.id, i.state) for i in store.all()]
            if (st, inst) != last:
                say(f"jobs={st} instances={inst}")
                last = (st, inst)
            if all(x in ("completed", "failed") for x in st):
                break
            cost = usage.cost_between(store, t0, time.time(), time.time())
            usd = cost["gpu"] + cost["disk"]
            if usd > Decimal(str(a.max_usd)) or time.time() - t0 > a.max_minutes * 60:
                reason = f"guard usd={usd:.3f} min={(time.time() - t0) / 60:.0f}"
                say("ABORT " + reason)
                break
            time.sleep(5)
    finally:
        logs = []
        for i in store.all():
            if i.state not in ("destroyed", "lost"):
                try:
                    r = subprocess.run(["vastai", "logs", str(i.id), "--tail", "3000"], capture_output=True, text=True, timeout=120)
                    logs.append(r.stdout + r.stderr)
                except Exception:  # noqa: BLE001
                    pass
                try:
                    vast.destroy(i.id)
                    say(f"destroyed {i.id}")
                except Exception as exc:  # noqa: BLE001
                    say(f"DESTROY FAILED {i.id}: {type(exc).__name__}")
        srv.shutdown()
        a.out.mkdir(parents=True, exist_ok=True)
        red = "\n".join(logs)
        for n in (secret, os.environ.get("S3_SECRET_ACCESS_KEY", ""), os.environ.get("S3_ACCESS_KEY_ID", ""),
                  os.environ.get("GITHUB_RELEASE_TOKEN", "")):
            if n:
                red = red.replace(n, "***")
        (a.out / "worker-log.redacted.txt").write_text(red)
    for x, (name, _, _) in zip(jids, jobs_in):
        d = a.out / name if name else a.out
        d.mkdir(parents=True, exist_ok=True)
        for f in stub.jobs[x]["files"]:
            if f.get("s3_key"):
                s3.download_file(bucket, f["s3_key"], str(d / pathlib.Path(f["s3_key"]).name))
        if name:
            (d / "job.json").write_text(json.dumps({"status": stub.jobs[x]["status"], "error": stub.jobs[x].get("error")},
                                                   ensure_ascii=False))
    j = stub.jobs[jid]
    cost = usage.cost_between(store, t0 - 1, time.time(), time.time())
    for _ in range(12):
        left = [i for i in vast.list_instances() if i.label.startswith(PREFIX)]
        if not left:
            break
        time.sleep(5)
    rep = {"jobs": [stub.jobs[x]["status"] for x in jids], "job": j["status"], "error": j.get("error"), "files": j["files"], "aborted": reason,
           "usd": float(cost["gpu"] + cost["disk"]), "minutes": round((time.time() - t0) / 60, 1),
           "abtest_instances_left": len(left), "history": j["history"]}
    (a.out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str))
    say(f"DONE job={j['status']} usd={rep['usd']:.3f} left={len(left)} files={len(j['files'])}")
    return 0 if all(stub.jobs[x]["status"] == "completed" for x in jids) and not left else 1


if __name__ == "__main__":
    sys.exit(main())
