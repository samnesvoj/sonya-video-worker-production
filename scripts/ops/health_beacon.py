"""Tiny «I'm alive» beacon for the self-check (scripts/ops/health_run.py on the VPS).

  bot server 147.45.235.79 : systemd sonya-health-beacon.timer (5 min)
        python3 health_beacon.py --disk / --services sonya-accounting-bot,sonya-group-bot,sonya-ytrelay --s3 worker-v2/health/beacon/bot.json
        (S3 keys from /etc/sonya-ytrelay/env, the relay's own; boto3 from the relay venv)
  Mac mini                 : launchd group.sonya.healthbeacon (10 min), prints JSON piped over ssh to the VPS
        python3 health_beacon.py --disk /System/Volumes/Data
No secrets in the output: ts, host, disk total/free, service states.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import time


def collect(disk_path: str, services) -> dict:
    u = shutil.disk_usage(disk_path)
    st = {}
    for s in services:
        try:
            st[s] = subprocess.run(["systemctl", "is-active", s], capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
        except Exception:  # noqa: BLE001
            st[s] = "unknown"
    return {"ts": time.time(), "host": socket.gethostname()[:40], "disk": {"path": disk_path, "total_gb": round(u.total / 1e9, 2),
            "free_gb": round(u.free / 1e9, 2)}, "services": st}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--disk", default="/")
    ap.add_argument("--services", default="")
    ap.add_argument("--s3", default="")
    a = ap.parse_args()
    data = collect(a.disk, [s for s in a.services.split(",") if s])
    if not a.s3:
        print(json.dumps(data))
        return 0
    import boto3
    c = boto3.client("s3", endpoint_url=os.environ.get("S3_ENDPOINT_URL") or None, region_name=os.environ.get("S3_REGION") or None,
                     aws_access_key_id=os.environ["S3_ACCESS_KEY_ID"], aws_secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"])
    bucket = os.environ.get("S3_BUCKET_NAME") or os.environ.get("MODELS_S3_BUCKET")
    c.put_object(Bucket=bucket, Key=a.s3, Body=json.dumps(data).encode(), ContentType="application/json")
    print("beacon ok", a.s3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
