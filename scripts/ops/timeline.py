"""
timeline.py - start-up timeline of one worker instance (stdlib; best-effort; never raises into the caller).

Marks are appended as JSON lines to $SONYA_TIMELINE_FILE (default $SONYA_STATE_DIR/timeline.jsonl). The worker uploads the finished
timeline to S3 under  fleet-metrics/<WORKER_ID>/<boot-epoch>.json  so the CENTRAL dispatcher can read it (the dispatcher cannot reach
the instance's loopback). Content: stage names, epoch seconds, byte counts, GPU name, release SHA, image version. Never env values,
tokens, URLs, file names of user media, or transcripts.

CLI (used by the shell entrypoints):  python -m scripts.ops.timeline mark NAME | upload
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

SAFE_TYPES = (int, float, str, bool, dict, list, type(None))


def path() -> Path:
    return Path(os.environ.get("SONYA_TIMELINE_FILE") or (Path(os.environ.get("SONYA_STATE_DIR", "/run/sonya")) / "timeline.jsonl"))


def mark(name: str, **extra: Any) -> None:
    try:
        rec = {"name": name, "t": round(time.time(), 3)}
        rec.update({k: v for k, v in extra.items() if isinstance(v, SAFE_TYPES)})
        p = path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except Exception:  # noqa: BLE001 - metrics must never break start-up
        pass


def read(p: Path | None = None) -> List[Dict[str, Any]]:
    try:
        return [json.loads(l) for l in (p or path()).read_text().splitlines() if l.strip()]
    except (OSError, ValueError):
        return []


def snapshot() -> Dict[str, Any]:
    return {"worker_id": os.environ.get("WORKER_ID", ""), "release_sha": os.environ.get("SONYA_RELEASE_SHA", "")[:40],
            "image_version": os.environ.get("SONYA_IMAGE_VERSION", ""), "marks": read()}


def metric_key(snap: Dict[str, Any]) -> str:
    first = min((m["t"] for m in snap["marks"]), default=time.time())
    wid = "".join(c for c in snap["worker_id"] if c.isalnum() or c in "-_") or "unknown"
    return f"fleet-metrics/{wid}/{int(first)}.json"


def upload(client=None) -> bool:
    """Best-effort PUT of the current timeline to S3. Returns False (silently) on any problem."""
    try:
        snap = snapshot()
        if not snap["marks"]:
            return False
        if client is None:
            from scripts.prod_s3_storage import upload_bytes
            upload_bytes(json.dumps(snap).encode(), metric_key(snap), content_type="application/json")
            return True
        bucket = os.environ.get("S3_BUCKET_NAME") or os.environ.get("S3_BUCKET", "")
        client.put_object(Bucket=bucket, Key=metric_key(snap), Body=json.dumps(snap).encode(), ContentType="application/json")
        return True
    except Exception:  # noqa: BLE001
        return False


def upload_async() -> None:
    threading.Thread(target=upload, daemon=True).start()


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "mark":
        mark(sys.argv[2])
    elif len(sys.argv) >= 2 and sys.argv[1] == "upload":
        upload()
