"""Release pointer (owner 2026-11-18: a new Sonya version must not re-create the Vast servers).

The dispatcher publishes the release every NEW start should run:  s3://<S3_BUCKET_NAME>/worker-v2/release/current.json
    {"sha": "<40 hex>", "image": "<base image the release needs>", "ts": <epoch>}
deploy/vast/run_worker.sh reads it on every container start (create AND resume): another SHA for the same base image ->
bootstrap fetches + verifies that release (public repo, pinned SHA, manifest + lock checked) and the worker runs it.
Any problem (no pointer, S3 down, other image, bootstrap failure) -> the release the instance already has; never fatal.

  python3 -m scripts.ops.release_pointer get     -> prints the SHA to switch to ("" = stay), exit 0
  python3 -m scripts.ops.release_pointer put SHA IMAGE   (dispatcher side; tests)
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Optional

KEY = os.environ.get("SONYA_RELEASE_POINTER_KEY", "worker-v2/release/current.json")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _client(env=os.environ):
    import boto3
    from botocore.config import Config
    return boto3.client("s3", endpoint_url=env.get("S3_ENDPOINT_URL") or None, region_name=env.get("S3_REGION") or None,
                        aws_access_key_id=env.get("S3_ACCESS_KEY_ID"), aws_secret_access_key=env.get("S3_SECRET_ACCESS_KEY"),
                        config=Config(connect_timeout=3, read_timeout=5, retries={"max_attempts": 2}))


def _bucket(env=os.environ) -> str:
    return env.get("S3_BUCKET_NAME") or env.get("S3_BUCKET") or ""


def read(client=None, env=os.environ) -> Optional[dict]:
    bucket = _bucket(env)
    if not bucket:
        return None
    c = client or _client(env)
    try:
        return json.loads(c.get_object(Bucket=bucket, Key=KEY)["Body"].read())
    except Exception:  # noqa: BLE001 - no pointer / S3 trouble = stay on the release we have
        return None


def decide(pointer: Optional[dict], current_sha: str, image: str) -> str:
    """SHA to switch to, or '' to stay."""
    if not isinstance(pointer, dict):
        return ""
    sha = str(pointer.get("sha") or "").strip().lower()
    if not SHA_RE.match(sha) or sha == (current_sha or "").strip().lower():
        return ""
    want = str(pointer.get("image") or "")
    if want and image and want != image:      # that release needs another base image: only a new server can run it
        return ""
    return sha


def publish(sha: str, image: str, client=None, env=os.environ) -> bool:
    """Dispatcher: write the pointer if it differs. True = written."""
    if not SHA_RE.match(sha or ""):
        raise ValueError("release pointer needs a full 40-hex SHA")
    bucket = _bucket(env)
    if not bucket:
        return False
    c = client or _client(env)
    cur = read(c, env)
    if isinstance(cur, dict) and cur.get("sha") == sha and (cur.get("image") or "") == (image or ""):
        return False
    body = json.dumps({"sha": sha, "image": image or "", "ts": int(time.time())}).encode()
    c.put_object(Bucket=bucket, Key=KEY, Body=body, ContentType="application/json")
    return True


def main(argv) -> int:
    if len(argv) >= 2 and argv[1] == "get":
        # SONYA_WORKER_IMAGE is not in the container; Vast sets nothing either -> compare with what the dispatcher passed
        print(decide(read(), os.environ.get("SONYA_RELEASE_SHA", ""), os.environ.get("SONYA_BASE_IMAGE_REF", "")))
        return 0
    if len(argv) >= 4 and argv[1] == "put":
        print("written" if publish(argv[2], argv[3]) else "unchanged")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
