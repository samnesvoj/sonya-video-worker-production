"""
model_fetch.py - parallel, verified, resumable-by-retry model download for the worker start-up path.

Why not scripts/model_downloader.py as-is: it downloads one file at a time, trusts any existing file (a truncated file left by an
interrupted download is treated as "present"), writes straight to the final path and has no checksum. Here:
  * all models of the enabled modes are fetched concurrently (files in parallel, each with boto3 multipart concurrency)
  * every download goes to <file>.part, is size- and SHA-256-checked against release/models.lock.json, then atomically renamed
  * a verified file gets a small sidecar (<file>.verified) so a RESUME / restart does not re-hash multi-GB files; any size/mtime
    change invalidates it
  * transient errors and hash mismatches are retried with back-off; a required model that cannot be verified fails readiness;
    an optional one that is absent (404) is skipped
  * no URLs, keys or file contents are ever logged - only model keys, sizes and durations
Models with no pin in the lock (lock still empty) are accepted as before but marked `unverified`, and readiness refuses them when
SONYA_REQUIRE_MODEL_LOCK=1.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

log = logging.getLogger("sonya.model_fetch")
CHUNK = 8 << 20


class ModelFetchError(RuntimeError):
    pass


@dataclass
class Item:
    key: str
    s3_path: Optional[str]
    local: Path
    optional: bool
    sha256: Optional[str] = None
    size: Optional[int] = None
    modes: List[str] = field(default_factory=list)


@dataclass
class Result:
    key: str
    status: str                 # verified | unverified | skipped_optional | cached | failed
    bytes: int = 0
    seconds: float = 0.0
    attempts: int = 0
    error: str = ""


def plan(root: Path, modes: List[str], models_dir: Path, lock: Optional[dict] = None) -> List[Item]:
    """One Item per distinct local file across the enabled modes (shared models are fetched once)."""
    pins = (lock or {}).get("models", {})
    items: Dict[Path, Item] = {}
    for mode in modes:
        cfg = root / "modes" / mode / "mode.yaml"
        if not cfg.is_file():
            continue
        for key, spec in ((yaml.safe_load(cfg.read_text()) or {}).get("models") or {}).items():
            lp = Path(spec.get("local_path") or key)
            if lp.parts and lp.parts[0] == "models":
                lp = Path(*lp.parts[1:]) if len(lp.parts) > 1 else Path(key)
            local = lp if lp.is_absolute() else models_dir / lp
            it = items.get(local)
            if it is None:
                pin = pins.get(key) or {}
                it = items[local] = Item(key, spec.get("s3_path"), local, bool(spec.get("optional", True)), pin.get("sha256"), pin.get("size"))
            else:
                it.optional = it.optional and bool(spec.get("optional", True))     # required by any mode => required
            it.modes.append(mode)
    return list(items.values())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _sidecar(p: Path) -> Path:
    return p.with_name(p.name + ".verified")


def _cached_ok(it: Item) -> bool:
    """True if the file is present AND (when pinned) provably the pinned bytes, without re-hashing if the sidecar still matches."""
    p = it.local
    if not p.is_file() or p.stat().st_size == 0:
        return False
    if it.sha256 is None:
        return True                                           # legacy behaviour; reported as 'unverified'
    st = p.stat()
    if it.size is not None and st.st_size != it.size:
        return False
    sc = _sidecar(p)
    if sc.is_file():
        try:
            d = json.loads(sc.read_text())
            if d.get("sha256") == it.sha256 and d.get("size") == st.st_size and d.get("mtime_ns") == st.st_mtime_ns:
                return True
        except ValueError:
            pass
    if sha256_file(p) == it.sha256:
        _write_sidecar(it)
        return True
    return False


def _write_sidecar(it: Item) -> None:
    st = it.local.stat()
    _sidecar(it.local).write_text(json.dumps({"sha256": it.sha256, "size": st.st_size, "mtime_ns": st.st_mtime_ns}))


def make_client():
    import boto3
    from botocore.config import Config
    return boto3.client("s3", endpoint_url=os.environ["S3_ENDPOINT_URL"], aws_access_key_id=os.environ.get("S3_ACCESS_KEY_ID"),
                        aws_secret_access_key=os.environ.get("S3_SECRET_ACCESS_KEY"), region_name=os.environ.get("S3_REGION", "ru-1"),
                        config=Config(max_pool_connections=64, retries={"max_attempts": 3, "mode": "standard"}, connect_timeout=15, read_timeout=60))


def _download_one(it: Item, client, bucket: str, retries: int, sleep: Callable[[float], None]) -> Result:
    from boto3.s3.transfer import TransferConfig
    cfg = TransferConfig(multipart_threshold=64 << 20, multipart_chunksize=32 << 20, max_concurrency=int(os.environ.get("SONYA_MODEL_STREAMS", "16")), use_threads=True)
    t0, last = time.time(), ""
    for attempt in range(1, retries + 1):
        part = it.local.with_name(it.local.name + ".part")
        try:
            it.local.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, it.s3_path, str(part), Config=cfg)
            size = part.stat().st_size
            if it.size is not None and size != it.size:
                raise ModelFetchError(f"size mismatch {size} != {it.size}")
            if it.sha256 and sha256_file(part) != it.sha256:
                raise ModelFetchError("sha256 mismatch")
            os.replace(part, it.local)
            if it.sha256:
                _write_sidecar(it)
            return Result(it.key, "verified" if it.sha256 else "unverified", size, time.time() - t0, attempt)
        except Exception as exc:  # noqa: BLE001
            part.unlink(missing_ok=True)
            code = getattr(getattr(exc, "response", None), "get", lambda *_: None)("Error") or {}
            if code.get("Code") in ("404", "NoSuchKey", "NotFound"):
                last = "not found"
                break                                          # retrying a missing key is pointless
            last = type(exc).__name__ + (f": {exc}" if isinstance(exc, ModelFetchError) else "")
            if attempt < retries:
                sleep(min(2 ** attempt, 20))
    return Result(it.key, "failed", 0, time.time() - t0, attempt, last)


def fetch_all(items: List[Item], *, client=None, bucket: Optional[str] = None, workers: Optional[int] = None, retries: int = 3,
              sleep: Callable[[float], None] = time.sleep) -> List[Result]:
    """Returns one Result per item. Raises ModelFetchError if a REQUIRED model is not present and verified."""
    results: Dict[str, Result] = {}
    todo: List[Item] = []
    for it in items:
        if _cached_ok(it):
            results[it.key] = Result(it.key, "cached" if it.sha256 else "unverified", it.local.stat().st_size)
        elif not it.s3_path:
            results[it.key] = Result(it.key, "failed" if not it.optional else "skipped_optional", error="no s3_path")
        else:
            todo.append(it)
    if todo:
        client = client or make_client()
        bucket = bucket or (os.environ.get("MODELS_S3_BUCKET") or os.environ.get("S3_BUCKET_NAME") or os.environ.get("S3_BUCKET", ""))
        if not bucket:
            raise ModelFetchError("MODELS_S3_BUCKET not set")
        n = workers or min(int(os.environ.get("SONYA_MODEL_WORKERS", "4")), len(todo))
        with ThreadPoolExecutor(max_workers=n) as pool:
            futs = {pool.submit(_download_one, it, client, bucket, retries, sleep): it for it in todo}
            for f in as_completed(futs):
                r = f.result()
                if r.status == "failed" and futs[f].optional and r.error == "not found":
                    r = Result(r.key, "skipped_optional", 0, r.seconds, r.attempts, r.error)
                results[r.key] = r
    bad = [r for it in items for r in [results[it.key]] if r.status == "failed"]
    if bad:
        raise ModelFetchError("required model(s) not available: " + ", ".join(f"{r.key} ({r.error})" for r in bad))
    return [results[it.key] for it in items]
