import hashlib
import json
import os
import time

import boto3
import pytest
from moto import mock_aws

from dispatcher import startup_metrics as sm
from scripts.ops import model_fetch, timeline

BUCKET = "models-bucket"


def _setup(tmp_path, files):
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=BUCKET)
    items, pins = [], {}
    for name, data in files.items():
        s3.put_object(Bucket=BUCKET, Key=f"models/{name}", Body=data)
        pins[name] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
        items.append(model_fetch.Item(name, f"models/{name}", tmp_path / "m" / f"{name}.pt", False, pins[name]["sha256"], len(data), ["virality"]))
    return s3, items


@mock_aws
def test_parallel_verified_download_and_atomic_rename(tmp_path):
    files = {f"m{i}": os.urandom(300_000) + bytes([i]) for i in range(5)}
    s3, items = _setup(tmp_path, files)
    res = model_fetch.fetch_all(items, client=s3, bucket=BUCKET, workers=4)
    assert [r.status for r in res] == ["verified"] * 5 and sum(r.bytes for r in res) == sum(len(v) for v in files.values())
    for it in items:
        assert it.local.read_bytes() == files[it.key] and not it.local.with_name(it.local.name + ".part").exists()


@mock_aws
def test_hash_mismatch_is_retried_then_fails_and_leaves_no_file(tmp_path):
    s3, items = _setup(tmp_path, {"a": b"x" * 5000})
    items[0].sha256 = "0" * 64                                   # the pin does not match what is in S3
    with pytest.raises(model_fetch.ModelFetchError, match="not available"):
        model_fetch.fetch_all(items, client=s3, bucket=BUCKET, retries=3, sleep=lambda s: None)
    assert not items[0].local.exists() and not items[0].local.with_name("a.pt.part").exists()


@mock_aws
def test_truncated_file_from_an_interrupted_download_is_not_trusted(tmp_path):
    data = os.urandom(200_000)
    s3, items = _setup(tmp_path, {"a": data})
    items[0].local.parent.mkdir(parents=True)
    items[0].local.write_bytes(data[:1000])                      # what the OLD downloader would have accepted as "present"
    res = model_fetch.fetch_all(items, client=s3, bucket=BUCKET)
    assert res[0].status == "verified" and items[0].local.read_bytes() == data


@mock_aws
def test_verified_sidecar_skips_rehash_and_detects_changes(tmp_path, monkeypatch):
    data = os.urandom(100_000)
    s3, items = _setup(tmp_path, {"a": data})
    model_fetch.fetch_all(items, client=s3, bucket=BUCKET)
    calls = {"n": 0}
    real = model_fetch.sha256_file
    monkeypatch.setattr(model_fetch, "sha256_file", lambda p: calls.__setitem__("n", calls["n"] + 1) or real(p))
    res = model_fetch.fetch_all(items, client=s3, bucket=BUCKET)           # RESUME / restart path
    assert res[0].status == "cached" and calls["n"] == 0
    items[0].local.write_bytes(os.urandom(100_000))                        # content changed behind our back, same size
    res = model_fetch.fetch_all(items, client=s3, bucket=BUCKET)
    assert res[0].status == "verified" and items[0].local.read_bytes() == data and calls["n"] >= 1


@mock_aws
def test_optional_missing_is_skipped_required_missing_fails(tmp_path):
    s3, items = _setup(tmp_path, {"a": b"1" * 100})
    gone = model_fetch.Item("gone", "models/gone", tmp_path / "m" / "gone.pt", True)
    assert [r.status for r in model_fetch.fetch_all([items[0], gone], client=s3, bucket=BUCKET)] == ["verified", "skipped_optional"]
    gone.optional = False
    with pytest.raises(model_fetch.ModelFetchError):
        model_fetch.fetch_all([gone], client=s3, bucket=BUCKET, retries=1)


def test_plan_dedupes_shared_models_and_never_plans_trailer(tmp_path):
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    items = model_fetch.plan(root, ["virality", "streamer"], tmp_path)
    paths = [str(i.local.relative_to(tmp_path)) for i in items]
    assert len(paths) == len(set(paths)) and "common/yolo11n.pt" in paths and not any("trailer" in p for p in paths)
    shared = next(i for i in items if i.local.name == "yolo11n.pt")
    assert set(shared.modes) == {"virality", "streamer"}
    pinned = model_fetch.plan(root, ["virality"], tmp_path, {"models": {"yolo11n": {"sha256": "a" * 64, "size": 5}}})
    assert next(i for i in pinned if i.key == "yolo11n").sha256 == "a" * 64


def test_unpinned_models_are_flagged_unverified(tmp_path):
    p = tmp_path / "m" / "x.pt"
    p.parent.mkdir(); p.write_bytes(b"z")
    res = model_fetch.fetch_all([model_fetch.Item("x", "models/x", p, True)])
    assert res[0].status == "unverified"


# ---------------------------------------------------------------- timeline + dispatcher breakdown
def test_timeline_marks_and_s3_key_contain_no_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("SONYA_TIMELINE_FILE", str(tmp_path / "tl.jsonl"))
    monkeypatch.setenv("WORKER_ID", "sonya-pool-abc12345")
    monkeypatch.setenv("WORKER_SECRET", "SUPERSECRET-LEAKCHECK")
    monkeypatch.setenv("S3_SECRET_ACCESS_KEY", "S3SECRET-LEAKCHECK")
    timeline.mark("entrypoint_start", release_cached="no")
    timeline.mark("models_done", bytes_downloaded=123, evil=object())      # non-JSON-safe extras are dropped
    snap = timeline.snapshot()
    blob = json.dumps(snap)
    assert "LEAKCHECK" not in blob and snap["worker_id"] == "sonya-pool-abc12345" and "evil" not in blob
    assert timeline.metric_key(snap).startswith("fleet-metrics/sonya-pool-abc12345/")
    timeline.mark("x" * 3)                                                  # never raises


@mock_aws
def test_timeline_upload_and_dispatcher_reads_it_back(tmp_path, monkeypatch):
    s3 = boto3.client("s3", region_name="us-east-1"); s3.create_bucket(Bucket="metrics-bucket")
    monkeypatch.setenv("SONYA_TIMELINE_FILE", str(tmp_path / "tl.jsonl")); monkeypatch.setenv("WORKER_ID", "sonya-pool-1"); monkeypatch.setenv("S3_BUCKET_NAME", "metrics-bucket")
    timeline.mark("entrypoint_start"); timeline.mark("first_job_processing")
    assert timeline.upload(client=s3) is True
    assert [m["name"] for m in sm.read_timeline_from_s3(s3, "metrics-bucket", "sonya-pool-1")] == ["entrypoint_start", "first_job_processing"]
    monkeypatch.setattr(timeline, "path", lambda: tmp_path / "missing" / "x.jsonl")
    assert timeline.upload(client=s3) is False                              # nothing to upload -> quietly False


def test_breakdown_stages_and_user_wait():
    t0 = 1000.0
    events = [{"event": "created", "ts": t0}, {"event": "running", "ts": t0 + 90}]
    marks = [{"name": "entrypoint_start", "t": t0 + 92, "release_cached": "no"}, {"name": "bootstrap_done", "t": t0 + 96},
             {"name": "run_worker_start", "t": t0 + 96.2}, {"name": "preflight_done", "t": t0 + 98}, {"name": "models_start", "t": t0 + 98.5},
             {"name": "models_done", "t": t0 + 140, "bytes_downloaded": 5_000_000_000}, {"name": "cuda_init_start", "t": t0 + 140.1},
             {"name": "cuda_init_done", "t": t0 + 144, "ms": 3900, "gpu": "RTX 3090"}, {"name": "readiness_ok", "t": t0 + 145},
             {"name": "worker_started", "t": t0 + 145.5}, {"name": "prewarm_done", "t": t0 + 152}, {"name": "first_job_claimed", "t": t0 + 160},
             {"name": "first_job_processing", "t": t0 + 163}]
    b = sm.breakdown(events, marks, job_queued_at=t0 - 5)
    s = b["stages"]
    assert (s["vast_provision_and_pull"], s["bootstrap"], s["models"], s["cuda_init"], s["claim_to_processing"]) == (90.0, 4.0, 41.5, 3.9, 3.0)
    assert s["request_to_ready"] == 152.0 and s["user_wait"] == 168.0 and b["gpu"] == "RTX 3090" and not b["release_cached"] and b["flags"] == []
    summ = sm.summarize([b], "cold")
    assert summ["models"]["p50"] == 41.5 and sm.summarize([b], "resume") == {}


def test_breakdown_flags_clock_skew_and_missing_marks():
    b = sm.breakdown([{"event": "created", "ts": 100.0}, {"event": "running", "ts": 200.0}], [{"name": "entrypoint_start", "t": 150.0}])
    assert b["stages"]["container_boot"] == 0.0 and any("skew" in f for f in b["flags"]) and b["stages"]["models"] is None
