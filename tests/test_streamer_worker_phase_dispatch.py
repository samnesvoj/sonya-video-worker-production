"""
gpu_worker.py's streamer_phase="analyze"/"compose" branch in process_job()
(REAL STREAMER BATCH FLOW). No real GPU/ffmpeg/S3/DB — every dependency
process_job() touches (download_file, ensure_models_for_mode, upload_file,
build_output_key, the _do_* backend dispatchers, get_runner) is
monkeypatched at its call site in scripts.gpu_worker, same convention as
tests/test_worker_heartbeat_pulse.py. modes.streamer.runner.analyze() /
compose_one() are imported locally inside process_job() (not at module
level), so they're monkeypatched on modes.streamer.runner itself — a
`from module import name` performed inside a function body re-resolves
`module.name` on every call, so this reaches it correctly.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import modes.streamer.runner as streamer_runner
from scripts import gpu_worker


def _quiet_worker(monkeypatch):
    """Stub out every side-effecting dependency process_job() calls that
    isn't the thing under test in a given case."""
    monkeypatch.setattr(gpu_worker, "_update_status", lambda job_id, status: None)
    monkeypatch.setattr(gpu_worker, "_do_heartbeat", lambda job_id: None)
    monkeypatch.setattr(gpu_worker, "ensure_models_for_mode", lambda mode: True)
    monkeypatch.setattr(gpu_worker, "download_file",
                         lambda s3_key, local_path: Path(local_path).write_bytes(b"\x00" * 16) or Path(local_path))


def _job(**overrides):
    job = {
        "id": "job-1", "mode": "streamer", "user_id": "user-1",
        "params": {}, "s3_input_key": "users/user-1/jobs/job-1/streamer/input/source.mp4",
    }
    job.update(overrides)
    return job


# ── analyze phase ────────────────────────────────────────────────────────

def test_analyze_phase_submits_segments_and_completes_with_manifest(monkeypatch):
    _quiet_worker(monkeypatch)

    analyze_result = {
        "segments": [{"start_sec": 1.0, "duration_sec": 5.0, "score": 1.0, "source": "x"}],
        "crop_hints": {"anchor": "center"}, "warnings": [],
        "webcam_boxes_found": 2, "active_speaker_segs": 1,
    }
    monkeypatch.setattr(streamer_runner, "analyze", lambda **kw: analyze_result)

    submitted = {}
    monkeypatch.setattr(gpu_worker, "_do_submit_streamer_segments",
                         lambda batch_id, result: submitted.update(batch_id=batch_id, result=result))
    monkeypatch.setattr(gpu_worker, "build_output_key",
                         lambda user_id, job_id, mode, filename: f"users/{user_id}/jobs/{job_id}/{mode}/output/{filename}")
    monkeypatch.setattr(gpu_worker, "upload_file", lambda local_path, s3_key, content_type=None: None)

    completed = {}
    monkeypatch.setattr(gpu_worker, "_do_complete_job",
                         lambda **kw: completed.update(kw))
    monkeypatch.setattr(gpu_worker, "_do_fail_job",
                         lambda *a, **kw: (_ for _ in ()).throw(AssertionError("must not fail")))

    job = _job(params={"streamer_phase": "analyze", "streamer_batch_id": "batch-1"})
    gpu_worker.process_job(job, "worker-1")

    assert submitted == {"batch_id": "batch-1", "result": analyze_result}
    assert completed["job_id"] == "job-1"
    assert completed["clip_count"] == 1
    assert completed["s3_output_key"].endswith("analysis.json")


def test_analyze_phase_segments_submit_failure_fails_job_retryable(monkeypatch):
    _quiet_worker(monkeypatch)
    monkeypatch.setattr(streamer_runner, "analyze",
                         lambda **kw: {"segments": [], "crop_hints": {}, "warnings": []})

    def _boom(batch_id, result):
        raise RuntimeError("backend unreachable")
    monkeypatch.setattr(gpu_worker, "_do_submit_streamer_segments", _boom)

    failed = {}
    monkeypatch.setattr(gpu_worker, "_do_fail_job",
                         lambda job_id, error_code, error_message, retry: failed.update(
                             job_id=job_id, error_code=error_code, retry=retry))
    monkeypatch.setattr(gpu_worker, "_do_complete_job",
                         lambda **kw: (_ for _ in ()).throw(AssertionError("must not complete")))

    job = _job(params={"streamer_phase": "analyze", "streamer_batch_id": "batch-1"})
    gpu_worker.process_job(job, "worker-1")

    assert failed["error_code"] == "STREAMER_SEGMENTS_SUBMIT_FAILED"
    assert failed["retry"] is True


def test_analyze_phase_missing_batch_id_fails_job_non_retryable(monkeypatch):
    _quiet_worker(monkeypatch)
    monkeypatch.setattr(streamer_runner, "analyze",
                         lambda **kw: {"segments": [], "crop_hints": {}, "warnings": []})
    monkeypatch.setattr(gpu_worker, "_do_submit_streamer_segments",
                         lambda *a, **kw: (_ for _ in ()).throw(AssertionError("must not be called")))

    failed = {}
    monkeypatch.setattr(gpu_worker, "_do_fail_job",
                         lambda job_id, error_code, error_message, retry: failed.update(
                             error_code=error_code, retry=retry))

    job = _job(params={"streamer_phase": "analyze"})  # no streamer_batch_id
    gpu_worker.process_job(job, "worker-1")

    assert failed["error_code"] == "STREAMER_BATCH_ID_MISSING"
    assert failed["retry"] is False


# ── compose phase ────────────────────────────────────────────────────────

def test_compose_phase_calls_compose_one_and_completes_normally(monkeypatch, tmp_path):
    _quiet_worker(monkeypatch)

    seen_call = {}

    def _fake_compose_one(input_video_path, output_path, segment, crop_hints=None):
        seen_call["segment"] = segment
        seen_call["crop_hints"] = crop_hints
        Path(output_path).write_bytes(b"\x00" * 32)
        return output_path

    monkeypatch.setattr(streamer_runner, "compose_one", _fake_compose_one)
    monkeypatch.setattr(gpu_worker, "build_output_key",
                         lambda user_id, job_id, mode, filename: f"users/{user_id}/jobs/{job_id}/{mode}/output/{filename}")
    monkeypatch.setattr(gpu_worker, "upload_file", lambda local_path, s3_key, content_type=None: None)
    monkeypatch.setattr(gpu_worker, "_do_add_file", lambda **kw: None)

    completed = {}
    monkeypatch.setattr(gpu_worker, "_do_complete_job", lambda **kw: completed.update(kw))

    job = _job(params={
        "streamer_phase": "compose",
        "streamer_batch_id": "batch-1",
        "streamer_segment": {"start_sec": 12.0, "duration_sec": 8.0},
        "streamer_crop_hints": {"anchor": "left"},
    })
    gpu_worker.process_job(job, "worker-1")

    assert seen_call["segment"] == {"start_sec": 12.0, "duration_sec": 8.0}
    assert seen_call["crop_hints"] == {"anchor": "left"}
    assert completed["job_id"] == "job-1"
    assert completed["clip_count"] == 1  # exactly one .mp4 -> classified "output"


# ── unaffected: a plain streamer job with no batch behind it ───────────

def test_plain_streamer_job_without_phase_uses_legacy_run(monkeypatch, tmp_path):
    """
    A "streamer" job whose params carry no streamer_phase (e.g. created
    via the old /api/generation/jobs path, not a batch) must still go
    through get_runner()/run() exactly as before this change.
    """
    _quiet_worker(monkeypatch)

    called = {}

    def _fake_run(input_video_path, output_dir, params, progress_callback=None):
        called["ran"] = True
        (Path(output_dir) / "stream_clip_01.mp4").write_bytes(b"\x00" * 8)
        return {"clips": ["stream_clip_01.mp4"], "mode": "streamer", "warnings": []}

    monkeypatch.setattr(gpu_worker, "get_runner", lambda mode: _fake_run)
    monkeypatch.setattr(gpu_worker, "build_output_key",
                         lambda user_id, job_id, mode, filename: f"users/{user_id}/jobs/{job_id}/{mode}/output/{filename}")
    monkeypatch.setattr(gpu_worker, "upload_file", lambda local_path, s3_key, content_type=None: None)
    monkeypatch.setattr(gpu_worker, "_do_add_file", lambda **kw: None)
    monkeypatch.setattr(gpu_worker, "_do_complete_job", lambda **kw: None)

    job = _job(params={})
    gpu_worker.process_job(job, "worker-1")

    assert called.get("ran") is True


@pytest.fixture(autouse=True)
def _streamer_opted_in(monkeypatch):
    """streamer is an opt-in mode in this worker (see scripts/mode_policy.py); these tests exercise its phase dispatch."""
    monkeypatch.setenv("ENABLED_MODES", "streamer")
