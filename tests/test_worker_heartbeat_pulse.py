"""
PHASE A heartbeat (E, worker-side): gpu_worker.py's _HeartbeatPulse thread
must actually call the heartbeat function periodically while running, and
must stop promptly (not just "eventually") when stopped — so it can never
outlive process_job()'s try/finally around a mode run.

WORKER_BACKEND_MODE is left at its "db" default (see conftest.py's own env
setup) so gpu_worker imports scripts.prod_job_store — that import is
patched out entirely here (touch_job_heartbeat replaced with a recording
stub) so this test needs no DB either.
"""
from __future__ import annotations

import time

from scripts import gpu_worker


def test_heartbeat_pulse_calls_heartbeat_periodically(monkeypatch):
    calls = []
    monkeypatch.setattr(gpu_worker, "_do_heartbeat", lambda job_id: calls.append(job_id))

    pulse = gpu_worker._HeartbeatPulse("job-123", interval_sec=0.02)
    pulse.start()
    try:
        # A handful of intervals -- comfortably enough to observe multiple
        # calls without making the test slow.
        time.sleep(0.11)
    finally:
        pulse.stop()

    assert len(calls) >= 3
    assert all(job_id == "job-123" for job_id in calls)


def test_heartbeat_pulse_stops_promptly(monkeypatch):
    calls = []
    monkeypatch.setattr(gpu_worker, "_do_heartbeat", lambda job_id: calls.append(job_id))

    pulse = gpu_worker._HeartbeatPulse("job-123", interval_sec=0.02)
    pulse.start()
    time.sleep(0.05)
    pulse.stop()
    count_at_stop = len(calls)

    # Nothing further should arrive once stopped, even after waiting out
    # several more intervals.
    time.sleep(0.1)
    assert len(calls) == count_at_stop


def test_heartbeat_pulse_stop_without_start_is_a_noop():
    pulse = gpu_worker._HeartbeatPulse("job-123", interval_sec=1)
    pulse.stop()  # must not raise


def test_heartbeat_pulse_start_is_idempotent(monkeypatch):
    """Calling start() twice must not spawn a second thread double-firing
    heartbeats."""
    calls = []
    monkeypatch.setattr(gpu_worker, "_do_heartbeat", lambda job_id: calls.append(job_id))

    pulse = gpu_worker._HeartbeatPulse("job-123", interval_sec=0.02)
    pulse.start()
    pulse.start()  # second call must be a no-op
    try:
        time.sleep(0.07)
    finally:
        pulse.stop()

    # With one thread at 0.02s intervals over ~0.07s we'd expect ~3 calls;
    # two overlapping threads would roughly double that.
    assert len(calls) < 6


def test_heartbeat_failure_does_not_kill_the_pulse_thread(monkeypatch):
    """_do_heartbeat already swallows its own exceptions (see gpu_worker.py)
    -- confirm a failure on one beat doesn't stop subsequent ones."""
    calls = []

    def _flaky(job_id):
        calls.append(job_id)
        if len(calls) == 1:
            raise RuntimeError("transient DB hiccup")

    monkeypatch.setattr(gpu_worker, "_do_heartbeat", _flaky)

    pulse = gpu_worker._HeartbeatPulse("job-123", interval_sec=0.02)
    pulse.start()
    try:
        time.sleep(0.4)
    finally:
        pulse.stop()

    assert len(calls) >= 3
