import os

import pytest

from tools.stub_worker_api import StubState, serve_in_thread

SECRET = "throwaway-secret-LEAKCHECK"


@pytest.fixture
def api(monkeypatch):
    st = StubState()
    srv, url = serve_in_thread(st, SECRET)
    monkeypatch.setenv("BACKEND_API_URL", url); monkeypatch.setenv("WORKER_SECRET", SECRET)
    yield st, url
    srv.shutdown()


def test_the_real_worker_client_runs_a_full_job_lifecycle_against_the_stub(api):
    """gpu_worker's own _BackendAPIClient (not a re-implementation) talks to the stub."""
    from scripts import gpu_worker
    st, _ = api
    jid = st.enqueue("virality", "fast-start-test/r1/input.mp4")
    c = gpu_worker._BackendAPIClient()
    assert c.claim_job(None, "sonya-pool-ab", ["beep"]) is None                       # mode filter honoured
    job = c.claim_job(None, "sonya-pool-ab", ["virality", "beep"])
    assert job["id"] == jid and job["mode"] == "virality" and job["s3_input_key"].endswith("input.mp4")
    c.update_status(jid, "mode_running"); c.heartbeat(jid)
    snap = st.snapshot(["virality"], "sonya-pool-")
    assert snap.active_by_worker == {"sonya-pool-ab": 1} and snap.pending == 0
    c.add_file(jid, "u", "output", "outputs/x.mp4", "x.mp4", "video/mp4", 123)
    c.complete_job(jid, "outputs/x.mp4", 1, 5000)
    assert st.jobs[jid]["status"] == "completed" and st.jobs[jid]["files"][0]["size_bytes"] == 123
    assert st.snapshot(["virality"], "sonya-pool-").last_terminal_by_worker["sonya-pool-ab"] > 0


def test_retryable_failure_requeues_and_permanent_failure_does_not(api):
    from scripts import gpu_worker
    st, _ = api
    a = st.enqueue("virality", "k"); c = gpu_worker._BackendAPIClient()
    c.claim_job(None, "w", None); c.fail_job(a, "WORKER_SHUTDOWN", "x", True)
    assert st.jobs[a]["status"] == "queued" and st.failures[-1]["code"] == "WORKER_SHUTDOWN"
    c.claim_job(None, "w", None); c.fail_job(a, "RUNNER_FAILED", "x", False)
    assert st.jobs[a]["status"] == "failed"


def test_wrong_secret_oversized_and_unknown_paths_are_refused(api):
    import requests
    _, url = api
    assert requests.post(url + "/api/worker/claim", json={}, headers={"Authorization": "Bearer nope"}).status_code == 401
    assert requests.post(url + "/api/worker/claim", data=b"x" * 70000, headers={"Authorization": f"Bearer {SECRET}"}).status_code == 413
    assert requests.post(url + "/api/internal/anything", json={}, headers={"Authorization": f"Bearer {SECRET}"}).status_code == 404
    assert requests.post(url + "/api/worker/jobs/not-a-uuid/status", json={}, headers={"Authorization": f"Bearer {SECRET}"}).status_code == 404


def test_server_binds_loopback_by_default():
    st = StubState(); srv, url = serve_in_thread(st, "s")
    try:
        assert url.startswith("http://127.0.0.1:") and srv.server_address[0] == "127.0.0.1"
    finally:
        srv.shutdown()
