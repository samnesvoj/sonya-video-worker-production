import pytest

from scripts import gpu_worker, mode_policy, mode_registry
from scripts.mode_policy import ModeDisabledError


def test_trailer_is_hard_disabled_even_if_requested(monkeypatch):
    monkeypatch.setenv("ENABLED_MODES", "trailer_film_breaker,trailer_cinematic,virality")
    assert mode_policy.enabled_modes() == ["virality"]
    for m in ("trailer_film_breaker", "trailer_cinematic", "trailer_future", "TRAILER_FILM_BREAKER"):
        with pytest.raises(ModeDisabledError):
            mode_policy.assert_mode_allowed(m)
        with pytest.raises(ModeDisabledError):
            mode_registry.get_runner(m)


def test_default_enables_only_evidenced_modes(monkeypatch):
    monkeypatch.delenv("ENABLED_MODES", raising=False)
    assert mode_policy.enabled_modes() == ["virality", "beep"]
    for m in ("stories", "educational", "streamer", "sonya_gen"):
        with pytest.raises(ModeDisabledError):
            mode_policy.assert_mode_allowed(m)


def test_opt_in_modes_need_explicit_env(monkeypatch):
    monkeypatch.setenv("ENABLED_MODES", "virality,stories,bogus")
    assert mode_policy.enabled_modes() == ["virality", "stories"]


def test_claim_filter_is_never_empty(monkeypatch):
    monkeypatch.setenv("ENABLED_MODES", "trailer_film_breaker")
    with pytest.raises(ModeDisabledError):
        mode_policy.claimable_modes()


def test_trailer_code_is_preserved_but_unrunnable():
    assert "trailer_film_breaker" in mode_registry.list_modes()
    with pytest.raises(ModeDisabledError):
        mode_registry.get_runner("trailer_film_breaker")


def test_worker_refuses_trailer_job_before_downloading(monkeypatch):
    calls = {"fail": [], "dl": 0}
    monkeypatch.setattr(gpu_worker, "_do_fail_job", lambda *a: calls["fail"].append(a))
    monkeypatch.setattr(gpu_worker, "download_file", lambda *a, **k: calls.__setitem__("dl", calls["dl"] + 1))
    gpu_worker.process_job({"id": "j1", "mode": "trailer_film_breaker", "user_id": "u", "s3_input_key": "in/x.mp4", "params": {}}, "w")
    assert calls["dl"] == 0
    assert calls["fail"][0][1] == "mode_disabled" and calls["fail"][0][3] is False


def test_poll_sends_explicit_allowlist(monkeypatch):
    monkeypatch.delenv("ENABLED_MODES", raising=False)
    monkeypatch.delenv("SONYA_REQUIRE_READY", raising=False)
    seen = {}

    class Client:
        def claim_job(self, job_id, worker_id, modes=None):
            seen["modes"] = modes
            gpu_worker._STOP.set()

    monkeypatch.setattr(gpu_worker, "_WORKER_MODE", "api")
    monkeypatch.setattr(gpu_worker, "_get_api_client", lambda: Client())
    gpu_worker._STOP.clear()
    try:
        gpu_worker.run_poll("w")
    finally:
        gpu_worker._STOP.clear()
    assert seen["modes"] == ["virality", "beep"]


def test_poll_refuses_without_ready_file(monkeypatch, tmp_path):
    monkeypatch.setenv("SONYA_REQUIRE_READY", "1")
    monkeypatch.setenv("SONYA_READY_FILE", str(tmp_path / "nope"))
    gpu_worker._STOP.clear()
    with pytest.raises(SystemExit):
        gpu_worker.run_poll("w")


def test_sigterm_requeues_active_job(monkeypatch):
    failed = []
    monkeypatch.setattr(gpu_worker, "_do_fail_job", lambda *a: failed.append(a))
    gpu_worker._ACTIVE_JOB["id"] = "job-9"
    try:
        with pytest.raises(SystemExit) as ei:
            gpu_worker._on_sigterm(15, None)
    finally:
        gpu_worker._STOP.clear()
        gpu_worker._ACTIVE_JOB["id"] = None
    assert ei.value.code == 143
    assert failed == [("job-9", "WORKER_SHUTDOWN", "worker terminated while job was running", True)]


def test_manifest_matches_policy():
    import json, pathlib
    m = json.loads((pathlib.Path(__file__).resolve().parents[1] / "release/manifest.json").read_text())
    assert m["enabled_modes"] == list(mode_policy.DEFAULT_ENABLED_MODES)
    assert m["opt_in_modes"] == list(mode_policy.OPT_IN_MODES)
    assert set(m["disabled_modes"]) == set(mode_policy.HARD_DISABLED_MODES)
