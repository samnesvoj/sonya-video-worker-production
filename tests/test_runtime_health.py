import importlib, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bootstrap"))


def test_ready_only_when_worker_alive_and_ready_file(monkeypatch, tmp_path):
    monkeypatch.setenv("SONYA_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SONYA_READY_FILE", str(tmp_path / "ready"))
    monkeypatch.setenv("SONYA_WORKER_PID_FILE", str(tmp_path / "worker.pid"))
    import healthd
    h = importlib.reload(healthd)
    assert h.status_for("/readyz")[0] == 503
    (tmp_path / "worker.pid").write_text(str(os.getpid()))
    assert h.status_for("/healthz")[0] == 200 and h.status_for("/readyz") == (503, "not_ready")
    (tmp_path / "ready").write_text("ready")
    assert h.status_for("/readyz") == (200, "ready") and h.status_for("/x")[0] == 404


def test_health_binds_loopback_only():
    src = (ROOT / "bootstrap/healthd.py").read_text()
    assert '("127.0.0.1"' in src and "0.0.0.0" not in src


def _env(tmp_path, **extra):
    e = {k: v for k, v in os.environ.items() if not k.startswith(("S3_", "BACKEND_", "WORKER_", "ENABLED_"))}
    e.update(SONYA_SKIP_GPU_CHECK="1", SONYA_READY_FILE=str(tmp_path / "ready"), MODELS_LOCAL_DIR=str(tmp_path / "m"),
             BACKEND_API_URL="https://api.example", WORKER_SECRET="SECRET-VALUE-XYZ", S3_ENDPOINT_URL="http://127.0.0.1:9",
             S3_ACCESS_KEY_ID="AKIA-LEAK-CHECK", S3_SECRET_ACCESS_KEY="S3-SECRET-LEAK-CHECK", S3_REGION="r",
             S3_BUCKET_NAME="b", MODELS_S3_BUCKET="b", **extra)
    return e


def test_readiness_blocks_without_env(tmp_path):
    e = _env(tmp_path)
    e.pop("WORKER_SECRET")
    r = subprocess.run([sys.executable, "-m", "scripts.ops.readiness"], cwd=ROOT, env=e, capture_output=True, text=True)
    assert r.returncode == 1 and not (tmp_path / "ready").exists() and "missing env" in r.stdout


def test_readiness_never_logs_secrets_and_ignores_trailer(tmp_path):
    e = _env(tmp_path, ENABLED_MODES="trailer_film_breaker,virality")
    r = subprocess.run([sys.executable, "-m", "scripts.ops.readiness"], cwd=ROOT, env=e, capture_output=True, text=True)
    out = r.stdout + r.stderr
    assert not any(s in out for s in ("SECRET-VALUE-XYZ", "AKIA-LEAK-CHECK", "S3-SECRET-LEAK-CHECK"))
    assert "enabled modes: virality" in out
