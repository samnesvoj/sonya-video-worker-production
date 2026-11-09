"""run_worker.sh starts SONYA_WORKER_CONCURRENCY worker processes under one WORKER_ID and stops them all together."""
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _bash():
    for b in ("/bin/bash", "/opt/homebrew/bin/bash", "/usr/local/bin/bash", shutil.which("bash") or ""):
        if b and Path(b).exists():
            out = subprocess.run([b, "-c", "echo ${BASH_VERSINFO[0]}${BASH_VERSINFO[1]}"], capture_output=True, text=True).stdout.strip()
            if out.isdigit() and int(out) >= 51:          # wait -n PID... needs bash 5.1 (the image has 5.2)
                return b
    pytest.skip("bash >= 5.1 not available")


def _fake_release(tmp: Path, worker_body: str) -> Path:
    rel = tmp / "rel"
    (rel / "scripts" / "ops").mkdir(parents=True)
    (rel / "release").mkdir()
    (rel / "release" / "manifest.json").write_text('{"version": "test"}')
    (rel / "scripts" / "__init__.py").write_text("")
    (rel / "scripts" / "ops" / "__init__.py").write_text("")
    (rel / "scripts" / "ops" / "timeline.py").write_text("")
    (rel / "scripts" / "ops" / "readiness.py").write_text("import os, pathlib; pathlib.Path(os.environ['SONYA_READY_FILE']).write_text('ok')\n")
    (rel / "scripts" / "prod_preflight_check.py").write_text("")
    (rel / "scripts" / "gpu_worker.py").write_text(worker_body)
    boot = tmp / "boot"
    boot.mkdir()
    (boot / "healthd.py").write_text("import time\nwhile True: time.sleep(1)\n")
    return rel


def _env(tmp, rel, conc):
    return {**os.environ, "SONYA_RELEASE_DIR": str(rel), "SONYA_STATE_DIR": str(tmp / "state"), "SONYA_BOOTSTRAP_DIR": str(tmp / "boot"),
            "TMPDIR": str(tmp / "work"), "MODELS_LOCAL_DIR": str(tmp / "models"), "SONYA_WORKER_CONCURRENCY": str(conc),
            "WORKER_ID": "sonya-pool-test", "OUT": str(tmp / "out"), "PYTHONDONTWRITEBYTECODE": "1"}


WORKER = """import os, sys, signal, time, pathlib
out = pathlib.Path(os.environ['OUT']); out.mkdir(exist_ok=True)
slot = os.environ['SONYA_WORKER_SLOT']
(out / f'slot{slot}').write_text(' '.join(sys.argv[1:]))
signal.signal(signal.SIGTERM, lambda *a: ((out / f'term{slot}').write_text('1'), sys.exit(143)))
while True: time.sleep(0.1)
"""


def test_three_worker_processes_same_worker_id_and_all_get_sigterm(tmp_path):
    bash = _bash()
    (tmp_path / "state").mkdir()
    rel = _fake_release(tmp_path, WORKER)
    p = subprocess.Popen([bash, str(ROOT / "deploy/vast/run_worker.sh")], env=_env(tmp_path, rel, 3), stderr=subprocess.PIPE, text=True)
    out = tmp_path / "out"
    for _ in range(100):
        if out.exists() and len(list(out.glob("slot*"))) == 3:
            break
        time.sleep(0.1)
    assert sorted(f.name for f in out.glob("slot*")) == ["slot1", "slot2", "slot3"]
    assert all("--worker-id sonya-pool-test" in f.read_text() for f in out.glob("slot*"))
    p.send_signal(signal.SIGTERM)
    assert p.wait(timeout=15) == 143
    assert sorted(f.name for f in out.glob("term*")) == ["term1", "term2", "term3"]      # every job is requeued


def test_one_worker_dying_stops_the_others(tmp_path):
    bash = _bash()
    (tmp_path / "state").mkdir()
    body = WORKER.replace("while True: time.sleep(0.1)", "time.sleep(1.5) if slot == '2' else None\nif slot == '2': sys.exit(7)\nwhile True: time.sleep(0.1)")
    rel = _fake_release(tmp_path, body)
    p = subprocess.run([bash, str(ROOT / "deploy/vast/run_worker.sh")], env=_env(tmp_path, rel, 2), capture_output=True, text=True, timeout=30)
    assert p.returncode == 7
    assert (tmp_path / "out" / "term1").exists()


def test_bad_concurrency_is_refused(tmp_path):
    bash = _bash()
    (tmp_path / "state").mkdir()
    rel = _fake_release(tmp_path, WORKER)
    p = subprocess.run([bash, str(ROOT / "deploy/vast/run_worker.sh")], env=_env(tmp_path, rel, 33), capture_output=True, text=True, timeout=30)
    assert p.returncode == 3 and "1..32" in p.stderr
