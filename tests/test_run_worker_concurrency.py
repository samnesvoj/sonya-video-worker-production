"""run_worker.sh starts SONYA_WORKER_CONCURRENCY worker processes under one WORKER_ID, pins each to a card on a multi-GPU
server (fake 2 / 4 GPUs via the readiness plan) and stops them all together."""
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


def _fake_release(tmp: Path, worker_body: str, readiness: str = "") -> Path:
    rel = tmp / "rel"
    (rel / "scripts" / "ops").mkdir(parents=True)
    shutil.copy(ROOT / "scripts" / "ops" / "gpu_plan.py", rel / "scripts" / "ops" / "gpu_plan.py")
    (rel / "release").mkdir()
    (rel / "release" / "manifest.json").write_text('{"version": "test"}')
    (rel / "scripts" / "__init__.py").write_text("")
    (rel / "scripts" / "ops" / "__init__.py").write_text("")
    (rel / "scripts" / "ops" / "timeline.py").write_text("")
    (rel / "scripts" / "ops" / "readiness.py").write_text(readiness + "import os, pathlib; pathlib.Path(os.environ['SONYA_READY_FILE']).write_text('ok')\n")
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
(out / f'gpu{slot}').write_text(os.environ.get('CUDA_VISIBLE_DEVICES', '-') + ' ' + os.environ.get('SONYA_NVENC_MAX_SESSIONS', '-'))
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


# readiness on a fake multi-GPU server: the real gpu_plan decides, from probe results the stub fills in
PLAN_READINESS = """import json, os, pathlib, sys
sys.path.insert(0, os.getcwd())
from scripts.ops import gpu_plan
gpus = gpu_plan._fake_gpus(os.environ['FAKE_GPUS'])
for i in [int(x) for x in os.environ.get('FAKE_BROKEN', '').split(',') if x]:
    gpus[i].cuda_ok, gpus[i].error = False, 'CUDA probe failed (simulated)'
plan = gpu_plan.build_plan(gpus, threads=64, ram=256 * 1024)
gpu_plan.plan_path().write_text(json.dumps(plan))
"""


def _run_plan(tmp_path, conc, gpus, broken="", extra=None):
    bash = _bash()
    (tmp_path / "state").mkdir()
    rel = _fake_release(tmp_path, WORKER, PLAN_READINESS)
    env = {**_env(tmp_path, rel, conc), "FAKE_GPUS": gpus, "FAKE_BROKEN": broken, **(extra or {})}
    p = subprocess.Popen([bash, str(ROOT / "deploy/vast/run_worker.sh")], env=env, stderr=subprocess.PIPE, text=True)
    out = tmp_path / "out"
    want = None
    for _ in range(300):
        plan = tmp_path / "state" / "gpu_plan.json"
        if want is None and plan.exists():
            want = len(__import__("json").loads(plan.read_text())["workers"])
        if want and out.exists() and len(list(out.glob("gpu*"))) == want:
            break
        time.sleep(0.1)
    p.send_signal(signal.SIGTERM)
    _, err = p.communicate(timeout=30)
    assert p.returncode == 143, err
    return {int(f.name[3:]): f.read_text().split() for f in out.glob("gpu*")}


def test_two_gpus_four_workers_two_per_card_pinned_by_uuid(tmp_path):
    got = _run_plan(tmp_path, 4, "2xA40:46068")
    cards = [got[k][0] for k in sorted(got)]
    assert len(got) == 4 and cards[0] != cards[1] and cards[0] == cards[2] and cards[1] == cards[3]
    assert all(c.startswith("GPU-fake") for c in cards)
    assert all(v[1] == "-" for v in got.values())                     # A40: no NVENC session cap


def test_auto_on_four_gpus_uses_every_card(tmp_path):
    got = _run_plan(tmp_path, "auto", "4xL40S:46068", extra={"SONYA_WORKER_CONCURRENCY_MAX": "8"})
    cards = [v[0] for v in got.values()]
    assert len(got) == 8 and sorted(cards.count(c) for c in set(cards)) == [2, 2, 2, 2]


def test_broken_card_gets_no_worker_the_others_work(tmp_path):
    got = _run_plan(tmp_path, 4, "2xA40:46068", broken="1")
    assert len(got) == 4 and {v[0] for v in got.values()} == {"GPU-fake0000-0000-0000-0000-000000000000"}


def test_sigterm_while_workers_are_still_starting_stops_the_started_ones(tmp_path):
    bash = _bash()
    (tmp_path / "state").mkdir()
    rel = _fake_release(tmp_path, WORKER)
    p = subprocess.Popen([bash, str(ROOT / "deploy/vast/run_worker.sh")], env=_env(tmp_path, rel, 6), stderr=subprocess.PIPE, text=True)
    out = tmp_path / "out"
    for _ in range(100):
        if out.exists() and list(out.glob("slot*")):
            break
        time.sleep(0.1)
    p.send_signal(signal.SIGTERM)                 # worker 2..6 not started yet (2 s stagger)
    p.communicate(timeout=30)
    assert p.returncode == 143
    started = {f.name[4:] for f in out.glob("slot*")}
    assert started and {f.name[4:] for f in out.glob("term*")} == started


def test_one_gpu_is_not_pinned_like_before_and_geforce_gets_a_session_cap(tmp_path):
    got = _run_plan(tmp_path, 2, "1xNVIDIA GeForce RTX 3060:12288@570")
    assert got == {1: ["-", "8"], 2: ["-", "8"]}
