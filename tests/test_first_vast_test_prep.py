"""First Vast test (owner decisions 2026-11-08): webcam detector off and not fetched, lock covers every required model, plain read-only token."""
import json
import pathlib
import subprocess

from dispatcher.token_provider import static_token_provider
from scripts.ops import model_fetch
from tools import models_publish as mp

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_webcam_detector_is_not_fetched_while_disabled(monkeypatch, tmp_path):
    lock = json.loads((ROOT / "release/models.lock.json").read_text())
    monkeypatch.setenv("WEBCAM_DETECTOR_ENABLED", "false")
    keys = {i.key for i in model_fetch.plan(ROOT, ["virality", "beep"], tmp_path, lock)}
    assert "webcam_detector" not in keys and {"gigaam_v3_e2e_rnnt_ckpt", "yolo11n"} <= keys
    monkeypatch.setenv("WEBCAM_DETECTOR_ENABLED", "true")
    assert "webcam_detector" in {i.key for i in model_fetch.plan(ROOT, ["virality"], tmp_path, lock)}


def test_lock_pins_every_model_the_gpu_test_needs_with_real_hashes():
    lock = json.loads((ROOT / "release/models.lock.json").read_text())["models"]
    man = json.loads((ROOT / "release/models.required.json").read_text())["models"]
    need = {m["key"] for m in man if m.get("required_for_gpu_test", True)}
    gen = {m["key"] for m in man if m.get("used_by") == ["sonya_gen"]}      # SONYA Gen models: pinned, not needed by the virality/beep test
    assert need | gen == set(lock) and "webcam_detector" not in lock
    for m in man:
        if m["key"] in lock:
            pin = lock[m["key"]]
            assert len(pin["sha256"]) == 64 and pin["size"] > 0 and pin["s3_path"] == m["s3_path"]
            if m.get("public_sha256"):
                assert pin["sha256"] == m["public_sha256"] and pin["size"] == m["public_size"]


def test_run_worker_defaults_gemini_layout_and_webcam_off():
    s = (ROOT / "deploy/vast/run_worker.sh").read_text()
    assert 'GEMINI_LAYOUT_ENABLED="${GEMINI_LAYOUT_ENABLED:-false}"' in s and 'WEBCAM_DETECTOR_ENABLED="${WEBCAM_DETECTOR_ENABLED:-false}"' in s
    subprocess.run(["bash", "-n", str(ROOT / "deploy/vast/run_worker.sh")], check=True)


def test_models_publish_skip_leaves_a_model_out_of_the_lock(tmp_path):
    for m in mp.load_manifest()["models"]:
        p = tmp_path / m["local_path"]; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b"x")
    out = tmp_path / "lock.json"
    custom = sum([["--accept-custom", m["key"]] for m in mp.load_manifest()["models"]], [])
    assert mp.main(["lock", "--src", str(tmp_path), "--out", str(out), "--skip", "webcam_detector"] + custom) == 0
    assert "webcam_detector" not in json.loads(out.read_text())["models"]


def test_static_token_provider_returns_the_token_and_refuses_empty():
    assert static_token_provider("t0k")() == "t0k"
    try:
        static_token_provider("")
    except ValueError:
        pass
    else:
        raise AssertionError("empty token accepted")
