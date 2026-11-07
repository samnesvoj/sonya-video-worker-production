"""Face detector preflight (OpenCV Haar) and its effect on Desktop crop."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from scripts.shared.crop import detector_check as dc

REPO = Path(__file__).resolve().parents[1]
cv2 = pytest.importorskip("cv2")


def test_current_opencv_and_cascade_pass():
    r = dc.face_detector_preflight()
    assert r["ok"], r
    assert r["opencv_version"].startswith("4.") and r["cascade"].endswith(dc.CASCADE_NAME)


def fake_cv2(version="4.14.0", data_dir=None, empty=False):
    m = types.ModuleType("cv2")
    m.__version__ = version
    m.data = types.SimpleNamespace(haarcascades=data_dir) if data_dir is not None else None
    m.CascadeClassifier = lambda p: types.SimpleNamespace(empty=lambda: empty)
    return m


def test_missing_cascade(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2(data_dir=str(tmp_path) + "/"))
    r = dc.face_detector_preflight()
    assert not r["ok"] and "not found" in r["reason"]


def test_broken_cascade(monkeypatch, tmp_path):
    (tmp_path / dc.CASCADE_NAME).write_text("<not a cascade/>")
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2(data_dir=str(tmp_path) + "/", empty=True))
    r = dc.face_detector_preflight()
    assert not r["ok"] and "empty classifier" in r["reason"]


def test_real_opencv_with_garbage_cascade_is_reported(monkeypatch, tmp_path):
    (tmp_path / dc.CASCADE_NAME).write_text("<opencv_storage>garbage</opencv_storage>")
    monkeypatch.setattr(cv2.data, "haarcascades", str(tmp_path) + "/")
    r = dc.face_detector_preflight()
    assert not r["ok"] and ("CascadeClassifier failed" in r["reason"] or "empty" in r["reason"])


def test_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", None)
    r = dc.face_detector_preflight()
    assert not r["ok"] and "import failed" in r["reason"]


def test_opencv5_rejected(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2(version="5.0.0", data_dir=str(tmp_path)))
    r = dc.face_detector_preflight()
    assert not r["ok"] and "expected 4.x" in r["reason"]


def test_headless_without_display():
    env = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY")}
    out = subprocess.run([sys.executable, "-c",
                          "import sys, json; sys.path.insert(0, sys.argv[1]);"
                          "from scripts.shared.crop.detector_check import face_detector_preflight as f;"
                          "print(json.dumps(f()))", str(REPO)], env=env, capture_output=True, text=True)
    assert out.returncode == 0 and json.loads(out.stdout)["ok"]


@pytest.mark.skipif(not (REPO / "desktop").is_dir(), reason="desktop client is not part of the GPU worker repo")
def test_failed_preflight_vision_child_never_runs_detector(monkeypatch, tmp_path):
    sys.path.insert(0, str(REPO / "desktop"))
    from sonya_desktop.modes import vision_child
    from scripts.shared.crop import cropper
    monkeypatch.setattr(dc, "face_detector_preflight", lambda: {"ok": False, "reason": "OpenCV 5.0.0 — expected 4.x"})
    ran = []
    monkeypatch.setattr(cropper, "analyze_faces", lambda *a, **k: ran.append(1))
    req = tmp_path / "r.json"
    req.write_text(json.dumps({"media": "x.mp4", "out_path": str(tmp_path / "v.json"), "repo_root": str(REPO),
                               "sample_fps": 2.0, "max_width": 640, "key": {}}))
    assert vision_child.main(str(req)) == 3 and ran == [] and not (tmp_path / "v.json").exists()


def test_detector_unavailable_gives_center_crop_with_reason(monkeypatch, tmp_path):
    from modes.virality import runner
    from scripts.shared.crop import smart_crop_adapter as sca
    monkeypatch.setattr(runner, "_source_dims", lambda p: (1920, 1080))
    tracks = []
    monkeypatch.setattr(sca, "render_tracked", lambda *a, **k: tracks.append(a[4]) or a[1])
    runner._compose_tracked("in.mp4", [{"start": 0, "duration": 5}], tmp_path,
                            {"mode": "tracked", "analysis_path": None, "analysis_status": "unavailable",
                             "analysis_error": "detector unavailable: OpenCV 5.0.0 — expected 4.x"})
    seg = json.loads((tmp_path / "selected_segments.json").read_text())["segments"][0]
    assert seg["crop_status"] == "center_fallback" and seg["crop_fallback_reason"] == "detector_unavailable"
    assert "OpenCV 5.0.0" in seg["crop_detector_error"] and tracks[0]["crop_status"] == "center_fallback"


def test_worker_requirements_pin_one_opencv_4_version():
    import re
    for f in ("requirements-worker.txt", "deploy/docker/requirements-worker-fast.txt"):
        pins = re.findall(r"^(opencv-[a-z-]+)==(\S+)", (REPO / f).read_text(), re.M)
        assert pins and {v for _, v in pins} == {"4.14.0.94"}, (f, pins)
        assert not re.search(r"^opencv-[a-z-]+\s*$", (REPO / f).read_text(), re.M)  # nothing unpinned
