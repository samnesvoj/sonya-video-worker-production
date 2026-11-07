import os
from pathlib import Path

import pytest

from scripts.shared import model_paths
from scripts.shared.enhancers import sonya_enhancer
from scripts.shared.transcription import transcriber


def test_models_root_priority_and_legacy_default(monkeypatch, tmp_path):
    monkeypatch.delenv("SONYA_MODELS_DIR", raising=False); monkeypatch.delenv("MODELS_LOCAL_DIR", raising=False)
    assert model_paths.models_root() == Path(model_paths.__file__).resolve().parents[2] / "models"      # unchanged dev behaviour
    monkeypatch.setenv("MODELS_LOCAL_DIR", str(tmp_path / "a"))
    assert model_paths.models_root() == tmp_path / "a"
    monkeypatch.setenv("SONYA_MODELS_DIR", str(tmp_path / "b"))
    assert model_paths.model_file("common", "x.pt") == tmp_path / "b" / "common" / "x.pt"


def test_enhancer_finds_verified_downloads_in_models_local_dir_not_in_the_release_dir(monkeypatch, tmp_path):
    """Regression for the Vast layout: code in /opt/sonya/releases/<sha>, models in /opt/sonya/models."""
    seen = {}
    from scripts.shared.vision import yolo_common
    monkeypatch.setattr(yolo_common, "detect_common", lambda v, m: seen.setdefault("yolo", m) and [])
    monkeypatch.setattr(yolo_common, "detect_pose", lambda v, m: seen.setdefault("pose", m) and [])
    for k in ("GEMINI_LAYOUT_ENABLED", "WORD_LEVEL_SUBTITLES_ENABLED", "WEBCAM_DETECTOR_ENABLED"):
        monkeypatch.setenv(k, "false")
    monkeypatch.setenv("MODELS_LOCAL_DIR", str(tmp_path))
    (tmp_path / "common").mkdir()
    (tmp_path / "common" / "yolo11n.pt").write_bytes(b"x")
    monkeypatch.setattr("scripts.shared.crop.smart_crop_adapter.get_crop_hints", lambda *a, **k: {})
    out = sonya_enhancer.enrich_video_for_mode("v.mp4", str(tmp_path), "virality")
    assert seen.get("yolo") == str(tmp_path / "common" / "yolo11n.pt")
    assert any("yolo11n-pose.pt not found" in w for w in out["warnings"])                             # absent model is still only a warning


def _fake_whisper(monkeypatch):
    calls = []
    monkeypatch.setattr(transcriber, "_WhisperModel", lambda path, **kw: calls.append(path) or object(), raising=False)
    return calls


def test_whisper_loads_from_models_local_dir_and_honours_model_override(monkeypatch, tmp_path):
    calls = _fake_whisper(monkeypatch)
    monkeypatch.setenv("MODELS_LOCAL_DIR", str(tmp_path)); monkeypatch.setenv("SONYA_ASR_OFFLINE", "1")
    (tmp_path / "whisper-base").mkdir(); (tmp_path / "whisper-base" / "model.bin").write_bytes(b"x")
    (tmp_path / "whisper-large-v3").mkdir(); (tmp_path / "whisper-large-v3" / "model.bin").write_bytes(b"x")
    monkeypatch.delenv("SONYA_WHISPER_MODEL", raising=False)
    assert transcriber.Transcriber._load_faster_whisper("whisper-1", "cpu", "int8", str(tmp_path / "nolegacy"))
    assert calls[-1] == str(tmp_path / "whisper-base")                                                 # default mapping unchanged
    monkeypatch.setenv("SONYA_WHISPER_MODEL", "large-v3")
    transcriber.Transcriber._load_faster_whisper("whisper-1", "cpu", "int8", str(tmp_path / "nolegacy"))
    assert calls[-1] == str(tmp_path / "whisper-large-v3")


def test_offline_mode_never_downloads_from_huggingface(monkeypatch, tmp_path):
    calls = _fake_whisper(monkeypatch)
    monkeypatch.setenv("MODELS_LOCAL_DIR", str(tmp_path)); monkeypatch.setenv("SONYA_ASR_OFFLINE", "1")
    assert transcriber.Transcriber._load_faster_whisper("whisper-1", "cpu", "int8", str(tmp_path / "nolegacy")) is None
    assert calls == []                                                                                  # no WhisperModel('base') => no HF download
