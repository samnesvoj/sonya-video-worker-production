"""Worker reproducibility manifest: commit/image from env, secrets masked, models identified, never fails."""
import json

from scripts.shared import worker_manifest as wm


def test_image_build_env_wins_and_secret_values_never_appear(monkeypatch, tmp_path):
    pinned = "ghcr.io/samnesvoj/sonya-worker:fast@sha256:" + "cd" * 32
    monkeypatch.setenv("SONYA_GIT_SHA", "0123456789abcdef")
    monkeypatch.setenv("SONYA_IMAGE", pinned)
    monkeypatch.setenv("WORKER_SECRET", "super-secret-value")
    monkeypatch.setenv("SONYA_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("WORKER_LOOP", "true")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-xyz")
    monkeypatch.setenv("SONYA_MODELS_DIR", str(tmp_path))
    (tmp_path / "yolo").mkdir()
    (tmp_path / "yolo" / "yolo11n.pt").write_bytes(b"w" * 1000)
    m = wm.build_manifest()
    text = json.dumps(m)
    assert m["commit"] == {"sha": "0123456789abcdef", "source": "image"}
    assert m["image"] == {"ref": pinned, "digest": "sha256:" + "cd" * 32, "pinned": True}
    assert "super-secret-value" not in text and "sk-should-not-leak" not in text and "sk-or-xyz" not in text
    assert m["env"]["WORKER_SECRET"] == "***" and m["env"]["WORKER_LOOP"] == "true"
    assert "OPENROUTER_API_KEY" not in m["env"]
    f = [x for x in m["models"]["files"] if x["path"] == "yolo/yolo11n.pt"][0]
    assert f["size"] == 1000 and len(f["id"]) == 32
    assert m["modes"]["boundaries"]["VERSION"] and m["modes"]["smart_crop_adapter"]["CROP_VERSION_V2"]
    assert m["deps"]["count"] and len(m["deps"]["sha256"]) == 64


def test_model_id_changes_with_content_and_cli_writes_file(monkeypatch, tmp_path):
    a, b = tmp_path / "a.onnx", tmp_path / "b.onnx"
    a.write_bytes(b"1" * 10)
    b.write_bytes(b"2" * 10)
    assert wm._file_id(a) != wm._file_id(b)
    monkeypatch.delenv("SONYA_GIT_SHA", raising=False)
    out = tmp_path / "m.json"
    assert wm.main(["--out", str(out), "--no-models"]) == 0
    m = json.loads(out.read_text())
    assert m["manifest_version"] == wm.MANIFEST_VERSION and m["models"] is None
    assert m["commit"]["source"] in ("git", None)


def test_missing_tools_are_reported_not_raised(monkeypatch):
    monkeypatch.setenv("SONYA_FFMPEG", "/nonexistent/ffmpeg")
    assert wm._ffmpeg() == {"path": "/nonexistent/ffmpeg", "version": None}
