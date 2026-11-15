"""Desktop ingest normalization in the worker (scripts/shared/source_normalize.py)."""
from __future__ import annotations

from pathlib import Path

from scripts.shared import source_normalize as sn


def _raw(vcodec="h264", acodec="aac", r="30/1", avg="30/1", pix="yuv420p", rotate=None):
    v = {"codec_type": "video", "codec_name": vcodec, "r_frame_rate": r, "avg_frame_rate": avg, "pix_fmt": pix,
         "width": 1920, "height": 1080}
    if rotate is not None:
        v["side_data_list"] = [{"rotation": rotate}]
    return {"streams": [v, {"codec_type": "audio", "codec_name": acodec}], "format": {"duration": "10"}}


def test_reasons_match_desktop_rules():
    assert sn.normalize_reasons(sn.summarize_probe(_raw())) == []
    assert sn.normalize_reasons(sn.summarize_probe(_raw(vcodec="av1"))) == ["vcodec:av1"]
    assert sn.normalize_reasons(sn.summarize_probe(_raw(avg="2997/100", r="30/1"))) == []          # 0.1 %: CFR
    assert sn.normalize_reasons(sn.summarize_probe(_raw(avg="25/1", r="30/1"))) == ["vfr"]
    assert sn.normalize_reasons(sn.summarize_probe(_raw(rotate=-90))) == ["rotation:270"]
    assert sn.normalize_reasons(sn.summarize_probe(_raw(acodec="pcm_s16le", pix="yuv422p10le"))) == \
        ["acodec:pcm_s16le", "pix_fmt:yuv422p10le"]


def test_cmd_is_cfr_aac_with_the_given_encoder():
    cmd = sn.normalize_cmd("ffmpeg", Path("a.webm"), Path("b.mp4"), sn.summarize_probe(_raw(avg="2997/100")),
                           ["-c:v", "h264_nvenc"])
    assert cmd[cmd.index("-vf") + 1] == "fps=29.97" and "h264_nvenc" in cmd and cmd[-1] == "b.mp4"
    assert cmd[cmd.index("-c:a") + 1] == "aac"


def test_other_modes_and_rollback_are_untouched(monkeypatch, tmp_path):
    assert sn.normalize_source("x.mp4", str(tmp_path), "educational") == {"path": "x.mp4", "reasons": [],
                                                                          "normalized": False}
    monkeypatch.setenv("SONYA_NORMALIZE", "never")
    assert sn.normalize_source("x.mp4", str(tmp_path), "beep")["path"] == "x.mp4"


def test_friendly_source_is_not_re_encoded(monkeypatch, tmp_path):
    monkeypatch.setattr(sn, "probe", lambda path, fp="ffprobe": _raw())
    monkeypatch.setattr(sn.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("encoded")))
    assert sn.normalize_source("x.mp4", str(tmp_path), "virality")["normalized"] is False


def _long(monkeypatch, tmp_path, rc=0, stderr="", family="nvenc", used="nvenc"):
    raw = _raw(vcodec="vp9")
    raw["format"]["duration"] = "3300"
    monkeypatch.setattr(sn, "probe", lambda path, fp="ffprobe": raw)
    from scripts.shared import cut_profile
    monkeypatch.setattr(cut_profile, "encoder_family", lambda ff=None: family)
    seen = {}

    def fake_run(make_cmd, args, **kw):
        seen.update(kw, args=args)
        return sn.subprocess.CompletedProcess([], rc, "", stderr), used, 1.0
    monkeypatch.setattr(cut_profile, "run_encode", fake_run)
    src = tmp_path / "s.webm"
    src.write_bytes(b"x" * 1000)
    return str(src), seen


def test_long_source_cpu_is_ultrafast_with_watchdog(monkeypatch, tmp_path):
    src, seen = _long(monkeypatch, tmp_path, family="cpu")
    assert sn.normalize_source(src, str(tmp_path / "ing"), "virality")["normalized"] is True
    assert seen["cpu_preset"] == "ultrafast" and seen["timeout"] == 1650
    assert seen["args"][seen["args"].index("-preset") + 1] == "ultrafast"


def test_long_source_timeout_and_disk_are_clear_errors(monkeypatch, tmp_path):
    import pytest
    src, _ = _long(monkeypatch, tmp_path, rc=124)
    with pytest.raises(sn.NormalizeError) as e:
        sn.normalize_source(src, str(tmp_path / "ing"), "virality")
    assert e.value.code == "NORMALIZE_TIMEOUT" and "длинного" in e.value.message
    src, _ = _long(monkeypatch, tmp_path, rc=1, stderr="Error: No space left on device")
    with pytest.raises(sn.NormalizeError) as e:
        sn.normalize_source(src, str(tmp_path / "ing"), "virality")
    assert e.value.code == "NORMALIZE_NO_SPACE"
    src, _ = _long(monkeypatch, tmp_path)
    monkeypatch.setattr(sn.shutil, "disk_usage", lambda p: type("U", (), {"free": 1})())
    with pytest.raises(sn.NormalizeError) as e:
        sn.normalize_source(src, str(tmp_path / "ing"), "virality")
    assert e.value.code == "NORMALIZE_NO_SPACE"


def test_fast_normalize_gpu_decode_and_fast_preset(monkeypatch):
    """Load test 2026-11-16: 19-min 1440p VP9 took 382 s with the clip preset + CPU decode -> NVDEC + p4 at cq 17."""
    monkeypatch.delenv("SONYA_NORMALIZE_FAST", raising=False)
    a = sn.fast_nvenc_args()
    assert a[a.index("-preset") + 1] == "p4" and a[a.index("-cq") + 1] == "17" and "-multipass" not in a
    cmd = sn.normalize_cmd("ffmpeg", Path("a.webm"), Path("b.mp4"), sn.summarize_probe(_raw(avg="30/1")), a, hw_decode=True)
    assert cmd[cmd.index("-hwaccel") + 1] == "cuda" and cmd.index("-hwaccel") < cmd.index("-i")
    assert "-hwaccel" not in sn.normalize_cmd("ffmpeg", Path("a"), Path("b"), sn.summarize_probe(_raw(avg="30/1")), a)
    monkeypatch.setenv("SONYA_NORMALIZE_FAST", "0")
    assert not sn.fast_enabled()
