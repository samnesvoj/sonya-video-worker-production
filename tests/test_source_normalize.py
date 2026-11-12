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
