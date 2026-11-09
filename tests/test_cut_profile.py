"""The Mac mini recipe for production virality jobs (scripts/shared/cut_profile.py) = what the SONYA Desktop
daemon passes to the same runner (sonya-desktop desktop/sonya_desktop/modes/worker.py)."""
from __future__ import annotations

import json

import pytest

from scripts.shared import cut_profile as cp
from tools.fast_start_test import SITE_VIRALITY_PARAMS


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("SONYA_CUT_PROFILE", raising=False)
    monkeypatch.setenv("SONYA_RENDER_ARGS", '["-c:v", "libx264"]')     # no encoder probe in unit tests


@pytest.fixture
def video(tmp_path):
    p = tmp_path / "in.mp4"
    p.write_bytes(b"not a real video")
    return str(p)


def test_site_job_gets_the_desktop_settings(video, tmp_path):
    out = cp.apply_profile("virality", dict(SITE_VIRALITY_PARAMS), video, str(tmp_path / "w"),
                           ass_supported=True, editor=False)
    assert out["selection"] == "story" and out["story"] == {"strategy": "qa"}
    crop = out["crop"]
    assert crop["mode"] == "tracked" and crop["analysis"] == "per_clip" and crop["layout"] == "smart"
    assert crop["subtitles"] == {"mode": "burn", "burn_supported": True, "style": "highlight", "hook": False,
                                 "font": "dela"}
    assert len(crop["source_sha256"]) == 64 and crop["clip_cache_dir"].startswith(str(tmp_path / "w"))
    assert out["cut_profile"] == cp.PROFILE_VERSION


def test_hooks_and_styles_map_like_the_site(video, tmp_path):
    p = dict(SITE_VIRALITY_PARAMS, clipType="hooks", settings={"subtitleStyle": "basic"})
    out = cp.apply_profile("virality", p, video, str(tmp_path), ass_supported=True, editor=True)
    assert out["story"] == {"strategy": "hooks", "editor": True}
    assert out["crop"]["subtitles"]["style"] == "minimal" and "font" not in out["crop"]["subtitles"]


def test_legacy_settings_in_a_job_no_longer_select_the_old_path(video, tmp_path):
    p = dict(SITE_VIRALITY_PARAMS, crop={"mode": "legacy"}, llm="on")
    out = cp.apply_profile("virality", p, video, str(tmp_path), ass_supported=True, editor=False)
    assert out["crop"]["mode"] == "tracked" and out["selection"] == "story" and "llm" not in out


def test_desktop_style_job_is_left_alone_and_there_is_no_rollback_switch(video, tmp_path, monkeypatch):
    p = {"selection": "story", "crop": {"mode": "tracked"}}
    assert cp.apply_profile("virality", p, video, str(tmp_path)) is p
    assert cp.apply_profile("beep", {}, video, str(tmp_path)) == {}
    monkeypatch.setenv("SONYA_CUT_PROFILE", "legacy")      # the old switch does nothing any more
    out = cp.apply_profile("virality", dict(SITE_VIRALITY_PARAMS), video, str(tmp_path))
    assert out["selection"] == "story" and out["crop"]["mode"] == "tracked"
    half = {"selection": "story", "crop": {"mode": "legacy"}}  # half-Desktop settings get the full recipe
    assert cp.apply_profile("virality", half, video, str(tmp_path))["crop"]["mode"] == "tracked"


def test_encoder_args_are_the_desktop_quality_preset_at_6mbps():
    # sonya_desktop.media.encoders.encoder_args(family, "h264", "quality", "6M")
    assert cp.encoder_args("nvenc") == ["-c:v", "h264_nvenc", "-preset", "p6", "-rc", "vbr", "-cq", "20",
                                        "-b:v", "6M", "-pix_fmt", "yuv420p"]
    assert cp.encoder_args("cpu") == ["-c:v", "libx264", "-preset", "slow", "-crf", "19", "-b:v", "6M",
                                      "-pix_fmt", "yuv420p"]


def test_render_args_probe_falls_back_to_cpu(monkeypatch):
    monkeypatch.delenv("SONYA_RENDER_ARGS")
    monkeypatch.setattr(cp, "_encoder_works", lambda ff, fam: False)
    assert cp.ensure_render_args("ffmpeg") == "cpu"
    assert json.loads(__import__("os").environ["SONYA_RENDER_ARGS"])[:2] == ["-c:v", "libx264"]


def test_no_words_fails_instead_of_the_old_path(video, tmp_path, monkeypatch):
    from scripts.shared.enhancers import sonya_enhancer
    from modes.virality import runner
    monkeypatch.setattr(sonya_enhancer, "enrich_video_for_mode",
                        lambda **kw: {"warnings": ["word_timestamps: ASR down"], "word_timestamps": []})
    monkeypatch.setattr(cp, "ffmpeg_has_ass", lambda ffmpeg=None: True)
    monkeypatch.setattr(cp, "editor_available", lambda env=None: False)
    with pytest.raises(RuntimeError, match="ASR_FAILED"):
        runner.run(video, str(tmp_path / "output"), dict(SITE_VIRALITY_PARAMS))


def test_worker_progress_accepts_both_argument_orders():
    """BEEP calls progress_callback(0.2, "asr: …"), virality (stage, fraction): the worker must take both
    (before: AttributeError 'float'.lower() killed every BEEP job right after ASR)."""
    from scripts.gpu_worker import _progress_args
    assert _progress_args("word_timestamps", 0.72) == ("word_timestamps", 0.72)
    assert _progress_args(0.2, "asr: gigaam 120 words") == ("asr: gigaam 120 words", 0.2)
    assert _progress_args(None, None) == ("running", 0.0)
