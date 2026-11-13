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
    # libx264: -b:v is ignored next to -crf, so the 6 Mb/s cap is a VBV cap (streamer v2 measured 7.1 Mb/s without it)
    assert cp.encoder_args("cpu") == ["-c:v", "libx264", "-preset", "slow", "-crf", "19", "-maxrate", "6M",
                                      "-bufsize", "12M", "-pix_fmt", "yuv420p"]
    assert cp.encoder_args("cpu", None) == ["-c:v", "libx264", "-preset", "slow", "-crf", "19", "-pix_fmt", "yuv420p"]


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


def test_run_encode_nvenc_failure_or_hang_falls_back_to_libx264(monkeypatch):
    """GeForce: max 8 NVENC sessions; L40S: cuCtxCreate errors / a hung NVENC render. The clip is re-encoded on CPU."""
    import subprocess
    from scripts.shared import cut_profile as cp
    calls = []

    def run(cmd, **kw):
        calls.append((cmd, kw.get("timeout")))
        if "h264_nvenc" in cmd:
            if len(calls) == 1:
                return subprocess.CompletedProcess(cmd, 1, "", "OpenEncodeSessionEx failed: incompatible client key (21)")
            raise subprocess.TimeoutExpired(cmd, kw["timeout"])
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(cp.subprocess, "run", run)
    make = lambda a: ["ffmpeg", "-i", "in.mp4", *a, "out.mp4"]
    for _ in range(2):                                            # 1st: error, 2nd: hang (timeout)
        r, used, _s = cp.run_encode(make, cp.encoder_args("nvenc"), timeout=cp.encode_timeout(45))
        assert r.returncode == 0 and used == "nvenc->cpu"
        assert calls[-1][0] == make(cp.encoder_args("cpu", "6M")) and calls[-1][1] == 450
    r, used, _s = cp.run_encode(make, cp.encoder_args("cpu"))
    assert used == "cpu" and len(calls) == 5                     # libx264 is never retried
    assert cp.encode_timeout(5) == 300                            # floor for short clips


def test_encoder_family_trusts_the_readiness_probe(monkeypatch):
    monkeypatch.setattr(cp, "_FAMILY", {})
    monkeypatch.setattr(cp, "_encoder_works", lambda ff, fam: (_ for _ in ()).throw(AssertionError("probed again")))
    monkeypatch.setenv("SONYA_NVENC_PROBED", "ok")
    assert cp.encoder_family("ffmpeg-a") == "nvenc"
    monkeypatch.setenv("SONYA_NVENC_PROBED", "fail")
    assert cp.encoder_family("ffmpeg-b") == "cpu"


def test_run_encode_all_nvenc_sessions_busy_goes_straight_to_libx264(monkeypatch, tmp_path):
    """GeForce session cap (feat/multi-gpu nvenc_guard) inside run_encode: no free slot -> libx264 at once, reported as nvenc->cpu."""
    import contextlib
    import subprocess
    from scripts.shared import nvenc_guard
    calls = []
    monkeypatch.setattr(cp.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0, "", ""))
    monkeypatch.setattr(nvenc_guard, "session", lambda limit=None: contextlib.nullcontext(False))
    monkeypatch.setattr(nvenc_guard, "max_sessions", lambda: 3)
    make = lambda a: ["ffmpeg", *a]
    r, used, _s = cp.run_encode(make, cp.encoder_args("nvenc", "6M"))
    assert used == "nvenc->cpu" and calls == [make(cp.encoder_args("cpu", "6M"))]


def test_site_job_gets_eight_clips_by_default(video, tmp_path, monkeypatch):
    """Owner 2026-11-12: «8 клипов по видео» without any max_clips from the site; an explicit value wins."""
    monkeypatch.delenv("SONYA_DEFAULT_MAX_CLIPS", raising=False)
    out = cp.apply_profile("virality", dict(SITE_VIRALITY_PARAMS), video, str(tmp_path), ass_supported=True, editor=False)
    assert out["max_clips"] == 8
    out = cp.apply_profile("virality", dict(SITE_VIRALITY_PARAMS, max_clips=2), video, str(tmp_path),
                           ass_supported=True, editor=False)
    assert out["max_clips"] == 2
    assert cp.default_max_clips({}) == 8 and cp.default_max_clips({"max_clips": "x"}) == 8
    monkeypatch.setenv("SONYA_DEFAULT_MAX_CLIPS", "5")
    assert cp.default_max_clips({}) == 5
