"""scripts/shared/nvenc_guard.py: the GeForce NVENC session cap never fails a job - extra encodes go to libx264 (the same
Desktop CPU args every NVENC-less server uses); no cap / no NVENC in the command -> the command runs unchanged."""
import subprocess
import sys
import textwrap
import time

import pytest

from scripts.shared import nvenc_guard as ng
from scripts.shared.cut_profile import encoder_args

HEAD = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostdin", "-ss", "1.000", "-i", "in.mp4", "-t", "30.000", "-vf", "crop=1"]
TAIL = ["-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "out.mp4"]


@pytest.fixture
def calls(monkeypatch):
    seen = []
    results = []

    def run(cmd, **kw):
        seen.append(list(cmd))
        return results.pop(0) if results else subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(ng.subprocess, "run", run)
    return seen, results


@pytest.mark.parametrize("bitrate", ["6M", None])
def test_to_cpu_is_exactly_the_desktop_cpu_encoder(bitrate):
    nv = HEAD + encoder_args("nvenc", bitrate) + TAIL
    assert ng.to_cpu(nv) == HEAD + encoder_args("cpu", bitrate) + TAIL


def test_to_cpu_adds_overwrite_and_pix_fmt_when_missing():
    out = ng.to_cpu(["ffmpeg", "-i", "a.mp4", "-c:v", "h264_nvenc", "-preset", "p4", "-b:v", "5M", "b.mp4"])
    assert out == ["ffmpeg", "-y", "-i", "a.mp4", "-c:v", "libx264", "-preset", "slow", "-crf", "18",
                   "-maxrate", "5M", "-bufsize", "10M", "-pix_fmt", "yuv420p", "b.mp4"]
    assert ng.to_cpu(["ffmpeg", "-c:v", "libx264", "o.mp4"]) == ["ffmpeg", "-c:v", "libx264", "o.mp4"]


def test_no_nvenc_in_command_runs_unchanged(calls, monkeypatch):
    seen, _ = calls
    monkeypatch.setenv("SONYA_NVENC_MAX_SESSIONS", "1")
    cmd = HEAD + encoder_args("cpu") + TAIL
    ng.run(cmd, capture_output=True, text=True)
    assert seen == [cmd]


def test_no_cap_nvenc_runs_unchanged(calls, monkeypatch):
    seen, _ = calls
    monkeypatch.delenv("SONYA_NVENC_MAX_SESSIONS", raising=False)
    cmd = HEAD + encoder_args("nvenc") + TAIL
    assert ng.run(cmd, capture_output=True, text=True).returncode == 0
    assert seen == [cmd]


def test_cap_reached_this_encode_goes_to_the_cpu(calls, monkeypatch, tmp_path):
    seen, _ = calls
    monkeypatch.setenv("SONYA_NVENC_MAX_SESSIONS", "1")
    monkeypatch.setenv("SONYA_NVENC_SLOT_DIR", str(tmp_path))
    cmd = HEAD + encoder_args("nvenc") + TAIL
    with ng.session() as got:                     # another clip render holds the only session
        assert got is True
        ng.run(cmd, capture_output=True, text=True)
    assert seen == [HEAD + encoder_args("cpu") + TAIL]
    ng.run(cmd, capture_output=True, text=True)   # released -> NVENC again
    assert seen[-1] == cmd


def test_slots_are_shared_between_worker_processes(calls, monkeypatch, tmp_path):
    seen, _ = calls
    monkeypatch.setenv("SONYA_NVENC_MAX_SESSIONS", "2")
    monkeypatch.setenv("SONYA_NVENC_SLOT_DIR", str(tmp_path))
    holder = subprocess.Popen([sys.executable, "-c", textwrap.dedent(f"""
        import fcntl, os, sys, time
        fds = []
        for k in range(2):
            f = os.open({str(tmp_path)!r} + f'/slot-{{k}}.lock', os.O_RDWR | os.O_CREAT, 0o600)
            fcntl.flock(f, fcntl.LOCK_EX); fds.append(f)
        print('held', flush=True); time.sleep(30)
        """)], stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        ng.run(HEAD + encoder_args("nvenc") + TAIL, capture_output=True, text=True)
        assert seen[-1] == HEAD + encoder_args("cpu") + TAIL
    finally:
        holder.kill()
        holder.wait()
    time.sleep(0.05)
    ng.run(HEAD + encoder_args("nvenc") + TAIL, capture_output=True, text=True)
    assert "h264_nvenc" in seen[-1]                 # the dead process released its slots


def test_refused_session_is_retried_on_the_cpu(calls, monkeypatch):
    seen, results = calls
    monkeypatch.delenv("SONYA_NVENC_MAX_SESSIONS", raising=False)
    results.append(subprocess.CompletedProcess([], 1, "", "[h264_nvenc @ 0x1] OpenEncodeSessionEx failed: incompatible client key (21)"))
    cmd = HEAD + encoder_args("nvenc") + TAIL
    assert ng.run(cmd, capture_output=True, text=True).returncode == 0
    assert seen == [cmd, HEAD + encoder_args("cpu") + TAIL]


def test_other_failures_are_not_retried(calls, monkeypatch):
    seen, results = calls
    monkeypatch.delenv("SONYA_NVENC_MAX_SESSIONS", raising=False)
    results.append(subprocess.CompletedProcess([], 1, "", "in.mp4: Invalid data found when processing input"))
    assert ng.run(HEAD + encoder_args("nvenc") + TAIL, capture_output=True, text=True).returncode == 1
    assert len(seen) == 1


def test_render_tracked_goes_through_the_guard(calls, monkeypatch, tmp_path):
    """The Cut renderer: same command as before when NVENC is free (moments, crop, subtitles untouched)."""
    seen, _ = calls
    from scripts.shared.crop import smart_crop_adapter as sca
    monkeypatch.delenv("SONYA_NVENC_MAX_SESSIONS", raising=False)
    track = {"layout": "fit"}
    sca.render_tracked("in.mp4", str(tmp_path / "o.mp4"), 1.0, 2.0, track, encoder_args=encoder_args("nvenc"))
    assert "h264_nvenc" in seen[0] and seen[0][-1] == str(tmp_path / "o.mp4")
    monkeypatch.setenv("SONYA_NVENC_MAX_SESSIONS", "1")
    monkeypatch.setenv("SONYA_NVENC_SLOT_DIR", str(tmp_path / "slots"))
    with ng.session():
        sca.render_tracked("in.mp4", str(tmp_path / "o.mp4"), 1.0, 2.0, track, encoder_args=encoder_args("nvenc"))
    assert ng.to_cpu(seen[0]) == seen[1]
