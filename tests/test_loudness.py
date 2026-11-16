"""Even clip loudness (scripts/shared/loudness.py): -14 LUFS by one linear gain + limiter."""
import shutil
import subprocess

import pytest

from scripts.shared import loudness as L


def test_target_env():
    assert L.target_lufs({}) == -14.0
    assert L.target_lufs({"SONYA_CLIP_LUFS": "off"}) is None
    assert L.target_lufs({"SONYA_CLIP_LUFS": "-16"}) == -16.0
    assert L.target_lufs({"SONYA_CLIP_LUFS": "loud"}) == -14.0
    assert L.target_lufs({"SONYA_CLIP_LUFS": "-2"}) == -14.0     # nonsense target -> default


def test_gain_rules():
    assert L.gain_db(-24.1, -14.0) == 10.1          # quiet street interview (prod aed1964f)
    assert L.gain_db(-7.4, -14.0) == -6.6           # mastered YouTube show (prod b53e8674)
    assert L.gain_db(-14.2, -14.0) is None          # already right
    assert L.gain_db(-40.0, -14.0) == 12.0          # boost is capped
    assert L.gain_db(-70.0, -14.0) is None          # silence is not pumped up
    assert L.gain_db(None, -14.0) is None
    assert L.gain_db(-24.0, None) is None           # switched off


def test_filter():
    assert L.audio_filter(None) is None
    f = L.audio_filter(10.1)
    assert f.startswith("volume=10.10dB,") and "alimiter=limit=0.84" in f


def test_parse_summary():
    err = "[Parsed_ebur128_0] t: 1 M: -20 S: -20 I: -21.0 LUFS\nSummary:\n  Integrated loudness:\n    I:         -17.3 LUFS\n"
    assert L.parse_integrated(err) == -17.3
    assert L.parse_integrated("I: -inf LUFS") is None
    assert L.parse_integrated("") is None
    # Debian ffmpeg 5.1 (Vast image): a filter that failed to init still prints a 0.0 summary -> must not read as "loud"
    assert L.parse_integrated("Summary:\n  Integrated loudness:\n    I:           0.0 LUFS\n") is None


def test_measure_failed_ffmpeg_is_none(monkeypatch):
    class R:
        returncode, stderr = 1, "    I:           0.0 LUFS\n"
    monkeypatch.setattr(L.subprocess, "run", lambda *a, **k: R())
    assert L.measure("ffmpeg", "x.mp4", 0, 5) is None


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
@pytest.mark.parametrize("amp", ["0.3", "0.5,volume=12dB"])   # ≈ -25 and ≈ -10 LUFS
def test_real_ffmpeg_lands_near_target(tmp_path, amp):
    """A quiet and a loud source both end within 1 LU of -14 after the gain + limiter."""
    src = tmp_path / f"src_{amp}.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    f"anoisesrc=c=pink:d=8:r=48000:a={amp}", str(src)], check=True)
    g = L.clip_gain("ffmpeg", str(src), 1.0, 6.0)
    assert g is not None
    out = tmp_path / "out.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "1", "-i", str(src), "-t", "6",
                    "-af", L.audio_filter(g), str(out)], check=True)
    assert abs(L.measure("ffmpeg", str(out)) - (-14.0)) <= 1.0
