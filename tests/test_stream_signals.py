"""scripts/shared/stream_signals.py: per-second stream signals (no models, tiny synthetic media)."""
from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from scripts.shared import stream_signals as ss

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def test_robust_z_centres_on_median_and_masks():
    z = ss.robust_z([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 50], mask=[True] * 10 + [False])
    assert z[-1] == 0.0                                   # masked out
    assert max(z) <= 6.0 and min(z) >= -6.0


def test_robust_z_too_few_values_is_flat():
    assert ss.robust_z([1, 2, 3]) == [0.0, 0.0, 0.0]


@pytest.mark.parametrize("word,kind", [
    ("ахахаха", "laugh"), ("Хахах!", "laugh"), ("ору", "laugh"),
    ("Вау!", "react"), ("жесть", "react"),
    ("аааааа", "shout"), ("нееет", "shout"),
    ("стол", None), ("", None),
])
def test_reaction_kind(word, kind):
    assert ss.reaction_kind(word) == kind


def test_reaction_kind_mat():
    assert ss.reaction_kind("блять") in ("mat", "react")


def _w(text, t):
    return {"word": text, "start": t, "end": t + 0.3}


def test_speech_signals_hits_and_phrases():
    words = [_w("да", 5.0), _w("ладно", 5.4), _w("ахахах", 7.0), _w("ничего", 9.0), _w("Что?!", 12.0)]
    out = ss.speech_signals(words, 20)
    kinds = {k for _, k, _ in out["hits"]}
    assert {"laugh", "phrase"} <= kinds
    assert out["react"][7] >= 1.5 and out["react"][5] >= 1.0
    assert out["react"][12] > 0                           # exclamation
    assert len(out["rate"]) == 20


def test_chat_times_youtube_live_chat(tmp_path):
    lines = []
    for ms in (1000, 2500, 61000):
        lines.append(json.dumps({"replayChatItemAction": {"videoOffsetTimeMsec": str(ms), "actions": [
            {"addChatItemAction": {"item": {"liveChatTextMessageRenderer": {"message": {}}}}}]}}))
    lines.append(json.dumps({"replayChatItemAction": {"videoOffsetTimeMsec": "5000", "actions": [
        {"addChatItemAction": {"item": {"liveChatViewerEngagementMessageRenderer": {}}}}]}}))   # not a message
    p = tmp_path / "x.live_chat.json"
    p.write_text("\n".join(lines), encoding="utf-8")
    assert ss.chat_times(str(p)) == [1.0, 2.5, 61.0]


def test_chat_times_twitch_and_plain(tmp_path):
    p = tmp_path / "tw.json"
    p.write_text(json.dumps({"comments": [{"content_offset_seconds": 3.5}, {"content_offset_seconds": 1}]}))
    assert ss.chat_times(str(p)) == [1.0, 3.5]
    q = tmp_path / "plain.json"
    q.write_text(json.dumps([{"t": 2}, 1, "x"]))
    assert ss.chat_times(str(q)) == [1.0, 2.0]


def test_chat_signals_burst_and_too_few():
    times = [float(t) for t in range(0, 600, 5)] + [306.0 + i * 0.05 for i in range(80)]
    c = ss.chat_signals(times, 600, lag_sec=6.0)
    peak = max(range(600), key=lambda i: c["chat"][i])
    assert 295 <= peak <= 305                              # shifted back by the viewers' lag
    assert ss.chat_signals([1.0, 2.0], 600)["chat"] == [0.0] * 600


@needs_ffmpeg
def test_audio_signals_loud_high_voice(tmp_path):
    """20 s: quiet 120 Hz hum, then 4 s of a loud 300 Hz tone (a «shout»), then quiet again."""
    src = tmp_path / "a.wav"
    expr = "if(between(t,8,12),0.8*sin(2*PI*300*t),0.05*sin(2*PI*120*t))"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    f"aevalsrc='{expr}':s=8000:d=20", str(src)], check=True)
    cache = tmp_path / "audio.json"
    a = ss.audio_signals(str(src), cache_path=str(cache))
    assert len(a["db"]) == 20
    assert max(range(20), key=lambda i: a["loud"][i]) in range(8, 12)
    assert 280 <= a["pitch_hz"][10] <= 320
    assert 110 <= a["pitch_hz"][2] <= 130
    assert cache.is_file() and ss.audio_signals(str(src), cache_path=str(cache)) == a


@needs_ffmpeg
def test_visual_signals_flash_and_cut(tmp_path):
    """10 s dark video with a white flash at 5 s."""
    src = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=0x202020:s=320x180:r=30:d=10",
                    "-vf", "drawbox=x=0:y=0:w=iw:h=ih:color=white:t=fill:enable='between(t,5,5.3)'",
                    "-pix_fmt", "yuv420p", str(src)], check=True)
    v = ss.visual_signals(str(src))
    assert len(v["cut"]) == 10
    assert v["cut"][5] == 1.0
