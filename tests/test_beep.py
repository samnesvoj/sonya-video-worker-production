"""SONYA BEEP: intervals, effects (checked on the real audio), masked subtitles, ASR mandatory."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from scripts.shared import beep as bp
from scripts.shared.profanity import Hit

FF = shutil.which("ffmpeg")
needs_ff = pytest.mark.skipif(not FF, reason="ffmpeg")


def hit(s, e, i=0):
    return Hit(i, "x", s, e, "mat")


def test_intervals_pad_merge_clamp():
    cfg = bp.BeepConfig()
    iv = bp.censor_intervals([hit(1.0, 1.3), hit(1.4, 1.6), hit(5.0, 5.2), hit(9.95, 10.2)], 10.0, cfg)
    assert iv == [[0.95, 1.68], [4.95, 5.28], [9.9, 10.0]]
    assert bp.censor_intervals([], 10.0, cfg) == []


def test_config_validation():
    with pytest.raises(ValueError):
        bp.BeepConfig(effect="loud")
    with pytest.raises(ValueError):
        bp.BeepConfig(effect="custom")
    with pytest.raises(ValueError):
        bp.BeepConfig(subtitles="yes")


def make_video(path, dur=4.0, freq=440):
    subprocess.run([FF, "-loglevel", "error", "-f", "lavfi", "-i", f"testsrc2=size=320x240:rate=25:duration={dur}",
                    "-f", "lavfi", "-i", f"sine=frequency={freq}:sample_rate=48000:duration={dur}",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)], check=True)


def pcm(path):
    raw = subprocess.run([FF, "-loglevel", "error", "-i", str(path), "-f", "s16le", "-ac", "1", "-ar", "48000", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


def band(x, a, b, sr=48000):
    return x[int(a * sr):int(b * sr)]


def dom_freq(x, sr=48000):
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return np.fft.rfftfreq(len(x), 1 / sr)[int(np.argmax(spec))]


@needs_ff
@pytest.mark.parametrize("effect", ["mute", "beep", "custom"])
def test_effects_on_real_audio(tmp_path, effect):
    src = tmp_path / "in.mp4"
    make_video(src)
    asset = tmp_path / "boing.wav"
    subprocess.run([FF, "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=2500:sample_rate=48000:duration=2",
                    str(asset)], check=True)
    cfg = bp.BeepConfig(effect=effect, asset=str(asset) if effect == "custom" else None)
    out = tmp_path / "o.mp4"
    bp.render(str(src), str(out), [[1.0, 1.6], [2.5, 2.9]], cfg)
    x = pcm(out)
    inside, outside = band(x, 1.1, 1.5), band(x, 0.2, 0.8)
    assert abs(dom_freq(outside) - 440) < 15                        # untouched outside
    if effect == "mute":
        assert np.sqrt((inside ** 2).mean()) < 1e-3
    else:
        want = 1000 if effect == "beep" else 2500
        assert abs(dom_freq(inside) - want) < 15 and abs(dom_freq(band(x, 2.55, 2.85)) - want) < 15
        assert np.abs(np.fft.rfft(inside))[int(440 * len(inside) / 48000)] < \
            0.05 * np.abs(np.fft.rfft(outside))[int(440 * len(outside) / 48000)]  # original gone
    # picture is copied bit for bit (no re-encode without subtitles)
    def vhash(p):
        return subprocess.run([FF, "-loglevel", "error", "-i", str(p), "-map", "0:v", "-c", "copy", "-f", "md5", "-"],
                              capture_output=True, text=True, check=True).stdout
    assert vhash(src) == vhash(out)


@needs_ff
def test_no_hits_keeps_audio_and_video(tmp_path):
    src = tmp_path / "in.mp4"
    make_video(src)
    out = tmp_path / "o.mp4"
    bp.render(str(src), str(out), [], bp.BeepConfig())
    assert abs(dom_freq(band(pcm(out), 1.0, 2.0)) - 440) < 15


def test_mode_registry_has_beep():
    from scripts.mode_registry import list_modes
    assert "beep" in list_modes()


def test_runner_refuses_without_asr(tmp_path):
    from modes.beep.runner import run
    with pytest.raises(RuntimeError, match="BEEP_FAILED"):
        run("in.mp4", str(tmp_path), {})
    empty = tmp_path / "w.json"
    empty.write_text(json.dumps({"words": []}))
    with pytest.raises(RuntimeError, match="BEEP_FAILED"):
        run("in.mp4", str(tmp_path), {"word_timestamps_path": str(empty)})


@needs_ff
def test_runner_end_to_end_with_masked_srt(tmp_path):
    from modes.beep.runner import run
    src = tmp_path / "in.mp4"
    make_video(src)
    words = [{"word": " Это", "start": 0.2, "end": 0.5}, {"word": " пиздец,", "start": 0.6, "end": 1.1},
             {"word": " рубля", "start": 1.3, "end": 1.7}, {"word": " блять.", "start": 2.4, "end": 2.8}]
    wp = tmp_path / "w.json"
    wp.write_text(json.dumps({"words": words}, ensure_ascii=False))
    res = run(str(src), str(tmp_path), {"word_timestamps_path": str(wp), "beep": {"subtitles": "srt"}})
    rep = json.loads((tmp_path / "beep_report.json").read_text())
    assert [h["word"] for h in rep["hits"]] == ["пиздец,", "блять."] and rep["video"] == "stream_copy"
    assert rep["intervals"] == [[0.55, 1.18], [2.35, 2.88]]
    srt = (tmp_path / "beeped.srt").read_text()
    assert "п*****," in srt and "б****." in srt and "рубля" in srt and "пиздец" not in srt
    assert set(res["extras"]) == {"beeped.srt", "beep_report.json"} and Path(res["clips"][0]).is_file()
    assert abs(dom_freq(band(pcm(tmp_path / "beeped.mp4"), 0.65, 1.05)) - 1000) < 15


def test_every_library_sound_is_distinct_normalised_and_licence_free():
    from scripts.shared import beep_sounds as bs
    seen = set()
    for s in bs.SOUNDS:
        x = bs.render(s, 2.0)
        assert len(x) == 96000 and abs(float(np.abs(x).max()) - bs.PEAK) < 1e-3, s
        assert np.sqrt((x ** 2).mean()) > 0.03, s                 # audible throughout (fill sound)
        key = hash(np.round(x[:4800], 3).tobytes())
        assert key not in seen, s
        seen.add(key)
    assert bs.wav_bytes("cuckoo", 1.5)[:4] == b"RIFF"


@needs_ff
@pytest.mark.parametrize("sound", ["bike_horn", "boing", "car_horn", "cuckoo"])
def test_library_sound_replaces_the_word(tmp_path, sound):
    src = tmp_path / "in.mp4"
    make_video(src)
    out = tmp_path / "o.mp4"
    bp.render(str(src), str(out), [[1.0, 1.8]], bp.BeepConfig(sound=sound))
    x = pcm(out)
    inside, outside = band(x, 1.1, 1.7), band(x, 0.2, 0.8)
    assert np.sqrt((inside ** 2).mean()) > 0.05                    # the effect is there
    k = int(440 * len(inside) / 48000)
    assert np.abs(np.fft.rfft(inside))[k] < 0.1 * np.abs(np.fft.rfft(outside))[int(440 * len(outside) / 48000)]
    assert abs(dom_freq(outside) - 440) < 15
    assert not list(tmp_path.glob("*.wav"))                         # temp preset file cleaned up


def test_unknown_sound_is_refused():
    with pytest.raises(ValueError):
        bp.BeepConfig(sound="fart")
