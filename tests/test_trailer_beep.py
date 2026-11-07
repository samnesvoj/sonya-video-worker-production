"""BEEP step inside the Trailer: mask format, intervals from the narration timing, bleeped voice, subtitles masked."""
import shutil
import subprocess
import wave

import numpy as np
import pytest

from modes.trailer_cinematic import censor, subs
from scripts.shared.profanity import mask_word


def _vt():
    return {"total": 10.0, "voice_end": 6.0, "segments": [
        {"id": "s1", "t0": 0.5, "t1": 3.0, "sentences": [
            {"text": "Этот парень просто пиздец.", "t0": 0.5, "t1": 3.0, "sec": 2.5}]},
        {"id": "s2", "t0": 3.5, "t1": 6.0, "sentences": [
            {"text": "Ну и сука, блять, дело.", "t0": 3.5, "t1": 6.0, "sec": 2.5}]}]}


def test_mask_hash_format():
    assert mask_word("пиздец.", "hash") == "пи###ц."
    assert mask_word("Блять,", "hash") == "Бл#ть,"
    assert mask_word("хуй", "hash") == "х##"
    assert mask_word("суки", "hash") == "с##и"
    assert mask_word("хуй-то", "hash") == "х##-то"          # only the profane part of a hyphenated word


def test_mask_default_is_the_old_beep_format():
    assert mask_word("хуйня") == "х****"
    assert mask_word("хуйня!") == "х****!"
    with pytest.raises(ValueError):
        mask_word("хуй", "nope")


def test_plan_masks_subtitles_and_finds_intervals():
    p = censor.plan(_vt())
    masked = [w for w, _, _ in p.words]
    assert "пи###ц." in masked and "с##а," in masked and "бл#ть," in masked
    assert not any(w.lower().startswith(("пизд", "сук", "бля")) for w in masked)
    assert len(p.hits) == 3 and len(p.intervals) >= 2          # сука + блять sit next to each other and may merge
    for a, b in p.intervals:
        assert 0.5 <= a < b <= 6.0
    # sentence edges stay clean (validator's sentence-edge level check)
    assert all(not (a < 0.5 + censor.CLAMP or b > 6.0 - censor.CLAMP + 1e-6) for a, b in p.intervals)


def test_asr_word_time_replaces_estimate():
    asr = [{"word": "пиздец", "start": 2.0, "end": 2.5}]
    p = censor.plan(_vt(), asr)
    r = [x for x in p.refined if x["word"].startswith("пиздец")][0]
    assert r["source"] == "asr" and (r["start"], r["end"]) == (2.0, 2.5)
    p2 = censor.plan(_vt(), None)
    assert all(x["source"] == "estimate" for x in p2.refined)


def test_clean_text_has_no_bleeps():
    vt = {"total": 5.0, "voice_end": 3.0, "segments": [{"id": "a", "t0": 0.5, "t1": 3.0, "sentences": [
        {"text": "Это обычное предложение про корабль.", "t0": 0.5, "t1": 3.0, "sec": 2.5}]}]}
    p = censor.plan(vt)
    assert p.intervals == [] and p.hits == []


def test_without_profanity():
    assert censor.without_profanity("Ну и пиздец, парень.") == "Ну и парень."


def test_subtitles_fit_with_masks():
    p = censor.plan(_vt())
    ass = subs.to_ass(p.words, 10.0)
    assert "ПИ###Ц" in ass and "ПИЗДЕЦ" not in ass and "БЛ#ТЬ" in ass


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg")
def test_beep_voice_replaces_the_word(tmp_path):
    src = tmp_path / "v.wav"
    sr = 48000
    t = np.arange(sr * 4) / sr
    sig = (0.5 * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2")
    with wave.open(str(src), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(sig.tobytes())
    out = censor.beep_voice(str(src), [[1.0, 2.0]], str(tmp_path / "o.wav"))
    with wave.open(out, "rb") as w:
        y = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(float).reshape(-1, w.getnchannels())[:, 0]
    seg = y[int(1.1 * sr):int(1.9 * sr)]
    spec = np.abs(np.fft.rfft(seg))
    assert abs(np.argmax(spec) / (len(seg) / sr) - 1000) < 30          # 1 kHz beep inside, the 220 Hz voice is gone
    before = y[int(0.2 * sr):int(0.8 * sr)]
    assert abs(np.argmax(np.abs(np.fft.rfft(before))) / (len(before) / sr) - 220) < 30


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg")
def test_beep_voice_keeps_level_outside_the_bleep(tmp_path):
    """mono voice + stereo beep source must not be downmixed by amix (-3 dB regression)."""
    src = tmp_path / "v.wav"
    sr = 48000
    t = np.arange(sr * 4) / sr
    sig = (0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2")
    with wave.open(str(src), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(sig.tobytes())
    out = censor.beep_voice(str(src), [[1.0, 1.5]], str(tmp_path / "o.wav"))
    with wave.open(out, "rb") as w:
        ch = w.getnchannels()
        y = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(float).reshape(-1, ch)
    ref = subprocess.run(["ffmpeg", "-v", "error", "-i", str(src), "-af", "aformat=channel_layouts=stereo", "-f", "s16le", "-"],
                         capture_output=True).stdout
    r = np.frombuffer(ref, dtype="<i2").astype(float).reshape(-1, 2)
    a, b = y[int(2.5 * sr):int(3.5 * sr)], r[int(2.5 * sr):int(3.5 * sr)]
    assert abs(20 * np.log10(np.sqrt((a ** 2).mean()) / np.sqrt((b ** 2).mean()))) < 0.3
