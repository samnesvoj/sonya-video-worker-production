"""trailer_cinematic v1: TTS interface + calibration fit, whole-run cost forecast, regression for shared
modules (registry untouched — v1 is CLI-only). No network, no models."""
import wave

import numpy as np
import pytest

from scripts.shared import tts


def test_run_forecast_for_15_min_is_far_under_one_dollar():
    from modes.trailer_cinematic.runner import forecast_run
    fc = forecast_run([60000], "google/gemini-3.8-flash")
    assert fc["usd_with_repairs"] < 1.0 and fc["calls"] == 7


# ── TTS ─────────────────────────────────────────────────────────────────────

class _FakeTTS(tts.TTSProvider):
    name, voice = "fake", "v"

    def _synth(self, text, out_wav):                # 0.2 s silence + 0.06 s per char + 0.25 s per pause
        c, p = tts.text_features(tts.normalize_ru(text))
        sr = 8000
        sec = 0.1 + 0.06 * c + 0.25 * p
        x = np.zeros(int((sec + 0.4) * sr), dtype="<i2")
        x[int(0.2 * sr):int((0.2 + sec) * sr)] = 8000
        with wave.open(out_wav, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(x.tobytes())


def test_tts_interface_trims_silence_and_calibrates(tmp_path):
    p = _FakeTTS()
    wav, sec = p.synthesize("Всё началось.", str(tmp_path / "a.wav"))
    c, n = tts.text_features("Всё началось.")
    assert wav.endswith("a.wav") and abs(sec - (0.1 + 0.06 * c + 0.25 * n) - 0.06) < 0.01   # edges trimmed
    r = p.calibrate(str(tmp_path))
    assert abs(r["chars_per_sec"] - 1 / 0.06) < 0.5 and abs(r["pause_sec"] - 0.25) < 0.05
    assert p.max_chars(3.0) > 0 and abs(p.estimate_sec("Кто он такой?") - (0.16 + 0.06 * 13 + 0.25)) < 0.1


def test_tts_providers():
    assert set(tts.PROVIDERS) == {"silero", "xtts", "elevenlabs"}
    el = tts.get_provider("elevenlabs")
    assert not el.available() and not el.test_only
    with pytest.raises(tts.TTSUnavailable):
        el.synthesize("x", "/tmp/x.wav")
    s = tts.get_provider("silero")
    assert s.info()["test_only"] is True and "TEST ONLY" in s.license
    assert tts.normalize_ru("Ему 25 лет — «и всё»") == "Ему двадцать пять лет , и всё"
    with pytest.raises(ValueError):
        tts.get_provider("nope")


def test_shared_registry_and_worker_untouched():
    """Regression: v1 is CLI-only — the shared registry and the desktop ASR mode set are unchanged."""
    from scripts.mode_registry import list_modes
    assert "trailer_cinematic" not in list_modes()
    assert {"trailer_film_breaker", "virality", "beep"} <= set(list_modes())
