"""
tts.py
======
Text-to-speech behind one small interface, so a mode (Trailer) never knows which engine speaks.

    p = get_provider("silero")              # | "xtts" | "elevenlabs"
    wav, sec = p.synthesize("Текст.")        # measured length of the voiced audio (edge silence trimmed)
    model = p.calibrate()                     # sec ≈ base + chars/cps + pauses × pause_sec, fitted per voice
    p.estimate_sec("Фраза."), p.max_chars(3.0)

Providers
---------
silero      Silero TTS v4_ru from the local torch-hub cache (no network, CPU, fast).
            TEST ONLY — non-commercial license (CC BY-NC-SA 4.0); replace in production.
xtts        Coqui XTTS-v2 in its own venv (SONYA_XTTS_PYTHON), run as a child process.
            LICENSE: Coqui Public Model License (non-commercial) — TEST-ONLY.
elevenlabs  Interface stub for the production voice (keys/voice ids live server-side);
            raises TTSUnavailable until it is implemented — never a silent fallback.

Every provider reports `license` and `test_only`, and the Trailer report carries them.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Dict, Optional

VERSION = "tts-v2"
# calibration set: different lengths and punctuation, so the fit separates speed from pauses
CALIBRATION_TEXTS = (
    "Всё началось с одного звонка.",
    "Он думал, что это будет обычная поездка.",
    "Но одна встреча изменила всё, и теперь пути назад уже нет.",
    "Кто он такой? Зачем он вернулся? И почему именно сейчас?",
    "Десять лет тишины, одна ошибка, и прошлое снова стучится в дверь.",
    "Они смеялись. Пока не поняли, что это не шутка.",
)
PAUSE_CHARS = ",.!?;:…"


class TTSUnavailable(RuntimeError):
    pass


def wav_duration(path: str) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def trim_silence(path: str, threshold_db: float = -45.0, keep: float = 0.03) -> float:
    """Cut leading/trailing silence in place (16-bit mono WAV); returns the new length in seconds."""
    import numpy as np
    with wave.open(str(path), "rb") as w:
        sr, n = w.getframerate(), w.getnframes()
        data = np.frombuffer(w.readframes(n), dtype="<i2")
    if not len(data):
        return 0.0
    thr = 32767 * 10 ** (threshold_db / 20)
    loud = np.flatnonzero(np.abs(data.astype(np.int32)) > thr)
    if not len(loud):
        return len(data) / sr
    a = max(0, loud[0] - int(keep * sr))
    b = min(len(data), loud[-1] + int(keep * sr))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data[a:b].tobytes())
    return (b - a) / sr


def text_features(text: str) -> tuple:
    t = re.sub(r"\s+", " ", text).strip()
    return len(t), sum(t.count(c) for c in PAUSE_CHARS)


def fit_rate(samples) -> Dict[str, float]:
    """samples: [(chars, pauses, sec)] → least squares sec = base + chars·k + pauses·pause_sec."""
    import numpy as np
    X = np.array([[1.0, c, p] for c, p, _ in samples])
    y = np.array([s for _, _, s in samples])
    (base, k, pause), *_ = np.linalg.lstsq(X, y, rcond=None)
    k = max(k, 1e-3)
    pred = X @ np.array([base, k, pause])
    return {"base_sec": round(float(base), 4), "chars_per_sec": round(1.0 / float(k), 3),
            "pause_sec": round(float(pause), 4),
            "max_abs_err_sec": round(float(np.max(np.abs(pred - y))), 3), "samples": len(samples)}


_ONES = ["ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять", "десять",
         "одиннадцать", "двенадцать", "тринадцать", "четырнадцать", "пятнадцать", "шестнадцать",
         "семнадцать", "восемнадцать", "девятнадцать"]
_TENS = ["", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят", "восемьдесят",
         "девяносто"]


def _num_ru(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    return str(n)    # long numbers stay digits: the Writer is told to spell them out


def normalize_ru(text: str) -> str:
    """What Silero can read: digits < 100 spelled out, «—» as a pause, no markup."""
    text = re.sub(r"\b(\d{1,2})\b", lambda m: _num_ru(int(m.group(1))), text)
    text = text.replace("—", ",").replace("–", ",").replace("«", "").replace("»", "").replace('"', "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


class TTSProvider:
    """synthesize(text) -> (wav_path, duration_sec). Duration is measured on the written WAV."""
    name = "base"
    license = ""
    test_only = True
    voice = ""
    rate: Optional[Dict[str, float]] = None

    def _synth(self, text: str, out_wav: str) -> None:
        raise NotImplementedError

    def synthesize(self, text: str, out_wav: Optional[str] = None):
        if out_wav is None:
            fd, out_wav = tempfile.mkstemp(suffix=".wav", prefix=f"sonya-tts-{self.name}-")
            os.close(fd)
        self._synth(text, out_wav)
        return out_wav, trim_silence(out_wav)

    def available(self) -> bool:
        return True

    def info(self) -> Dict[str, object]:
        return {"provider": self.name, "voice": self.voice, "license": self.license,
                "test_only": self.test_only, "version": VERSION, "rate": self.rate}

    def calibrate(self, cache_dir: Optional[str] = None) -> Dict[str, float]:
        """Fit this voice's real speech model on CALIBRATION_TEXTS (cached per provider+voice+version)."""
        cache = Path(cache_dir or tempfile.gettempdir()) / f"sonya-tts-rate-{self.name}-{self.voice}-{VERSION}.json"
        try:
            self.rate = json.loads(cache.read_text())
            return self.rate
        except (OSError, ValueError):
            pass
        samples = []
        with tempfile.TemporaryDirectory() as td:
            for i, t in enumerate(CALIBRATION_TEXTS):
                _, sec = self.synthesize(t, os.path.join(td, f"cal{i}.wav"))
                c, p = text_features(normalize_ru(t))
                samples.append((c, p, sec))
        self.rate = fit_rate(samples)
        try:
            cache.write_text(json.dumps(self.rate))
        except OSError:
            pass
        return self.rate

    def estimate_sec(self, text: str) -> float:
        r = self.rate or self.calibrate()
        c, p = text_features(normalize_ru(text))
        return max(0.3, r["base_sec"] + c / r["chars_per_sec"] + p * r["pause_sec"])

    def max_chars(self, seconds: float, pauses: int = 1) -> int:
        r = self.rate or self.calibrate()
        return max(0, int((seconds - r["base_sec"] - pauses * r["pause_sec"]) * r["chars_per_sec"]))


class SileroProvider(TTSProvider):
    name = "silero"
    license = "TEST ONLY — non-commercial license (Silero CC BY-NC-SA 4.0); replace in production"
    test_only = True
    MODEL = Path(os.environ.get("SONYA_SILERO_MODEL") or
                 Path.home() / ".cache/torch/hub/snakers4_silero-models_master/src/silero/model/v4_ru.pt")
    SR = 48000

    def __init__(self, voice: str = "aidar"):
        self.voice = voice
        self._model = None

    def available(self) -> bool:
        return self.MODEL.is_file()

    def _load(self):
        if self._model is None:
            if not self.available():
                raise TTSUnavailable(f"Silero model not found: {self.MODEL}")
            import torch
            torch.set_num_threads(max(1, min(8, os.cpu_count() or 4)))
            imp = torch.package.PackageImporter(str(self.MODEL))
            self._model = imp.load_pickle("tts_models", "model")
            self._model.to(torch.device("cpu"))
        return self._model

    def _synth(self, text: str, out_wav: str) -> None:
        import numpy as np
        model = self._load()
        audio = model.apply_tts(text=normalize_ru(text), speaker=self.voice, sample_rate=self.SR,
                                put_accent=True, put_yo=True)
        pcm = (np.clip(audio.numpy(), -1, 1) * 32767).astype("<i2")
        with wave.open(str(out_wav), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.SR)
            w.writeframes(pcm.tobytes())


_XTTS_CHILD = r"""
import json, os, sys
os.environ["COQUI_TOS_AGREED"] = "1"
from TTS.api import TTS
req = json.load(open(sys.argv[1]))
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")
for it in req["items"]:
    tts.tts_to_file(text=it["text"], speaker=req["speaker"], language="ru", file_path=it["out"])
    print("ok", it["out"], flush=True)
"""


class XTTSProvider(TTSProvider):
    """Coqui XTTS-v2 through its own venv. Batch-friendly: synthesize_many() loads the model once."""
    name = "xtts"
    license = "Coqui Public Model License (non-commercial) — test-only"
    test_only = True

    def __init__(self, voice: str = "Damien Black"):
        self.voice = voice
        self.python = os.environ.get("SONYA_XTTS_PYTHON", "/tmp/voice/xtts/bin/python")

    def available(self) -> bool:
        return Path(self.python).is_file()

    def synthesize_many(self, items, timeout: float = 1800.0) -> None:
        if not self.available():
            raise TTSUnavailable(f"XTTS venv not found: {self.python}")
        with tempfile.TemporaryDirectory() as td:
            req, child = Path(td) / "req.json", Path(td) / "xtts_child.py"
            req.write_text(json.dumps({"speaker": self.voice, "items": [
                {"text": normalize_ru(t), "out": str(o)} for t, o in items]}, ensure_ascii=False))
            child.write_text(_XTTS_CHILD)
            r = subprocess.run([self.python, "-I", str(child), str(req)], capture_output=True, text=True,
                               timeout=timeout)
            if r.returncode != 0:
                raise TTSUnavailable(f"XTTS failed: {r.stderr[-400:]}")

    def _synth(self, text: str, out_wav: str) -> None:
        self.synthesize_many([(text, out_wav)])


class ElevenLabsProvider(TTSProvider):
    """Production voice — interface only (owner: no ElevenLabs yet). voiceKey → voice_id mapping and
    the API key stay server-side (docs/TRAILER_PHASE2_BACKEND.md §5)."""
    name = "elevenlabs"
    license = "ElevenLabs commercial (paid plan)"
    test_only = False

    def __init__(self, voice: str = "trailer_voice_b"):
        self.voice = voice

    def available(self) -> bool:
        return False

    def _synth(self, text: str, out_wav: str) -> None:
        raise TTSUnavailable("ElevenLabs provider is not implemented yet (no key, no paid TTS today)")


PROVIDERS = {"silero": SileroProvider, "xtts": XTTSProvider, "elevenlabs": ElevenLabsProvider}


def get_provider(name: str = "silero", voice: Optional[str] = None) -> TTSProvider:
    cls = PROVIDERS.get(name)
    if cls is None:
        raise ValueError(f"unknown TTS provider {name!r}; one of {sorted(PROVIDERS)}")
    return cls(voice) if voice else cls()
