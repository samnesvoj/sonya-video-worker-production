"""
beep_sounds.py
==============
SONYA's own censor sound library, synthesised from code (numpy) — no third-party
recordings, so no licence questions. Each preset renders a 4 s "fill" track:
continuous tones stay continuous, one-shot effects (quack, boing, cuckoo…)
repeat at a natural rate, so any bleep interval just takes the first N seconds.

    render(sound_id, seconds) -> float32 mono @ 48 kHz
    write_wav(sound_id, path)  -> 16-bit PCM WAV
"""
from __future__ import annotations

import io
import wave
from typing import Callable, Dict, List

import numpy as np

SR = 48000
LIBRARY_VERSION = "sonya-sounds-v2"   # v2: duck + dolphin removed (owner: sounded awful)


def _t(sec: float) -> np.ndarray:
    return np.arange(int(sec * SR)) / SR


def _env(n: int, attack: float = 0.005, release: float = 0.02) -> np.ndarray:
    e = np.ones(n)
    a, r = max(1, int(attack * SR)), max(1, int(release * SR))
    e[:a] = np.linspace(0, 1, a)
    e[-r:] = np.minimum(e[-r:], np.linspace(1, 0, r))
    return e


def _repeat(one: np.ndarray, period: float, sec: float) -> np.ndarray:
    out = np.zeros(int(sec * SR))
    step = int(period * SR)
    for s in range(0, len(out), step):
        k = min(len(one), len(out) - s)
        out[s:s + k] += one[:k]
    return out


def _square(f, t):
    return np.sign(np.sin(2 * np.pi * f * t))


def _saw(phase):
    return 2 * (phase - np.floor(phase + 0.5))


def _bandpass(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    X[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(X, len(x))


def tv_beep(sec):            # broadcast standard: 1 kHz sine
    return np.sin(2 * np.pi * 1000 * _t(sec))


def high_beep(sec):
    return np.sin(2 * np.pi * 1400 * _t(sec))


def bleep_bloop(sec):        # two tones alternating every 120 ms
    t = _t(sec)
    f = np.where((t // 0.12) % 2 == 0, 880.0, 1320.0)
    return np.sin(2 * np.pi * np.cumsum(f) / SR)


def car_horn(sec):           # two detuned square waves (typical 400/500 Hz horn pair)
    t = _t(sec)
    x = _square(405, t) + _square(510, t)
    return _bandpass(x, 200, 3500) * 0.5


def bike_horn(sec):          # "honk-honk": squeezed rubber horn
    one_t = _t(0.28)
    f = 520 + 40 * np.sin(2 * np.pi * 9 * one_t)
    honk = _bandpass(_saw(np.cumsum(f) / SR), 300, 3000) * _env(len(one_t), .01, .08)
    return _repeat(honk, 0.36, sec)


def boing(sec):              # spring: pitch vibrato decaying
    one_t = _t(0.6)
    f = 180 + 140 * np.exp(-one_t * 6) * np.sin(2 * np.pi * 14 * one_t) + 120 * np.exp(-one_t * 3)
    b = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-one_t * 4) * _env(len(one_t), .002, .05)
    return _repeat(b, 0.65, sec)


def cuckoo(sec):             # two falling notes, like a cuckoo clock (E5 -> C5)
    def note(f, d):
        tt = _t(d)
        return (np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(4 * np.pi * f * tt)) * _env(len(tt), .005, .08)
    one = np.concatenate([note(659.3, 0.16), np.zeros(int(0.04 * SR)), note(523.3, 0.22)])
    return _repeat(one, 0.55, sec)


def buzzer(sec):             # game-show "wrong answer" buzzer
    t = _t(sec)
    return _bandpass(_square(150, t) + 0.5 * _square(151.5, t), 80, 4000) * 0.6


def radio_static(sec):
    rng = np.random.default_rng(7)
    x = rng.standard_normal(int(sec * SR))
    return _bandpass(x, 300, 6000) * 0.5 * (0.8 + 0.2 * np.sin(2 * np.pi * 7 * _t(sec)))


def record_scratch(sec):     # back-and-forth scratch: noise with fast pitch sweeps
    rng = np.random.default_rng(11)
    t = _t(sec)
    n = rng.standard_normal(len(t))
    sweep = 0.5 + 0.5 * np.sin(2 * np.pi * 4.5 * t)
    lo = _bandpass(n, 400, 1800)
    hi = _bandpass(n, 1800, 7000)
    return (lo * (1 - sweep) + hi * sweep) * (0.4 + 0.6 * np.abs(np.sin(2 * np.pi * 4.5 * t)))


def robot_glitch(sec):       # bit-crushed square, random pitch every 60 ms
    rng = np.random.default_rng(3)
    t = _t(sec)
    steps = rng.choice([220, 330, 440, 660, 880, 990], size=int(sec / 0.06) + 1)
    f = steps[(t // 0.06).astype(int)]
    x = _square(1, np.cumsum(f) / SR)
    return np.round(x * 3) / 3 * 0.6


def whistle(sec):            # referee-style whistle: high tone with fast trill
    t = _t(sec)
    f = 2600 + 120 * np.sin(2 * np.pi * 28 * t)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * (0.85 + 0.15 * np.sin(2 * np.pi * 28 * t))


SOUNDS: Dict[str, Dict[str, object]] = {
    "tv_beep": {"name": "ТВ-писк (классика, 1 кГц)", "fn": tv_beep},
    "high_beep": {"name": "Высокий писк", "fn": high_beep},
    "bleep_bloop": {"name": "Бип-буп", "fn": bleep_bloop},
    "car_horn": {"name": "Клаксон", "fn": car_horn},
    "bike_horn": {"name": "Гудок-пищалка", "fn": bike_horn},
    "boing": {"name": "Боинг (пружина)", "fn": boing},
    "cuckoo": {"name": "Кукушка", "fn": cuckoo},
    "buzzer": {"name": "Зуммер «неправильно»", "fn": buzzer},
    "radio_static": {"name": "Шипение радио", "fn": radio_static},
    "record_scratch": {"name": "Скретч пластинки", "fn": record_scratch},
    "robot_glitch": {"name": "Робот-глитч", "fn": robot_glitch},
    "whistle": {"name": "Свисток", "fn": whistle},
}
PEAK = 0.5                   # ≈ -6 dBFS, the same loudness feel for every preset


def catalog() -> List[Dict[str, str]]:
    return [{"id": k, "name": str(v["name"])} for k, v in SOUNDS.items()]


def render(sound_id: str, seconds: float = 4.0) -> np.ndarray:
    if sound_id not in SOUNDS:
        raise ValueError(f"unknown sound {sound_id!r}")
    fn: Callable[[float], np.ndarray] = SOUNDS[sound_id]["fn"]  # type: ignore[assignment]
    x = np.asarray(fn(seconds), dtype=np.float64)
    m = float(np.max(np.abs(x))) or 1.0
    return (x / m * PEAK).astype(np.float32)


def wav_bytes(sound_id: str, seconds: float = 4.0) -> bytes:
    pcm = (np.clip(render(sound_id, seconds), -1, 1) * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm)
    return buf.getvalue()


def write_wav(sound_id: str, path, seconds: float = 4.0) -> str:
    with open(path, "wb") as f:
        f.write(wav_bytes(sound_id, seconds))
    return str(path)
