"""
stream_signals.py
=================
Cheap per-second signals of a long stream VOD for moment search (modes/streamer).

Each signal is a list with one value per second of the source, z-scored against the stream's own
median / MAD (a loud streamer and a quiet one are judged on their own scale):

  audio_signals()   one ffmpeg pass, mono 8 kHz PCM:
                      loud   — RMS loudness (dBFS) z-score
                      pitch  — median voice pitch (autocorrelation, 75–500 Hz) z-score over voiced seconds;
                               shouting / squealing raises it, a calm read-out stays near 0
                      voiced — share of 40 ms frames with a clear pitch (speech / screaming vs game noise)
  speech_signals()  from ASR words: rate (words/s z-score) and lexical reaction hits per second
                      (mat, laughter «ахаха», «что?!», «да ладно», «вау», «а-а-а», exclamation marks)
  visual_signals()  one low-res ffmpeg pass (gray 64x36, 4 fps): cut — hard scene changes per second,
                      flash — sudden brightness jumps (explosions, flashbangs, kill effects)
  chat_signals()    chat messages per second (Twitch / YouTube live chat replay), when the VOD has chat

Every signal is cached as JSON next to the job's work dir. No model downloads, no network.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

VERSION = "stream-signals-v1"
RATE = 8000
FRAME = 320                     # 40 ms at 8 kHz
LAG_MIN, LAG_MAX = 16, 106      # 500 Hz .. 75 Hz


# ── helpers ─────────────────────────────────────────────────────────────────

def robust_z(values: Sequence[float], mask: Optional[Sequence[bool]] = None, clip: float = 6.0) -> List[float]:
    """(x - median) / (1.4826 · MAD) over the masked values; unmasked → 0. Clipped to ±clip."""
    a = np.asarray(values, dtype=np.float64)
    m = np.ones(len(a), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    live = a[m & np.isfinite(a)]
    if live.size < 10:
        return [0.0] * len(a)
    med = float(np.median(live))
    mad = float(np.median(np.abs(live - med))) or 1e-6
    z = (a - med) / (1.4826 * mad)
    z[~m | ~np.isfinite(a)] = 0.0
    return [round(float(v), 3) for v in np.clip(z, -clip, clip)]


def _load_cache(path: Optional[str], kind: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.isfile(path):
        return None
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return d if d.get("version") == VERSION and d.get("kind") == kind else None
    except (OSError, ValueError):
        return None


def _save_cache(path: Optional[str], kind: str, data: Dict[str, Any]) -> None:
    if not path:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.partial"
    Path(tmp).write_text(json.dumps(dict(data, version=VERSION, kind=kind)), encoding="utf-8")
    os.replace(tmp, path)


# ── audio ───────────────────────────────────────────────────────────────────

def _pitch_frames(x: np.ndarray) -> np.ndarray:
    """Pitch (Hz) per 40 ms frame of a (n, FRAME) float array; 0 = unvoiced. Normalized autocorrelation."""
    x = x - x.mean(axis=1, keepdims=True)
    x = x * np.hanning(FRAME)[None, :]
    spec = np.fft.rfft(x, n=1024, axis=1)
    ac = np.fft.irfft(np.abs(spec) ** 2, axis=1)[:, :LAG_MAX + 1]
    e0 = ac[:, 0] + 1e-9
    seg = ac[:, LAG_MIN:LAG_MAX + 1] / e0[:, None]
    lag = np.argmax(seg, axis=1)
    peak = seg[np.arange(len(seg)), lag]
    energy = np.sqrt(e0 / FRAME)
    voiced = (peak > 0.45) & (energy > 150.0)          # int16 scale: quieter frames are background
    return np.where(voiced, RATE / (lag + LAG_MIN), 0.0)


def audio_signals(media: str, ffmpeg: str = "ffmpeg", cache_path: Optional[str] = None) -> Dict[str, List[float]]:
    """-> {"db": [...], "loud": z, "pitch_hz": [...], "pitch": z, "voiced": [0..1]} per second."""
    c = _load_cache(cache_path, "audio")
    if c:
        return {k: c[k] for k in ("db", "loud", "pitch_hz", "pitch", "voiced")}
    p = subprocess.Popen([ffmpeg, "-v", "error", "-nostdin", "-i", media, "-vn", "-ac", "1", "-ar", str(RATE),
                          "-f", "s16le", "-"], stdout=subprocess.PIPE)
    assert p.stdout is not None
    db: List[float] = []
    pitch: List[float] = []
    voiced: List[float] = []
    while True:
        buf = p.stdout.read(RATE * 2 * 60)              # one minute at a time
        if not buf:
            break
        a = np.frombuffer(buf[: len(buf) // 2 * 2], dtype=np.int16).astype(np.float64)
        n_sec = int(math.ceil(len(a) / RATE))
        a = np.pad(a, (0, n_sec * RATE - len(a)))
        secs = a.reshape(n_sec, RATE)
        ms = np.mean(secs * secs, axis=1)
        db += [round(float(10 * math.log10(v / (32768.0 ** 2) + 1e-10)), 2) for v in ms]
        f0 = _pitch_frames(secs.reshape(-1, FRAME)).reshape(n_sec, RATE // FRAME)
        for row in f0:
            v = row[row > 0]
            voiced.append(round(len(v) / len(row), 3))
            pitch.append(round(float(np.median(v)), 1) if len(v) >= 3 else 0.0)
    p.wait()
    if p.returncode != 0 and not db:
        raise RuntimeError("audio decode failed")
    out = {"db": db, "loud": robust_z(db, [d > -60 for d in db]),
           "pitch_hz": pitch, "pitch": robust_z(pitch, [v >= 0.2 and f > 0 for v, f in zip(voiced, pitch)]),
           "voiced": voiced}
    _save_cache(cache_path, "audio", out)
    return out


# ── speech (ASR words) ──────────────────────────────────────────────────────

# reactions a viewer clips: laughter, surprise, disbelief, panic, triumph (lower-case, «ё» → «е»)
_LAUGH = re.compile(r"^(а?ха){2,}х?а?$|^(хе){2,}|^(хи){2,}|^ахах|^хаха|^ору$|^ржу")
_REACT_WORDS = {
    "вау", "офигеть", "охренеть", "ахренеть", "обалдеть", "жесть", "капец", "кошмар", "ужас", "имба", "изи",
    "гг", "господи", "мамочки", "мама", "помогите", "куда", "серьезно", "реально", "нифига", "ничего",
    "нормально", "красава", "красавчик", "пушка", "бомба", "ааа", "ооо", "эээ", "блин", "блять", "сука",
    "нет", "стоп", "подожди", "беги", "лови", "убил", "убили", "умер", "ваншот", "клатч", "эйс",
}
_REACT_PHRASES = re.compile(r"(?i)\b(да ладно|не может быть|ты чего|ты что|что это было|как так|"
                            r"ну нет|нет нет|да да да|вот это да|ни хрена|ни фига|я в шоке|что происходит|"
                            r"какого|что за)\b")


# owner 2026-11-10 ~06:00 (Buster examples): guests with a lead-in, a surprise entrance / costume, crude jokes
_GUEST_LEAD = re.compile(r"(сейчас|щас|скоро|сегодня|ща)\s+(к нам\s+)?(зайдет|зайдут|придет|придут|приедет|"
                         r"приедут|подключится|присоединится|будет гость|позвоню|позвоним|наберу|наберем)|"
                         r"\b(у нас|к нам)\s+(сегодня\s+)?(гост|зайдет|придет|приехал|пришел)|\bвстречайте\b|"
                         r"\bждем\s+(гост|его|ее)|\bкто\s+к\s+нам\b")
_GUEST_IN = re.compile(r"^(привет|приветик|здарова|здорово|здравствуй|здравствуйте|хай|салам|салют|заходи|"
                       r"заходите|проходи|проходите|присаживайся|садись|добропожаловать)$")
_REVEAL = re.compile(r"\b(смотрите|зацените|оцените|как я вам|что это|кто это|что на мне|мой образ|в образе|"
                     r"костюм|сюрприз|не узнали|узнаете|переоделся|переоделась|нарядился)\b")


def _norm(w: str) -> str:
    return re.sub(r"[^\w-]", "", w.lower().replace("ё", "е"))


def reaction_kind(text: str) -> Optional[str]:
    """One ASR word → "laugh" | "mat" | "react" | "shout" (repeated letters «аааа», «нееет») | None."""
    t = _norm(text)
    if not t:
        return None
    if _LAUGH.search(t):
        return "laugh"
    from .profanity import classify
    if classify(t) == "mat":
        return "mat"
    if t in _REACT_WORDS:
        return "react"
    if re.search(r"(\w)\1\1", t):                       # «аааа», «нееет», «дааа»
        return "shout"
    return None


def speech_signals(words: Sequence[Dict[str, Any]], n_sec: int) -> Dict[str, Any]:
    """-> {"rate": z words/s (3 s window), "react": hits per second (weighted), "hits": [(t, kind, word)]}."""
    count = np.zeros(n_sec)
    react = np.zeros(n_sec)
    hits: List[Any] = []
    weights = {"laugh": 1.5, "mat": 0.8, "react": 0.7, "shout": 1.0, "phrase": 1.0, "excl": 0.4,
               "guest_lead": 0.6, "guest_in": 0.5, "reveal": 0.8, "joke": 1.2}
    seq = [w for w in words if w.get("start") is not None]
    for i, w in enumerate(seq):
        s = int(float(w["start"]))
        if not 0 <= s < n_sec:
            continue
        txt = str(w.get("word", w.get("text", "")))
        count[s] += 1
        kind = reaction_kind(txt)
        if kind:
            react[s] += weights[kind]
            hits.append((round(float(w["start"]), 2), kind, txt.strip()))
        if "!" in txt or "?!" in txt:
            react[s] += weights["excl"]
        if i + 1 < len(seq):
            pair = f"{_norm(txt)} {_norm(str(seq[i + 1].get('word', seq[i + 1].get('text', ''))))}"
            if _REACT_PHRASES.search(pair):
                react[s] += weights["phrase"]
                hits.append((round(float(w["start"]), 2), "phrase", pair))
        win2 = " ".join(_norm(str(x.get("word", x.get("text", "")))) for x in seq[i:i + 3])
        if _GUEST_LEAD.search(win2) and not any(h[1] == "guest_lead" and float(w["start"]) - h[0] < 8 for h in hits):
            react[s] += weights["guest_lead"]
            hits.append((round(float(w["start"]), 2), "guest_lead", win2))
        if _GUEST_IN.match(_norm(txt).replace("-", "")):
            react[s] += weights["guest_in"]
            hits.append((round(float(w["start"]), 2), "guest_in", txt.strip()))
        if _REVEAL.search(win2) and not any(h[1] == "reveal" and float(w["start"]) - h[0] < 5 for h in hits):
            react[s] += weights["reveal"]
            hits.append((round(float(w["start"]), 2), "reveal", win2))
    # a crude joke: mat / a vulgar line followed by laughter within 4 s
    mats = [h for h in hits if h[1] == "mat"]
    for t, kind, txt in [h for h in hits if h[1] == "laugh"]:
        if any(0.0 <= t - m[0] <= 4.0 for m in mats) and not any(h[1] == "joke" and abs(h[0] - t) < 6 for h in hits):
            s = int(t)
            if 0 <= s < n_sec:
                react[s] += weights["joke"]
            hits.append((t, "joke", txt))
    k = np.ones(3) / 3.0
    rate = np.convolve(count, k, mode="same")
    return {"rate": robust_z(rate, rate > 0.3), "react": [round(float(v), 2) for v in react], "hits": hits}


# ── visual (game events) ────────────────────────────────────────────────────

VIS_FPS = 4
VIS_W, VIS_H = 64, 36


def visual_signals(media: str, ffmpeg: str = "ffmpeg", cache_path: Optional[str] = None,
                   hwaccel: Optional[str] = None) -> Dict[str, List[float]]:
    """-> {"cut": [0/1 hard cut in that second], "flash": z of the biggest brightness jump, "motion": z}.
    One decode of the whole source at 4 fps, 64x36 gray (the decoder still reads every frame; on a GPU box
    pass hwaccel="cuda", on a Mac "videotoolbox")."""
    c = _load_cache(cache_path, "visual")
    if c:
        return {k: c[k] for k in ("cut", "flash", "motion")}
    cmd = [ffmpeg, "-v", "error", "-nostdin"]
    if hwaccel:
        cmd += ["-hwaccel", hwaccel]
    # B-frames nobody references are not decoded (−30 % CPU, measured on a 1080p60 VOD); 4 fps sampling
    # does not need them
    cmd += ["-skip_frame:v", "noref", "-i", media, "-an", "-vf", f"fps={VIS_FPS},scale={VIS_W}:{VIS_H}:flags=area,format=gray",
            "-f", "rawvideo", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    assert p.stdout is not None
    raw = p.stdout.read()
    p.wait()
    fsz = VIS_W * VIS_H
    n = len(raw) // fsz
    if n < 2:
        raise RuntimeError("video decode failed")
    f = np.frombuffer(raw[: n * fsz], dtype=np.uint8).reshape(n, VIS_H, VIS_W).astype(np.float32)
    luma = f.mean(axis=(1, 2))
    diff = np.abs(f[1:] - f[:-1]).mean(axis=(1, 2))
    diff = np.concatenate([[0.0], diff])
    jump = np.concatenate([[0.0], luma[1:] - luma[:-1]])
    n_sec = int(math.ceil(n / VIS_FPS))
    pad = n_sec * VIS_FPS - n
    d = np.pad(diff, (0, pad)).reshape(n_sec, VIS_FPS)
    j = np.pad(jump, (0, pad)).reshape(n_sec, VIS_FPS)
    med = float(np.median(diff)) or 1.0
    cut = (d.max(axis=1) > max(25.0, 6.0 * med)).astype(float)
    flash = j.max(axis=1)                                # brightening only (explosion / flash / kill effect)
    out = {"cut": [float(v) for v in cut],
           "flash": robust_z(flash, flash > 0.5),
           "motion": robust_z(d.mean(axis=1))}
    _save_cache(cache_path, "visual", out)
    return out


# ── chat ────────────────────────────────────────────────────────────────────

def chat_times(path: str) -> List[float]:
    """Message offsets (s) from a chat replay file:
      * yt-dlp YouTube live chat (``*.live_chat.json``, one JSON object per line, replayChatItemAction);
      * Twitch VOD chat (TwitchDownloader / rechat JSON: {"comments": [{"content_offset_seconds": ..}]});
      * a plain JSON list of seconds or of {"t": seconds}."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    out: List[float] = []
    try:
        data = json.loads(text)
    except ValueError:
        data = None
    if isinstance(data, dict) and isinstance(data.get("comments"), list):
        for c in data["comments"]:
            v = c.get("content_offset_seconds") if isinstance(c, dict) else None
            if isinstance(v, (int, float)):
                out.append(float(v))
        return sorted(out)
    if isinstance(data, list):
        for v in data:
            v = v.get("t") if isinstance(v, dict) else v
            if isinstance(v, (int, float)):
                out.append(float(v))
        return sorted(out)
    for line in text.splitlines():                     # yt-dlp live_chat: JSON lines
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        act = (obj.get("replayChatItemAction") or {})
        off = act.get("videoOffsetTimeMsec")
        items = [a.get("addChatItemAction", {}).get("item", {}) for a in act.get("actions") or []]
        if off is None or not any("liveChatTextMessageRenderer" in it or "liveChatPaidMessageRenderer" in it
                                  for it in items):
            continue
        try:
            out.append(float(off) / 1000.0)
        except (TypeError, ValueError):
            continue
    return sorted(out)


def chat_signals(times: Iterable[float], n_sec: int, lag_sec: float = 6.0) -> Dict[str, Any]:
    """Messages per second (5 s window), shifted back by the viewers' reaction lag, z-scored.
    -> {"chat": z, "messages": total} ; all zeros when there are too few messages to mean anything."""
    t = [x - lag_sec for x in times]
    cnt = np.zeros(n_sec)
    for x in t:
        i = int(x)
        if 0 <= i < n_sec:
            cnt[i] += 1
    total = int(cnt.sum())
    if total < max(50, n_sec // 60):                    # < ~1 message a minute: no signal
        return {"chat": [0.0] * n_sec, "messages": total}
    rate = np.convolve(cnt, np.ones(5) / 5.0, mode="same")
    return {"chat": robust_z(rate), "messages": total}
