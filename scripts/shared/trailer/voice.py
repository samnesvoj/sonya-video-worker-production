"""
Trailer narrator voice: local Qwen3-TTS (Apache-2.0) voice clone of a fixed SONYA narrator (male / female reference in
voices/, designed once with Qwen3-TTS VoiceDesign) → one WAV per segment, then:

  * edge silence trimmed, long inner pauses shortened (≤ 0.28 s) — the TikTok narrator never breathes long;
  * tempo fitted to the target pace (3.1–3.4 words/s, atempo ≤ 1.45 so the voice stays natural);
  * ONE GigaAM pass over all segments (1 s of silence between them) → word times; every segment is checked against its
    text (≥ 82 % of the words heard, the last word heard = no clipped words). A failed segment is synthesised again with
    another seed (at most 2 more times); still failing → the best take is kept and reported.

The TTS runs in its own process/venv (tts_child.py; ensure_runtime installs it once per host, like the Gen ComfyUI venv),
so the worker's own dependency lock is untouched.
"""
from __future__ import annotations

import difflib
import json
import logging
import os
import re
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

VOICE_VERSION = "trailer-voice-v1"
HERE = Path(__file__).resolve().parent
VOICES_DIR = HERE / "voices"
LOCK = HERE / "tts.lock.txt"
MODEL_NAME = os.environ.get("SONYA_TTS_MODEL", "Qwen3-TTS-12Hz-1.7B-Base")
PACE = (3.1, 3.4)                 # words per second of the voice (owner formula 2.9–3.4, energetic → the upper half)
MAX_TEMPO = 1.45
MIN_TEMPO = 0.92
MAX_PAUSE = 0.28
SR = 48000
WORD_RE = re.compile(r"[\w]+(?:[-'][\w]+)*", re.U)


class VoiceError(RuntimeError):
    code = "TRAILER_VOICE_FAILED"


def voices() -> Dict[str, Dict[str, str]]:
    out = {}
    for g in ("male", "female"):
        wav, txt = VOICES_DIR / f"{g}.flac", VOICES_DIR / f"{g}.txt"
        if wav.is_file() and txt.is_file():
            out[g] = {"wav": str(wav), "text": txt.read_text(encoding="utf-8").strip()}
    return out


# ── runtime (venv + model) ────────────────────────────────────────────────────────────────────────────────────────

def ensure_runtime(log: Callable[[str], None] = logger.info) -> Tuple[str, str]:
    """(python of the TTS venv, model dir). SONYA_TTS_PYTHON / SONYA_TTS_MODEL_DIR override (the Mac mini tests)."""
    py = os.environ.get("SONYA_TTS_PYTHON")
    if not py:
        root = Path(os.environ.get("SONYA_TTS_DIR", "/work/tts"))
        py = str(root / "venv" / "bin" / "python")
        mark = root / "venv" / ".sonya-ok"
        want = LOCK.read_text() if LOCK.is_file() else ""
        if not (mark.is_file() and mark.read_text() == want):
            import shutil
            import time
            t = time.time()
            shutil.rmtree(root / "venv", ignore_errors=True)
            root.mkdir(parents=True, exist_ok=True)
            subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(root / "venv")], check=True)
            r = subprocess.run([py, "-m", "pip", "install", "-q", "--disable-pip-version-check", "--no-deps",
                                "--require-hashes", "-r", str(LOCK)], capture_output=True, text=True, timeout=900)
            if r.returncode != 0:
                raise VoiceError(f"TTS venv install failed: {r.stderr[-400:]}")
            mark.write_text(want)
            log(f"TTS venv installed in {time.time() - t:.0f} s")
    mdir = os.environ.get("SONYA_TTS_MODEL_DIR")
    if not mdir:
        from scripts.model_downloader import download_models_for_mode
        if not download_models_for_mode(HERE / "models.yaml"):
            raise VoiceError("TTS model download failed")
        base = Path(os.environ.get("MODELS_LOCAL_DIR", "/opt/sonya/models"))
        mdir = str(base / "tts" / MODEL_NAME)
    if not Path(mdir, "config.json").is_file():
        raise VoiceError(f"TTS model missing in {mdir}")
    return py, mdir


# ── audio helpers ─────────────────────────────────────────────────────────────────────────────────────────────────

def wav_sec(path: str) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def read_mono(path: str):
    import numpy as np
    with wave.open(str(path), "rb") as w:
        sr, n, ch = w.getframerate(), w.getnframes(), w.getnchannels()
        x = np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def write_mono(path: str, x, sr: int) -> None:
    import numpy as np
    y = (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(y.tobytes())


def tighten(x, sr: int, max_pause: float = MAX_PAUSE, thr_db: float = -42.0, edge: float = 0.04):
    """Trim edge silence, shorten inner pauses longer than max_pause (keeps max_pause of each, cross-faded)."""
    import numpy as np
    hop = int(sr * 0.01)
    if x.size < hop * 4:
        return x
    n = x.size // hop
    rms = np.sqrt(np.mean(x[:n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    loud = 20 * np.log10(rms) > thr_db
    if not loud.any():
        return x
    first, last = int(np.argmax(loud)), int(n - 1 - np.argmax(loud[::-1]))
    keep: List[Tuple[int, int]] = []
    i = first
    while i <= last:
        if loud[i]:
            j = i
            while j <= last and loud[j]:
                j += 1
            keep.append((i, j))
            i = j
        else:
            j = i
            while j <= last and not loud[j]:
                j += 1
            gap = j - i
            mp = int(max_pause * 100)
            if gap <= mp:
                keep.append((i, j))
            else:
                keep.append((i, i + mp // 2))
                keep.append((j - (mp - mp // 2), j))
            i = j
    e = int(edge * 100)
    a0 = max(0, first - e) * hop
    b1 = min(x.size, (last + 1 + e) * hop)
    pieces = []
    fade = int(0.006 * sr)
    for k, (s, t) in enumerate(keep):
        s0 = a0 if k == 0 else s * hop
        t1 = b1 if k == len(keep) - 1 else t * hop
        seg = x[s0:t1].copy()
        if seg.size > 2 * fade and 0 < k:
            seg[:fade] *= np.linspace(0, 1, fade)
        if seg.size > 2 * fade and k < len(keep) - 1:
            seg[-fade:] *= np.linspace(1, 0, fade)
        pieces.append(seg)
    return np.concatenate(pieces) if pieces else x


def ffmpeg_tempo(src: str, dst: str, tempo: float, ffmpeg: str = "ffmpeg") -> None:
    af = f"aresample={SR},atempo={tempo:.4f}" if abs(tempo - 1.0) > 0.005 else f"aresample={SR}"
    r = subprocess.run([ffmpeg, "-y", "-v", "error", "-nostdin", "-i", src, "-af", af, "-ac", "1", "-c:a", "pcm_s16le", dst],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise VoiceError(f"tempo failed: {r.stderr[-200:]}")


# ── text ↔ heard words ────────────────────────────────────────────────────────────────────────────────────────────

def norm_word(w: str) -> str:
    return re.sub(r"[^\w]", "", w.lower().replace("ё", "е"))


def words_of(text: str) -> List[str]:
    return WORD_RE.findall(text or "")


def heard_ratio(text_words: Sequence[str], heard: Sequence[str]) -> Tuple[float, bool]:
    """(share of the text's words heard in order, last word heard). Fuzzy per word (ASR spelling)."""
    a = [norm_word(w) for w in text_words if norm_word(w)]
    b = [norm_word(w) for w in heard if norm_word(w)]
    if not a:
        return 1.0, True
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    hit = sum(bl.size for bl in sm.get_matching_blocks())
    # near misses (stems): a replaced word that shares its first 4 letters counts
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace":
            for x, y in zip(a[i1:i2], b[j1:j2]):
                if x[:4] == y[:4] or difflib.SequenceMatcher(a=x, b=y).ratio() >= 0.7:
                    hit += 1
    tail = b[-3:]
    last_ok = any(a[-1][:4] == y[:4] or difflib.SequenceMatcher(a=a[-1], b=y).ratio() >= 0.7 for y in tail)
    return min(1.0, hit / len(a)), last_ok


def align(text_words: Sequence[str], heard: Sequence[Dict[str, Any]], seg_start: float, seg_end: float) -> List[Tuple[str, float, float]]:
    """The TEXT's words (correct spelling) with times from the heard words: equal/replaced runs take the heard times,
    missing ones are interpolated between neighbours. -> [(word, start, end)] inside [seg_start, seg_end]."""
    a = [norm_word(w) for w in text_words]
    b = [norm_word(h["word"]) for h in heard]
    times: List[Optional[Tuple[float, float]]] = [None] * len(a)
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal" or (tag == "replace" and i2 - i1 == j2 - j1):
            for k in range(i2 - i1):
                times[i1 + k] = (heard[j1 + k]["start"], heard[j1 + k]["end"])
        elif tag == "replace":
            s, e = heard[j1]["start"], heard[j2 - 1]["end"]
            n = i2 - i1
            for k in range(n):
                times[i1 + k] = (s + (e - s) * k / n, s + (e - s) * (k + 1) / n)
    # interpolate the gaps
    out: List[Tuple[str, float, float]] = []
    k = 0
    while k < len(a):
        if times[k] is not None:
            out.append((text_words[k], *times[k]))
            k += 1
            continue
        j = k
        while j < len(a) and times[j] is None:
            j += 1
        lo = times[k - 1][1] if k > 0 and times[k - 1] else seg_start
        hi = times[j][0] if j < len(a) and times[j] else seg_end
        n = j - k
        for m in range(n):
            times[k + m] = (lo + (hi - lo) * m / n, lo + (hi - lo) * (m + 1) / n)
            out.append((text_words[k + m], *times[k + m]))
        k = j
    # monotonic, inside the segment
    res, prev = [], seg_start
    for w, s, e in out:
        s = min(max(s, prev), seg_end)
        e = min(max(e, s + 0.05), seg_end)
        res.append((w, round(s, 3), round(e, 3)))
        prev = s
    return res


# ── the stage ─────────────────────────────────────────────────────────────────────────────────────────────────────

def run_tts(py: str, model_dir: str, voice: Dict[str, str], items: List[Dict[str, Any]], log: Callable[[str], None],
            timeout: float = 1500.0) -> Dict[str, Any]:
    req = {"model": model_dir, "ref_wav": voice["wav"], "ref_text": voice["text"], "items": items,
           "batch": int(os.environ.get("SONYA_TTS_BATCH", "4")), "temperature": 0.8, "top_p": 0.9}
    if os.environ.get("SONYA_TTS_DEVICE"):
        req["device"] = os.environ["SONYA_TTS_DEVICE"]
    env = dict(os.environ, PYTHONPATH="", TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1")
    r = subprocess.run([py, str(HERE / "tts_child.py")], input=json.dumps(req), capture_output=True, text=True,
                       timeout=timeout, env=env)
    res: Dict[str, Any] = {"items": {}}
    for line in r.stdout.splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("done"):
            res.update(d)
        elif d.get("out"):
            res["items"][d["out"]] = d
    if r.returncode != 0 or not res.get("done"):
        raise VoiceError(f"TTS child failed (rc={r.returncode}): {r.stderr[-500:]}")
    return res


def synthesize(segments: Sequence[Dict[str, Any]], gender: str, work: str, *, ffmpeg: str = "ffmpeg",
               transcribe: Optional[Callable[[str], Dict[str, Any]]] = None, pace: Tuple[float, float] = PACE,
               log: Callable[[str], None] = logger.info) -> Dict[str, Any]:
    """segments [{"n", "text"}] → {"segments": {n: {"wav", "sec", "words": [(w, s, e)], "heard", "last_ok", "tempo"}},
    "tts": {...}}. WAVs: 48 kHz mono, tempo-fitted, word times relative to the WAV start."""
    vs = voices()
    voice = vs.get(gender) or vs.get("male")
    if not voice:
        raise VoiceError("no narrator voice reference in voices/")
    py, mdir = ensure_runtime(log)
    wd = Path(work)
    wd.mkdir(parents=True, exist_ok=True)
    if transcribe is None:
        from ..asr import transcribe as _t
        transcribe = lambda p: _t(p, "ru")      # noqa: E731
    todo = {s["n"]: s for s in segments}
    best: Dict[int, Dict[str, Any]] = {}
    stats = {"model": Path(mdir).name, "voice": gender if gender in vs else "male", "tries": 0, "load_s": 0.0, "gen_s": 0.0}
    for attempt in range(3):
        if not todo:
            break
        items = [{"text": s["text"], "out": str(wd / f"raw_{n}_{attempt}.wav"), "seed": 1000 * attempt + n} for n, s in todo.items()]
        res = run_tts(py, mdir, voice, items, log)
        stats["tries"] += 1
        stats["load_s"] += res.get("load_s", 0.0)
        stats["gen_s"] += res.get("gen_s", 0.0)
        stats["device"] = res.get("device")
        takes = {}
        for n, s in todo.items():
            raw = str(wd / f"raw_{n}_{attempt}.wav")
            if not (res["items"].get(raw) or {}).get("ok") or not Path(raw).is_file():
                continue
            takes[n] = fit(raw, s["text"], str(wd / f"seg_{n}_{attempt}.wav"), pace, ffmpeg)
        if not takes:
            continue
        heard = asr_many([(n, t["wav"]) for n, t in takes.items()], str(wd / f"asr_{attempt}.wav"), transcribe, ffmpeg)
        for n, t in takes.items():
            tw = words_of(todo[n]["text"])
            hw = heard.get(n, [])
            ratio, last_ok = heard_ratio(tw, [h["word"] for h in hw])
            t.update(heard=round(ratio, 3), last_ok=last_ok, words=align(tw, hw, 0.0, t["sec"]), attempt=attempt)
            good = ratio >= 0.82 and last_ok
            old = best.get(n)
            if old is None or (good, t["heard"]) > (old["heard"] >= 0.82 and old["last_ok"], old["heard"]):
                best[n] = t
        todo = {n: s for n, s in todo.items() if not (n in best and best[n]["heard"] >= 0.82 and best[n]["last_ok"])}
        if todo:
            log(f"voice: re-synthesise {sorted(todo)} (heard " +
                ", ".join(f"{n}:{best[n]['heard'] if n in best else '-'}" for n in sorted(todo)) + ")")
    missing = [s["n"] for s in segments if s["n"] not in best]
    if missing:
        raise VoiceError(f"no voice for segments {missing}")
    words = sum(len(words_of(s["text"])) for s in segments)
    sec = sum(best[s["n"]]["sec"] for s in segments)
    stats.update(words=words, voice_sec=round(sec, 2), pace=round(words / max(0.1, sec), 2),
                 weak=[n for n, t in best.items() if not (t["heard"] >= 0.82 and t["last_ok"])])
    log(f"voice: {len(best)} segments, {sec:.1f} s, {stats['pace']} words/s, tries {stats['tries']}, "
        f"model load {stats['load_s']:.0f} s, synth {stats['gen_s']:.0f} s, weak {stats['weak']}")
    return {"segments": best, "tts": stats, "version": VOICE_VERSION}


def fit(raw: str, text: str, out: str, pace: Tuple[float, float], ffmpeg: str = "ffmpeg") -> Dict[str, Any]:
    """Tighten pauses, then atempo so the segment lands inside the pace window (spoken words / s)."""
    import numpy as np
    x, sr = read_mono(raw)
    y = tighten(x, sr)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0:
        y = y * min(4.0, 0.89 / peak)                      # −1 dBFS peak; loudness is set in the final mix
    tight = out + ".tight.wav"
    write_mono(tight, y, sr)
    n = max(1, len(words_of(text)))
    sec0 = y.size / float(sr)
    rate = n / max(0.2, sec0)
    tempo = 1.0
    if rate < pace[0]:
        tempo = min(MAX_TEMPO, pace[0] / rate * 1.02)
    elif rate > pace[1] * 1.12:
        tempo = max(MIN_TEMPO, pace[1] / rate)
    ffmpeg_tempo(tight, out, tempo, ffmpeg)
    os.unlink(tight)
    sec = wav_sec(out)
    return {"wav": out, "sec": round(sec, 3), "tempo": round(tempo, 3), "raw_rate": round(rate, 2),
            "rate": round(n / max(0.2, sec), 2)}


def asr_many(takes: List[Tuple[int, str]], joined: str, transcribe: Callable[[str], Dict[str, Any]],
             ffmpeg: str = "ffmpeg", gap: float = 1.0) -> Dict[int, List[Dict[str, Any]]]:
    """All takes in one 16 kHz file with `gap` s of silence between → one ASR call → words per take (relative times)."""
    import numpy as np
    pieces, spans, t = [], [], 0.0
    sil = np.zeros(int(16000 * gap), dtype=np.float32)
    for n, wav in takes:
        r = subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-i", wav, "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
                           capture_output=True)
        x = np.frombuffer(r.stdout, dtype=np.float32)
        spans.append((n, t, t + x.size / 16000.0))
        pieces += [x, sil]
        t += x.size / 16000.0 + gap
    write_mono(joined, np.concatenate(pieces) if pieces else sil, 16000)
    words = (transcribe(joined) or {}).get("words") or []
    out: Dict[int, List[Dict[str, Any]]] = {n: [] for n, _, _ in spans}
    for w in words:
        mid = (float(w["start"]) + float(w["end"])) / 2
        for n, a, b in spans:
            if a - 0.3 <= mid <= b + 0.3:
                out[n].append({"word": str(w.get("word") or w.get("text") or ""), "start": round(max(0.0, float(w["start"]) - a), 3),
                               "end": round(max(0.0, float(w["end"]) - a), 3)})
                break
    return out
