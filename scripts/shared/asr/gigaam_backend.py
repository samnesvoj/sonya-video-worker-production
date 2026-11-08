"""
gigaam_backend.py - GigaAM v3 e2e RNNT (Russian) on CUDA / MPS / CPU, offline only.

Ported from the proven Mac mini backend (sonya-desktop `desktop/sonya_desktop/asr/backends.py`,
`GigaAmBackend` + `speech_chunks` + `_load_16k` + `merge_punct_tokens` + `fix_rnnt_word_times`).
The chunking, batching, decoding and post-processing are the same code; the differences are:
  1. weights come ONLY from `<models root>/gigaam/` (our S3, sha256-verified by the worker's
     model fetch); a missing file is an error - the upstream downloader is never reached;
  2. a word emitted in the SAME 40 ms RNNT frame as the previous one (zero length after the
     reference post-processing, silently dropped by Cut) gets the last 40 ms of the previous word
     (`split_zero_length_words`). On the 1 h 51 min reference transcript this touches 2 of 18 959
     words; everything else is identical to the reference. Post-processing tag: ts4.
Owner decision (docs/OWNER_DECISIONS.md): GigaAM is the primary ASR of SONYA; speed matters.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

MODEL_ID = "v3_e2e_rnnt"                 # e2e = punctuation + normalisation (sentence boundaries for Cut)
POSTPROC_TAG = "ts4"                     # ts3 (reference) + same-frame split; part of any ASR cache key
MODEL_NAME = f"gigaam-{MODEL_ID}-{POSTPROC_TAG}"
FILES = (f"{MODEL_ID}.ckpt", f"{MODEL_ID}_tokenizer.model")
SAMPLE_RATE = 16000
BATCH = 8
START_LEAD = 0.15     # RNNT emits a word ~160 ms after it starts (median vs whisper, 20k words; CUT_PLAYBOOK)
MIN_WORD = 0.12       # near-zero-length RNNT words get a real extent (the boundary engine needs it)

_VENDOR = Path(__file__).resolve().parents[2] / "third_party"     # scripts/third_party/gigaam (pinned upstream commit)
_cache: Dict[str, Any] = {}


class GigaAMUnavailable(RuntimeError):
    """Weights missing / wrong size or the runtime (torch, torchaudio...) is not importable."""


# ---------------------------------------------------------------- reference post-processing (unchanged)
def speech_chunks(x, sr: int, min_sec: float = 14.0, max_sec: float = 22.0):
    """[(start, end)] sample ranges <= max_sec, each cut at the quietest 30 ms frame
    between min_sec and max_sec (pauses), so no word is split by a chunk edge."""
    import numpy as np
    hop = max(1, int(0.03 * sr))
    if len(x) <= max_sec * sr:
        return [(0, len(x))]
    rms = np.sqrt(np.convolve(np.asarray(x, dtype=np.float32) ** 2, np.ones(hop) / hop, "same"))[::hop]
    cuts, pos = [0], 0
    while pos + max_sec * sr < len(x):
        a, b = int((pos + min_sec * sr) // hop), int((pos + max_sec * sr) // hop)
        pos = (a + int(np.argmin(rms[a:b]))) * hop
        cuts.append(pos)
    cuts.append(len(x))
    return list(zip(cuts, cuts[1:]))


def merge_punct_tokens(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """GigaAM emits dialogue dashes / stray quotes as words of their own («прикол. —»).
    Glue them to the next word (an opening dash/quote) or the previous one (closing)."""
    out: List[Dict[str, Any]] = []
    pending = ""
    for w in words:
        t = w["word"].strip()
        if not any(ch.isalnum() for ch in t):
            if t in ("—", "–", "-", "«", "„", "("):
                pending += t + (" " if t in ("—", "–", "-") else "")
            elif out:
                out[-1] = dict(out[-1], word=out[-1]["word"] + t)
            continue
        if pending:
            w = dict(w, word=" " + pending + t)
            pending = ""
        out.append(w)
    return out


def fix_rnnt_word_times(words: List[Dict[str, Any]], lead: float = START_LEAD, min_word: float = MIN_WORD) -> None:
    """In place: move each word start earlier by the measured emission lag (never before
    the previous word's end), and give near-zero-length words a minimal duration."""
    prev_end = 0.0
    for i, w in enumerate(words):
        s, e = float(w["start"]), float(w["end"])
        s = max(prev_end, s - lead)
        nxt = float(words[i + 1]["start"]) - lead if i + 1 < len(words) else e + min_word
        if e - s < min_word:
            e = max(e, min(s + min_word, nxt - 0.01))
        w["start"], w["end"] = round(s, 3), round(max(e, s), 3)
        prev_end = w["end"]


# ---------------------------------------------------------------- worker addition (ts4)
FRAME_SEC = 0.04      # one GigaAM encoder frame


def split_zero_length_words(words: List[Dict[str, Any]]) -> int:
    """In place: a word left with end <= start (it shared one RNNT frame with the previous word)
    takes the LAST frame (40 ms, at most half) of the previous word's span, so Cut keeps it.
    The previous word loses at most 40 ms at its end (BEEP pads intervals by 50-80 ms anyway).
    Returns how many words were fixed."""
    n = 0
    for i, w in enumerate(words):
        if float(w["end"]) > float(w["start"]):
            continue
        p = words[i - 1] if i else None
        span = float(p["end"]) - float(p["start"]) if p else 0.0
        if p and span >= 0.02:
            cut = round(float(p["end"]) - min(FRAME_SEC, span / 2), 3)
            w["start"], w["end"], p["end"] = cut, round(max(float(w["end"]), float(p["end"])), 3), cut
        else:
            w["end"] = round(float(w["start"]) + 0.01, 3)
        n += 1
    return n


# ---------------------------------------------------------------- runtime
def weights_dir(models_root: Optional[Path] = None) -> Path:
    if models_root is None:
        from scripts.shared.model_paths import models_root as _root
        models_root = _root()
    return Path(models_root) / "gigaam"


def check_weights(models_root: Optional[Path] = None) -> Path:
    """-> the gigaam/ dir; raises if a file is missing or empty (never downloads)."""
    d = weights_dir(models_root)
    missing = [f for f in FILES if not (d / f).is_file() or (d / f).stat().st_size == 0]
    if missing:
        raise GigaAMUnavailable(f"GigaAM weights missing in {d}: {missing} (they come from our S3 via the model fetch; "
                                f"nothing is downloaded at run time)")
    return d


def _import_gigaam():
    if _VENDOR.is_dir() and str(_VENDOR) not in sys.path:
        sys.path.insert(0, str(_VENDOR))          # the pinned upstream copy wins over any installed one
    try:
        import gigaam  # noqa: F401
        from gigaam.utils import AudioDataset  # noqa: F401
    except Exception as e:  # torch / torchaudio / hydra missing or ABI mismatch
        raise GigaAMUnavailable(f"cannot import gigaam: {e}") from e
    return gigaam


def device_name() -> str:
    forced = os.environ.get("SONYA_ASR_DEVICE", "").strip()
    if forced:
        return forced
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"


def load_model(models_root: Optional[Path] = None, device: Optional[str] = None):
    d = check_weights(models_root)
    dev = device or device_name()
    key = f"{d}|{dev}"
    if key not in _cache:
        gigaam = _import_gigaam()
        # download_root points at our verified files; upstream returns early when they exist (no network).
        _cache.clear()
        _cache[key] = gigaam.load_model(MODEL_ID, device=dev, fp16_encoder=(dev != "cpu"), use_flash=False,
                                        download_root=str(d))
    return _cache[key]


def load_16k(path: str):
    """Mono float32 @ 16 kHz via ffmpeg (any container)."""
    import numpy as np
    ff = os.environ.get("SONYA_FFMPEG") or shutil.which("ffmpeg") or "ffmpeg"
    raw = subprocess.run([ff, "-nostdin", "-loglevel", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32)


def transcribe(audio_path: str, models_root: Optional[Path] = None, device: Optional[str] = None,
               progress: Optional[Callable[[float], None]] = None, batch: int = BATCH) -> Dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader
    model = load_model(models_root, device)
    from gigaam.utils import AudioDataset
    x = load_16k(audio_path)
    spans = speech_chunks(x, SAMPLE_RATE)
    segs = [torch.from_numpy(x[a:b].copy()) for a, b in spans]
    dl = DataLoader(AudioDataset(segs, tokenizer=None), batch_size=batch, shuffle=False, collate_fn=AudioDataset.collate)
    out: List[Dict[str, Any]] = []
    idx = 0
    with torch.inference_mode():
        for wav_pad, wav_lens in dl:
            wav_pad = wav_pad.to(model._device).to(model._dtype)
            wav_lens = wav_lens.to(model._device)
            enc, enc_len = model.forward(wav_pad, wav_lens)
            for text, words in model._decode(enc, enc_len, wav_lens, True):
                a, b = spans[idx]
                idx += 1
                t0, t1 = a / SAMPLE_RATE, b / SAMPLE_RATE
                if not text.strip():
                    continue
                out.append({"start": round(t0, 3), "end": round(t1, 3), "text": text,
                            "words": [{"word": " " + w.text, "start": round(t0 + w.start, 3), "end": round(t0 + w.end, 3)}
                                      for w in (words or [])]})
            if progress:
                progress(idx / len(spans))
    for seg in out:
        seg["words"] = merge_punct_tokens(seg["words"])
    flat = [w for seg in out for w in seg["words"]]
    fix_rnnt_word_times(flat)
    split_zero_length_words(flat)
    return {"text": " ".join(s["text"].strip() for s in out).strip(), "language": "ru", "backend": "gigaam",
            "model": MODEL_NAME, "segments": out, "words": flat,
            "device": str(model._device), "audio_sec": round(len(x) / SAMPLE_RATE, 3)}
