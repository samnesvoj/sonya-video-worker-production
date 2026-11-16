"""
scripts.shared.asr - the ASR entry point of the worker (contract: scripts/shared/asr_contract.py, docs/ASR_INTERFACE.md).

    transcribe(audio_path, language="ru") -> {"text", "language", "backend", "model", "segments", "words"}

Owner decision 2026-11-07 (docs/OWNER_DECISIONS.md): GigaAM v3 (`v3_e2e_rnnt`) is the PRIMARY ASR for virality and beep,
on CUDA as on the Mac mini; faster-whisper is the fallback only.
  SONYA_ASR_BACKEND   gigaam (default) | faster_whisper_cuda | faster_whisper_cpu
  SONYA_ASR_FALLBACK  1 = if GigaAM cannot run (weights/runtime missing) use faster-whisper and mark the result
                      `fallback_from`; default 0 = raise (a silent quality drop is worse than a clear failure)
GigaAM is Russian-only: for another explicit language the whisper path is used (as in the Desktop backend order);
language None/"auto": a quick Whisper language check on three 20 s samples (SONYA_ASR_DETECT=off disables it) -
a clearly non-Russian video goes to Whisper in its own language; anything unsure stays SONYA_ASR_LANGUAGE (default "ru").
Production 2026-11-15: an English video went to GigaAM and got subtitles like «Into for differtent», «Beascantestent».
Nothing here downloads a model: GigaAM reads $MODELS_LOCAL_DIR/gigaam/, faster-whisper honours SONYA_ASR_OFFLINE=1.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Optional

from scripts.shared.asr_contract import backend_name

log = logging.getLogger(__name__)
GIGAAM_LANGS = ("ru",)


def _whisper(audio_path: str, language: Optional[str], backend: str) -> Dict[str, Any]:
    from scripts.shared.transcription.transcriber import Transcriber
    cuda = backend == "faster_whisper_cuda"
    t = Transcriber(device="cuda" if cuda else "cpu", compute_type="float16" if cuda else "int8", language=language)
    res = t.transcribe(audio_file=audio_path, language=language)
    model = os.environ.get("SONYA_WHISPER_MODEL", "").strip() or "base"
    return dict(res, backend=backend, model=f"faster-whisper-{model}")


SITE_LANGS = {"ru": "ru", "en": "en", "none": "auto", "auto": "auto", "": "auto"}
DETECT_MIN_PROB = 0.7          # below this a non-Russian guess is not trusted (Russian is the default audience)


def job_language(params: Optional[Dict[str, Any]]) -> str:
    """The language the user picked: params.language (API/Desktop) or the site's chip settings.translation
    (Авто=none / ru / en). -> "ru" | "en" | ... | "auto"."""
    p = params or {}
    raw = p.get("language")
    if not raw:
        settings = p.get("settings") if isinstance(p.get("settings"), dict) else {}
        raw = settings.get("translation")
    raw = str(raw or "").strip().lower()
    return SITE_LANGS.get(raw, raw if 2 <= len(raw) <= 3 and raw.isalpha() else "auto")


def _samples(audio_path: str, seconds: float = 20.0, count: int = 3):
    """16 kHz mono float32 audio of `count` spots (20/50/80 % of the file) or None."""
    import subprocess
    import numpy as np
    ff = os.environ.get("SONYA_FFMPEG") or "ffmpeg"
    try:
        dur = float(subprocess.run([os.environ.get("SONYA_FFPROBE") or "ffprobe", "-v", "error", "-show_entries",
                                    "format=duration", "-of", "csv=p=0", audio_path],
                                   capture_output=True, text=True, timeout=60).stdout.strip() or 0)
    except Exception:  # noqa: BLE001
        dur = 0.0
    starts = [max(0.0, dur * f - seconds / 2) for f in (0.2, 0.5, 0.8)][:count] if dur > seconds * 2 else [0.0]
    out = []
    for st in starts:
        r = subprocess.run([ff, "-v", "error", "-ss", f"{st:.2f}", "-t", f"{seconds:.2f}", "-i", audio_path,
                            "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "-"], capture_output=True, timeout=120)
        if r.returncode == 0 and r.stdout:
            out.append(np.frombuffer(r.stdout, dtype=np.float32))
    return out or None


def detect_language(audio_path: str) -> Optional[str]:
    """Whisper's language guess (majority of 3 samples, each >= DETECT_MIN_PROB) or None when unsure/unavailable."""
    if os.environ.get("SONYA_ASR_DETECT", "on").strip().lower() in ("off", "0", "false", "no"):
        return None
    try:
        from scripts.shared.transcription.transcriber import Transcriber
        cuda = _cuda()
        model = Transcriber(device="cuda" if cuda else "cpu", compute_type="float16" if cuda else "int8")._model
        samples = _samples(audio_path) if model is not None else None
        if not samples:
            return None
        votes: Dict[str, int] = {}
        for a in samples:
            _segs, info = model.transcribe(a, language=None, beam_size=1, vad_filter=True, without_timestamps=True)
            if info.language and (info.language_probability or 0) >= DETECT_MIN_PROB:
                votes[info.language] = votes.get(info.language, 0) + 1
        if not votes:
            return None
        lang, n = max(votes.items(), key=lambda kv: kv[1])
        log.info("[asr] language check: %s", votes)
        return lang if n * 2 > len(samples) else None
    except Exception as e:  # noqa: BLE001 - the check must never fail the job; default language then
        log.warning("[asr] language check failed (%s) - using the default", type(e).__name__)
        return None


def resolve_language(audio_path: str, language: Optional[str]) -> str:
    lang = (language or "auto").strip().lower()
    if lang in ("auto", "none", ""):
        default = os.environ.get("SONYA_ASR_LANGUAGE", "ru").strip().lower() or "ru"
        lang = detect_language(audio_path) or default
    return lang


def transcribe(audio_path: str, language: Optional[str] = "ru", backend: Optional[str] = None,
               progress: Optional[Callable[[float], None]] = None) -> Dict[str, Any]:
    name = backend or backend_name()
    lang = resolve_language(audio_path, language)
    if name == "gigaam" and lang in GIGAAM_LANGS:
        from scripts.shared.asr import gigaam_backend as g
        try:
            return g.transcribe(audio_path, progress=progress)
        except g.GigaAMUnavailable as e:
            if os.environ.get("SONYA_ASR_FALLBACK", "0") != "1":
                raise
            log.warning("GigaAM unavailable (%s) - falling back to faster-whisper (SONYA_ASR_FALLBACK=1)", e)
            res = _whisper(audio_path, lang, "faster_whisper_cuda" if _cuda() else "faster_whisper_cpu")
            return dict(res, fallback_from="gigaam", fallback_reason=str(e))
    if name == "gigaam":                       # non-Russian request: GigaAM cannot do it
        name = "faster_whisper_cuda" if _cuda() else "faster_whisper_cpu"
    return _whisper(audio_path, lang, name)


def _cuda() -> bool:
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:  # noqa: BLE001
        return False
