"""
SONYA BEEP runner: ASR words → profanity hits → censor intervals → mute/beep/custom
→ optional masked subtitles (srt / burned) → one output video + beep_report.json.

ASR is mandatory: without real word timestamps the run fails (BEEP FAILED) —
never a «clean» output that was not actually checked.
"""
from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


def _duration(path: str) -> float:
    import os
    ffprobe = os.path.join(os.path.dirname(os.environ["SONYA_FFMPEG"]), "ffprobe") \
        if os.environ.get("SONYA_FFMPEG") else "ffprobe"
    out = subprocess.run([ffprobe, "-v", "error", "-show_entries", "format=duration:stream=width,height",
                          "-of", "json", path], capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    v = next((s for s in d.get("streams", []) if s.get("width")), {})
    return float(d["format"]["duration"]), int(v.get("width") or 0), int(v.get("height") or 0)


def _run_asr(input_video_path: str, out_dir: Path, language: Optional[str],
             progress_callback: Optional[Callable[..., None]]) -> str:
    """No precomputed words (the worker path): run the SONYA ASR (GigaAM by default, scripts/shared/asr) and keep its
    words next to the output. Any ASR failure is a BEEP failure - never a «clean» video that was not checked."""
    from scripts.shared import asr
    try:
        res = asr.transcribe(input_video_path, language=language)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"BEEP_FAILED: ASR failed — nothing was checked ({e})") from e
    path = out_dir / "asr_words.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: res.get(k) for k in ("backend", "model", "language", "words")}, ensure_ascii=False),
                    encoding="utf-8")
    if progress_callback:
        progress_callback(0.2, f"asr: {res.get('backend')} {len(res.get('words') or [])} words")
    return str(path)


def run(input_video_path: str, output_dir: str, params: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[..., None]] = None) -> Dict[str, Any]:
    from scripts.shared import beep as bp
    from scripts.shared.profanity import DETECTOR_VERSION, detect, mask_word
    params = params or {}
    out_dir = Path(output_dir)
    wpath = params.get("word_timestamps_path")
    if not wpath:
        wpath = _run_asr(input_video_path, out_dir, params.get("language"), progress_callback)
    if not Path(wpath).is_file():
        raise RuntimeError("BEEP_FAILED: no ASR word timestamps — nothing was checked")
    words = json.loads(Path(wpath).read_text(encoding="utf-8")).get("words") or []
    if not words:
        raise RuntimeError("BEEP_FAILED: ASR returned no words")
    cfg = bp.BeepConfig(**(params.get("beep") or {}))
    duration, W, H = _duration(input_video_path)
    hits = detect(words, cfg.categories)
    intervals = bp.censor_intervals(hits, duration, cfg)
    if progress_callback:
        progress_callback(0.3, f"beep: {len(hits)} words")

    extras = []
    ass = None
    if cfg.subtitles != "off":
        from scripts.shared.subtitles import SubtitleStyle, build_cues, clip_words, to_ass, write_atomic
        bad = {h.index for h in hits}
        masked = [dict(w, word=(" " + mask_word(str(w.get("word") or ""))) if i in bad else w.get("word"))
                  for i, w in enumerate(words)]
        style = SubtitleStyle(size=max(24, round(H * 0.05)), margin_v=round(H * 0.07),
                              margin_lr=round(W * 0.08), max_chars_per_line=42, outline=3.0)
        cues = build_cues(clip_words(masked, 0.0, duration), style, duration)
        srt_path = out_dir / "beeped.srt"
        write_atomic(srt_path, bp.srt(cues))
        extras.append(srt_path.name)
        if cfg.subtitles == "burn":
            ass_path = out_dir / "beeped.ass"
            write_atomic(ass_path, to_ass(cues, style, play_res=(W, H)))
            ass = str(ass_path)
            extras.append(ass_path.name)

    out = out_dir / "beeped.mp4"
    bp.render(input_video_path, str(out), intervals, cfg, subtitle_ass=ass)
    report = {"beep_version": bp.BEEP_VERSION, "detector_version": DETECTOR_VERSION, "config": asdict(cfg),
              "words_checked": len(words), "hits": [dict(asdict(h)) for h in hits],
              "intervals": intervals, "censored_sec": round(sum(b - a for a, b in intervals), 3),
              "duration": duration, "video": "stream_copy" if ass is None else "reencoded_for_subtitles"}
    (out_dir / "beep_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    extras.append("beep_report.json")
    return {"clips": [str(out)], "mode": "beep", "warnings": [], "extras": extras,
            "selection": {"beep": {k: report[k] for k in ("words_checked", "censored_sec")} | {"hits": len(hits)}}}
