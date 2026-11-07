"""
modes/streamer/runner.py
========================
Streamer mode (beta).

yuvelirochka integration:
  - modes/streamer/legacy/analyzer.py  (yuvelirochka analyzer)
  - modes/streamer/legacy/clipper.py   (yuvelirochka clipper)
  - scripts/shared/crop/cropper.py     (SmartCropper face-following)
  - scripts/shared/vision/webcam_detector.py (person bounding boxes)
  - scripts/shared/speaker/lip_sync_detector.py (active speaker)

webcam_detector is required (optional=false in mode.yaml).
If model not found → error logged, pipeline runs in degraded mode.

PHASE A split (long-form / two-phase architecture prep):
  analyze(...)      — the one expensive pass over the whole input video
                       (enrich_video_for_mode + yuvelirochka analyzer/
                       clipper, or the fallback heuristic). Returns ALL
                       candidate segments, unfiltered — does not pick a
                       top-N subset and does not compose anything.
  compose_one(...)  — cheap, repeatable: composes exactly ONE clip from an
                       already-known segment. Never calls
                       enrich_video_for_mode() or the analyzer/clipper.
  run(...)          — unchanged production entry point / compatibility
                       wrapper: analyze() once, pick top max_clips by score
                       exactly as before, compose_one() each selected
                       segment. No human checkpoint here yet — a future
                       batch API would call analyze()/compose_one()
                       directly instead of going through run().

Segment contract (canonical, internal to this module and its callers):
  {"start_sec": float, "duration_sec": float, "score": float, "source": str}
  Normalized exactly once, at the point analyze() gets raw_segments back
  from the analyzer/clipper or the fallback — nothing past that boundary
  (compose_one, run(), a future batch caller) should ever see
  "start"/"end"/"start_time"/"offset" again. compose_vertical_clip()'s own
  start_time/duration parameter names are the one place this contract is
  translated back, inside compose_one().
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))


def analyze(
    input_video_path: str,
    output_dir: str,
    params: Optional[Dict[str, Any]] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    One-time, expensive pass over the whole input video. Runs
    enrich_video_for_mode() (transcription, webcam/person detection,
    active-speaker detection, etc.) exactly once, then the yuvelirochka
    analyzer/clipper (or the fallback heuristic) over that enrichment to
    produce candidate segments.

    Does NOT select a top-N subset and does NOT compose any clips — see
    compose_one() for that. This is the single expensive step a future
    two-phase batch job would run exactly once per source video, however
    many clips end up getting generated from it afterwards.

    Returns:
        {
            "segments": [{"start_sec", "duration_sec", "score", "source"}, ...],
                # ALL candidates, unranked/unfiltered by this function
            "crop_hints": dict,        # pass through to compose_one()
            "warnings": [str, ...],
            "webcam_boxes_found": int,
            "active_speaker_segs": int,
        }
    """
    if params is None:
        params = {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from scripts.shared.enhancers.sonya_enhancer import enrich_video_for_mode
    enrichment = enrich_video_for_mode(
        input_video_path=input_video_path,
        output_dir=str(output_dir),
        mode="streamer",
        params=params,
        progress_callback=progress_callback,
    )

    # ── yuvelirochka enrichment data ──────────────────────────────────────────
    webcam_layout          = enrichment.get("webcam_layout") or {}
    active_speaker_segs    = enrichment.get("active_speaker_segments") or []
    word_timestamps        = enrichment.get("word_timestamps") or []
    crop_hints              = enrichment.get("crop_hints") or {}
    warnings                = list(enrichment.get("warnings") or [])

    webcam_boxes = webcam_layout.get("boxes", [])

    logger.info(
        "[streamer] webcam_boxes=%d active_speaker_segs=%d words=%d",
        len(webcam_boxes), len(active_speaker_segs), len(word_timestamps),
    )

    # Build transcript dict for yuvelirochka analyzer
    transcript = {
        "words": word_timestamps,
        "text": " ".join(w.get("word", "") for w in word_timestamps),
        "segments": [],
    }

    # ── yuvelirochka legacy analyzer + clipper ────────────────────────────────
    try:
        from modes.streamer.legacy.analyzer import analyze as _legacy_analyze
        from modes.streamer.legacy.clipper import clip_segments

        # analyzer.py from yuvelirochka — accepts enriched webcam/lip-sync data
        analysis = _legacy_analyze(
            input_video_path,
            webcam_boxes=webcam_boxes,
            lip_sync=active_speaker_segs,
            transcript=transcript,
        )
        raw_segments = clip_segments(analysis, params=params)
        logger.info("[streamer] yuvelirochka analyzer: %d segments", len(raw_segments))
    except Exception as exc:
        logger.warning("[streamer] yuvelirochka analyzer/clipper unavailable (%s) — fallback", exc)
        warnings.append(f"streamer_analyzer_unavailable: {exc}")
        raw_segments = _fallback_segments(active_speaker_segs, webcam_boxes, word_timestamps)

    # Normalize to the canonical start_sec/duration_sec contract exactly
    # once, right here at the analyzer/fallback boundary (see module
    # docstring). raw_segments' own "start"/"duration"/"score" shape is
    # yuvelirochka's (or _fallback_segments'), left untouched at the source.
    segments = [
        {
            "start_sec": float(seg.get("start", 0)),
            "duration_sec": float(seg.get("duration", 30.0)),
            "score": float(seg.get("score", 0)),
            "source": seg.get("source", "analyzer"),
        }
        for seg in raw_segments
    ]

    return {
        "segments": segments,
        "crop_hints": crop_hints,
        "warnings": warnings,
        "webcam_boxes_found": len(webcam_boxes),
        "active_speaker_segs": len(active_speaker_segs),
    }


def compose_one(
    input_video_path: str,
    output_path: str,
    segment: Dict[str, Any],
    crop_hints: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Compose exactly ONE clip from an already-known segment.

    Deliberately does NOT call enrich_video_for_mode() or the analyzer/
    clipper — this is the cheap, repeatable half of the pipeline a future
    per-clip job would run, reusing analyze()'s crop_hints instead of
    recomputing them. Raises on failure (same as compose_vertical_clip) —
    callers decide how to handle a single failed clip; see run() below for
    the existing per-clip try/except/warn pattern.

    `segment` must carry the canonical start_sec/duration_sec contract
    (see analyze() / module docstring). The one and only translation to
    compose_vertical_clip()'s own start_time/duration parameter names
    happens right here.
    """
    from scripts.shared.crop.smart_crop_adapter import compose_vertical_clip

    return compose_vertical_clip(
        input_video_path=input_video_path,
        output_path=output_path,
        crop_hints=crop_hints,
        start_time=float(segment["start_sec"]),
        duration=float(segment["duration_sec"]),
    )


def run(
    input_video_path: str,
    output_dir: str,
    params: Optional[Dict[str, Any]] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Current production entry point — compatibility wrapper over
    analyze()/compose_one(). Behavior is unchanged from before the split:
    analyze() once, pick the top params["max_clips"] (default 5) segments
    by score exactly as before, compose_one() each. No human checkpoint
    here yet — see module docstring.
    """
    if params is None:
        params = {}

    result = analyze(
        input_video_path=input_video_path,
        output_dir=output_dir,
        params=params,
        progress_callback=progress_callback,
    )
    segments   = result["segments"]
    crop_hints = result["crop_hints"]
    warnings   = list(result["warnings"])

    # ── Select and compose with SmartCropper ─────────────────────────────────
    max_clips = params.get("max_clips", 5)
    top = sorted(segments, key=lambda s: float(s.get("score", 0)), reverse=True)[:max_clips]

    output_dir_path = Path(output_dir)
    output_paths = []
    for i, seg in enumerate(top):
        out = str(output_dir_path / f"stream_clip_{i+1:02d}.mp4")
        try:
            compose_one(input_video_path, out, seg, crop_hints=crop_hints)
            output_paths.append(out)
        except Exception as exc:
            logger.warning("[streamer] Clip %d failed: %s", i + 1, exc)
            warnings.append(f"clip_{i+1}_failed: {exc}")

    from scripts.shared.mode_output import write_selected_segments
    names = [str(output_dir_path / f"stream_clip_{i+1:02d}.mp4") for i in range(len(top))]
    write_selected_segments(output_dir, "streamer", top,
                            [n if n in output_paths else None for n in names])

    return {
        "clips": output_paths,
        "mode": "streamer",
        "beta": True,
        "webcam_boxes_found": result["webcam_boxes_found"],
        "active_speaker_segs": result["active_speaker_segs"],
        "warnings": warnings,
    }


def _fallback_segments(
    active_speaker_segs: List[Dict],
    webcam_boxes: List[Dict],
    word_timestamps: List[Dict],
) -> List[Dict]:
    """Fallback segment selection from available enrichment."""
    segments = []

    # Use active speaker segments as clips
    for sp in active_speaker_segs:
        sp_start = float(sp.get("start", 0))
        sp_end   = float(sp.get("end", sp_start + 30))
        if sp_end - sp_start >= 5:
            segments.append({
                "start":    sp_start,
                "duration": sp_end - sp_start,
                "score":    2.0 + len([
                    b for b in webcam_boxes
                    if abs(float(b.get("frame", 0)) / 30 - sp_start) < 5
                ]) * 0.1,
                "source": "speaker_fallback",
            })

    if not segments:
        segments.append({"start": 0, "duration": 30.0, "score": 1.0, "source": "default"})

    return segments
