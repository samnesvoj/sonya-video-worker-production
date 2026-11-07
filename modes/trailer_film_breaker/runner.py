"""
modes/trailer_film_breaker/runner.py
=====================================
Production trailer mode runner.

Enrichment via sonya_enhancer:
  - GeminiVideoAnalyzer  → gemini_viral_moments (scene selection)
  - GeminiLayoutAnalyzer → layout_segments (crop strategy)
  - SmartCropper/GeminiComposer → compose_vertical_clip (smart crop)
  - Transcriber          → word_timestamps (subtitle alignment)
  - YOLO detections      → yolo_detections, pose_detections (hero/frame)

Legacy GPU scripts NOT used here.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def run(
    input_video_path: str,
    output_dir: str,
    params: Optional[Dict[str, Any]] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    """Entry point for trailer_film_breaker mode."""
    if params is None:
        params = {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[trailer_film_breaker] Enriching: %s", input_video_path)

    from scripts.shared.enhancers.sonya_enhancer import enrich_video_for_mode
    enrichment = enrich_video_for_mode(
        input_video_path=input_video_path,
        output_dir=str(output_dir),
        mode="trailer_film_breaker",
        params=params,
        progress_callback=progress_callback,
    )

    # ── Extract enrichment data ───────────────────────────────────────────────
    gemini_viral_moments  = enrichment.get("gemini_viral_moments") or []
    layout_segments       = enrichment.get("layout_segments") or []
    crop_hints            = enrichment.get("crop_hints") or {}
    word_timestamps       = enrichment.get("word_timestamps") or []
    yolo_detections       = enrichment.get("yolo_detections") or []
    pose_detections       = enrichment.get("pose_detections") or []
    warnings              = enrichment.get("warnings") or []

    logger.info(
        "[trailer] gemini_moments=%d layout_segs=%d words=%d yolo_frames=%d pose_frames=%d",
        len(gemini_viral_moments), len(layout_segments),
        len(word_timestamps), len(yolo_detections), len(pose_detections),
    )

    # ── Score and rank segments ───────────────────────────────────────────────
    segments = _score_segments(
        word_timestamps=word_timestamps,
        gemini_viral_moments=gemini_viral_moments,
        yolo_detections=yolo_detections,
        pose_detections=pose_detections,
    )

    max_clips = params.get("max_clips", 5)
    clip_duration = params.get("clip_duration", 15.0)
    top = sorted(segments, key=lambda s: s.get("score", 0), reverse=True)[:max_clips]

    # ── Compose vertical clips ────────────────────────────────────────────────
    from scripts.shared.crop.smart_crop_adapter import compose_vertical_clip
    output_paths = []
    for i, seg in enumerate(top):
        out = str(output_dir / f"trailer_clip_{i+1:02d}.mp4")
        try:
            compose_vertical_clip(
                input_video_path=input_video_path,
                output_path=out,
                crop_hints=crop_hints,
                start_time=seg.get("start", 0),
                duration=min(seg.get("duration", clip_duration), clip_duration),
            )
            output_paths.append(out)
            logger.info("[trailer] Clip %d/%d produced: %s", i + 1, len(top), out)
        except Exception as exc:
            logger.warning("[trailer] Clip %d failed: %s", i + 1, exc)
            warnings.append(f"clip_{i+1}_failed: {exc}")

    from scripts.shared.mode_output import write_selected_segments
    names = [str(output_dir / f"trailer_clip_{i+1:02d}.mp4") for i in range(len(top))]
    write_selected_segments(str(output_dir), "trailer_film_breaker", top,
                            [n if n in output_paths else None for n in names],
                            max_duration=clip_duration)

    return {
        "clips": output_paths,
        "mode": "trailer_film_breaker",
        "enrichment_keys": [k for k, v in enrichment.items() if v and k != "warnings"],
        "warnings": warnings,
        "subtitle_words": len(word_timestamps),
        "gemini_moments_used": len(gemini_viral_moments),
        "layout": layout_segments[0].get("layout", "unknown") if layout_segments else "unknown",
    }


def _score_segments(
    word_timestamps: List[Dict],
    gemini_viral_moments: List[Dict],
    yolo_detections: List[Dict],
    pose_detections: List[Dict],
) -> List[Dict]:
    """
    Score candidate segments using all enrichment layers.

    Scoring:
      - Gemini viral moment → base score from relevance field
      - Word density in window → speech activity score
      - YOLO frame coverage → visual action boost
      - Pose detection → human action boost
    """
    segments: List[Dict] = []

    # ── Gemini viral moments as primary candidates ────────────────────────────
    for moment in gemini_viral_moments:
        start = float(moment.get("start", moment.get("start_time", 0)))
        dur   = float(moment.get("duration", moment.get("clip_duration", 15.0)))
        relevance = float(moment.get("relevance", moment.get("score", 0.5)))
        seg = {
            "start":    start,
            "duration": dur,
            "score":    relevance * 10.0,
            "source":   "gemini",
        }
        # Word density boost in this window
        if word_timestamps:
            window_words = [
                w for w in word_timestamps
                if start <= float(w.get("start", 0)) < start + dur
            ]
            seg["score"] += len(window_words) * 0.1
        segments.append(seg)

    # ── Sliding window over word timestamps ──────────────────────────────────
    if word_timestamps:
        window = 15.0
        step   = 5.0
        t_start = float(word_timestamps[0].get("start", 0))
        t_end   = float(word_timestamps[-1].get("end", word_timestamps[-1].get("start", 0)))
        t = t_start
        while t + window <= t_end:
            window_words = [
                w for w in word_timestamps
                if t <= float(w.get("start", 0)) < t + window
            ]
            if not window_words:
                t += step
                continue
            score = len(window_words) * 0.4
            # Boost for YOLO detections in this window (proxy: if detections exist)
            if yolo_detections:
                score += 0.5
            if pose_detections:
                score += 0.3
            segments.append({"start": t, "duration": window, "score": score, "source": "words"})
            t += step

    # ── Ensure at least one fallback segment ─────────────────────────────────
    if not segments:
        segments.append({"start": 0, "duration": 15.0, "score": 1.0, "source": "fallback"})

    return segments
