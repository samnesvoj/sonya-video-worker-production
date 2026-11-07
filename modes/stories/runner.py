"""
modes/stories/runner.py
=======================
Stories mode: story_mode_v1 + llm_segment_analysis + yuvelirochka visual layer.

Visual layer from yuvelirochka:
  - gemini_viral_moments  → turning points in narrative
  - layout_segments       → scene type annotation
  - yolo_detections       → visual events
  - active_speaker_segs   → speaking scene detection
  - word_timestamps       → transcript-based narrative segmentation
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))


def run(
    input_video_path: str,
    output_dir: str,
    params: Optional[Dict[str, Any]] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    if params is None:
        params = {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from scripts.shared.enhancers.sonya_enhancer import enrich_video_for_mode
    enrichment = enrich_video_for_mode(
        input_video_path=input_video_path,
        output_dir=str(output_dir),
        mode="stories",
        params=params,
        progress_callback=progress_callback,
    )

    # ── Enrichment data ───────────────────────────────────────────────────────
    word_timestamps       = enrichment.get("word_timestamps") or []
    layout_segments       = enrichment.get("layout_segments") or []
    gemini_viral_moments  = enrichment.get("gemini_viral_moments") or []
    yolo_detections       = enrichment.get("yolo_detections") or []
    active_speaker_segs   = enrichment.get("active_speaker_segments") or []
    crop_hints            = enrichment.get("crop_hints") or {}

    # Build transcript dict compatible with story_mode_v1
    transcript = _build_transcript_dict(word_timestamps)

    logger.info(
        "[stories] words=%d layout_segs=%d gemini_moments=%d yolo_frames=%d speakers=%d",
        len(word_timestamps), len(layout_segments),
        len(gemini_viral_moments), len(yolo_detections), len(active_speaker_segs),
    )

    # ── story_mode_v1 + llm_segment_analysis ─────────────────────────────────
    try:
        from scripts.legacy_gpu.story_mode_v1 import StoryMode
        from scripts.legacy_gpu.llm_segment_analysis import analyze_segments

        story = StoryMode(params=params)
        raw_segments = story.extract(input_video_path, transcript=transcript)

        # Feed all visual layers to llm_segment_analysis
        enriched = analyze_segments(
            raw_segments,
            visual_events=yolo_detections,
            layout_segments=layout_segments,
            gemini_moments=gemini_viral_moments,
        )
        logger.info("[stories] story_mode extracted %d segments", len(enriched))
    except Exception as exc:
        logger.warning("[stories] story_mode_v1/llm_segment_analysis unavailable (%s) — fallback", exc)
        enrichment.setdefault("warnings", []).append(f"stories_engine_unavailable: {exc}")
        enriched = _fallback_segments(gemini_viral_moments, word_timestamps, active_speaker_segs)

    # ── Annotate segments with visual context ─────────────────────────────────
    enriched = _annotate_with_visual_context(
        enriched, layout_segments, gemini_viral_moments, active_speaker_segs
    )

    # ── Select and compose ────────────────────────────────────────────────────
    max_clips = params.get("max_clips", 5)
    top = sorted(enriched, key=lambda s: float(s.get("score", 0)), reverse=True)[:max_clips]

    from scripts.shared.crop.smart_crop_adapter import compose_vertical_clip
    output_paths = []
    for i, seg in enumerate(top):
        out = str(output_dir / f"story_clip_{i+1:02d}.mp4")
        try:
            compose_vertical_clip(
                input_video_path=input_video_path,
                output_path=out,
                crop_hints=crop_hints,
                start_time=float(seg.get("start", 0)),
                duration=float(seg.get("duration", 60.0)),
            )
            output_paths.append(out)
        except Exception as exc:
            logger.warning("[stories] Clip %d failed: %s", i + 1, exc)

    from scripts.shared.mode_output import write_selected_segments
    names = [str(output_dir / f"story_clip_{i+1:02d}.mp4") for i in range(len(top))]
    write_selected_segments(str(output_dir), "stories", top,
                            [n if n in output_paths else None for n in names])

    return {
        "clips": output_paths,
        "mode": "stories",
        "warnings": enrichment.get("warnings", []),
        "segments_analyzed": len(enriched),
    }


def _build_transcript_dict(word_timestamps: List[Dict]) -> Dict:
    """Convert word_timestamps list to transcript dict compatible with story_mode_v1."""
    if not word_timestamps:
        return {"text": "", "segments": [], "words": []}
    text = " ".join(w.get("word", "") for w in word_timestamps)
    return {"text": text, "words": word_timestamps, "segments": []}


def _fallback_segments(
    gemini_moments: List[Dict],
    word_timestamps: List[Dict],
    active_speaker_segs: List[Dict],
) -> List[Dict]:
    """Generate fallback segments from available enrichment when story_mode fails."""
    segments = []

    for m in gemini_moments:
        segments.append({
            "start":    float(m.get("start", m.get("start_time", 0))),
            "duration": float(m.get("duration", m.get("clip_duration", 60.0))),
            "score":    float(m.get("relevance", 0.6)) * 8,
            "source":   "gemini_fallback",
        })

    for sp in active_speaker_segs:
        sp_start = float(sp.get("start", 0))
        sp_end   = float(sp.get("end", sp_start + 60))
        segments.append({
            "start":    sp_start,
            "duration": sp_end - sp_start,
            "score":    3.0,
            "source":   "speaker_fallback",
        })

    if not segments:
        segments.append({"start": 0, "duration": 60.0, "score": 1.0, "source": "default"})

    return segments


def _annotate_with_visual_context(
    segments: List[Dict],
    layout_segments: List[Dict],
    gemini_moments: List[Dict],
    active_speaker_segs: List[Dict],
) -> List[Dict]:
    """Boost segment scores based on visual context from yuvelirochka."""
    for seg in segments:
        start = float(seg.get("start", 0))
        end   = start + float(seg.get("duration", 60))
        boost = 0.0

        # Gemini turning-point overlap → strong story signal
        for m in gemini_moments:
            m_start = float(m.get("start", m.get("start_time", 0)))
            m_end   = m_start + float(m.get("duration", m.get("clip_duration", 0)))
            if m_start < end and m_end > start:
                boost += 1.5

        # Layout scene type
        for ls in layout_segments:
            ls_start = float(ls.get("start", 0))
            ls_end   = float(ls.get("end", ls_start + 1))
            if ls_start < end and ls_end > start:
                lt = ls.get("layout", ls.get("type", ""))
                if lt in ("single_speaker", "talking_head", "person"):
                    boost += 0.5

        # Active speaker in segment → speaking scene
        for as_seg in active_speaker_segs:
            as_start = float(as_seg.get("start", 0))
            as_end   = float(as_seg.get("end", as_start))
            if as_start < end and as_end > start:
                boost += 0.3

        seg["score"] = float(seg.get("score", 0)) + boost

    return segments
