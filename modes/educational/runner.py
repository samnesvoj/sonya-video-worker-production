"""
modes/educational/runner.py
============================
Educational mode: educational_mode_v5 + yuvelirochka layout-aware processing.

Layout strategy from GeminiLayoutAnalyzer:
  screen_share (>50%)   → preserve_screen  (letterbox — do NOT aggressive crop)
  single_speaker (>50%) → person_focus     (SmartCropper face-center crop)
  other                 → default          (smart crop)

Word timestamps from Transcriber used for subtitle alignment.
Crop hints from layout prevent screen content being cut off.
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
        mode="educational",
        params=params,
        progress_callback=progress_callback,
    )

    # ── Enrichment data ───────────────────────────────────────────────────────
    layout_segments  = enrichment.get("layout_segments") or []
    word_timestamps  = enrichment.get("word_timestamps") or []
    crop_hints       = enrichment.get("crop_hints") or {}
    yolo_detections  = enrichment.get("yolo_detections") or []

    # ── Detect dominant layout from GeminiLayoutAnalyzer output ──────────────
    layout_mode = _detect_layout(layout_segments)
    logger.info("[educational] dominant layout: %s | words=%d", layout_mode, len(word_timestamps))

    # ── Build transcript for educational_mode_v5 ──────────────────────────────
    transcript = {"words": word_timestamps, "text": " ".join(w.get("word", "") for w in word_timestamps)}

    # ── educational_mode_v5 ───────────────────────────────────────────────────
    try:
        from scripts.legacy_gpu.educational_mode_v5 import EducationalMode
        edu = EducationalMode(params={**params, "layout_mode": layout_mode})
        segments = edu.extract(input_video_path, transcript=transcript)
        logger.info("[educational] educational_mode_v5: %d segments", len(segments))
    except Exception as exc:
        logger.warning("[educational] educational_mode_v5 unavailable (%s) — fallback", exc)
        enrichment.setdefault("warnings", []).append(f"educational_engine_unavailable: {exc}")
        segments = _fallback_segments(word_timestamps, layout_segments)

    max_clips = params.get("max_clips", 10)
    top = sorted(segments, key=lambda s: float(s.get("score", 0)), reverse=True)[:max_clips]

    # ── Build layout-aware crop hints ─────────────────────────────────────────
    # Merge layout strategy into crop_hints so smart_crop_adapter respects it
    layout_crop_hints = _build_layout_crop_hints(layout_mode, crop_hints, layout_segments)

    from scripts.shared.crop.smart_crop_adapter import compose_vertical_clip
    output_paths = []
    for i, seg in enumerate(top):
        out = str(output_dir / f"edu_clip_{i+1:02d}.mp4")
        seg_start    = float(seg.get("start", 0))
        seg_duration = float(seg.get("duration", 60.0))

        # Find subtitle words for this segment (for metadata)
        seg_words = [
            w for w in word_timestamps
            if seg_start <= float(w.get("start", 0)) < seg_start + seg_duration
        ]

        try:
            compose_vertical_clip(
                input_video_path=input_video_path,
                output_path=out,
                crop_hints=layout_crop_hints,
                start_time=seg_start,
                duration=seg_duration,
            )
            output_paths.append(out)
            logger.info(
                "[educational] Clip %d: start=%.1fs dur=%.1fs words=%d layout=%s",
                i + 1, seg_start, seg_duration, len(seg_words), layout_mode,
            )
        except Exception as exc:
            logger.warning("[educational] Clip %d failed: %s", i + 1, exc)

    from scripts.shared.mode_output import write_selected_segments
    names = [str(output_dir / f"edu_clip_{i+1:02d}.mp4") for i in range(len(top))]
    write_selected_segments(str(output_dir), "educational", top,
                            [n if n in output_paths else None for n in names])

    return {
        "clips": output_paths,
        "mode": "educational",
        "layout": layout_mode,
        "word_timestamps_count": len(word_timestamps),
        "warnings": enrichment.get("warnings", []),
    }


def _detect_layout(layout_segments: List[Dict]) -> str:
    """Determine dominant layout from GeminiLayoutAnalyzer segments."""
    if not layout_segments:
        return "unknown"
    types = [s.get("layout", s.get("type", "")) for s in layout_segments]
    dominant = max(set(types), key=types.count) if types else ""
    if dominant in ("screen_share", "screen"):
        return "preserve_screen"
    if dominant in ("single_speaker", "talking_head", "person"):
        return "person_focus"
    return "default"


def _build_layout_crop_hints(
    layout_mode: str,
    base_crop_hints: Dict,
    layout_segments: List[Dict],
) -> Dict:
    """
    Build crop hints that reflect the educational layout.
    preserve_screen → letterbox (safe: keep screen content intact)
    person_focus    → person_center (SmartCropper face-follow)
    """
    hints = {**base_crop_hints}
    hints["layout_mode"] = layout_mode

    if layout_mode == "preserve_screen":
        # Letterbox: do NOT crop into screen content
        hints["strategy"] = "letterbox"
        hints["aggressive_crop"] = False
        hints["padding_color"] = "black"
    elif layout_mode == "person_focus":
        hints["strategy"] = "person_center"
        hints["aggressive_crop"] = False
    else:
        hints.setdefault("strategy", "default")

    return hints


def _fallback_segments(word_timestamps: List[Dict], layout_segments: List[Dict]) -> List[Dict]:
    """Fallback segments when educational_mode_v5 fails."""
    segments = []

    # Segment by layout changes if available
    for ls in layout_segments:
        ls_start = float(ls.get("start", 0))
        ls_end   = float(ls.get("end", ls_start + 60))
        if ls_end - ls_start >= 10:
            segments.append({
                "start":    ls_start,
                "duration": ls_end - ls_start,
                "score":    2.0,
                "source":   "layout_fallback",
            })

    if not segments:
        segments.append({"start": 0, "duration": 60.0, "score": 1.0, "source": "default"})

    return segments
