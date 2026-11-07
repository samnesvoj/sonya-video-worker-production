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
import os
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

        overlays_opt = params.get("edu_overlays")
        other_aspect = isinstance(overlays_opt, dict) and overlays_opt.get("aspect", "9:16") != "9:16"
        try:
            if not other_aspect:    # compose_vertical_clip only makes 9:16; other aspects re-cut the source themselves
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
            if overlays_opt:
                done = _apply_overlays(out, input_video_path, word_timestamps, seg_start, seg_duration, i + 1,
                                       params, enrichment)
                if other_aspect:
                    if not done:    # no plain clip exists yet in this path: the user still gets one
                        compose_vertical_clip(input_video_path=input_video_path, output_path=out,
                                              crop_hints=layout_crop_hints, start_time=seg_start,
                                              duration=seg_duration)
                    output_paths.append(out)
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


def _apply_overlays(clip_path: str, input_video_path: str, words: List[Dict], seg_start: float, seg_duration: float,
                    index: int, params: Dict[str, Any], enrichment: Dict[str, Any]) -> bool:
    """Educational v2 (params["edu_overlays"], default off). The plain clip stays in place if anything goes wrong."""
    options = dict(params["edu_overlays"]) if isinstance(params["edu_overlays"], dict) else {}
    options.setdefault("llm_opt_in", params.get("llm") == "on")
    options.setdefault("video_id", Path(input_video_path).stem)
    try:
        from modes.educational.overlays import apply_overlays
        from modes.educational.overlays.compose import render_base
        tmp = str(Path(clip_path).with_name(Path(clip_path).stem + ".v2.mp4"))
        base = clip_path
        aspect = options.get("aspect", "9:16")
        if aspect != "9:16":
            base = str(Path(clip_path).with_name(Path(clip_path).stem + ".base.mp4"))
            render_base(input_video_path, base, seg_start, seg_duration, aspect)
        report = apply_overlays(base, tmp, words, seg_start, seg_duration, options)
        os.replace(tmp, clip_path)
        sidecar = Path(tmp).with_suffix(".overlays.json")
        if sidecar.exists():
            os.replace(sidecar, Path(clip_path).with_suffix(".overlays.json"))
        if base != clip_path:
            os.unlink(base)
        enrichment.setdefault("warnings", []).extend(report.get("warnings", []))
        return True
    except Exception as exc:  # noqa: BLE001 — a card must never cost the user the clip
        logger.warning("[educational] overlays skipped for clip %d: %s", index, type(exc).__name__)
        enrichment.setdefault("warnings", []).append(f"edu_overlays_failed: {type(exc).__name__}")
        return False


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
