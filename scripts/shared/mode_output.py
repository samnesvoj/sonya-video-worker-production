"""
mode_output.py
==============
Writes `selected_segments.json` next to a mode's clips: which source
interval each produced clip was cut from (+ score/source of the candidate).

Why: runners only return clip paths, so after a human reviews the clips
there is no record of the intervals behind them — and golden tests
(desktop/sonya_desktop/golden) need exactly that. gpu_worker uploads .json
files as "enrichment_json", so this file never counts as a clip.

Pure bookkeeping: it does not influence which segments are selected.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

FILENAME = "selected_segments.json"
_BOUNDARY_KEYS = ("semantic_start", "semantic_end", "render_start", "render_end", "boundary_status",
                  "boundary_version", "boundary_reasons", "boundary_params",
                  "candidate_id", "selection_reason",
                  "crop_version", "crop_status", "crop_fallback_reason", "crop_detector", "crop_cache",
                  "crop_detector_error", "render_encoder", "render_s",
                  "source_width", "source_height", "output_aspect", "crop_keyframes", "crop_cuts", "crop_shots",
                  "subtitle_status", "subtitle_version", "subtitle_style", "subtitle_path", "subtitle_cues",
                  "subtitle_words", "subtitle_first_word", "subtitle_last_word", "subtitle_burn",
                  "subtitle_centered_cues", "subtitle_moved_for_text", "crop_text_zones",
                  "hook", "keywords", "editor_score", "subtitle_hook", "ad_check", "promo_version",
                  "promo_integration", "promo_insert_at", "promo_asset_sec", "promo_asset_audio",
                  # streamer v2 layout: where the webcam / chat / game windows are (source px)
                  "crop_layout", "crop_cam", "crop_game", "webcam_rect", "webcam_source", "layout_chat",
                  "game_chat_overlap", "webcam_detector_box",
                  # streamer v2 dynamics: jump cuts, hook title, SONYA's own ad
                  "tight_sec", "jump_cuts", "body_sec", "cut_sec", "output_duration", "render_pieces",
                  "sonya_ad", "sonya_ad_gain_db", "clip_lufs", "ad_lufs")
SCHEMA_VERSION = 1


def _interval(seg: Dict[str, Any]) -> Dict[str, float]:
    start = float(seg.get("start_sec", seg.get("start", 0.0)))
    if "end" in seg:
        end = float(seg["end"])
    else:
        end = start + float(seg.get("duration_sec", seg.get("duration", 0.0)))
    return {"start": round(start, 3), "end": round(end, 3)}


# optional per-clip facts a mode may report (Educational v4: the title, the place in a series, the frame layout)
_CLIP_KEYS = ("title", "topic_title", "series_index", "series_total", "frame_layout")


def write_selected_segments(output_dir: str, mode: str, selected: Sequence[Dict[str, Any]],
                            clip_paths: Sequence[Optional[str]],
                            max_duration: Optional[float] = None) -> Optional[str]:
    """selected[i] produced clip_paths[i] (None if that clip failed).

    Never raises: bookkeeping must not fail a run that produced clips.
    """
    try:
        items: List[Dict[str, Any]] = []
        for i, seg in enumerate(selected):
            iv = _interval(seg)
            if max_duration is not None:
                iv["end"] = round(min(iv["end"], iv["start"] + float(max_duration)), 3)
            path = clip_paths[i] if i < len(clip_paths) else None
            item = {"index": i + 1, "file": os.path.basename(path) if path else None,
                    **iv, "score": seg.get("score"), "source": seg.get("source")}
            # boundary engine metadata (scripts/shared/boundaries.py), when applied
            for k in _BOUNDARY_KEYS + _CLIP_KEYS:
                if k in seg:
                    item[k] = seg[k]
            items.append(item)
        out = Path(output_dir) / FILENAME
        out.write_text(json.dumps({"schema_version": SCHEMA_VERSION, "mode": mode,
                                   "segments": items}, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        return str(out)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("[mode_output] could not write %s: %s", FILENAME, exc)
        return None
