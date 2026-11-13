"""
modes/educational/runner.py
============================
Educational mode v4: whole-sentence lesson clips (edu-select-v4 on top of the educational_mode_v5 analysis, optional
Gemini editor — analysis only), exact boundaries on ASR words (scripts/shared/boundaries, the same as Cut), one of four
templates (subtitles + cards + title; modes/educational/templates.py) + layout-aware processing.

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

    # ── Educational v4: which parts become clips ─────────────────────────────
    # one rouble ledger per job: the editor and every clip's card planner share the per-video limit
    params = {**params, "edu_ledger_path": params.get("edu_ledger_path") or str(output_dir / "edu_llm_ledger.json")}
    selection = select_segments(input_video_path, word_timestamps, params, output_dir, enrichment)
    top = selection["segments"]

    # ── Build layout-aware crop hints ─────────────────────────────────────────
    # Merge layout strategy into crop_hints so smart_crop_adapter respects it
    layout_crop_hints = _build_layout_crop_hints(layout_mode, crop_hints, layout_segments)
    look = _look_options(params)

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

        overlays_opt = _overlay_options(params, look, seg, len(top))
        other_aspect = isinstance(overlays_opt, dict) and overlays_opt.get("aspect", "9:16") != "9:16"
        try:
            if not other_aspect:    # compose_vertical_clip only makes 9:16; other aspects re-cut the source themselves
                frame, why = _frame_choice(params, input_video_path, seg_start, seg_duration, layout_mode)
                seg["frame_layout"], seg["frame_reason"] = frame, why
                st = None
                if frame == "stage":
                    try:
                        from modes.educational.stage import render_stage
                        from modes.educational.templates import get_template
                        st = render_stage(input_video_path, out, seg_start, seg_duration,
                                          get_template(look["template"]).stage)
                        if overlays_opt is not None:
                            overlays_opt.update(on_stage=True, protect_rect=st["video_rect"])
                    except Exception as exc:  # noqa: BLE001 — the user still gets the clip, cropped as before
                        logger.warning("[educational] stage render failed for clip %d (%s) — smart crop", i + 1, exc)
                        enrichment.setdefault("warnings", []).append(f"edu_stage_failed: {type(exc).__name__}")
                        seg["frame_layout"] = "crop"
                if st is None:
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
                                       {**params, "edu_overlays": overlays_opt}, enrichment)
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
        "selection_method": selection["method"],
        "template": look["template"],
        "llm": selection.get("llm"),
        "warnings": enrichment.get("warnings", []),
    }


def _frame_choice(params: Dict[str, Any], src: str, start: float, duration: float, layout_mode: str):
    """«stage» (whole slide/board on the template background) or «crop» (talking head). params.edu.frame forces it."""
    edu = params.get("edu") if isinstance(params.get("edu"), dict) else {}
    forced = edu.get("frame")
    if forced in ("stage", "crop"):
        return forced, {"why": "forced"}
    if forced == "legacy":
        return "crop", {"why": "legacy"}
    from modes.educational.stage import choose
    return choose(src, start, duration, layout_mode)


def _asr_segments(words: List[Dict]) -> List[Dict]:
    """v5 reads ASR segments (sentences); the worker hands over words: group them with the shared sentence rules."""
    from scripts.shared.boundaries import group_sentences, normalize_words
    ws, _ = normalize_words(words)
    out = []
    for a, b in group_sentences(ws):
        out.append({"start": ws[a].start, "end": ws[b].end, "text": " ".join(w.text for w in ws[a:b + 1]),
                    "words": [{"word": w.text, "start": w.start, "end": w.end} for w in ws[a:b + 1]]})
    return out


def _llm_on(params: Dict[str, Any]) -> bool:
    """Gemini (analysis only): params.edu.editor wins; then params.llm; otherwise the same rule as Cut — on when a
    provider is configured on this worker (TIMEWEB / OPENROUTER key) and SONYA_LLM_EDITOR is not off."""
    edu = params.get("edu") if isinstance(params.get("edu"), dict) else {}
    if "editor" in edu:
        return bool(edu["editor"])
    if params.get("llm") in ("on", "off"):
        return params.get("llm") == "on"
    try:
        from scripts.shared.cut_profile import editor_available
        return editor_available()
    except Exception:  # noqa: BLE001
        return False


def select_segments(input_video_path: str, words: List[Dict], params: Dict[str, Any], output_dir: Path,
                    enrichment: Dict[str, Any]) -> Dict[str, Any]:
    """v5 analysis + edu-select-v4 (+ optional Gemini editor) + exact boundaries on ASR words.
    Never silent: any degradation is written to warnings and to edu_selection.json."""
    import json
    warnings = enrichment.setdefault("warnings", [])
    trace: Dict[str, Any] = {"words": len(words)}
    from scripts.shared.cut_profile import default_max_clips
    max_clips = default_max_clips(params)          # 8 by default (owner 2026-11-12)
    v5_result = None
    try:
        from scripts.legacy_gpu.educational_mode_v5 import run_educational_mode_v5
        v5_result = run_educational_mode_v5(input_video_path, _asr_segments(words), None, None)
        trace["v5"] = {"status": "ok", "stats": v5_result.get("stats"),
                       "topics": len(v5_result.get("topic_segments") or [])}
    except Exception as exc:  # noqa: BLE001 — selection still works on the words alone
        logger.warning("[educational] v5 analysis failed (%s) — selection on words only", exc)
        warnings.append(f"educational_v5_failed: {type(exc).__name__}: {str(exc)[:160]}")
        trace["v5"] = {"status": "failed", "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
    segments: List[Dict] = []
    method = "fallback"
    try:
        from modes.educational import selection as sl
        edu = params.get("edu") if isinstance(params.get("edu"), dict) else {}
        cfg = sl.EduConfig(max_clips=max_clips, editor=_llm_on(params),
                           **{k: v for k, v in edu.items() if k in ("min_duration", "max_duration", "skip_intro_sec")})
        editor_fn = None
        if cfg.editor:
            from modes.educational import editor as ed
            from modes.educational.overlays import budget as bg
            ledger = bg.Ledger(params.get("edu_ledger_path"))
            editor_fn = ed.make_editor(video_id=Path(input_video_path).stem, ledger=ledger, want=max_clips)
        segments, meta = sl.select_lesson_clips(words, cfg, v5_result, editor_fn)
        trace["selection"] = meta
        if meta.get("llm", {}).get("status") not in ("off", "ok"):
            warnings.append(f"educational_editor_{meta['llm'].get('status')}")
        if segments:
            method = meta.get("selection_method", "local")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[educational] edu-select-v4 failed: %s", exc)
        warnings.append(f"educational_selection_failed: {type(exc).__name__}: {str(exc)[:160]}")
        trace["selection_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    if not segments:
        warnings.append("educational_engine_unavailable: no complete explanation found — fallback first minute / "
                        "layout cuts")
        logger.warning("[educational] FALLBACK: no v4 segments — the old first-60-s / layout segments are used")
        segments = _fallback_segments(words, enrichment.get("layout_segments") or [])[:max_clips]
    else:
        from scripts.shared.boundaries import BoundaryConfig, refine_segments
        # priority order for overlap resolution = score order; then back to the series order
        by_score = sorted(segments, key=lambda s: -float(s.get("score", 0)))
        refined = refine_segments(by_score, words, BoundaryConfig(**(params.get("boundaries") or {})))
        segments = sorted([s for s in refined if s.get("boundary_status") != "dropped_overlap"],
                          key=lambda s: s["start"])
        for k, s in enumerate(segments, 1):
            s["series_index"], s["series_total"] = k, len(segments)
    trace["method"] = method
    trace["segments"] = [{k: s.get(k) for k in ("start", "duration", "title", "topic_title", "segment_type",
                                                "selection_reason", "boundary_status", "source")} for s in segments]
    try:
        (Path(output_dir) / "edu_selection.json").write_text(json.dumps(trace, ensure_ascii=False, indent=1,
                                                                        default=str), encoding="utf-8")
    except OSError:
        pass
    return {"segments": segments, "method": method, "llm": (trace.get("selection") or {}).get("llm")}


def _look_options(params: Dict[str, Any]) -> Dict[str, Any]:
    """Template 1–4 from the job (params.template or the site's settings.edu_template); titles and subtitles on."""
    from modes.educational.templates import get_template
    settings = params.get("settings") if isinstance(params.get("settings"), dict) else {}
    raw = params.get("template", settings.get("edu_template", settings.get("template")))
    tpl = get_template(raw)
    subs = params.get("subtitles", settings.get("subtitles", True))
    if str(settings.get("subtitleStyle") or "").lower() in ("none", "off"):      # the site's subtitles switch
        subs = False
    return {"template": tpl.id, "subtitles": subs not in (False, "off", "false", 0),
            "title": params.get("edu_title", settings.get("edu_title", True)) not in (False, "off", "false", 0)}


def _overlay_options(params: Dict[str, Any], look: Dict[str, Any], seg: Dict[str, Any], total: int) -> Dict[str, Any]:
    """Every v4 clip goes through the overlay compositor: template subtitles + title; cards only when the job asks
    for them (params.edu_overlays, v2 semantics) — they need the LLM planner."""
    base = params.get("edu_overlays")
    opts: Dict[str, Any] = dict(base) if isinstance(base, dict) else {}
    edu = params.get("edu") if isinstance(params.get("edu"), dict) else {}
    llm = _llm_on(params)
    # cards need the LLM planner: on with the LLM unless the job says otherwise (edu.cards / edu_overlays=False)
    opts.setdefault("cards", bool(base) if base is not None else bool(edu.get("cards", llm)))
    opts.setdefault("llm_opt_in", llm)
    opts.setdefault("preset", "deep")       # terms, facts, formulas, steps, charts — each still has to pass evidence
    opts.setdefault("template", look["template"])
    opts.setdefault("burn_subtitles", look["subtitles"])
    if look["title"] and seg.get("title"):
        opts.setdefault("title", seg["title"])
    if seg.get("series_total", 0) > 1:
        opts.setdefault("part", (seg["series_index"], seg["series_total"]))
    return opts


def _apply_overlays(clip_path: str, input_video_path: str, words: List[Dict], seg_start: float, seg_duration: float,
                    index: int, params: Dict[str, Any], enrichment: Dict[str, Any]) -> bool:
    """Educational v2 (params["edu_overlays"], default off). The plain clip stays in place if anything goes wrong."""
    options = dict(params["edu_overlays"]) if isinstance(params["edu_overlays"], dict) else {}
    options.setdefault("llm_opt_in", _llm_on(params))
    options.setdefault("ledger_path", params.get("edu_ledger_path"))
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
