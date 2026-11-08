"""
modes/virality/runner.py
========================
Virality mode: hook_mode_v1 + modes_scoring + yuvelirochka visual boosts.

Visual boosts applied to hook_mode segments:
  - gemini_viral_moments  → overlap boost +0.3 per overlapping moment
  - person in first 3s    → webcam_layout boxes boost +0.2
  - layout_segments       → dominant layout awareness
  - active_speaker_segs   → speaking boost +0.15
  - pose_detections       → human action boost +0.1
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

    # production jobs carry no Cut settings → the Mac mini reference recipe (scripts/shared/cut_profile.py);
    # its caches live next to output/, never inside it (everything in output/ is uploaded)
    from scripts.shared.cut_profile import apply_profile
    params = apply_profile("virality", params, input_video_path, str(output_dir.parent / "cut_work"))

    from scripts.shared.enhancers.sonya_enhancer import enrich_video_for_mode
    enrichment = enrich_video_for_mode(
        input_video_path=input_video_path,
        output_dir=str(output_dir),
        mode="virality",
        params=params,
        progress_callback=progress_callback,
    )

    clips, selection = _select_and_compose(
        input_video_path=input_video_path,
        enrichment=enrichment,
        params=params,
        output_dir=output_dir,
    )

    if params.get("cut_profile"):
        selection = dict(selection or {}, cut_profile=params["cut_profile"])
    return {
        "clips": clips,
        "mode": "virality",
        "warnings": enrichment.get("warnings", []),
        "selection": selection,
    }


def _select_and_compose(
    input_video_path: str,
    enrichment: Dict[str, Any],
    params: Dict[str, Any],
    output_dir: Path,
) -> List[str]:
    """Select clips, refine boundaries on ASR words, compose.

    With ASR words: candidates are complete phrases from the transcript and
    the selection (Gemini on opt-in, else deterministic local ranking) comes
    from scripts/shared/cut_selection.py — timestamps only from ASR.
    Without words: legacy hook_mode_v1 path (+ visual boosts), whose
    fallback is the first 30 s; selection_status says so explicitly.
    """
    max_clips = params.get("max_clips", 3)
    words = enrichment.get("word_timestamps") or []
    if params.get("cut_profile") and not words:
        # Mac mini recipe: ASR is mandatory (Desktop _ensure_asr) - never the old no-words hook_mode path
        raise RuntimeError("ASR_FAILED: no word timestamps - Cut does not run without speech recognition ("
                           + "; ".join(str(w) for w in enrichment.get("warnings", []))[:300] + ")")
    if words:   # owner-confirmed name fixes (Вавилон→Эвелон near «стример»…) before selection and subtitles
        from scripts.shared.glossary import apply as apply_glossary
        words = apply_glossary(words, params.get("glossary"))
        enrichment["word_timestamps"] = words
    selection: Dict[str, Any] = {}
    asr_segments: List[Dict[str, Any]] = []
    if params.get("promo") and not (params.get("selection") == "story" and params.get("llm") != "on"
                                     and ((params.get("crop") or {}).get("mode") == "tracked")):
        # advertiser rules + overlay exist only on the story selector + tracked renderer: refuse
        # instead of shipping clips that ignore the conditions or lack the marking
        raise ValueError("promo requires selection=story and crop.mode=tracked")
    if words and params.get("selection") == "story" and params.get("llm") != "on":
        # Desktop: whole stories / question + answer, 20–90 s (cut-local-rank-v3)
        from scripts.shared.cut_selection import StoryConfig, select_story_clips
        story = dict(params.get("story") or {})
        if story.get("strategy") in ("viral", "hooks", "story"):
            # loudness per second (one ffmpeg pass, cached in the project): emotional peaks for ranking
            try:
                from scripts.shared.audio_features import loudness_per_second, zscores
                from scripts.shared.crop.smart_crop_adapter import ffmpeg_bin
                cache = (params.get("crop") or {}).get("clip_cache_dir")
                cache = str(Path(cache).parent / "audio_energy.json") if cache else None
                story["energy"] = zscores(loudness_per_second(input_video_path, ffmpeg_bin(), cache))
            except Exception as exc:             # ranking still works on words alone; say so
                logger.warning("[virality] audio energy unavailable: %s", exc)
        if params.get("promo"):          # advertiser conditions shape WHICH moments are picked
            story["promo"] = params["promo"]
        asr_segments, selection = select_story_clips(words, max_clips, StoryConfig(**story))
    elif words:
        from scripts.shared.cut_selection import CandidateConfig, select_clips
        asr_segments, selection = select_clips(
            words, max_clips, llm_opt_in=params.get("llm") == "on",
            cfg=CandidateConfig(**(params.get("candidates") or {})))
    if asr_segments:
        top = asr_segments
    else:
        selection["selection_status"] = ("no_asr_legacy_fallback" if not words
                                         else "no_asr_candidates_legacy_fallback")
        top = _legacy_segments(input_video_path, enrichment, params)

    # ── Exact boundaries on real ASR words (does not change WHICH moments) ──
    from scripts.shared.boundaries import BoundaryConfig, refine_segments
    selected = refine_segments(top[:max_clips], words,
                               BoundaryConfig(**(params.get("boundaries") or {})))
    return _compose(input_video_path, enrichment, selected, output_dir, params), selection


def _legacy_segments(input_video_path: str, enrichment: Dict[str, Any],
                     params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """hook_mode_v1 + yuvelirochka visual boosts (unchanged legacy behaviour)."""
    # ── Base segments from hook_mode_v1 ──────────────────────────────────────
    try:
        from scripts.legacy_gpu.hook_mode_v1 import HookMode
        hook_mode = HookMode(params=params)
        raw_segments = hook_mode.extract_segments(input_video_path)
        logger.info("[virality] hook_mode_v1: %d raw segments", len(raw_segments))
    except Exception as exc:
        logger.warning("[virality] hook_mode_v1 unavailable (%s) — fallback segments", exc)
        enrichment.setdefault("warnings", []).append(f"virality_engine_unavailable: {exc}")
        raw_segments = [{"start": 0.0, "duration": 30.0, "score": 0.5}]

    # ── yuvelirochka visual boosts ────────────────────────────────────────────
    gemini_viral_moments   = enrichment.get("gemini_viral_moments") or []
    layout_segments        = enrichment.get("layout_segments") or []
    pose_detections        = enrichment.get("pose_detections") or []
    webcam_layout          = enrichment.get("webcam_layout") or {}
    active_speaker_segs    = enrichment.get("active_speaker_segments") or []
    crop_hints             = enrichment.get("crop_hints") or {}

    webcam_boxes = webcam_layout.get("boxes", [])

    for seg in raw_segments:
        start = float(seg.get("start", 0))
        end   = start + float(seg.get("duration", 30))
        boost = 0.0

        # Person visible in first 3 seconds → strong hook signal
        if start <= 3.0 and webcam_boxes:
            boost += 0.2

        # Gemini viral moment overlap
        for m in gemini_viral_moments:
            m_start = float(m.get("start", m.get("start_time", 0)))
            m_end   = m_start + float(m.get("duration", m.get("clip_duration", 0)))
            if m_start < end and m_end > start:
                boost += 0.3

        # Layout signal: person-focused layout → virality boost
        for ls in layout_segments:
            ls_start = float(ls.get("start", 0))
            ls_end   = float(ls.get("end", ls_start + 1))
            if ls_start < end and ls_end > start:
                layout_type = ls.get("layout", ls.get("type", ""))
                if layout_type in ("single_speaker", "person", "talking_head"):
                    boost += 0.1

        # Active speaker in segment
        for as_seg in active_speaker_segs:
            as_start = float(as_seg.get("start", 0))
            as_end   = float(as_seg.get("end", as_start))
            if as_start < end and as_end > start:
                boost += 0.15

        # Human pose detected (action visible)
        if pose_detections:
            boost += 0.1

        seg["score"] = float(seg.get("score", 0)) + boost

    # ── Select top segments ───────────────────────────────────────────────────
    return sorted(raw_segments, key=lambda s: float(s.get("score", 0)), reverse=True)


def _load_face_analysis(crop: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The face analysis, or None (→ centered fallback) if it is missing,
    unreadable or malformed — a bad cache file must never fail the run."""
    path = crop.get("analysis_path")
    if not path:
        return None
    import json
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        meta = data.get("meta") or {}
        if not isinstance(data.get("samples"), list) or not meta.get("width") or not meta.get("height"):
            raise ValueError("malformed face analysis")
        return data
    except (OSError, ValueError, AttributeError) as exc:
        logger.warning("[virality] face analysis unusable (%s) — centered crop", exc)
        crop["analysis_status"] = "unreadable"
        return None


def _source_dims(path: str) -> "tuple[int, int]":
    import json
    import subprocess
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height", "-of", "json", path], capture_output=True, text=True)
    st = (json.loads(out.stdout or "{}").get("streams") or [{}])[0]
    return int(st.get("width") or 0), int(st.get("height") or 0)


def _compose(input_video_path: str, enrichment: Dict[str, Any], selected: List[Dict[str, Any]],
             output_dir: Path, params: Optional[Dict[str, Any]] = None) -> List[str]:
    crop_hints = enrichment.get("crop_hints") or {}
    crop = (params or {}).get("crop") or {}
    if crop.get("mode") == "tracked":
        if params.get("promo"):
            crop = dict(crop, promo=params["promo"])     # overlay drawn with the subtitles
        return _compose_tracked(input_video_path, selected, output_dir, crop,
                                enrichment.get("word_timestamps") or [])
    # ── Compose vertical clips ────────────────────────────────────────────────
    from scripts.shared.crop.smart_crop_adapter import compose_vertical_clip
    output_paths = []
    for i, seg in enumerate(selected):
        out = str(output_dir / f"viral_clip_{i+1:02d}.mp4")
        if seg.get("boundary_status") == "dropped_overlap":
            logger.info("[virality] Clip %d dropped: overlaps a higher-ranked clip", i + 1)
            continue
        try:
            compose_vertical_clip(
                input_video_path=input_video_path,
                output_path=out,
                crop_hints=crop_hints,
                start_time=float(seg.get("start", 0)),
                duration=float(seg.get("duration", 30.0)),
            )
            output_paths.append(out)
        except Exception as exc:
            logger.warning("[virality] Clip %d failed: %s", i + 1, exc)

    from scripts.shared.mode_output import write_selected_segments
    names = [str(output_dir / f"viral_clip_{i+1:02d}.mp4") for i in range(len(selected))]
    write_selected_segments(str(output_dir), "virality", selected,
                            [n if n in output_paths else None for n in names])
    return output_paths


def _clip_analysis(crop: Dict[str, Any], video: str, start: float, dur: float):
    """Per-clip cuts + faces (scripts/shared/crop/clip_analysis), cached per
    (source sha256, interval, version). -> (analysis | None, cache, error)."""
    import json
    import os
    from scripts.shared.crop import clip_analysis as ca
    st = ca.model_ok()
    if not st["ok"]:
        return None, None, "detector_unavailable: " + st["reason"]
    cache_dir, sha = crop.get("clip_cache_dir"), crop.get("source_sha256")
    path = None
    if cache_dir and sha:
        path = os.path.join(cache_dir, f"{sha[:16]}-{start:.3f}-{dur:.3f}-{ca.CLIP_ANALYSIS_VERSION}.json")
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            m = data.get("meta") or {}
            if m.get("version") == ca.CLIP_ANALYSIS_VERSION and isinstance(data.get("samples"), list) \
                    and abs(float(m.get("start", -1)) - start) < 1e-3 and m.get("width") and m.get("height"):
                return data, "hit", None
        except (OSError, ValueError, AttributeError, TypeError):
            pass
    try:
        data = ca.analyze_clip(video, start, dur)
    except Exception as exc:                      # a broken analysis never fails the run: centred crop
        logger.warning("[virality] clip analysis failed (%s) — centered crop", exc)
        return None, None, f"analysis_failed: {str(exc)[:200]}"
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = f"{path}.{os.getpid()}.partial"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        os.replace(tmp, path)
    return data, "miss", None


def _insert_integration(clip: str, promo: Dict[str, Any], words: List[Dict[str, Any]],
                        start: float, dur: float) -> Dict[str, Any]:
    """The advertiser's own video into a rendered clip, at a pause near the middle; the half-screen
    variant falls back to a midroll when the clip is too short for it to play out (offer rule:
    «интеграция должна успеть полностью воспроизвестись»)."""
    import os
    import subprocess
    from scripts.shared.promo import PromoSpec, check_asset, insert_asset_cmd, media_info, pick_insert_time
    from scripts.shared.crop.smart_crop_adapter import _env_encoder_args, ffmpeg_bin
    from scripts.shared.subtitle_styles import FONTS, FONTS_DIR
    spec = PromoSpec.from_dict(promo)
    ff = ffmpeg_bin()
    probe = os.path.join(os.path.dirname(ff), "ffprobe") if os.path.dirname(ff) else "ffprobe"
    root = promo.get("asset_root") or os.path.dirname(os.path.dirname(clip))
    info = check_asset(spec.asset, root, probe)
    clip_info = media_info(clip, probe)
    mode = spec.integration
    t = pick_insert_time(words, start, dur, need_after=info["duration"] if mode == "half" else 0.0)
    if mode == "half" and t + info["duration"] > clip_info["duration"] - 0.2:
        mode = "midroll"            # too short for the half-screen one to play out
        t = pick_insert_time(words, start, dur)
    font = str(FONTS_DIR / FONTS["dela"][1])
    tmp = clip + ".promo.mp4"
    cmd = insert_asset_cmd(clip, info["path"], tmp, mode, t, info["duration"], clip_info["duration"],
                           clip_info["has_audio"], spec.marking(), font, ffmpeg=ff,
                           encoder_args=_env_encoder_args() or None)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"integration failed: {r.stderr[-400:]}")
    os.replace(tmp, clip)
    return {"promo_integration": mode, "promo_insert_at": t, "promo_asset_sec": round(info["duration"], 3),
            "promo_asset_audio": info["has_audio"]}


def _compose_tracked(input_video_path: str, selected: List[Dict[str, Any]], output_dir: Path,
                     crop: Dict[str, Any], words: Optional[List[Dict[str, Any]]] = None) -> List[str]:
    """9:16 along a crop track computed from ONE face analysis of the source
    (scripts/shared/crop/cropper.analyze_faces, done by the caller). No
    detector runs here; missing analysis/subject → centered window with an
    explicit crop_status / fallback_reason."""
    from scripts.shared.crop.smart_crop_adapter import (crop_track, crop_track_v2, fit_track, game_track,
                                                         render_tracked, text_zones)
    from scripts.shared.mode_output import write_selected_segments

    from scripts.shared.subtitles import subtitles_for_clip

    fit = crop.get("layout") in ("fit", "blur")
    per_clip = crop.get("analysis") == "per_clip" and not fit
    analysis = None if per_clip else _load_face_analysis(crop)
    meta = (analysis or {}).get("meta") or {}
    W, H = (meta.get("width"), meta.get("height")) if analysis else _source_dims(input_video_path)
    subs = crop.get("subtitles") or {}
    def one(i: int, seg: Dict[str, Any]) -> Optional[str]:
        """Analysis → track → subtitles → render for one clip (runs in a worker thread)."""
        out = str(output_dir / f"viral_clip_{i+1:02d}.mp4")
        if seg.get("boundary_status") == "dropped_overlap":
            return None
        start, dur = float(seg.get("start", 0)), float(seg.get("duration", 30.0))
        zones: List[List[float]] = []
        cache = crop.get("cache")
        if fit:
            track = fit_track(W, H, dur, "blur" if crop.get("layout") == "blur" else "black")
        elif per_clip:
            clip_an, cache, err = _clip_analysis(crop, input_video_path, start, dur)
            track = (game_track(clip_an, dur, width=W, height=H) if crop.get("layout") == "game" else None) \
                or crop_track_v2(clip_an, dur, width=W, height=H)
            if clip_an is None:
                track["fallback_reason"] = err.split(":", 1)[0] if err else "analysis_failed"
                seg["crop_detector_error"] = err
            seg.update({"crop_cuts": track.get("cuts", []),
                        "crop_shots": [{k: sh.get(k) for k in ("start", "end", "target", "faces", "speaker_switch")}
                                       for sh in track.get("shots", [])]})
            zones = text_zones(clip_an, track)
            seg["crop_text_zones"] = zones
        else:
            track = crop_track(analysis, start, dur, width=W, height=H)
        if not fit and not per_clip and analysis is None and crop.get("analysis_status"):
            st = crop["analysis_status"]
            track["fallback_reason"] = "detector_unavailable" if st == "unavailable" else f"analysis_{st}"
            if crop.get("analysis_error"):
                seg["crop_detector_error"] = crop["analysis_error"]
        seg.update({"crop_version": track["crop_version"], "crop_status": track["crop_status"],
                    "crop_fallback_reason": track["fallback_reason"], "crop_detector": track["detector"],
                    "crop_cache": cache, "source_width": W, "source_height": H,
                    "output_aspect": "9:16", "crop_keyframes": track["keyframes"]})
        sub_path = None
        if subs.get("mode") == "burn":
            sm = subtitles_for_clip(words, start, dur, Path(out).with_suffix(".ass"),
                                    centered=(track.get("split") or {}).get("intervals") or track.get("seam"),
                                    avoid=zones,
                                    preset=subs.get("style"), keywords=seg.get("keywords"),
                                    hook=seg.get("hook") if subs.get("hook", False) else None,
                                    font=subs.get("font"), promo=crop.get("promo"))
            burn = sm.get("subtitle_status") in ("ok", "promo_only") and bool(subs.get("burn_supported"))
            sm["subtitle_burn"] = ("burned" if burn else
                                   "ffmpeg_without_libass" if sm.get("subtitle_status") == "ok" else "none")
            if sm.get("subtitle_path"):
                sm["subtitle_path"] = Path(sm["subtitle_path"]).name
            seg.update(sm)
            sub_path = str(Path(out).with_suffix(".ass")) if burn else None
        try:
            render_tracked(input_video_path, out, start, dur, track, subtitle_path=sub_path)
            if crop.get("promo") and crop["promo"].get("integration"):
                seg.update(_insert_integration(out, crop["promo"], words or [], start, dur))
            return out
        except Exception as exc:
            logger.warning("[virality] Clip %d failed: %s", i + 1, exc)
            return None

    # Two clips at a time: one analyses (CPU) while the other renders (hardware encoder).
    import os
    from concurrent.futures import ThreadPoolExecutor
    workers = max(1, int(os.environ.get("SONYA_CLIP_WORKERS", "2")))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda a: one(*a), enumerate(selected)))
    output_paths: List[str] = [r for r in results if r]
    names = [str(output_dir / f"viral_clip_{i+1:02d}.mp4") for i in range(len(selected))]
    write_selected_segments(str(output_dir), "virality", selected,
                            [n if n in output_paths else None for n in names])
    return output_paths
