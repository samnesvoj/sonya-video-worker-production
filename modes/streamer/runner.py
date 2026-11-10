"""
modes/streamer/runner.py
========================
Streamer mode v2 (2026-11-10): highlights from a long stream VOD, rendered like SONYA Cut.

Why v2 (measured on real streams, see docs/STREAMER_V2.md):
  * v1 never found moments: the yuvelirochka analyzer/clipper it imported does not exist
    (`from modes.streamer.legacy.analyzer import analyze` raises ImportError), so every stream fell
    back to ONE clip of the first 30 s (`_fallback_segments` with lip sync off);
  * v1 looked bad: three lossy encodes (ffmpeg cut at CRF 18 → OpenCV `mp4v` frame writer with no
    bitrate → ffmpeg-python default CRF 23), a Haar face crop at the source's own size (608x1080 from
    1080p, never 1080x1920), no webcam / game layout at all.

v2 pipeline (analysis ONCE, render per clip):
  analyze()     ASR words (GigaAM) + per-second signals of the whole VOD (scripts/shared/stream_signals:
                voice loudness/pitch, speech rate, reaction words, game cuts/flashes, chat density when the VOD
                has chat) → moments from whole sentences, 25–60 s, hook first (scripts/shared/stream_moments)
                → optional Gemini editor (analysis only, ≤ 10 ₽) → word-exact boundaries (scripts/shared/boundaries).
  compose_one() per-clip YuNet analysis → game layout «веб-камера сверху, игра снизу» when a steady webcam
                face is found (optionally the webcam_detector box), else the Cut person/centre crop →
                burned subtitles → ONE ffmpeg pass from the original source (lanczos, 6 Mb/s NVENC / libx264).

Segment contract (canonical): {"start_sec", "duration_sec", "score", "source", ...extra}. Extra keys used by
compose_one(): "words" (the clip's ASR words, for subtitles), "hook", "keywords".
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

STREAMER_VERSION = "streamer-v2"


def _flag(name: str, default: str = "true") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def streamer_crop(params: Dict[str, Any], input_video_path: str, work_dir: str) -> Dict[str, Any]:
    """Crop/render settings of the streamer recipe (Cut renderer with the game layout)."""
    from scripts.shared.cut_profile import SITE_STYLES, DEFAULT_STYLE, _sha256, ffmpeg_has_ass
    settings = params.get("settings") if isinstance(params.get("settings"), dict) else {}
    crop: Dict[str, Any] = {"mode": "tracked", "analysis": "per_clip", "layout": "game",
                            "source_sha256": params.get("source_sha256") or _sha256(input_video_path),
                            "clip_cache_dir": str(Path(work_dir) / "vision" / "clips"),
                            "webcam_detector": bool(params.get("webcam_detector",
                                                               _flag("STREAMER_WEBCAM_DETECTOR", "false")))}
    crop.update({k: v for k, v in (params.get("crop") or {}).items() if k in ("layout", "webcam_detector")})
    # owner 2026-11-10: dynamic clips — pauses > 0.7 s cut out, a hook title for the first ~3 s
    crop["jump_cuts"] = bool(params.get("jump_cuts", settings.get("jumpCuts", _flag("STREAMER_JUMP_CUTS", "true"))))
    crop["jump_gap"] = float(params.get("jump_gap", 0.7))
    crop["hook_title"] = bool(params.get("hook_title", settings.get("hookTitle", _flag("STREAMER_HOOK_TITLE", "true"))))
    ad = sonya_ad_settings(params, settings)
    if ad:
        crop["sonya_ad"] = ad
    style = str(settings.get("subtitleStyle") or DEFAULT_STYLE).lower()
    if style not in ("none", "off"):
        preset, font = SITE_STYLES.get(style, SITE_STYLES[DEFAULT_STYLE])
        crop["subtitles"] = {"mode": "burn", "burn_supported": ffmpeg_has_ass(), "style": preset, "hook": False,
                             **({"font": font} if font else {})}
    return crop


SONYA_AD_ASSET = "sonya_ad_streamer.mp4"        # models dir: common/sonya_ad_streamer.mp4 (mode.yaml, optional)


def sonya_ad_settings(params: Dict[str, Any], settings: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """SONYA's own short ad inside each clip (owner 2026-11-10: «вставляй между делом рекламу SONYA», at ~40 %
    of the clip on a pause between phrases). On / off: params["sonya_ad"] > settings["sonyaAd"] (site toggle) >
    STREAMER_SONYA_AD (default off). Asset: params["sonya_ad_asset"] > STREAMER_SONYA_AD_ASSET > models dir.
    -> {"path", "at"} or None (off, or the file is missing: the clip is rendered without it, with a warning)."""
    on = params.get("sonya_ad", settings.get("sonyaAd"))
    if on is None:
        on = _flag("STREAMER_SONYA_AD", "false")
    if isinstance(on, dict):
        cfg, on = dict(on), on.get("enabled", True)
    else:
        cfg = {}
    if not on:
        return None
    path = cfg.get("asset") or params.get("sonya_ad_asset") or os.environ.get("STREAMER_SONYA_AD_ASSET")
    if not path:
        from scripts.shared.model_paths import model_file
        path = str(model_file("common", SONYA_AD_ASSET))
    try:
        at = min(0.6, max(0.25, float(cfg.get("at", params.get("sonya_ad_at", 0.4)))))
    except (TypeError, ValueError):
        at = 0.4
    return {"path": str(path), "at": at}


def _loudness(path: str, pieces: List[Any], ff: str) -> Optional[float]:
    """Integrated loudness (LUFS) of the clip's audio pieces (ebur128), None when unknown."""
    import re
    import subprocess
    vals = []
    for a, d in pieces[:6]:
        r = subprocess.run([ff, "-hide_banner", "-nostdin", "-ss", f"{a:.3f}", "-t", f"{d:.3f}", "-i", path, "-vn",
                            "-af", "ebur128", "-f", "null", "-"], capture_output=True, text=True, timeout=120)
        m = re.findall(r"I:\s+(-?[0-9.]+) LUFS", r.stderr or "")
        if m and float(m[-1]) > -70:
            vals.append((float(m[-1]), d))
    if not vals:
        return None
    import math
    tot = sum(d for _, d in vals)
    return round(10 * math.log10(sum(10 ** (v / 10) * d for v, d in vals) / tot), 2)


def _source_info(path: str) -> Dict[str, Any]:
    import subprocess
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "stream=codec_type,width,height,avg_frame_rate:format=duration", "-of", "json", path],
                         capture_output=True, text=True)
    d = json.loads(out.stdout or "{}")
    sts = d.get("streams") or []
    st = next((x for x in sts if x.get("codec_type") == "video"), sts[0] if sts else {})
    try:
        n, _, dd = str(st.get("avg_frame_rate") or "30/1").partition("/")
        fps = float(n) / float(dd or 1)
    except (ValueError, ZeroDivisionError):
        fps = 30.0
    fps = 30.0 if not 10 <= fps <= 61 else (60.0 if fps > 45 else (30.0 if fps > 27 else fps))
    return {"width": int(st.get("width") or 0), "height": int(st.get("height") or 0),
            "duration": float((d.get("format") or {}).get("duration") or 0.0), "fps": fps,
            "has_audio": any(x.get("codec_type") == "audio" for x in sts)}


def _chat_path(params: Dict[str, Any], input_video_path: str) -> Optional[str]:
    """Chat replay of the VOD: params["chat_path"], else a yt-dlp ``<name>.live_chat.json`` next to the source."""
    p = params.get("chat_path")
    if p and os.path.isfile(p):
        return p
    sib = Path(input_video_path).with_suffix(".live_chat.json")
    return str(sib) if sib.is_file() else None


def analyze(
    input_video_path: str,
    output_dir: str,
    params: Optional[Dict[str, Any]] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    """One expensive pass: ASR + stream signals → ranked moment segments (the editor's pool order).
    Does not render. Returns {"segments", "crop_hints", "warnings", "selection", "webcam_boxes_found",
    "active_speaker_segs"} (the last two kept for the batch API's analysis.json shape)."""
    params = dict(params or {})
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(params.get("work_dir") or (out.parent / "streamer_work"))
    crop = streamer_crop(params, input_video_path, str(work))

    from scripts.shared.enhancers.sonya_enhancer import enrich_video_for_mode
    enrichment = enrich_video_for_mode(input_video_path=input_video_path, output_dir=str(out), mode="streamer",
                                       params=dict(params, crop=crop), progress_callback=progress_callback)
    warnings: List[str] = list(enrichment.get("warnings") or [])
    words = enrichment.get("word_timestamps") or []
    if not words:
        raise RuntimeError("ASR_FAILED: no word timestamps - streamer moments need speech recognition ("
                           + "; ".join(warnings)[:300] + ")")
    from scripts.shared.glossary import apply as apply_glossary
    words = apply_glossary(words, params.get("glossary"))

    info = _source_info(input_video_path)
    n_sec = int(max(info["duration"], max(float(w.get("end", 0)) for w in words))) + 1

    from scripts.shared import stream_signals as ss
    from scripts.shared.crop.smart_crop_adapter import ffmpeg_bin
    ff = ffmpeg_bin()
    sig: Dict[str, Any] = {}
    try:
        a = ss.audio_signals(input_video_path, ff, str(work / "signals" / "audio.json"))
        sig.update(loud=a["loud"], pitch=a["pitch"])
    except Exception as exc:
        warnings.append(f"audio_signals_failed: {exc}")
    if progress_callback:
        progress_callback("analyzing", 0.80)
    sp = ss.speech_signals(words, n_sec)
    sig.update(rate=sp["rate"], react=sp["react"])
    if _flag("STREAMER_VISUAL_SIGNALS", "true") and params.get("visual_signals", True):
        try:
            hw = os.environ.get("STREAMER_HWACCEL") or None
            v = ss.visual_signals(input_video_path, ff, str(work / "signals" / "visual.json"), hwaccel=hw)
            sig.update(flash=v["flash"], cut=v["cut"])
        except Exception as exc:
            warnings.append(f"visual_signals_failed: {exc}")
    chat_meta: Dict[str, Any] = {"status": "no_chat"}
    cp = _chat_path(params, input_video_path)
    if cp:
        try:
            c = ss.chat_signals(ss.chat_times(cp), n_sec)
            chat_meta = {"status": "ok" if any(c["chat"]) else "too_few_messages", "messages": c["messages"]}
            if any(c["chat"]):
                sig["chat"] = c["chat"]
        except Exception as exc:
            chat_meta = {"status": f"error: {str(exc)[:120]}"}
    if progress_callback:
        progress_callback("analyzing", 0.88)

    from scripts.shared.stream_moments import MomentConfig, select_moments
    from scripts.shared.cut_profile import editor_available
    mc = dict(params.get("moments") or {})
    mc.setdefault("editor", bool(params.get("editor", editor_available())))
    mc.setdefault("context", str(params.get("stream_title") or params.get("title") or "")[:200])
    mc.setdefault("jump_gap", crop["jump_gap"])
    want = int(params.get("max_clips", 5))
    pool_n = int(params.get("pool_clips", max(want, 12)))
    segs, selection = select_moments(words, n_sec, sig, pool_n, MomentConfig(**mc), hits=sp["hits"])
    selection["chat"] = chat_meta
    if not segs:
        raise RuntimeError(f"NO_CLIPS: no complete moment found in the stream ({len(words)} words)")

    from scripts.shared.boundaries import BoundaryConfig, refine_segments
    bcfg = dict(params.get("boundaries") or {})
    bcfg.setdefault("max_duration", 62.0)
    refined = refine_segments(segs, words, BoundaryConfig(**bcfg))
    segments = []
    for s in refined:
        if s.get("boundary_status") == "dropped_overlap":
            continue
        st, du = float(s["start"]), float(s["duration"])
        clip_words = [w for w in words if float(w.get("end", 0)) > st - 0.5 and float(w.get("start", 0)) < st + du + 0.5]
        segments.append({"start_sec": st, "duration_sec": du, "score": float(s.get("score", 0.0)),
                         "source": s.get("source", "stream_moment"),
                         **{k: s[k] for k in ("candidate_id", "selection_reason", "hook", "keywords", "editor_score",
                                              "peak_t", "boundary_status", "tight_sec") if k in s},
                         "loud": [round(float(x), 2) for x in (sig.get("loud") or [])[int(st):int(st + du) + 2]],
                         "words": [{"word": w.get("word", w.get("text", "")), "start": w["start"], "end": w["end"]}
                                   for w in clip_words]})
    crop_hints = {"version": STREAMER_VERSION, "crop": crop, "width": info["width"], "height": info["height"],
                  "fps": info.get("fps", 30.0), "has_audio": info.get("has_audio", True),
                  "survey": _layout_survey(input_video_path, info["duration"], work, warnings)}
    return {"segments": segments, "crop_hints": crop_hints, "warnings": warnings, "selection": selection,
            "webcam_boxes_found": 0, "active_speaker_segs": 0}


def _layout_survey(video: str, duration: float, work: Path, warnings: List[str]) -> Dict[str, Any]:
    """Webcam rectangles over the whole VOD (stream_layout.survey), cached in the job's work dir."""
    from scripts.shared.crop import stream_layout as sl
    cache = work / "signals" / "layout_survey.json"
    try:
        d = json.loads(cache.read_text(encoding="utf-8"))
        if d.get("version") == sl.VERSION:
            return d
    except (OSError, ValueError):
        pass
    try:
        d = sl.survey(video, duration)
    except Exception as exc:                          # clips still find their own webcam
        warnings.append(f"layout_survey_failed: {str(exc)[:200]}")
        return {"rects": []}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(d), encoding="utf-8")
    return d


def _webcam_box(video: str, start: float, dur: float, W: int, H: int) -> Optional[List[int]]:
    """Webcam overlay box [x, y, w, h] from webcam_detector.pt on ~1 frame / 2 s of the clip (median of the
    best box per frame), or None (model missing / nothing steady found)."""
    from scripts.shared.model_paths import model_file
    mp = os.environ.get("STREAMER_WEBCAM_MODEL") or str(model_file("common", "webcam_detector.pt"))
    if not os.path.isfile(mp):
        return None
    import cv2
    from scripts.shared.vision.webcam_detector import WebcamDetector
    det = WebcamDetector(mp)
    det._load()
    if not det._model:
        return None
    cap = cv2.VideoCapture(video)
    boxes = []
    t = start + 0.5
    while t < start + dur:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            break
        r = det._model(frame, conf=0.4, verbose=False)[0]
        if r.boxes is not None and len(r.boxes):
            i = int(r.boxes.conf.argmax())
            x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
            if (x2 - x1) * (y2 - y1) < 0.35 * W * H:     # the whole frame is not a webcam overlay
                boxes.append((x1, y1, x2, y2))
        t += 2.0
    cap.release()
    if len(boxes) < 3:
        return None
    import numpy as np
    b = np.median(np.array(boxes), axis=0)
    spread = np.percentile(np.array(boxes), 90, axis=0) - np.percentile(np.array(boxes), 10, axis=0)
    if spread.max() > 0.06 * W:                         # a moving box is not an overlay
        return None
    x1, y1, x2, y2 = (int(v) for v in b)
    return [x1, y1, x2 - x1, y2 - y1]


def compose_one(
    input_video_path: str,
    output_path: str,
    segment: Dict[str, Any],
    crop_hints: Optional[Dict[str, Any]] = None,
) -> str:
    """Render ONE clip (no ASR, no whole-video analysis). Raises on failure. Updates ``segment`` in place
    with the crop / subtitle / render report (layout, webcam source, encoder, seconds)."""
    from modes.virality.runner import _clip_analysis
    from scripts.shared.crop.smart_crop_adapter import crop_track_v2, text_zones
    from scripts.shared.cut_profile import ensure_render_args
    from scripts.shared.subtitles import subtitles_for_clip

    hints = crop_hints or {}
    if hints.get("version") != STREAMER_VERSION:      # a v1 batch row: still render it the v2 way
        hints = {"crop": {"mode": "tracked", "analysis": "per_clip", "layout": "game"}}
    crop = dict(hints.get("crop") or {})
    W, H = int(hints.get("width") or 0), int(hints.get("height") or 0)
    if not W or not H:
        info = _source_info(input_video_path)
        W, H = info["width"], info["height"]
    start, dur = float(segment["start_sec"]), float(segment["duration_sec"])
    try:
        ensure_render_args()
    except Exception as exc:                          # libx264 default of render_tracked still works
        logger.warning("[streamer] encoder probe failed: %s", exc)

    clip_an, cache, err = _clip_analysis(crop, input_video_path, start, dur)
    track = None
    if crop.get("layout") == "game":
        from scripts.shared.crop import stream_layout as sl
        face = sl.webcam_face(clip_an, W, H)
        det_box = None
        if crop.get("webcam_detector"):
            det_box = _webcam_box(input_video_path, start, dur, W, H)
            segment["webcam_detector_box"] = det_box
        try:
            known = [r["rect"] for r in (hints.get("survey") or {}).get("rects") or [] if r.get("seen", 0) >= 2]
            lay = sl.analyze_layout(input_video_path, start, dur, face, W, H, detector_box=det_box, known_rects=known)
        except Exception as exc:                      # never lose the clip over the layout look
            logger.warning("[streamer] layout analysis failed: %s", exc)
            lay = {"webcam": None, "chat": [], "error": str(exc)[:200]}
        segment["layout_webcam"] = lay.get("webcam")
        segment["layout_chat"] = lay.get("chat")
        track = sl.stream_game_track(clip_an, lay, dur, width=W, height=H,
                                     survey_rects=(hints.get("survey") or {}).get("rects") or [])
    if track is None:
        track = crop_track_v2(clip_an, dur, width=W, height=H)
        track.setdefault("layout", "person")
    track.setdefault("webcam_source", None)
    if clip_an is None:
        track["fallback_reason"] = err.split(":", 1)[0] if err else "analysis_failed"
    zones = text_zones(clip_an, track) if track.get("layout") != "game" else []
    segment.update({"crop_version": track["crop_version"], "crop_layout": track.get("layout"),
                    "crop_status": track["crop_status"], "crop_fallback_reason": track.get("fallback_reason"),
                    "webcam_source": track.get("webcam_source"), "crop_cam": track.get("cam"),
                    "crop_game": track.get("game"), "webcam_rect": track.get("webcam_rect"),
                    "game_chat_overlap": track.get("game_overlap"),
                    "crop_cache": cache, "source_width": W, "source_height": H})
    # jump cuts: pauses > jump_gap s inside the clip are cut (a loud gap — a game event — stays)
    from scripts.shared import jumpcut as jc
    fps = float(hints.get("fps") or 30.0)
    words = segment.get("words") or []
    loud = [0.0] * int(start) + list(segment.get("loud") or [])
    if crop.get("jump_cuts", True) and words:
        pieces = jc.pieces(words, start, start + dur, max_gap=float(crop.get("jump_gap", 0.7)), loud=loud or None,
                           fps=fps)
    else:
        pieces = [(start, dur)]
    words_out = jc.remap_words(words, pieces)
    body = jc.total(pieces)
    segment.update(jump_cuts=len(pieces) - 1, body_sec=round(body, 3), cut_sec=round(dur - body, 3))
    insert = None
    ad = crop.get("sonya_ad")
    if ad:
        insert = _plan_ad(input_video_path, ad, words_out, pieces, segment)
    if insert:                                         # words after the ad move by its length
        words_out = [dict(w, start=w["start"] + (insert["dur"] if w["start"] >= insert["at"] else 0.0),
                          end=w["end"] + (insert["dur"] if w["start"] >= insert["at"] else 0.0)) for w in words_out]
    out_dur = body + (insert["dur"] if insert else 0.0)
    sub_path = None
    subs = crop.get("subtitles") or {}
    if subs.get("mode") == "burn":
        ass = Path(output_path).with_suffix(".ass")
        game = track.get("layout") == "game"
        centered = [[0.0, round(out_dur, 3)]] if game else None
        if not game:
            t2 = track if not crop.get("jump_cuts", True) else None
            centered = (((t2 or {}).get("split") or {}).get("intervals") or (t2 or {}).get("seam")) if t2 else None
        hook = (segment.get("hook") or None) if crop.get("hook_title", True) else None
        sm = subtitles_for_clip(words_out, 0.0, out_dur, ass, centered=centered, avoid=zones,
                                preset=subs.get("style"), keywords=segment.get("keywords"), hook=hook,
                                font=subs.get("font"), hook_top=70 if game else None)
        burn = sm.get("subtitle_status") == "ok" and bool(subs.get("burn_supported"))
        segment["subtitle_status"] = sm.get("subtitle_status")
        segment["subtitle_burn"] = "burned" if burn else "none"
        segment["subtitle_hook"] = sm.get("subtitle_hook")
        sub_path = str(ass) if burn else None
    from scripts.shared.crop.smart_crop_adapter import render_pieces
    render_pieces(input_video_path, output_path, pieces, track, start, fps=fps,
                  has_audio=bool(hints.get("has_audio", True)), insert=insert, subtitle_path=sub_path, report=segment)
    return output_path


def _plan_ad(video: str, ad: Dict[str, Any], words_out: List[Dict[str, Any]], pieces, segment: Dict[str, Any]
             ) -> Optional[Dict[str, Any]]:
    """Where and how loud the SONYA ad goes; None (with segment["sonya_ad"] = reason) when it cannot be clean."""
    from scripts.shared import jumpcut as jc
    from scripts.shared.promo import media_info
    from scripts.shared.crop.smart_crop_adapter import ffmpeg_bin
    path = ad.get("path") or ""
    if not os.path.isfile(path):
        segment["sonya_ad"] = "asset_missing"
        logger.warning("[streamer] SONYA ad asset not found: %s", path)
        return None
    ff = ffmpeg_bin()
    probe = os.path.join(os.path.dirname(ff), "ffprobe") if os.path.dirname(ff) else "ffprobe"
    try:
        info = media_info(path, probe)
    except ValueError as exc:
        segment["sonya_ad"] = f"asset_bad: {exc}"
        return None
    at = jc.pick_insert(words_out, pieces, at=float(ad.get("at", 0.4)))
    if at is None:
        segment["sonya_ad"] = "no_pause"
        return None
    clip_lufs = _loudness(video, list(pieces), ff)
    ad_lufs = _AD_LUFS.get(path)
    if ad_lufs is None:
        ad_lufs = _loudness(path, [(0.0, info["duration"])], ff)
        _AD_LUFS[path] = ad_lufs
    # the ad plays at the clip's own loudness (never louder than -14, never more than 6 dB quieter than itself)
    gain = 0.0
    if clip_lufs is not None and ad_lufs is not None:
        gain = max(-6.0, min(3.0, min(clip_lufs, -14.0) - ad_lufs))
    segment.update(sonya_ad="inserted", promo_insert_at=round(at, 3), promo_asset_sec=round(info["duration"], 3),
                   promo_integration="midroll", sonya_ad_gain_db=round(gain, 2), clip_lufs=clip_lufs, ad_lufs=ad_lufs)
    return {"path": path, "at": at, "dur": round(info["duration"], 3), "gain_db": gain, "has_audio": info["has_audio"]}


_AD_LUFS: Dict[str, Optional[float]] = {}


def run(
    input_video_path: str,
    output_dir: str,
    params: Optional[Dict[str, Any]] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    """Production entry point: analyze() once, render the top params["max_clips"] (default 5) moments,
    two at a time (one analyses on the CPU while the other encodes)."""
    params = dict(params or {})
    result = analyze(input_video_path, output_dir, params, progress_callback)
    segments, hints = result["segments"], result["crop_hints"]
    warnings = list(result["warnings"])
    max_clips = int(params.get("max_clips", 5))
    top = segments[:max_clips]                        # already in the editor's / local rank order
    out_dir = Path(output_dir)
    names = [str(out_dir / f"stream_clip_{i + 1:02d}.mp4") for i in range(len(top))]

    def one(i: int) -> Optional[str]:
        try:
            return compose_one(input_video_path, names[i], top[i], crop_hints=hints)
        except Exception as exc:
            logger.warning("[streamer] Clip %d failed: %s", i + 1, exc)
            warnings.append(f"clip_{i + 1}_failed: {str(exc)[:300]}")
            return None

    from concurrent.futures import ThreadPoolExecutor
    workers = max(1, int(os.environ.get("SONYA_CLIP_WORKERS", "2")))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        done = list(pool.map(one, range(len(top))))
    for n in names:                                   # subtitles are burned in: no side files in the upload
        for ext in (".ass", ".glass.json", ".glassmask.ass"):
            p = Path(n).with_suffix(ext)
            if p.exists():
                p.unlink()
    from scripts.shared.mode_output import write_selected_segments
    report = [{k: v for k, v in s.items() if k not in ("words", "loud")} for s in top]
    write_selected_segments(output_dir, "streamer", report, done)
    if progress_callback:
        progress_callback("done", 1.0)
    return {"clips": [p for p in done if p], "mode": "streamer", "version": STREAMER_VERSION, "beta": True,
            "selection": {k: v for k, v in result["selection"].items() if k != "params"},
            "webcam_boxes_found": 0, "active_speaker_segs": 0, "warnings": warnings}
