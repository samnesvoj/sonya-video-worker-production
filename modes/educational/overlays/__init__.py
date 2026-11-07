"""
Educational v2 overlays: formula, fact, term, diagram and chart cards tied to the speech of a clip.

Off unless a job passes params["edu_overlays"]; this flag does not enable the educational mode (scripts/mode_policy.py
decides that, and it stays opt-in). Any failure here leaves the plain clip in place and is reported as a warning.

    apply_overlays(base_clip, out_clip, words, clip_start, clip_duration, options) -> report dict

options: preset (minimal|balanced|deep), enabled_types, max_per_min, theme, intensity, size, position, burn_subtitles,
manifest (path of an edited manifest: replaces the planner), llm_opt_in, ledger_path, video_id, author_materials,
verified_sources, cache_dir, formula_backend, work_dir, providers/transport (tests).
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import budget as bg
from . import cards as cd
from . import compose as cp
from . import evidence as ev
from . import formula as fm
from . import manifest as mf
from . import placement as pl
from . import planner

logger = logging.getLogger(__name__)

SIZE_STEPS = (1.0, 0.86)        # try a smaller card before giving up; below this it stops being readable on a phone
BURNED_SUBS_SHARE = 0.25        # share of sampled frames with burned-in captions in our subtitle band
DEFAULTS = {"preset": "balanced", "max_per_min": 2.0, "theme": "modern-tech", "intensity": "normal", "burn_subtitles": True}


def _subtitle_style(f: pl.Frame):
    from scripts.shared.subtitles import SubtitleStyle
    if f.aspect == "9:16":
        return SubtitleStyle(font="Montserrat ExtraBold", version="style-edu-9x16")
    k = f.short / 1080.0
    return SubtitleStyle(font="Montserrat ExtraBold", version=f"style-edu-{f.aspect}", size=int(54 * k), outline=2.5, margin_v=int(46 * k),
                         margin_lr=int(90 * k), max_chars_per_line=46 if f.aspect == "16:9" else 30)


def _join_punctuation(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """ASR sometimes emits «—» or «,» as a word of its own: a subtitle line must not end up holding only that."""
    out: List[Dict[str, Any]] = []
    for w in words:
        text = str(w.get("word", "")).strip()
        if out and len(text) > 1 and text[0] == "-" and text[1].isalpha():      # Whisper writes «что -то»
            out[-1] = {**out[-1], "word": str(out[-1]["word"]).rstrip() + text, "end": w.get("end", out[-1].get("end"))}
            continue
        if out and text and not any(ch.isalnum() for ch in text):
            if text not in ("-", "–", "—"):
                out[-1] = {**out[-1], "word": str(out[-1]["word"]).rstrip() + text}
            continue
        out.append(w)
    return out


def _write_subtitles(words: List[Dict[str, Any]], duration: float, f: pl.Frame, path: Path) -> Optional[str]:
    from scripts.shared.subtitles import build_cues, clip_words, to_ass, write_atomic
    style = _subtitle_style(f)
    ws = clip_words(_join_punctuation(words), 0.0, duration)
    if not ws:
        return None
    write_atomic(path, to_ass(build_cues(ws, style, duration), style, play_res=(f.width, f.height)))
    return str(path)


def _subtitle_top(f: pl.Frame, burn: bool) -> int:
    style = _subtitle_style(f)
    band = int(2 * style.size * 1.25) + int(16 * f.short / 1080)
    bottom = f.height - style.margin_v
    return bottom - band if burn else bottom


def _load_items(options: Dict[str, Any], words: List[Dict[str, Any]], report: Dict[str, Any]) -> List[Dict[str, Any]]:
    path = options.get("manifest")
    if path:
        with open(path, encoding="utf-8") as fh:
            report["source"] = "edited_manifest"
            return list(json.load(fh).get("overlays", []))
    ledger = bg.Ledger(options.get("ledger_path"))
    res = planner.propose(words, video_id=str(options.get("video_id") or "video"), ledger=ledger,
                          opt_in=bool(options.get("llm_opt_in")), providers=options.get("providers"),
                          transport=options.get("transport"))
    report["source"] = "llm"
    report["llm"] = res["telemetry"]
    if res["error"]:
        report["warnings"].append(f"planner:{res['error']}")
    return res["items"]


def apply_overlays(base_clip: str, out_clip: str, words_abs: List[Dict[str, Any]], clip_start: float, clip_duration: float,
                   options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Writes out_clip and <out_clip stem>.overlays.json (the editable manifest with statuses and a profile)."""
    t_all = time.monotonic()
    opts = {**DEFAULTS, **(options or {})}
    prof: Dict[str, float] = {}
    report: Dict[str, Any] = {"warnings": []}
    info = cp.probe(base_clip)
    f = pl.Frame(info["width"], info["height"])
    duration = min(float(clip_duration), info["duration"])
    words = ev.clip_local_words(words_abs, float(clip_start), float(clip_start) + duration)

    t = time.monotonic()
    raw = _load_items(opts, words, report)
    prof["planning_sec"] = time.monotonic() - t

    for it in raw:
        style = dict(it.get("style") or {})
        style.setdefault("theme", opts["theme"])
        style.setdefault("intensity", opts["intensity"])
        if opts.get("size"):
            style.setdefault("size", opts["size"])
        it["style"] = style
        if opts.get("position"):
            it.setdefault("position", opts["position"])
    t = time.monotonic()
    items = mf.validate(raw, words, duration, preset=str(opts["preset"]), enabled_types=opts.get("enabled_types"),
                        author_materials=opts.get("author_materials") or (),
                        verified_sources=opts.get("verified_sources") or ())
    items = mf.apply_density(items, duration, max_per_min=float(opts["max_per_min"]))
    prof["validation_sec"] = time.monotonic() - t

    burn = bool(opts["burn_subtitles"])
    accepted = [i for i in items if i.get("status") == "ok"]
    work = Path(opts.get("work_dir") or (str(out_clip) + ".work"))
    work.mkdir(parents=True, exist_ok=True)
    analysis: Dict[str, Any] = {}
    if accepted:
        t = time.monotonic()
        from scripts.shared.crop.clip_analysis import analyze_clip
        analysis = analyze_clip(base_clip, 0.0, duration, face_fps=5.0)
        prof["frame_analysis_sec"] = time.monotonic() - t
    if burn and pl.burned_caption_share(base_clip, _subtitle_top(f, True), f.height, duration) >= BURNED_SUBS_SHARE:
        burn = False            # the source already has captions where ours would go: two layers of text are unreadable
        report["subtitles_skipped"] = "source_has_burned_text"
    layers: List[cp.PlacedLayer] = []
    prof["card_render_sec"] = prof["placement_sec"] = 0.0
    cache_hits = 0
    if accepted:
        # burned-in captions occupy the same band ours would: it stays reserved even when ours are switched off
        sub_top = _subtitle_top(f, burn or "subtitles_skipped" in report)
        for it in sorted(accepted, key=lambda x: x["start_ms"]):
            start, end = pl.snap_to_cuts(it["start_ms"] / 1000.0, it["end_ms"] / 1000.0, analysis.get("cuts", []))
            t = time.monotonic()
            edges = pl.edge_density_maps(base_clip, start, end)
            prof["placement_sec"] += time.monotonic() - t
            placed = None
            why: Dict[str, Any] = {}
            for k in SIZE_STEPS:
                sized = {**it, "style": {**it["style"], "size": float(it["style"].get("size", 1.0)) * k}}
                t = time.monotonic()
                try:
                    card, hit = cd.render_cached(sized, pl.card_width(f), f.short, pl.max_card_height(f),
                                                 opts.get("cache_dir"), opts.get("formula_backend"), pl.text_scale(f))
                except (cd.CardOverflow, fm.FormulaError) as e:
                    why = {"render": str(e)}
                    prof["card_render_sec"] += time.monotonic() - t
                    continue
                cache_hits += int(hit)
                prof["card_render_sec"] += time.monotonic() - t
                t = time.monotonic()
                rect, why = pl.choose(f, card.width, card.height, sub_top, analysis, start, end, edges=edges,
                                      preferred=str(it.get("position") or "auto"))
                prof["placement_sec"] += time.monotonic() - t
                if rect:
                    placed = (card, rect)
                    break
            it["placement"] = why
            if not placed:
                it["status"], it["reason"] = "skipped", "no_safe_zone"
                continue
            card, rect = placed
            it["start_ms"], it["end_ms"] = int(start * 1000), int(end * 1000)
            it["rect"] = list(rect)
            slide = cp.slide_px(f.short, it["style"].get("intensity", "normal"))
            for n, layer in enumerate(card.layers):
                at = max(start, layer.at_ms / 1000.0) if n else start
                if at >= end - 0.4:
                    at = max(start, end - 0.8)
                png = work / f"{it['id']}_{n:02d}.png"
                layer.image.save(png)
                layers.append(cp.PlacedLayer(str(png), rect[0], rect[1], at, end, slide if n == 0 else 0))
    drawn = [i for i in items if i.get("status") == "ok"]

    t = time.monotonic()
    subtitle = _write_subtitles(words, duration, f, work / "subs.ass") if burn else None
    from scripts.shared.subtitle_styles import FONTS_DIR
    if layers or subtitle:
        cp.composite(base_clip, out_clip, layers, subtitle_path=subtitle, fonts_dir=str(FONTS_DIR))
    elif str(base_clip) != str(out_clip):
        shutil.copyfile(base_clip, out_clip)
    prof["composite_encode_sec"] = time.monotonic() - t
    prof["total_sec"] = time.monotonic() - t_all
    report.update(drawn=len(drawn), rejected=sum(1 for i in items if i.get("status") == "rejected"),
                  review=sum(1 for i in items if i.get("status") == "review"),
                  skipped=sum(1 for i in items if i.get("status") == "skipped"), cache_hits=cache_hits,
                  profile={k: round(v, 3) for k, v in prof.items()}, subtitles=bool(subtitle))
    doc = mf.document(items, clip_duration=duration, preset=str(opts["preset"]), aspect=f.aspect, report=report)
    side = Path(out_clip).with_suffix(".overlays.json")
    side.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    report["manifest"] = str(side)
    if not opts.get("keep_work"):
        shutil.rmtree(work, ignore_errors=True)
    logger.info("[edu-overlays] drawn=%d rejected=%d review=%d skipped=%d", report["drawn"], report["rejected"],
                report["review"], report["skipped"])    # counts only: no transcript or card text in logs
    return report
