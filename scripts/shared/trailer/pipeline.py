"""
Trailer pipeline (one job): GigaAM words → script (Gemini ≤ 1.5 ₽) → narration (subtitles; the TTS slot is off) → edit plan →
render the main trailer (~60–100 s) + up to 2 shorts (30–45 s) → validation + contact sheets.

    run(video, words, out_dir, work_dir, params) -> {"clips": [...], "meta": {...}}

Never a silent half-result: a failed stage raises TrailerError with a code (TRAILER_SCRIPT_FAILED / TRAILER_VOICE_FAILED /
TRAILER_RENDER_FAILED); a failed short is skipped and reported (the main trailer stands).
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from . import edit, narration, render, script, shots as shots_mod

logger = logging.getLogger(__name__)

PIPELINE_VERSION = "trailer-prod-v2"
COLLAGE_SHOTS = 48


class TrailerError(RuntimeError):
    def __init__(self, code: str, msg: str):
        super().__init__(f"{code}: {msg}")
        self.code = code


def enabled(env: Optional[Dict[str, str]] = None) -> bool:
    """Worker kill switch: SONYA_TRAILER=off turns trailer jobs into a clear failure (no charge logic here — the API
    only lets admins create them until the owner approves)."""
    e = os.environ if env is None else env
    return e.get("SONYA_TRAILER", "on").strip().lower() not in ("off", "0", "false", "no")


def voice_mode(env: Optional[Dict[str, str]] = None) -> str:
    """SONYA_TRAILER_VOICE: «off» (default since 2026-11-17 — the owner picks the narrator voice later; the story is told
    by the subtitles over the film's own sound) or «qwen» (voice.py, the local TTS slot)."""
    e = os.environ if env is None else env
    v = e.get("SONYA_TRAILER_VOICE", "off").strip().lower()
    return "qwen" if v in ("qwen", "on", "1") else "off"


def voice_choice(params: Dict[str, Any], hero_gender: str) -> str:
    st = params.get("settings") if isinstance(params.get("settings"), dict) else {}
    v = str(params.get("trailer_voice") or st.get("trailerVoice") or st.get("voice") or "auto").lower()
    if v in ("male", "m", "мужской"):
        return "male"
    if v in ("female", "f", "женский"):
        return "female"
    return "female" if hero_gender == "f" else "male"


def run(video: str, words: Sequence[Dict[str, Any]], out_dir: str, work_dir: str, params: Optional[Dict[str, Any]] = None,
        *, chat: Optional[Callable[..., Any]] = None, transcribe: Optional[Callable[[str], Dict[str, Any]]] = None,
        log: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    params = params or {}
    log = log or (lambda m: logger.info("[trailer] %s", m))
    t_start = time.time()
    timings: Dict[str, float] = {}
    od, wd = Path(out_dir), Path(work_dir)
    od.mkdir(parents=True, exist_ok=True)
    wd.mkdir(parents=True, exist_ok=True)
    ffmpeg = os.environ.get("SONYA_FFMPEG") or "ffmpeg"
    info = render.probe(video)
    film_sec = info["duration"]
    src_w, src_h = shots_mod.probe_dims(video)
    if not src_w:
        raise TrailerError("TRAILER_RENDER_FAILED", "cannot read the video")

    # 1. script (frames_fn scans the chosen storyline and builds the collages the writer sees)
    state: Dict[str, Any] = {}

    def frames_fn(a: float, b: float):
        t = time.time()
        data = shots_mod.scan(video, a, b, ffmpeg=ffmpeg, cache=str(wd / "scan.json"))
        sc = shots_mod.Scan(data)
        src_shots = shots_mod.source_shots(sc, a, b)
        chosen = shots_mod.pick_for_collage(src_shots, COLLAGE_SHOTS)
        imgs = shots_mod.collages(video, chosen, ffmpeg=ffmpeg)
        state.update(scan=sc, src_shots=src_shots, k_shots={s["id"]: s for s in chosen})
        timings["scan"] = round(time.time() - t, 1)
        log(f"scan {b - a:.0f} s: {len(src_shots)} source shots, {len(chosen)} on {len(imgs)} collages, {timings['scan']} s")
        return [s["id"] for s in chosen], imgs

    t = time.time()
    try:
        sres = script.build_script(words, film_sec, frames_fn=frames_fn, chat=chat, title=str(params.get("title") or ""),
                                   log=log)
    except script.TrailerScriptError as exc:
        raise TrailerError("TRAILER_SCRIPT_FAILED", str(exc)) from exc
    timings["script"] = round(time.time() - t - timings.get("scan", 0.0), 1)
    sc = state["scan"]
    scr = sres["script"]
    segs = scr["segments"]
    by_id = {s["id"]: s for s in sres["sents"]}
    (wd / "script.json").write_text(json.dumps({k: v for k, v in sres.items() if k != "sents"}, ensure_ascii=False, indent=1))

    # 2. narration: the voice slot (subtitles-only until the owner picks a TTS)
    gender = voice_choice(params, sres["pick"]["hero_gender"])
    t = time.time()
    if voice_mode() == "off":
        vres = narration.reading([{"n": s["n"], "text": s["text"]} for s in segs])
        log(f"narration: subtitles only (no voice), {vres['tts']['voice_sec']} s, {vres['tts']['pace']} words/s")
    else:
        from . import voice
        try:
            vres = voice.synthesize([{"n": s["n"], "text": s["text"]} for s in segs], gender, str(wd / "voice"),
                                    ffmpeg=ffmpeg, transcribe=transcribe, log=log)
        except voice.VoiceError as exc:
            raise TrailerError("TRAILER_VOICE_FAILED", str(exc)) from exc
    timings["voice"] = round(time.time() - t, 1)

    # 3. edit + render the main trailer
    aspect = str(params.get("trailer_aspect") or "3:4")
    g = render.geometry(src_w, src_h, aspect)
    hook_sid = sres["pick"].get("hook")
    hook_t = by_id[hook_sid]["start"] if hook_sid in by_id else None
    tl = edit.timeline(segs, vres["segments"])
    t = time.time()
    plan = edit.plan(tl, sc, by_id, state["k_shots"], sres["episode"], g["cw"], hook_t)
    from ..cut_profile import encoder_args, encoder_family
    fam = "cpu"
    try:
        fam = encoder_family(ffmpeg)
    except Exception:                                       # noqa: BLE001
        pass
    piece_enc = (["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "17", "-b:v", "0", "-pix_fmt", "yuv420p"]
                 if fam == "nvenc" else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p"])
    final_enc = encoder_args(fam, "8M")
    preset, font = _sub_style(params)
    outputs: List[Dict[str, Any]] = []
    try:
        main = _render_one(video, tl, plan["shots"], g, scr["title"], preset, font, str(od / "trailer_main.mp4"), wd,
                           ffmpeg=ffmpeg, piece_enc=piece_enc, final_enc=final_enc, log=log)
    except Exception as exc:                                # noqa: BLE001
        raise TrailerError("TRAILER_RENDER_FAILED", f"{type(exc).__name__}: {exc}") from exc
    main.update(kind="main", segments=[s["n"] for s in segs], plan=plan["stats"], warnings=plan["warnings"])
    outputs.append(main)
    timings["render_main"] = round(time.time() - t, 1)

    # 4. shorts (the same voice and shots, other segment sets)
    t = time.time()
    n_shorts = int(params.get("trailer_shorts", 2))
    for k, keep in enumerate((scr.get("shorts") or [])[:max(0, n_shorts)]):
        try:
            sub_tl, sub_shots = edit.sub_timeline(tl, keep, plan["shots"])
            dur = sub_tl["total"]
            if not (24.0 <= dur <= 52.0):
                log(f"short {k + 1}: {dur:.1f} s outside 24–52 s, skipped")
                continue
            title = (scr.get("short_titles") or [""] * 3)[k] if k < len(scr.get("short_titles") or []) else scr["title"]
            o = _render_one(video, sub_tl, sub_shots, g, title or scr["title"], preset, font, str(od / f"trailer_short_{k + 1}.mp4"),
                            wd, ffmpeg=ffmpeg, piece_enc=piece_enc, final_enc=final_enc, log=log)
            o.update(kind="short", segments=list(keep))
            outputs.append(o)
        except Exception as exc:                            # noqa: BLE001 - a short never kills the trailer
            log(f"short {k + 1} failed: {type(exc).__name__}: {exc}")
    timings["render_shorts"] = round(time.time() - t, 1)
    timings["total"] = round(time.time() - t_start, 1)
    meta = {"version": PIPELINE_VERSION, "film_sec": round(film_sec, 1), "episode": sres["episode"],
            "pick": {k: sres["pick"].get(k) for k in ("hero", "hero_gender", "line", "genre", "why", "hook", "peak")},
            "title": scr["title"], "segments": [{k: s.get(k) for k in ("n", "fn", "text", "src", "shots", "gag", "fact_fixed")}
                                                for s in segs],
            "llm": sres["llm"], "voice": vres["tts"], "outputs": outputs, "timings": timings, "aspect": aspect,
            "subtitles": preset}
    (od / "trailer.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str))
    log(f"done: {len(outputs)} video(s), Gemini {sres['llm']['rub']} ₽, {timings}")
    return {"clips": [o["path"] for o in outputs], "meta": meta}


def _sub_style(params: Dict[str, Any]):
    try:
        from ..cut_profile import DEFAULT_STYLE, SITE_STYLES, site_subtitle_style
        st = params.get("settings") if isinstance(params.get("settings"), dict) else {}
        name = site_subtitle_style(st)
        if name in ("none", "off"):
            name = DEFAULT_STYLE                    # the trailer always has subtitles (the format needs them)
        preset, font = SITE_STYLES.get(name, SITE_STYLES[DEFAULT_STYLE])
        return preset, font
    except Exception:                               # noqa: BLE001
        return "highlight", None


def _render_one(video: str, tl: Dict[str, Any], shots: List[Dict[str, Any]], g: Dict[str, int], title: str, preset: str,
                font: Optional[str], out: str, wd: Path, *, ffmpeg: str, piece_enc, final_enc, log) -> Dict[str, Any]:
    t = time.time()
    pieces = render.render_shots(video, shots, g, str(wd / "pieces"), ffmpeg=ffmpeg, enc=piece_enc,
                                 workers=int(os.environ.get("SONYA_TRAILER_RENDER_WORKERS", "3")))
    stem = Path(out).stem
    wav = str(wd / f"{stem}.wav")
    mixinfo = render.mix(video, tl, shots, wav, ffmpeg=ffmpeg)
    ass_text, sub_info = render.fit_ass(render.ass(tl, title, preset, font))
    ass_path = wd / f"{stem}.ass"
    ass_path.write_text(ass_text, encoding="utf-8")
    segs = tl["segments"]
    flash = (segs[1]["t0"] - 0.3) if len(segs) > 1 and segs[0]["fn"] == "hook" else None
    enc = render.final(pieces, tl, wav, str(ass_path), out, str(wd), ffmpeg=ffmpeg, enc=final_enc, flash_at=flash)
    pr = render.probe(out)
    sheet = render.contact_sheet(out, str(wd / f"{stem}_sheet.jpg"), ffmpeg=ffmpeg)
    words = sum(len(s["words"]) for s in segs)
    res = {"path": out, "duration": round(pr["duration"], 2), "expected": tl["total"], "shots": len(shots),
           "words": words, "pace_voice": round(words / max(0.1, sum(s["t1"] - s["t0"] for s in segs)), 2),
           "pace_total": round(words / max(0.1, tl["voice_end"]), 2), "mix": mixinfo, "subs": sub_info, "encode": enc,
           "sheet": sheet, "render_s": round(time.time() - t, 1)}
    if abs(pr["duration"] - tl["total"]) > 0.25 or not pr["audio"] or pr["width"] != render.W:
        raise RuntimeError(f"output check failed: {pr} vs {tl['total']}")
    log(f"{stem}: {res['duration']} s, {len(shots)} shots, {res['pace_voice']} w/s, subs {sub_info}, {res['render_s']} s")
    return res
