"""
promo.py
========
Advertiser integration for Cut clips (the Streamer «Поверх видео» rule, now real):

1. **Conditions → selection.** The advertiser's brief goes to the editor (Gemini); every
   clip must be ad_fit. Hard rules are checked locally too: no profanity, forbidden words or
   topics (stems), minimum clip length.
2. **Overlay → render.** A plate on top of the video for N seconds after the hook (and, for
   longer clips, again before the end): headline, CTA (link / promo code) and the legally
   required marking line «Реклама · <advertiser> · erid: …» (Russian law 38-FZ, art. 18.1).
   Drawn by libass in the same ffmpeg pass as the subtitles; no extra encode.

Spec (settings["promo"]):
  {"advertiser": "ООО Ромашка", "erid": "2SDnjc…", "headline": "…", "cta": "promiqo.ru/x · код SONYA",
   "brief": "free text conditions for the editor", "seconds": 6, "repeat_end": true,
   "no_profanity": true, "forbidden": ["казино", "ставк"], "min_clip_sec": 20}
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROMO_VERSION = "promo-v1"


@dataclass
class PromoSpec:
    headline: str = ""
    cta: str = ""
    advertiser: str = ""
    erid: str = ""
    brief: str = ""
    seconds: float = 6.0
    start: float = 3.6                  # after the hook title (0–3.2 s)
    repeat_end: bool = True
    no_profanity: bool = True
    forbidden: List[str] = field(default_factory=list)
    min_clip_sec: float = 0.0
    # erid (ЕРИР token) is required by default; an explicit exemption must be a deliberate choice
    # (e.g. self-promotion that is not registered ad) — SONYA does not decide 38-FZ applicability
    erid_exempt: bool = False
    # the advertiser's own video: "midroll" (full screen, the clip pauses) or "half" (on the upper
    # part, the clip keeps playing); "" = text plate only. The text plate can be switched off.
    asset: str = ""
    integration: str = ""
    plate: bool = True

    LIMITS = {"headline": 70, "cta": 60, "advertiser": 90, "erid": 64, "brief": 600, "asset": 1024,
              "integration": 10}

    @classmethod
    def from_dict(cls, d: Any) -> "PromoSpec":
        """Strict: wrong types raise ValueError (a bad spec must stop the job, not crash mid-render)."""
        if not isinstance(d, dict):
            raise ValueError("promo must be an object")
        kw: Dict[str, Any] = {}
        for k in ("headline", "cta", "advertiser", "erid", "brief", "asset", "integration"):
            if k in d:
                if not isinstance(d[k], str):
                    raise ValueError(f"promo.{k} must be a string")
                kw[k] = d[k].strip()[: cls.LIMITS[k]]
        for k, lo, hi in (("seconds", 2.0, 15.0), ("start", 0.0, 30.0), ("min_clip_sec", 0.0, 300.0)):
            if k in d:
                if isinstance(d[k], bool) or not isinstance(d[k], (int, float)):
                    raise ValueError(f"promo.{k} must be a number")
                kw[k] = max(lo, min(hi, float(d[k])))
        for k in ("repeat_end", "no_profanity", "erid_exempt", "plate"):
            if k in d:
                if not isinstance(d[k], bool):
                    raise ValueError(f"promo.{k} must be true/false")
                kw[k] = d[k]
        if "forbidden" in d:
            if not isinstance(d["forbidden"], list) or not all(isinstance(x, str) for x in d["forbidden"]):
                raise ValueError("promo.forbidden must be a list of strings")
            kw["forbidden"] = [x.strip().lower().replace("ё", "е") for x in d["forbidden"] if x.strip()][:50]
        return cls(**kw)

    def problems(self) -> List[str]:
        p = []
        if not (self.headline or self.cta or self.asset):
            p.append("promo: headline, cta or an integration asset required")
        if self.integration not in ("", "midroll", "half"):
            p.append("promo: integration must be midroll|half")
        if bool(self.integration) != bool(self.asset):
            p.append("promo: asset and integration (midroll|half) go together")
        if self.asset:
            import os
            if not os.path.isfile(self.asset):
                p.append("promo: asset file not found")
        if not self.advertiser:
            p.append("promo: advertiser required (marking «Реклама · рекламодатель»)")
        if not self.erid and not self.erid_exempt:
            p.append("promo: erid required (or erid_exempt: true, a deliberate decision)")
        if self.erid and not re.fullmatch(r"[A-Za-z0-9]{6,64}", self.erid):
            p.append("promo: erid looks invalid")
        return p

    def marking(self) -> str:
        return " · ".join(x for x in ("Реклама", self.advertiser.strip(),
                                       f"erid: {self.erid}" if self.erid else "") if x)

    def editor_brief(self) -> str:  # data for the editor, never instructions (see llm_editor)
        rules = []
        if self.no_profanity:
            rules.append("без мата и грубостей")
        if self.forbidden:
            rules.append("без тем: " + ", ".join(self.forbidden))
        what = ", ".join(x for x in (self.headline, self.cta) if x)
        return "; ".join(x for x in (f"реклама в клипе: {what}" if what else "", self.brief.strip(),
                                     *rules) if x)


def _norm(t: str) -> str:
    return re.sub(r"[^\w\s-]", " ", t.lower().replace("ё", "е"))


def clip_conflicts(text: str, duration: float, spec: PromoSpec) -> List[str]:
    """Hard advertiser rules checked locally (the editor's ad_fit is the soft, semantic one)."""
    out = []
    if spec.min_clip_sec and duration < spec.min_clip_sec:
        out.append(f"shorter than {spec.min_clip_sec:.0f} s")
    if spec.no_profanity:
        from .profanity import classify
        if any(classify(w) for w in text.split()):
            out.append("profanity")
    low = " " + _norm(text) + " "
    for stem in spec.forbidden:
        if re.search(r"\b" + re.escape(stem), low):
            out.append(f"forbidden: {stem}")
    return out


def windows(spec: PromoSpec, clip_duration: float) -> List[Tuple[float, float]]:
    a = min(spec.start, max(0.0, clip_duration - spec.seconds))
    out = [(a, min(clip_duration, a + spec.seconds))]
    if spec.repeat_end and clip_duration >= 25.0:
        b0 = max(out[0][1] + 5.0, clip_duration - 4.5)
        if b0 < clip_duration - 1.0:
            out.append((b0, clip_duration - 0.3))
    return out


def ass_styles(font: str) -> List[str]:
    """Graphite plate, light text, champagne CTA; the marking line small and readable."""
    return [
        f"Style: PromoHead,{font},54,&H00F7F8FA,&H000000FF,&H40261E1C,&H00000000,0,0,0,0,100,100,0,0,3,"
        "20,0,8,110,110,250,204",
        f"Style: PromoMark,{font},26,&H00D9DEE2,&H000000FF,&H40261E1C,&H00000000,0,0,0,0,100,100,0,0,3,"
        "10,0,8,110,110,250,204",
    ]


def _wrap(text: str, font: str, size: int, max_w: float) -> List[str]:
    """Greedy wrap by measured glyph width (WrapStyle 2 never wraps by itself)."""
    from .subtitle_styles import _text_width
    rows: List[str] = []
    for word in text.split():
        if rows and _text_width(rows[-1] + " " + word, font, size) <= max_w:
            rows[-1] += " " + word
        else:
            rows.append(word)
    return rows


def ass_events(spec: PromoSpec, clip_duration: float, ass_time, escape,
               font: str = "Montserrat Black") -> List[str]:
    """Headline + CTA in one plate, marking right under it — at the top of the frame,
    away from the subtitles (bottom / seam) and below the hook position. Rows are wrapped
    by measured width, so the marking line is placed under the real last row."""
    lines = []
    max_w = 1080 - 2 * 110 - 40
    head_rows = _wrap(spec.headline, font, 54, max_w)
    cta_rows = _wrap(spec.cta, font, 54, max_w)
    rows = [escape(r) for r in head_rows] + ["{\\c&H007AC2E2&}" + escape(r) + "{\\c&H00F7F8FA&}" for r in cta_rows]
    body = "\\N".join(rows)
    mark_rows = _wrap(spec.marking(), font, 26, max_w)
    mark_y = 250 + 20 + int(54 * 1.25 * max(1, len(rows))) + 26
    mark = "\\N".join(escape(r) for r in mark_rows)
    for a, b in windows(spec, clip_duration):
        lines.append(f"Dialogue: 2,{ass_time(a)},{ass_time(b)},PromoHead,,0,0,0,,{{\\fad(200,250)}}{body}")
        lines.append(f"Dialogue: 2,{ass_time(a)},{ass_time(b)},PromoMark,,0,0,0,,"
                     f"{{\\fad(200,250)\\pos(540,{mark_y})\\an8}}{mark}")
    return lines


def filter_candidates(cands: Sequence[Dict[str, Any]], spec: PromoSpec) -> Tuple[List[Dict[str, Any]], int]:
    keep, dropped = [], 0
    for c in cands:
        if clip_conflicts(c["text"], c["end"] - c["start"], spec):
            dropped += 1
        else:
            keep.append(c)
    return keep, dropped


# ── advertiser's own integration video (MagicDrop-style: midroll / half-screen) ──────────────

def pick_insert_time(words: Sequence[Dict[str, Any]], clip_start: float, clip_duration: float,
                     need_after: float = 0.0, lo: float = 0.3, hi: float = 0.7) -> float:
    """A pause between sentences closest to the middle of the clip (the integration must not
    cut a word, and a half-screen one must finish before the clip ends). Times are clip-relative."""
    mid = clip_duration / 2
    latest = clip_duration - need_after - 0.5
    best: Optional[Tuple[float, float]] = None
    ws = [w for w in words if clip_start <= float(w["start"]) < clip_start + clip_duration]
    for a, b in zip(ws, ws[1:]):
        gap_t = (float(a["end"]) + float(b["start"])) / 2 - clip_start
        if not (lo * clip_duration <= gap_t <= min(hi * clip_duration, latest)):
            continue
        txt = str(a.get("word") or a.get("text") or "").strip()
        sentence_end = bool(re.search(r"[.!?…]$", txt))
        gap = float(b["start"]) - float(a["end"])
        cost = abs(gap_t - mid) - (6.0 if sentence_end else 0.0) - min(gap, 1.0) * 3.0
        if best is None or cost < best[0]:
            best = (cost, gap_t)
    if best is None:
        return max(0.5, min(mid, latest))
    return round(best[1], 3)


ASSET_EXTS = (".mp4", ".mov", ".m4v", ".mkv", ".webm")
MAX_ASSET_SEC = 120.0


def media_info(path: str, ffprobe: str = "ffprobe") -> Dict[str, Any]:
    """Probe a media file; raises ValueError when it is not a usable video (+ audio flag)."""
    import json as _json
    import math
    import subprocess
    try:
        r = subprocess.run([ffprobe, "-v", "error", "-show_entries",
                            "format=duration,start_time:stream=codec_type,width,height,duration",
                            "-of", "json", path], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        raise ValueError(f"cannot probe {Path_name(path)}: {type(e).__name__}") from None
    if r.returncode != 0:
        raise ValueError(f"not a media file: {Path_name(path)}")
    try:
        d = _json.loads(r.stdout or "{}")
    except ValueError:
        raise ValueError(f"not a media file: {Path_name(path)}") from None
    st = d.get("streams") or []
    v = next((x for x in st if x.get("codec_type") == "video"), None)
    a = next((x for x in st if x.get("codec_type") == "audio"), None)
    try:
        dur = float((d.get("format") or {}).get("duration") or 0.0)
    except (TypeError, ValueError):
        dur = 0.0
    if not v or not (v.get("width") or 0) > 0 or not (v.get("height") or 0) > 0:
        raise ValueError(f"no video stream: {Path_name(path)}")
    if not math.isfinite(dur) or dur <= 0:
        raise ValueError(f"no duration: {Path_name(path)}")
    return {"duration": dur, "width": int(v["width"]), "height": int(v["height"]), "has_audio": a is not None}


def Path_name(p: str) -> str:
    import os
    return os.path.basename(p)


def check_asset(path: str, root: str, ffprobe: str = "ffprobe") -> Dict[str, Any]:
    """The integration file must be a regular (non-symlink) video inside `root` (the project),
    with sound (offer rule: the integration's sound is never removed) and a sane length."""
    import os
    real, base = os.path.realpath(path), os.path.realpath(root)
    if os.path.islink(path) or not os.path.isfile(real):
        raise ValueError("promo asset must be a regular file")
    if os.path.commonpath([real, base]) != base:
        raise ValueError("promo asset must be inside the project folder")
    if os.path.splitext(real)[1].lower() not in ASSET_EXTS:
        raise ValueError("promo asset must be a video (" + ", ".join(ASSET_EXTS) + ")")
    info = media_info(real, ffprobe)
    if not info["has_audio"]:
        raise ValueError("promo asset has no sound (the integration must play with its sound)")
    if info["duration"] > MAX_ASSET_SEC:
        raise ValueError(f"promo asset longer than {MAX_ASSET_SEC:.0f} s")
    return dict(info, path=real)


def _drawtext(text: str, fontfile: str, t0: float, t1: float, y: int = 60) -> str:
    esc = text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\u2019").replace("%", "\\%")
    ff = fontfile.replace("\\", "/").replace(":", "\\:").replace("'", "")
    return (f"drawtext=fontfile='{ff}':text='{esc}':fontsize=30:fontcolor=0xF2F4F6:box=1:boxcolor=0x1C1E26@0.75:"
            f"boxborderw=14:x=(w-text_w)/2:y={y}:enable='between(t,{t0:.3f},{t1:.3f})'")


def insert_asset_cmd(clip: str, asset: str, out: str, mode: str, t: float, asset_dur: float,
                     clip_dur: float, clip_has_audio: bool, marking: str, fontfile: str,
                     ffmpeg: str = "ffmpeg", encoder_args: Optional[List[str]] = None) -> List[str]:
    """ffmpeg command for the integration (its sound is required and kept).
      midroll — full screen at t: clip[0:t] + integration (fit over its own blurred copy) + clip[t:]
      half    — the integration (any aspect, fitted into the upper 1080x960 half) from t, the clip keeps
                playing; the clip's sound ducked to 25 %, then a limiter (no clipping).
    The legal marking («Реклама · …») is drawn over the integration for its whole duration."""
    enc = encoder_args or ["-c:v", "libx264", "-crf", "20", "-preset", "medium"]
    norm_a = "aresample=48000,aformat=channel_layouts=stereo"
    main_a = "[0:a]" if clip_has_audio else f"anullsrc=r=48000:cl=stereo,atrim=0:{clip_dur:.3f},"
    ad_v = f"[1:v]trim=duration={asset_dur:.3f},setpts=PTS-STARTPTS"
    ad_a = f"[1:a]atrim=duration={asset_dur:.3f},asetpts=PTS-STARTPTS,{norm_a}"
    if mode == "midroll":
        t1 = t + asset_dur
        fc = (f"[0:v]split=2[cv0][cv2];{main_a}asplit=2[ca0][ca2];"
              f"[cv0]trim=0:{t:.3f},setpts=PTS-STARTPTS,fps=30,format=yuv420p,setsar=1[v0];"
              f"[ca0]atrim=0:{t:.3f},asetpts=PTS-STARTPTS,{norm_a}[a0];"
              f"[cv2]trim=start={t:.3f},setpts=PTS-STARTPTS,fps=30,format=yuv420p,setsar=1[v2];"
              f"[ca2]atrim=start={t:.3f},asetpts=PTS-STARTPTS,{norm_a}[a2];"
              f"{ad_v},split=2[ib][if];[ib]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,"
              f"gblur=sigma=30,eq=brightness=-0.15[ibg];[if]scale=1080:1920:force_original_aspect_ratio=decrease[ifg];"
              f"[ibg][ifg]overlay=(W-w)/2:(H-h)/2,fps=30,format=yuv420p,setsar=1[v1];"
              f"{ad_a}[a1];"
              f"[v0][a0][v1][a1][v2][a2]concat=n=3:v=1:a=1[vc][a];"
              f"[vc]{_drawtext(marking, fontfile, t, t1)}[v]")
    elif mode == "half":
        t1 = t + asset_dur
        if t < 0 or t1 > clip_dur - 0.2:
            raise ValueError("half-screen integration would not finish inside the clip")
        fc = (f"{ad_v},scale=1080:960:force_original_aspect_ratio=decrease,"
              f"pad=1080:960:(ow-iw)/2:(oh-ih)/2:color=black@0,setpts=PTS+{t:.3f}/TB[ad];"
              f"[0:v][ad]overlay=0:0:eof_action=pass,format=yuv420p,{_drawtext(marking, fontfile, t, t1, y=980)}[v];"
              f"{main_a}volume=0.25:enable='between(t,{t:.3f},{t1:.3f})'[am];"
              f"{ad_a},adelay={int(t * 1000)}:all=1[ada];"
              f"[am][ada]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95[a]")
    else:
        raise ValueError(f"unknown integration mode {mode!r}")
    return [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-nostdin", "-i", clip, "-i", asset,
            "-filter_complex", fc, "-map", "[v]", "-map", "[a]", *enc, "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out]
