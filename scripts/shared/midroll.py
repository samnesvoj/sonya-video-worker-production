"""SONYA «Мидролл» (clients 2026-11-22: «не только баннер, а ещё и мидролл»): the user's own ad video inside every clip.

The user uploads an ad (any format / orientation, 1-30 s) on the site (Пресеты → Эффекты → Реклама); the API stores it in S3
under ``users/<user_id>/midroll/`` and the job carries ``params["midroll"]``::

    {"key": "users/<uid>/midroll/<sha>.mp4", "place": "mid" | "pre" | "post",
     "at": "auto" | "after", "after_s": 20, "every": 1 | 2, "fit": "auto" | "blur" | "fit" | "fill",
     "max_total": 0 | 60 | 90, "dur": 8.4, "name": "promo.mp4"}

The worker calls ``apply_to_outputs`` after the mode rendered its clips (and after the mini banner), before previews /
upload. Mid-roll point: a real pause between two spoken words, from the clip's own burned-in ASS (one Dialogue per spoken
word = exact word times), nearest to the middle (``at=auto``) or the first good pause after ``after_s`` seconds; a sentence
end (. ! ? …) wins over a comma pause; never inside a word, never in the first / last 20 % (auto). No ASS → the quietest
moment of the middle third (audio RMS). The ad goes in with a short dissolve (video + audio cross-fade inside the pause).

Subtitles are burned into the clip's own frames, so they are simply not there while the ad plays. The ad is scaled to the
clip's frame (9:16): same shape → cropped to fill; other shape → centred on a blurred copy of itself (``blur``), or on a
dark plate (``fit``); its audio gets one linear gain to the clips' −14 LUFS (scripts/shared/loudness.py) + the limiter, a
silent ad gets a silent track. ``every=2`` → clips 1, 3, 5…; ``max_total`` → a clip that would get longer than that
keeps no ad (reported). Nothing here can fail a job: any problem → the clip goes out as rendered.
Kill switch for every job: SONYA_MIDROLL=off.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

VERSION = "midroll-v1"
MODES = ("virality", "hooks", "beep", "educational", "streamer")
PLACES = ("mid", "pre", "post")
FITS = ("auto", "blur", "fit", "fill")
KEY_RE = re.compile(r"^users/[0-9A-Za-z_-]{1,64}/midroll/[0-9a-f]{12,64}\.(mp4|mov|m4v|webm|mkv)$")
AD_MIN_S, AD_MAX_S = 1.0, 30.0
SHAPE_TOL = 0.08          # ad aspect within 8 % of the clip's → crop to fill (no visible blur border)
XFADE_MAX = 0.25          # dissolve into / out of the ad
XFADE_MIN = 0.08
EDGE_XFADE = 0.20         # pre / post roll dissolve
MIN_PAUSE = 0.18          # a gap between two words shorter than this is not a pause (speech rhythm)
AUTO_ZONE = (0.2, 0.8)    # auto: never in the first / last 20 % of the clip
SENT_END = re.compile(r"[.!?…]+[»\"')\]]*$")
COMMA_END = re.compile(r"[,;:—–-]+[»\"')\]]*$")
POSTER_AT: Dict[str, float] = {}      # clip path -> poster time (a pre-roll moves it past the ad)


def _off(v: Any) -> bool:
    return str(v).strip().lower() in ("0", "off", "false", "no", "none", "")


def config(params: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """params["midroll"] -> normalised settings or None (off / no ad / bad key / kill switch)."""
    if str(os.environ.get("SONYA_MIDROLL", "")).strip().lower() in ("0", "off", "false", "no"):
        return None
    raw = (params or {}).get("midroll")
    if not isinstance(raw, dict) or raw.get("on") is False:
        return None
    key = str(raw.get("key") or "")
    if not KEY_RE.match(key):
        return None
    place = str(raw.get("place") or "mid").lower()
    at = str(raw.get("at") or "auto").lower()
    fit = str(raw.get("fit") or "auto").lower()

    def num(k, lo, hi, default):
        try:
            return max(lo, min(hi, float(raw.get(k, default))))
        except (TypeError, ValueError):
            return default
    return {
        "key": key,
        "place": place if place in PLACES else "mid",
        "at": at if at in ("auto", "after") else "auto",
        "after_s": num("after_s", 3.0, 120.0, 20.0),
        "every": 2 if str(raw.get("every")) == "2" else 1,
        "fit": fit if fit in FITS else "auto",
        "max_total": int(num("max_total", 0, 600, 0)),
    }


# ── word times from the clip's ASS ──────────────────────────────────────────

def _secs(t: str) -> float:
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


_TAG = re.compile(r"\{[^}]*\}")


def _active_word(text: str) -> Optional[str]:
    """The spoken word of a word-cue (scripts/shared/subtitle_styles): the text right after the first override block that
    scales it (\\fscx106/112…) or recolours + resets scale (grow=100 styles: {\\c&H…&\\fscx100\\fscy100}WORD); per-letter
    colour tags of the gradient look are skipped. None = a plain phrase line."""
    for m in _TAG.finditer(text):
        blk = m.group(0)
        g = re.search(r"\\fscx(\d+)", blk)
        if not g or "\\t(" in blk:
            continue
        if int(g.group(1)) == 100 and "\\c" not in blk:
            continue
        rest = _TAG.sub("", text[m.end():]).replace("\\N", " ").split()
        if rest:
            return rest[0]
    return None


def words_from_ass(ass_text: str) -> List[Tuple[float, float, str, bool]]:
    """[(start, end, word, phrase)] of the spoken words: one Dialogue per word in the subtitle styles of
    scripts/shared/subtitle_styles (the active word is scaled up). Lines without that mark (a hook title, a promo plate,
    a plain one-cue-per-phrase style) count as one "word" with the phrase's last word (phrase=True): their gaps are still
    pauses, and the pause is looked for only in the phrase's last 1.2 s."""
    style_col = None
    fmt: List[str] = []
    out = []
    for ln in ass_text.splitlines():
        if ln.startswith("Format:") and "Text" in ln and ln.find("Start") > 0:
            fmt = [x.strip() for x in ln.split(":", 1)[1].split(",")]
            style_col = fmt.index("Style") if "Style" in fmt else 3
            continue
        if not ln.startswith("Dialogue:"):
            continue
        f = ln.split(":", 1)[1].split(",", 9)
        if len(f) < 10:
            continue
        style = f[style_col if style_col is not None else 3].strip().lower()
        text = f[9]
        if r"\p1" in text or style in ("hook", "title", "promo", "teaser", "part", "comment"):
            continue
        try:
            t0, t1 = _secs(f[1].strip()), _secs(f[2].strip())
        except ValueError:
            continue
        word = _active_word(text)
        phrase = word is None
        if phrase:
            plain = _TAG.sub("", text).replace("\\N", " ").split()
            word = plain[-1] if plain else ""
        out.append((t0, t1, word, phrase))
    out.sort()
    # merge: the same word can span several cues (karaoke repaint); keep word boundaries only
    merged: List[Tuple[float, float, str, bool]] = []
    for t0, t1, w, ph in out:
        if merged and t0 - merged[-1][1] < 0.02 and w == merged[-1][2] and ph == merged[-1][3]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], t1), w, ph)
        else:
            merged.append((t0, t1, w, ph))
    return merged


def rms_curve(clip: str, ffmpeg: str = "ffmpeg", step: float = 0.05) -> List[Tuple[float, float]]:
    """[(t, dB)] audio RMS of the whole clip in ``step`` windows (mono 16 kHz). [] = no audio / ffmpeg failed."""
    n = int(16000 * step)
    cmd = [ffmpeg, "-hide_banner", "-nostdin", "-nostats", "-i", clip, "-vn",
           "-af", f"aresample=16000,pan=mono|c0=0.5*c0+0.5*c1,asetnsamples={n},astats=metadata=1:reset=1,"
                  "ametadata=print:key=lavfi.astats.Overall.RMS_level:file=-", "-f", "null", "-"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except Exception:                                            # noqa: BLE001
        return []
    if r.returncode != 0:      # mono source: pan with c1 fails -> plain downmix
        cmd[cmd.index("-af") + 1] = cmd[cmd.index("-af") + 1].replace("pan=mono|c0=0.5*c0+0.5*c1,", "aformat=channel_layouts=mono,")
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        except Exception:                                        # noqa: BLE001
            return []
    out: List[Tuple[float, float]] = []
    t = None
    for ln in (r.stdout or "").splitlines():
        m = re.search(r"pts_time:([\d.]+)", ln)
        if m:
            t = float(m.group(1))
            continue
        m = re.search(r"RMS_level=(-?[\d.]+|-inf)", ln)
        if m and t is not None:
            out.append((t, -100.0 if m.group(1) == "-inf" else max(-100.0, float(m.group(1)))))
            t = None
    return out


def _quiet_threshold(curve: Sequence[Tuple[float, float]], words: Sequence[Tuple[float, float, str]]) -> float:
    """dB below which a window counts as a pause: the speech level (median of the loud half of the word windows) - 15 dB,
    never above -24 dB (a loud music bed is not a pause) nor below -55 dB."""
    spans = [(w[0], w[1]) for w in words]
    lv = sorted(db for t, db in curve if any(s <= t < e for s, e in spans)) or sorted(db for _, db in curve)
    if not lv:
        return -40.0
    speech = lv[int(len(lv) * 0.75)]
    return max(-55.0, min(-24.0, speech - 15.0))


def pauses(words: Sequence[Tuple[float, float, str]], duration: float,
           curve: Sequence[Tuple[float, float]] = ()) -> List[Dict[str, Any]]:
    """Real pauses between consecutive words: [{"a", "b", "gap", "kind"}] (a = speech stops, b = the next word starts).
    The ASS gives exact word STARTS (a word's cue is held until the next one, so its end is not the word's end); with the
    audio curve the pause is the longest quiet run between the previous word's start (+ 0.12 s) and the next word's
    start; without audio only the ASS gaps count. kind from the punctuation of the word before: sentence | comma | plain."""
    out = []
    step = (curve[1][0] - curve[0][0]) if len(curve) > 1 else 0.05
    thr = _quiet_threshold(curve, words) if curve else None
    for prev, nxt in zip(words, words[1:]):
        s0, e0, w0 = prev[0], prev[1], prev[2]
        s1 = nxt[0]
        kind = "sentence" if SENT_END.search(w0) else "comma" if COMMA_END.search(w0) else "plain"
        if curve:
            lo, hi = s0 + 0.12, s1
            if len(prev) > 3 and prev[3]:                       # a phrase cue: only its end, never inside the phrase
                lo = max(lo, s1 - 1.2)
            best, run = None, None
            for t, db in curve:
                if t < lo or t + step > hi + 1e-6:
                    continue
                if db < thr:
                    run = (run[0], t + step) if run else (t, t + step)
                    if best is None or run[1] - run[0] > best[1] - best[0]:
                        best = run
                else:
                    run = None
            if not best:
                continue
            a, b = best[0], min(best[1], s1)
        else:
            a, b = e0, s1
        gap = b - a
        if gap < MIN_PAUSE:
            continue
        out.append({"a": round(a, 3), "b": round(b, 3), "gap": round(gap, 3), "kind": kind})
    return [p for p in out if 0.5 < p["a"] and p["b"] < duration - 0.5]


def pick_pause(cands: Sequence[Dict[str, Any]], duration: float, at: str = "auto", after_s: float = 20.0) -> Optional[Dict[str, Any]]:
    """Best pause for the mid-roll. auto: closest to the middle inside AUTO_ZONE, a sentence end preferred (a comma pause
    must be 3x closer to win, a plain gap 6x), longer gaps a little. after: the first sentence pause after after_s
    (else comma / plain within +15 s, else the nearest pause after it)."""
    if not cands:
        return None
    weight = {"sentence": 1.0, "comma": 3.0, "plain": 6.0}
    if at == "after":
        later = [p for p in cands if p["a"] >= after_s - 0.3]
        for kind in ("sentence", "comma", "plain"):
            hit = [p for p in later if p["kind"] == kind and p["a"] <= after_s + 15.0]
            if hit:
                return dict(hit[0], why=f"{kind} pause after {after_s:g} s")
        if later:
            return dict(later[0], why=f"first pause after {after_s:g} s")
        at = "auto"                                              # the clip is shorter than that: the middle
    lo, hi = AUTO_ZONE[0] * duration, AUTO_ZONE[1] * duration
    mid = duration / 2.0
    zone = [p for p in cands if lo <= (p["a"] + p["b"]) / 2 <= hi] or list(cands)
    best = min(zone, key=lambda p: (abs((p["a"] + p["b"]) / 2 - mid) + 1.0) * weight[p["kind"]] / (1.0 + min(p["gap"], 1.2)))
    return dict(best, why=f"{best['kind']} pause near the middle")


def quiet_point(clip: str, duration: float, ffmpeg: str = "ffmpeg") -> Optional[float]:
    """No word times: the quietest 0.3 s of the middle third (audio RMS, 0.1 s windows) -> its centre, or None."""
    lo, hi = duration / 3.0, duration * 2.0 / 3.0
    cmd = [ffmpeg, "-hide_banner", "-nostdin", "-nostats", "-ss", f"{lo:.3f}", "-t", f"{hi - lo:.3f}", "-i", clip, "-vn",
           "-af", "aresample=16000,asetnsamples=1600,astats=metadata=1:reset=1,"
                  "ametadata=print:key=lavfi.astats.Overall.RMS_level:file=-", "-f", "null", "-"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except Exception:                                            # noqa: BLE001
        return None
    lv = []
    t = None
    for ln in (r.stdout or "").splitlines():
        m = re.search(r"pts_time:([\d.]+)", ln)
        if m:
            t = float(m.group(1))
            continue
        m = re.search(r"RMS_level=(-?[\d.]+|-inf)", ln)
        if m and t is not None:
            lv.append((t, -120.0 if m.group(1) == "-inf" else float(m.group(1))))
    if len(lv) < 3:
        return None
    best = min(range(1, len(lv) - 1), key=lambda i: lv[i - 1][1] + lv[i][1] + lv[i + 1][1])
    return round(lo + lv[best][0] + 0.05, 3)


# ── media ───────────────────────────────────────────────────────────────────

def probe(path: str) -> Dict[str, Any]:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "format=duration:stream=codec_type,width,height,r_frame_rate,sample_rate,channels,"
                          "color_primaries,color_transfer,color_space:stream_side_data=rotation",
                          "-of", "json", path], capture_output=True, text=True, timeout=60, check=True).stdout
    j = json.loads(out)
    v = next(s for s in j["streams"] if s["codec_type"] == "video")
    a = next((s for s in j["streams"] if s["codec_type"] == "audio"), None)
    w, h = int(v["width"]), int(v["height"])
    rot = 0
    for sd in v.get("side_data_list") or []:
        if "rotation" in sd:
            rot = abs(int(float(sd["rotation"]))) % 180
    if rot == 90:                                               # phone video: ffmpeg autorotates on decode
        w, h = h, w
    return {"w": w, "h": h, "fps": v.get("r_frame_rate") or "30/1", "dur": float(j["format"]["duration"]),
            "audio": a is not None, "sr": int(a.get("sample_rate") or 48000) if a else 48000,
            "ch": int(a.get("channels") or 2) if a else 2,
            "colors": {k: v.get(k) for k in ("color_primaries", "color_transfer", "color_space")}}


def fit_kind(fit: str, ad_w: int, ad_h: int, W: int, H: int) -> str:
    if fit in ("blur", "fit", "fill"):
        return fit
    return "fill" if abs((ad_w / float(ad_h)) / (W / float(H)) - 1.0) <= SHAPE_TOL else "blur"


def ad_video_chain(src: str, W: int, H: int, fps: str, kind: str, dur: float, label: str = "ad") -> str:
    """Filter chain: the ad's video -> exactly W x H, the clip's fps, yuv420p, SAR 1, ``dur`` seconds."""
    base = f"{src}fps={fps},setsar=1"
    if kind == "fill":
        v = f"{base},scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos,crop={W}:{H}"
    elif kind == "fit":
        v = (f"{base},scale={W}:{H}:force_original_aspect_ratio=decrease:flags=lanczos,"
             f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=0x101014")
    else:        # blur-fill: the ad itself, enlarged, blurred and dimmed behind the fitted ad
        bw, bh = max(2, W // 8 // 2 * 2), max(2, H // 8 // 2 * 2)       # blur on a small copy: cheap and soft
        v = (f"{base},split=2[{label}_bg0][{label}_fg0];"
             f"[{label}_bg0]scale={bw}:{bh}:force_original_aspect_ratio=increase,crop={bw}:{bh},"
             f"gblur=sigma=6,eq=brightness=-0.10:saturation=0.85,scale={W}:{H}:flags=bicubic[{label}_bg];"
             f"[{label}_fg0]scale={W}:{H}:force_original_aspect_ratio=decrease:flags=lanczos[{label}_fg];"
             f"[{label}_bg][{label}_fg]overlay=(W-w)/2:(H-h)/2")
    return f"{v},format=yuv420p,tpad=stop_mode=clone:stop_duration=2,trim=duration={dur:.3f},setpts=PTS-STARTPTS[{label}v]"


def ad_audio_chain(src: Optional[str], dur: float, gain: Optional[float], sr: int, label: str = "ad") -> str:
    """The ad's audio -> stereo, the clip's rate, one gain to the clips' loudness + limiter; no audio -> silence."""
    if src is None:
        return f"anullsrc=r={sr}:cl=stereo,atrim=duration={dur:.3f}[{label}a]"
    g = f"volume={gain:.2f}dB,alimiter=limit=0.84:level=false," if gain is not None else ""
    return (f"{src}aresample={sr},aformat=sample_fmts=fltp:channel_layouts=stereo,{g}"
            f"apad,atrim=duration={dur:.3f},asetpts=PTS-STARTPTS[{label}a]")


def splice_filter(info: Dict[str, Any], ad: Dict[str, Any], place: str, cut: float, xf: float, kind: str,
                  gain: Optional[float]) -> Tuple[str, float]:
    """filter_complex for clip [0] + ad [1] -> ([v], [a]) and the new duration.
    mid: clip[0, cut] ⨯dissolve⨯ ad ⨯dissolve⨯ clip[cut, end]; the dissolves are centred on the cut (inside the pause).
    pre / post: ad then clip / clip then ad, one dissolve."""
    W, H, fps, sr, D = info["w"], info["h"], info["fps"], info["sr"], info["dur"]
    A = ad["dur"]
    parts = [ad_video_chain("[1:v]", W, H, fps, kind, A), ad_audio_chain("[1:a]" if ad["audio"] else None, A, gain, sr)]
    cv = f"[0:v]fps={fps},setsar=1,format=yuv420p"
    ca = f"[0:a]aresample={sr},aformat=sample_fmts=fltp:channel_layouts=stereo" if info["audio"] else \
        f"anullsrc=r={sr}:cl=stereo,atrim=duration={D:.3f}"
    if place == "mid":
        a_end = cut + xf / 2                                    # first part runs a half-dissolve past the cut point …
        b_start = cut - xf / 2                                  # … the second starts half a dissolve before it
        parts += [f"{cv},split=2[c1][c2]", f"[c1]trim=0:{a_end:.3f},setpts=PTS-STARTPTS[v1]",
                  f"[c2]trim=start={b_start:.3f},setpts=PTS-STARTPTS[v2]",
                  f"{ca},asplit=2[d1][d2]", f"[d1]atrim=0:{a_end:.3f},asetpts=PTS-STARTPTS[a1]",
                  f"[d2]atrim=start={b_start:.3f},asetpts=PTS-STARTPTS[a2]",
                  f"[v1][adv]xfade=transition=fade:duration={xf:.3f}:offset={a_end - xf:.3f}[v12]",
                  f"[v12][v2]xfade=transition=fade:duration={xf:.3f}:offset={a_end + A - 2 * xf:.3f}[v]",
                  f"[a1][ada]acrossfade=d={xf:.3f}:c1=tri:c2=tri[a12]",
                  f"[a12][a2]acrossfade=d={xf:.3f}:c1=tri:c2=tri[a]"]
        new = a_end + A + (D - b_start) - 2 * xf
    else:
        xf = EDGE_XFADE
        parts += [f"{cv}[cv]", f"{ca}[ca]"]
        if place == "pre":
            parts += [f"[adv][cv]xfade=transition=fade:duration={xf:.3f}:offset={A - xf:.3f}[v]",
                      f"[ada][ca]acrossfade=d={xf:.3f}:c1=tri:c2=tri[a]"]
        else:
            parts += [f"[cv][adv]xfade=transition=fade:duration={xf:.3f}:offset={D - xf:.3f}[v]",
                      f"[ca][ada]acrossfade=d={xf:.3f}:c1=tri:c2=tri[a]"]
        new = A + D - xf
    return ";".join(parts), round(new, 3)


def plan_cut(info: Dict[str, Any], cfg: Dict[str, Any], ass_path: Optional[str], clip: str,
             ffmpeg: str = "ffmpeg") -> Dict[str, Any]:
    """{"cut", "xf", "pause", "from"} for a mid-roll, or {"status": "skipped", "reason"}."""
    D = info["dur"]
    words = []
    if ass_path and Path(ass_path).is_file():
        words = words_from_ass(Path(ass_path).read_text(encoding="utf-8", errors="replace"))
    curve = rms_curve(clip, ffmpeg) if (words and info["audio"]) else []
    p = pick_pause(pauses(words, D, curve), D, cfg["at"], cfg["after_s"]) if words else None
    if p:
        xf = max(XFADE_MIN, min(XFADE_MAX, p["gap"] * 0.6))
        return {"cut": round((p["a"] + p["b"]) / 2, 3), "xf": round(xf, 3), "pause": p, "from": "words"}
    if words:
        return {"status": "skipped", "reason": "no_pause_between_words"}
    q = quiet_point(clip, D, ffmpeg) if info["audio"] else None
    if q is None:
        q = round(D / 2.0, 3) if not info["audio"] else None
    if q is None:
        return {"status": "skipped", "reason": "no_word_times_no_quiet_moment"}
    return {"cut": q, "xf": XFADE_MIN, "pause": None, "from": "audio_rms" if info["audio"] else "middle"}


def ad_gain(ffmpeg: str, ad_path: str) -> Optional[float]:
    """One linear gain that brings the ad to the clips' loudness (SONYA_CLIP_LUFS, −14 LUFS)."""
    from scripts.shared import loudness
    target = loudness.target_lufs()
    if target is None:
        target = loudness.DEFAULT_TARGET           # clips untouched by ops: the ad still matches the platforms' level
    m = loudness.measure(ffmpeg, ad_path)
    if m is None or m <= loudness.SILENCE_LUFS:
        return None
    g = max(-20.0, min(loudness.MAX_BOOST_DB, target - m))
    return round(g, 2) if abs(g) >= loudness.MIN_CHANGE_DB else None


def apply(clip: str, ad_path: str, ad: Dict[str, Any], cfg: Dict[str, Any], *, ass_path: Optional[str] = None,
          gain: Optional[float] = None, ffmpeg: str = "ffmpeg") -> Dict[str, Any]:
    """Insert the ad into ``clip`` in place (one re-encode with the render's encoder). -> report."""
    from scripts.shared.cut_profile import encode_timeout, encoder_args, encoder_family, run_encode
    info = probe(clip)
    D, place = info["dur"], cfg["place"]
    rep: Dict[str, Any] = {"version": VERSION, "place": place, "clip_s": round(D, 2), "ad_s": round(ad["dur"], 2)}
    est = D + ad["dur"] - 2 * XFADE_MIN
    if cfg["max_total"] and est > cfg["max_total"] + 0.5:
        return dict(rep, status="skipped", reason=f"longer_than_{cfg['max_total']}s ({est:.0f} s)")
    cut, xf = 0.0, EDGE_XFADE
    if place == "mid":
        if D < 6.0:
            return dict(rep, status="skipped", reason=f"clip_too_short ({D:.1f} s)")
        pc = plan_cut(info, cfg, ass_path, clip, ffmpeg)
        if pc.get("status") == "skipped":
            return dict(rep, **pc)
        cut, xf = pc["cut"], pc["xf"]
        rep.update(at=cut, xfade=xf, cut_from=pc["from"], pause=pc["pause"])
    kind = fit_kind(cfg["fit"], ad["w"], ad["h"], info["w"], info["h"])
    fc, new_dur = splice_filter(info, ad, place, cut, xf, kind, gain)
    rep.update(fit=kind, gain_db=gain, total_s=new_dur)
    tmp = str(Path(clip).with_name(Path(clip).stem + ".mr.mp4"))
    col = [arg for k, flag in (("color_primaries", "-color_primaries"), ("color_transfer", "-color_trc"),
                               ("color_space", "-colorspace"))
           if info["colors"].get(k) not in (None, "", "unknown") for arg in (flag, info["colors"][k])]

    def cmd(a: List[str]) -> List[str]:
        return [ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-i", clip, "-i", ad_path,
                "-filter_complex", fc, "-map", "[v]", "-map", "[a]", *a, *col, "-r", info["fps"],
                "-c:a", "aac", "-b:a", "192k", "-ar", str(info["sr"]), "-map_metadata", "0", "-movflags", "+faststart", tmp]

    r, used, sec = run_encode(cmd, encoder_args(encoder_family(ffmpeg)), timeout=encode_timeout(new_dur), what="midroll")
    if r.returncode != 0 or not Path(tmp).is_file() or Path(tmp).stat().st_size == 0:
        Path(tmp).unlink(missing_ok=True)
        return dict(rep, status="failed", reason=(r.stderr or "")[-300:].strip())
    os.replace(tmp, clip)
    if place == "pre":
        POSTER_AT[str(clip)] = round(ad["dur"] + 0.5, 2)
    if place != "post" and ass_path and Path(ass_path).is_file():
        try:
            a = Path(ass_path)
            a.write_text(shift_ass(a.read_text(encoding="utf-8", errors="replace"), cut, new_dur - D), encoding="utf-8")
        except OSError:
            pass
    return dict(rep, status="ok", encoder=used, encode_s=sec)


def _ass_time(t: float) -> str:
    cs = int(round(max(0.0, t) * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def shift_ass(ass_text: str, cut: float, delta: float) -> str:
    """The clip's ASS after the ad went in at ``cut`` (delta = seconds added): later cues move by delta, a cue running
    over the cut ends there (the uploaded .ass must match the new clip for the editor)."""
    out = []
    for ln in ass_text.splitlines():
        if ln.startswith("Dialogue:"):
            head, rest = ln.split(":", 1)
            f = rest.split(",", 9)
            if len(f) == 10:
                try:
                    t0, t1 = _secs(f[1].strip()), _secs(f[2].strip())
                except ValueError:
                    out.append(ln)
                    continue
                if t0 >= cut - 0.005:
                    t0, t1 = t0 + delta, t1 + delta
                elif t1 > cut:
                    t1 = cut
                f[1], f[2] = _ass_time(t0), _ass_time(t1)
                ln = head + ":" + ",".join(f)
        out.append(ln)
    return "\n".join(out) + ("\n" if ass_text.endswith("\n") else "")


def warn_text(total: float) -> Optional[str]:
    """Platform note for a clip that got longer than 60 / 90 s with the ad."""
    if total > 90.5:
        return f"с рекламой {total:.0f} с — длиннее 90 с"
    if total > 60.5:
        return f"с рекламой {total:.0f} с — длиннее 60 с"
    return None


# ── the job ─────────────────────────────────────────────────────────────────

def fetch_ad(key: str, work: Path, download=None) -> Path:
    """The ad from S3 into the job's own folder (once per job)."""
    work.mkdir(parents=True, exist_ok=True)
    dst = work / Path(key).name
    if dst.is_file() and dst.stat().st_size > 0:
        return dst
    override = os.environ.get("SONYA_MIDROLL_ASSET")           # local tests only
    if override:
        return Path(override)
    if download is None:
        from scripts.prod_s3_storage import download_file as download
    part = dst.with_name(dst.name + ".part")
    download(key, part)
    os.replace(part, dst)
    return dst


def _record(output_dir: Path, reports: Dict[str, Dict[str, Any]]) -> None:
    sel = output_dir / "selected_segments.json"
    try:
        doc = json.loads(sel.read_text(encoding="utf-8"))
        for seg in doc.get("segments") or []:
            if seg.get("file") in reports:
                seg["midroll"] = reports[seg["file"]]
        sel.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    except (OSError, ValueError):
        pass


def apply_to_outputs(output_dir: str, mode: str, params: Optional[Mapping[str, Any]], *, work_dir: Optional[str] = None,
                     download=None, ffmpeg: str = "ffmpeg") -> Dict[str, Dict[str, Any]]:
    """The ad in every (or every second) finished clip of the job. -> {clip name: report}. Never raises."""
    out_dir = Path(output_dir)
    cfg = config(params) if mode in MODES else None
    reports: Dict[str, Dict[str, Any]] = {}
    if not cfg:
        return reports
    clips = sorted(c for c in out_dir.rglob("*.mp4") if not c.stem.endswith(("_preview", ".ad", ".mr")))
    if not clips:
        return reports
    try:
        ad_path = str(fetch_ad(cfg["key"], Path(work_dir or out_dir.parent) / "midroll", download))
        ad = probe(ad_path)
        if not (AD_MIN_S <= ad["dur"] <= AD_MAX_S + 0.5):
            raise ValueError(f"ad length {ad['dur']:.1f} s")
        ad["dur"] = min(ad["dur"], AD_MAX_S)
        gain = ad_gain(ffmpeg, ad_path) if ad["audio"] else None
    except Exception as exc:                                    # noqa: BLE001
        logger.error("[midroll] ad unavailable (%s): clips go out without it", str(exc)[:200])
        return {c.name: {"status": "failed", "reason": "ad_unavailable"} for c in clips}
    for i, c in enumerate(clips):
        if cfg["every"] == 2 and i % 2 == 1:
            reports[c.name] = {"status": "skipped", "reason": "every_second_clip"}
            continue
        ass = c.with_suffix(".ass")
        try:
            rep = apply(str(c), ad_path, ad, cfg, ass_path=str(ass) if ass.is_file() else None, gain=gain, ffmpeg=ffmpeg)
        except Exception as exc:                                # never lose a rendered clip over the ad
            rep = {"status": "failed", "reason": f"{type(exc).__name__}: {str(exc)[:160]}"}
        if rep.get("status") == "ok":
            w = warn_text(rep.get("total_s") or 0)
            if w:
                rep["warning"] = w
        reports[c.name] = rep
        logger.info("[midroll] %s: %s", c.name, {k: rep.get(k) for k in
                                                ("status", "reason", "place", "at", "xfade", "fit", "total_s", "encoder")})
    _record(out_dir, reports)
    return reports


def warnings(reports: Mapping[str, Mapping[str, Any]]) -> List[str]:
    """Job-level notes for the result: clips that got longer than 60 / 90 s."""
    return [f"{name}: {r['warning']}" for name, r in sorted(reports.items()) if r.get("warning")]
