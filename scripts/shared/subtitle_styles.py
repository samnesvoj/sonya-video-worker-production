"""
subtitle_styles.py
==================
Trendy short-video subtitle presets (Reels / TikTok / Shorts), rendered as ASS.

    highlight  — Hormozi-style: 2–4 bold words, the word being spoken turns yellow and grows
    pop        — 1–2 HUGE caps words at a time, each word pops in
    box        — CapCut «box»: black text on a white plate, spoken word on a purple plate
    mrbeast    — heavy yellow caps, thick black outline + shadow, 2–3 words
    minimal    — clean sentence case, soft shadow, one line (aesthetic / podcast)
    karaoke    — the whole phrase visible, words fill from grey to white as spoken

Fonts: Montserrat Black / ExtraBold / SemiBold and Dela Gothic One (owner's pick 2026-10-15, subset to
Latin + Cyrillic) — SIL OFL, scripts/shared/fonts;
the renderer passes that directory to libass (fontsdir), so no system install is needed.
Positions/avoid-text logic reuse subtitles.placement().
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .subtitles import Cue, SubtitleStyle, Word, _ass_time, ass_escape, placement

FONTS_DIR = Path(os.environ.get("SONYA_FONTS_DIR") or Path(__file__).resolve().parent / "fonts")
# subtitle font choices (Google Fonts, OFL; Cyrillic). ASS name = full name: libass falls back to a
# default sans when a static font without a Regular weight is named by its family only.
FONTS: Dict[str, Tuple[str, str]] = {
    "montserrat": ("Montserrat Black", "Montserrat-Black.ttf"),
    "unbounded": ("Unbounded ExtraBold", "Unbounded-ExtraBold.ttf"),
    "dela": ("Dela Gothic One", "DelaGothicOne-Regular.ttf"),
    "rubik": ("Rubik ExtraBold", "Rubik-ExtraBold.ttf"),
    "oswald": ("Oswald Bold", "Oswald-Bold.ttf"),
    "russo": ("Russo One", "RussoOne-Regular.ttf"),
    "manrope": ("Manrope ExtraBold", "Manrope-ExtraBold.ttf"),
    "onest": ("Onest ExtraBold", "Onest-ExtraBold.ttf"),
    "golos": ("Golos Text ExtraBold", "GolosText-ExtraBold.ttf"),
    "exo2": ("Exo 2 ExtraBold", "Exo2-ExtraBold.ttf"),
    "tektur": ("Tektur ExtraBold", "Tektur-ExtraBold.ttf"),
    "playfair": ("Playfair Display ExtraBold", "PlayfairDisplay-ExtraBold.ttf"),
}


def available_fonts() -> List[str]:
    return [k for k, (_, f) in FONTS.items() if (FONTS_DIR / f).is_file()]
STYLES_VERSION = "sub-styles-v1"


@dataclass(frozen=True)
class Preset:
    id: str
    name: str
    font: str
    size: int
    upper: bool
    max_words: int
    max_chars: int
    lines: int
    primary: str          # ASS &HAABBGGRR
    outline_color: str
    outline: float
    shadow: float
    mode: str             # "word" (per-word highlight) | "static" | "karaoke" | "box"
    active: str = ""      # colour of the spoken word
    grow: int = 100       # % scale of the spoken word
    pop: bool = False
    back: str = "&H00000000"
    border_style: int = 1  # 1 outline, 3 opaque box
    glass: bool = False    # liquid-glass pill: the video behind the text is blurred + frosted (renderer)


PRESETS: Dict[str, Preset] = {p.id: p for p in [
    Preset("highlight", "Подсветка слова (Hormozi)", "Montserrat Black", 82, True, 4, 16, 2,
           "&H00FFFFFF", "&H00000000", 6, 0, "word", active="&H0000E5FF", grow=112),
    Preset("pop", "Крупно по 1–2 слова", "Montserrat Black", 120, True, 2, 12, 1,
           "&H00FFFFFF", "&H00000000", 7, 3, "word", active="&H0066FF3C", grow=100, pop=True),
    Preset("box", "Плашка (CapCut)", "Montserrat ExtraBold", 70, False, 4, 18, 2,
           "&H00141414", "&H00FFFFFF", 14, 0, "box", active="&H00F05AB4", border_style=3,
           back="&H00FFFFFF"),
    Preset("mrbeast", "Жёлтый жирный (MrBeast)", "Montserrat Black", 96, True, 3, 14, 2,
           "&H0000D4FF", "&H00000000", 8, 5, "static"),
    Preset("minimal", "Минимализм", "Montserrat SemiBold", 60, False, 8, 30, 1,
           "&H00FFFFFF", "&H64000000", 1.2, 2, "static"),
    Preset("karaoke", "Караоке (заливка по словам)", "Montserrat ExtraBold", 72, False, 6, 22, 2,
           "&H00FFFFFF", "&H00000000", 5, 0, "karaoke"),
    # ── noble palettes (owner: «благородные, но чтоб было видно»): pearl / champagne / lavender /
    #    silver on a graphite outline with a soft shadow — readable on bright and dark footage ──
    Preset("champagne", "Жемчуг и шампань", "Montserrat ExtraBold", 76, False, 4, 18, 2,
           "&H00EAF1F4", "&H00221C1C", 3.5, 3, "word", active="&H007AC2E2", grow=106),
    Preset("lavender", "Жемчуг и холодная лаванда", "Montserrat ExtraBold", 76, False, 4, 18, 2,
           "&H00EAF1F4", "&H00221C1C", 3.5, 3, "word", active="&H00E8B3BF", grow=106),
    Preset("silver", "Серебро, капс", "Montserrat Black", 80, True, 3, 16, 2,
           "&H00E1DCD9", "&H00241E1C", 4, 4, "word", active="&H00FFFFFF", grow=110),
    # owner 2026-10-16: «цвет субтитров бордово-красные» — burgundy fill, the spoken word in a
    # brighter crimson, white outline so it reads on dark footage too
    Preset("burgundy", "Бордово-красные", "Montserrat Black", 82, True, 4, 16, 2,
           "&H00321B9E", "&H00FFFFFF", 5, 2, "word", active="&H003D26D7", grow=110),
    Preset("glass", "Liquid glass", "Montserrat ExtraBold", 66, False, 5, 20, 2,
           "&H00FAF8F7", "&H00000000", 0, 1.5, "word", active="&H007AC2E2", grow=100, glass=True),
]}
DEFAULT = "highlight"


def groups(words: List[Word], p: Preset, clip_duration: float) -> List[List[Word]]:
    """Short word groups: ≤ max_words, ≤ lines×max_chars, split at sentence/clause ends and pauses."""
    out: List[List[Word]] = []
    cur: List[Word] = []
    for w in words:
        if cur:
            text = " ".join(x.text for x in cur + [w])
            if (len(cur) >= p.max_words or len(text) > p.max_chars * p.lines
                    or w.start - cur[-1].end > 0.45 or cur[-1].text.endswith((".", "!", "?", "…"))
                    or (cur[-1].text.endswith((",", ":", ";")) and len(cur) >= 2)):
                out.append(cur)
                cur = []
        cur.append(w)
    if cur:
        out.append(cur)
    return out


def _wrap(ws: List[str], max_chars: int, lines: int) -> List[List[int]]:
    """Indices per line (greedy)."""
    rows: List[List[int]] = [[]]
    n = 0
    for i, w in enumerate(ws):
        if rows[-1] and n + 1 + len(w) > max_chars and len(rows) < lines:
            rows.append([])
            n = 0
        rows[-1].append(i)
        n += len(w) + (1 if n else 0)
    return rows


def _norm(t: str) -> str:
    import re
    return re.sub(r"[^\w-]", "", t).lower().replace("ё", "е")


def _wrap_hook(hook: str, max_w: float = 900.0, font: str = "Montserrat Black") -> str:
    """Balanced 1–2 lines (WrapStyle 2 never wraps by itself); shrink if one word is still too wide."""
    ws = hook.strip().split()
    width = lambda t: _text_width(t, font, 62)
    rows = [" ".join(ws)]
    if width(rows[0]) > max_w and len(ws) > 1:
        cut = min(range(1, len(ws)), key=lambda i: max(width(" ".join(ws[:i])), width(" ".join(ws[i:]))))
        rows = [" ".join(ws[:cut]), " ".join(ws[cut:])]
    widest = max(width(r) for r in rows)
    scale = "" if widest <= max_w else f"{{\\fscx{int(100 * max_w / widest)}\\fscy{int(100 * max_w / widest)}}}"
    return scale + "\\N".join(ass_escape(r) for r in rows)


def to_ass(words: List[Word], preset_id: str, clip_duration: float, play_res: Tuple[int, int] = (1080, 1920),
           centered=None, avoid=None, base: Optional[SubtitleStyle] = None,
           keywords: Optional[List[str]] = None, hook: Optional[str] = None, font: Optional[str] = None,
           promo: Optional[Dict] = None, hook_top: Optional[int] = None) -> str:
    """keywords: words to keep emphasised (Submagic-style); hook: a title shown on top for the first seconds
    (Opus-style), on a translucent graphite plate. hook_top: its top edge in px (default: the style's 230 px;
    the streamer layout puts it higher, over the webcam's headroom, so it does not cover the face)."""
    p = PRESETS.get(preset_id) or PRESETS[DEFAULT]
    hook_font = "Montserrat Black"
    if font and font in available_fonts():
        hook_font = FONTS[font][0]
        name = FONTS[font][0]
        # same line width as the preset's own font: wide faces (Unbounded, Dela) get a smaller size
        probe = "ГДЕ ТЫ БЕРЁШЬ СТОЛЬКО" if p.upper else "где ты берёшь столько"
        ratio = _text_width(probe, p.font, p.size) / max(1.0, _text_width(probe, name, p.size))
        p = replace(p, font=name, size=int(round(p.size * min(1.0, ratio))))
    w, h = play_res
    st = replace(base or SubtitleStyle(), font=p.font, size=p.size, outline=p.outline, shadow=p.shadow,
                 max_chars_per_line=p.max_chars, max_lines=p.lines)
    secondary = "&H00A0A0A0" if p.mode == "karaoke" else "&H000000FF"
    head = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {w}", f"PlayResY: {h}", "WrapStyle: 2",
        "ScaledBorderAndShadow: yes", "YCbCr Matrix: TV.709", "", "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding",
        f"Style: Sonya,{p.font},{p.size},{p.primary},{secondary},{p.outline_color},{p.back},0,0,0,0,100,100,"
        f"{1 if p.upper else 0},0,{p.border_style},{p.outline},{p.shadow},2,{st.margin_lr},{st.margin_lr},{st.margin_v},204",
        f"Style: Hook,{hook_font},62,&H00FAF8F7,&H000000FF,&H59221C1C,&H00000000,0,0,0,0,100,100,0,0,3,"
        "22,0,8,90,90,230,204",
        *(_promo_styles(hook_font) if promo else []),
        "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    ev: List[str] = []
    if hook:
        pos = f"{{\\an8\\pos({w // 2},{int(hook_top)})}}" if hook_top is not None else ""
        ev.append(f"Dialogue: 1,{_ass_time(0.0)},{_ass_time(min(3.2, clip_duration))},Hook,,0,0,0,,"
                  f"{{\\fad(150,300)}}{pos}{_wrap_hook(hook, font=hook_font)}")
    kw = {_norm(k) for k in keywords or [] if _norm(k)}
    gs = groups(words, p, clip_duration)
    for gi, g in enumerate(gs):
        g_end = min(gs[gi + 1][0].start if gi + 1 < len(gs) else clip_duration,
                    max(g[-1].end, g[0].start + 0.5), clip_duration)
        texts = [(x.text.upper() if p.upper else x.text) for x in g]
        rows = _wrap(texts, p.max_chars, p.lines)
        cue = Cue(g[0].start, g_end, [" ".join(texts[i] for i in r) for r in rows], len(g))
        pos = placement(cue, st, play_res, centered, avoid)

        def line(active: Optional[int]) -> str:
            parts = []
            for r in rows:
                seg = []
                for i in r:
                    t = ass_escape(texts[i])
                    if active == i and p.active:
                        tag = (f"{{\\c{p.active}&\\fscx{p.grow}\\fscy{p.grow}}}" if p.mode != "box" else
                               f"{{\\3c{p.active}&\\c&H00FFFFFF&}}")
                        reset = (f"{{\\c{p.primary}&\\fscx100\\fscy100}}" if p.mode != "box" else
                                 f"{{\\3c{p.outline_color}&\\c{p.primary}&}}")
                        seg.append(tag + t + reset)
                    elif _norm(texts[i]) in kw and p.active:
                        tag = f"{{\\c{p.active}&}}" if p.mode != "box" else f"{{\\3c{p.active}&\\c&H00FFFFFF&}}"
                        reset = f"{{\\c{p.primary}&}}" if p.mode != "box" else f"{{\\3c{p.outline_color}&\\c{p.primary}&}}"
                        seg.append(tag + t + reset)
                    else:
                        seg.append(t)
                parts.append(" ".join(seg))
            return "\\N".join(parts)

        if p.mode in ("word", "box"):
            for i, x in enumerate(g):
                s = x.start if i else g[0].start
                e = g[i + 1].start if i + 1 < len(g) else g_end
                if e <= s:
                    continue
                pop = "{\\fscx135\\fscy135\\t(0,90,\\fscx100\\fscy100)}" if p.pop else ""
                ev.append(f"Dialogue: 0,{_ass_time(s)},{_ass_time(e)},Sonya,,0,0,0,,{pos}{pop}{line(i)}")
        elif p.mode == "karaoke":
            k = []
            flat = [i for r in rows for i in r]
            for n, i in enumerate(flat):
                nxt = g[n + 1].start if n + 1 < len(g) else g[-1].end
                cs = max(1, int(round((nxt - g[n].start) * 100)))
                k.append(f"{{\\kf{cs}}}{ass_escape(texts[i])}")
            text = "\\N".join(" ".join(k[flat.index(i)] for i in r) for r in rows)
            ev.append(f"Dialogue: 0,{_ass_time(g[0].start)},{_ass_time(g_end)},Sonya,,0,0,0,,{pos}{text}")
        else:
            ev.append(f"Dialogue: 0,{_ass_time(g[0].start)},{_ass_time(g_end)},Sonya,,0,0,0,,{pos}{line(None)}")
    if promo:
        from .promo import PromoSpec, ass_events
        spec = PromoSpec.from_dict(promo)
        if spec.plate:
            ev += ass_events(spec, clip_duration, _ass_time, ass_escape, font=hook_font)
    return "\n".join(head + ev) + "\n"


def _promo_styles(font: str) -> List[str]:
    from .promo import ass_styles
    return ass_styles(font)


GLASS_PAD_X, GLASS_PAD_Y = 34, 20


def _text_width(text: str, font: str, size: int) -> float:
    try:
        from PIL import ImageFont
        file = {"Montserrat Black": "Montserrat-Black.ttf", "Montserrat ExtraBold": "Montserrat-ExtraBold.ttf",
                "Montserrat SemiBold": "Montserrat-SemiBold.ttf",
                **{n: f for n, f in FONTS.values()}}.get(font, "Montserrat-ExtraBold.ttf")
        return float(ImageFont.truetype(str(FONTS_DIR / file), size).getlength(text))
    except Exception:                       # Pillow missing: a safe over-estimate
        return 0.62 * size * len(text)


def glass_pills(ass_text: str, preset_id: str, play_res: Tuple[int, int] = (1080, 1920)) -> List[List[float]]:
    """[[t0, t1, x, y, w, h]] — one frosted pill per cue group, measured from the real glyph widths
    and placed where libass puts the text (default bottom / \\an2 / \\an5 / \\an8 overrides)."""
    import re
    p = PRESETS.get(preset_id)
    if not p or not p.glass:
        return []
    w, h = play_res
    st = SubtitleStyle()
    line_h = p.size * 1.22
    pills: Dict[Tuple[str, str], List[float]] = {}
    out: List[List[float]] = []
    for ln in ass_text.splitlines():
        if not ln.startswith("Dialogue:"):
            continue
        f = ln.split(",", 9)
        if f[3] == "Hook":                     # the hook title has its own plate
            continue
        t0, t1, text = f[1], f[2], f[9]
        m = re.match(r"\{\\an(\d)\\pos\((\d+),(\d+)\)\}", text)
        plain = re.sub(r"\{[^}]*\}", "", text)
        rows = plain.split("\\N")
        bw = max(_text_width(r, p.font, p.size) for r in rows) + 2 * GLASS_PAD_X
        bh = line_h * len(rows) + 2 * GLASS_PAD_Y
        if m and m.group(1) == "5":
            cy = float(m.group(3)); y = cy - bh / 2
        elif m and m.group(1) == "8":
            y = float(m.group(3)) - GLASS_PAD_Y
        else:
            bottom = float(m.group(3)) if m else h - st.margin_v
            y = bottom - bh + GLASS_PAD_Y
        key = (t0, plain)
        def secs(t):
            hh, mm, ss = t.split(":")
            return int(hh) * 3600 + int(mm) * 60 + float(ss)
        if out and out[-1][1] >= secs(t0) - 0.011 and abs(out[-1][4] - bw) < 2 and abs(out[-1][3] - y) < 2:
            out[-1][1] = secs(t1)             # same group, next word: extend the pill
            continue
        out.append([secs(t0), secs(t1), round((w - bw) / 2), round(y), round(bw), round(bh)])
    res = []
    for a, b, x, y, bw, bh in out:             # ffmpeg crop fails on a pill outside the frame
        x0, y0 = max(0, int(x)), max(0, int(y))
        bw, bh = min(int(bw), w - x0), min(int(bh), h - y0)
        res.append([round(a, 3), round(b, 3), x0 // 2 * 2, y0 // 2 * 2, bw // 2 * 2, bh // 2 * 2])
    return res


def glass_mask_ass(pills: List[List[float]], play_res: Tuple[int, int] = (1080, 1920)) -> str:
    """White rounded rectangles on transparent: libass draws the alpha mask of all pills in one
    filter (antialiased corners), instead of one per-pixel geq per pill (8 clips: 911 s)."""
    w, h = play_res
    lines = ["[Script Info]", "ScriptType: v4.00+", f"PlayResX: {w}", f"PlayResY: {h}", "WrapStyle: 2",
             "ScaledBorderAndShadow: yes", "", "[V4+ Styles]",
             "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
             "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
             "Alignment, MarginL, MarginR, MarginV, Encoding",
             "Style: M,Arial,20,&H00FFFFFF,&H00FFFFFF,&H00FFFFFF,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1",
             "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    for a, b, x, y, pw, ph in pills:
        if pw <= 0 or ph <= 0:
            continue
        r = min(ph / 2.0, 44.0)
        k = r * 0.448                           # cubic Bézier for a quarter circle
        R = lambda v: f"{v:.1f}"
        path = (f"m {R(r)} 0 l {R(pw - r)} 0 b {R(pw - k)} 0 {pw} {R(k)} {pw} {R(r)} l {pw} {R(ph - r)} "
                f"b {pw} {R(ph - k)} {R(pw - k)} {ph} {R(pw - r)} {ph} l {R(r)} {ph} b {R(k)} {ph} 0 {R(ph - k)} "
                f"0 {R(ph - r)} l 0 {R(r)} b 0 {R(k)} {R(k)} 0 {R(r)} 0")
        # inside ~84 % opaque frosted glass, a thin brighter rim
        lines.append(f"Dialogue: 0,{_ass_time(a)},{_ass_time(b)},M,,0,0,0,,"
                     f"{{\\an7\\pos({x},{y})\\1a&H28&\\bord1.5\\3a&H05&\\p1}}{path}")
    return "\n".join(lines) + "\n"


def catalog() -> List[Dict[str, str]]:
    return [{"id": p.id, "name": p.name} for p in PRESETS.values()]
