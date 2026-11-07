"""
subs.py — narration subtitles for the recap that never leave the frame (owner 2026-11-04: «текст выходит за рамки экрана»).

The shared subtitle styles wrap by CHARACTER count; Dela Gothic One is a very wide face and the spoken word grew to
110 %, so long Russian words ran past the 1080 px edge. Here every line is MEASURED with the real font file
(Pillow/FreeType, the same TTF that libass gets via fontsdir):

    width(line) × spoken-word growth + 2 × outline  ≤  safe width (1080 − 2 × SAFE_X)

Words are wrapped greedily by measured width, ≤ 2 lines per cue; the size steps down from BASE_SIZE to MIN_SIZE if a
single word does not fit; a phrase that needs a third line becomes another cue. No \\fscx growth (the active word is
only recoloured), so the measured width is the drawn width. `check_ass` re-measures a finished .ass file — the
validator uses it, deterministic, no rendering needed.

Font and colours stay the owner's: Dela Gothic One, burgundy fill, crimson spoken word, white outline.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Dict, List, Sequence, Tuple

PLAY_W, PLAY_H = 1080, 1920
SAFE_X = 96                       # side margin of the 9:16 safe zone (Reels / Shorts / TikTok UI keeps ~8–9 %)
SAFE_W = PLAY_W - 2 * SAFE_X      # 888 px for the text including its outline
MARGIN_V = 400                    # bottom margin: clear of the platform caption / buttons
BASE_SIZE, MIN_SIZE = 74, 36
PREF_MIN = 58                     # below this a multi-word cue is split instead of shrunk further
MAX_LINES = 2
OUTLINE, SHADOW = 5, 2
FONT_NAME, FONT_FILE = "Dela Gothic One", "DelaGothicOne-Regular.ttf"
PRIMARY, ACTIVE, OUTLINE_C = "&H00321B9E", "&H003D26D7", "&H00FFFFFF"      # burgundy / crimson / white (preset "burgundy")
MAX_CUE_WORDS = 5
MAX_CUE_SEC = 2.6


def _fonts_dir():
    from scripts.shared.subtitle_styles import FONTS_DIR
    return FONTS_DIR


@lru_cache(maxsize=32)
def _font(size: int):
    from PIL import ImageFont
    return ImageFont.truetype(str(_fonts_dir() / FONT_FILE), size)


def text_width(text: str, size: int) -> float:
    """Advance width in PlayRes pixels (PlayRes = 1080×1920 = the output, so 1 px = 1 px)."""
    return float(_font(size).getlength(text))


def line_px(text: str, size: int) -> float:
    return text_width(text, size) + 2 * OUTLINE + SHADOW


def wrap(words: Sequence[str], size: int, max_w: float = SAFE_W) -> List[List[int]]:
    """Greedy wrap by measured width → rows of word indices. A single word wider than max_w is its own row
    (the caller lowers the size then)."""
    rows: List[List[int]] = []
    cur: List[int] = []
    for i, w in enumerate(words):
        trial = " ".join(words[j] for j in cur + [i])
        if cur and line_px(trial, size) > max_w:
            rows.append(cur)
            cur = [i]
        else:
            cur.append(i)
    if cur:
        rows.append(cur)
    return rows


def fit(words: Sequence[str], max_w: float = SAFE_W) -> Tuple[int, List[List[int]]]:
    """Largest size (BASE…MIN) at which the words fit in ≤ MAX_LINES rows; (0, rows) when even MIN needs more rows."""
    for size in range(BASE_SIZE, MIN_SIZE - 1, -2):
        rows = wrap(words, size, max_w)
        if len(rows) <= MAX_LINES and all(line_px(" ".join(words[i] for i in r), size) <= max_w for r in rows):
            return size, rows
    return 0, wrap(words, MIN_SIZE, max_w)


def cues(words: Sequence[Tuple[str, float, float]], max_w: float = SAFE_W) -> List[Dict[str, Any]]:
    """words: (text, start, end) in speech order, sentence ends keep their punctuation. Groups ≤ MAX_CUE_WORDS words,
    ≤ MAX_CUE_SEC, broken at sentence ends; a group that does not fit in 2 lines at MIN_SIZE is split in halves."""
    groups: List[List[Tuple[str, float, float]]] = []
    cur: List[Tuple[str, float, float]] = []
    for w in words:
        cur.append(w)
        if (len(cur) >= MAX_CUE_WORDS or cur[-1][2] - cur[0][1] >= MAX_CUE_SEC or re.search(r"[.!?…]$", w[0])):
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    out: List[Dict[str, Any]] = []
    stack = list(reversed(groups))
    while stack:
        g = stack.pop()
        toks = [x[0].upper() for x in g]
        size, rows = fit(toks, max_w)
        if (size == 0 or size < PREF_MIN) and len(g) > 1:
            h = len(g) // 2
            stack += [g[h:], g[:h]]
            continue
        if size == 0:                       # one monster word: hard floor size, the checker reports it
            size = MIN_SIZE
        out.append({"words": g, "tokens": toks, "size": size, "rows": rows})
    return out


def _t(sec: float) -> str:
    cs = int(round(max(0.0, sec) * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def _esc(t: str) -> str:
    return t.replace("\\", "\\\\").replace("{", "(").replace("}", ")")


def to_ass(words: Sequence[Tuple[str, float, float]], total: float) -> str:
    """One Dialogue per spoken word (the active word recoloured); every Dialogue carries its own \\fs and explicit \\N
    breaks (WrapStyle 2 = no automatic wrapping), so what `check_ass` measures is what libass draws."""
    head = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {PLAY_W}", f"PlayResY: {PLAY_H}", "WrapStyle: 2",
        "ScaledBorderAndShadow: yes", "YCbCr Matrix: TV.709", "", "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding",
        f"Style: Recap,{FONT_NAME},{BASE_SIZE},{PRIMARY},&H000000FF,{OUTLINE_C},&H00000000,0,0,0,0,100,100,0,0,1,"
        f"{OUTLINE},{SHADOW},2,{SAFE_X},{SAFE_X},{MARGIN_V},204",
        "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    ev: List[str] = []
    cl = cues(words)
    for ci, c in enumerate(cl):
        g = c["words"]
        end_all = min(total, cl[ci + 1]["words"][0][1] if ci + 1 < len(cl) else max(g[-1][2], g[0][1] + 0.5))
        for k, (_, s, e) in enumerate(g):
            s2 = s if k else g[0][1]
            e2 = g[k + 1][1] if k + 1 < len(g) else end_all
            if e2 - s2 < 0.01:
                continue
            lines = []
            for r in c["rows"]:
                seg = []
                for i in r:
                    t = _esc(c["tokens"][i])
                    seg.append(f"{{\\c{ACTIVE}&}}{t}{{\\c{PRIMARY}&}}" if i == k else t)
                lines.append(" ".join(seg))
            ev.append(f"Dialogue: 0,{_t(s2)},{_t(e2)},Recap,,0,0,0,,{{\\fs{c['size']}}}" + "\\N".join(lines))
    return "\n".join(head + ev) + "\n"


_TAG = re.compile(r"\{[^}]*\}")


def check_ass(text: str, max_w: float = SAFE_W) -> Dict[str, Any]:
    """Deterministic: every rendered line of every Dialogue, measured with the real font at its \\fs, must fit the
    safe width; ≤ MAX_LINES lines; the style must not scale X. Returns {"ok", "worst_px", "max_px", "problems"}."""
    problems: List[str] = []
    worst = 0.0
    style_size = BASE_SIZE
    for line in text.splitlines():
        if line.startswith("Style:"):
            f = line.split(":", 1)[1].split(",")
            style_size = int(float(f[2]))
            if f[11].strip() not in ("100", "100.0"):
                problems.append(f"style ScaleX {f[11]} ≠ 100")
        if not line.startswith("Dialogue:"):
            continue
        body = line.split(",", 9)[9]
        if re.search(r"\\fsc[xy](?!100\b)\d", body):
            problems.append(f"glyph scaling in {body[:40]!r}")
        m = re.search(r"\\fs(\d+)", body)
        size = int(m.group(1)) if m else style_size
        rows = [_TAG.sub("", r) for r in body.split("\\N")]
        if len(rows) > MAX_LINES:
            problems.append(f"{len(rows)} lines: {rows}")
        for r in rows:
            px = line_px(r, size)
            worst = max(worst, px)
            if px > max_w + 0.5:
                problems.append(f"{px:.0f}px > {max_w:.0f}px at {size}: {r!r}")
    return {"ok": not problems, "worst_px": round(worst, 1), "max_px": max_w, "problems": problems[:10]}
