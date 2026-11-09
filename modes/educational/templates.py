"""
templates.py — four Educational looks. One template = subtitles + overlay cards + the title, one visual system.

    1 «Белый лист»  light: white plate, graphite ink, the spoken word in SONYA violet          (Onest)
    2 «Серебро»     light: no plate, white text with a soft shadow; silver cards              (Manrope)
    3 «Графит»      dark: graphite plate, near-white text, light-violet spoken word            (Golos Text)
    4 «Фиолет»      bold: Dela Gothic One (owner's subtitle pick), spoken word on a violet pill (Onest cards)

Owner rules (2026-11-09): no glass, no gold/beige/brown; white / silver / black + #877FC7; one dark template at least;
text never leaves the frame; readable on a phone; no Montserrat. Motion (Emil Kowalski): a cue enters in 200 ms with
a strong ease-out and a 10 px rise from 96 % scale, leaves faster (120 ms); nothing pops in from zero.

Everything is measured with Pillow on the same font files libass uses (scripts/shared/fonts); a line that would be
wider than the safe width is wrapped, and a word that alone is wider than it is shrunk — never cut by the frame edge.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFilter, ImageFont

TEMPLATES_VERSION = "edu-templates-v1"
FONTS = Path(__file__).resolve().parents[2] / "scripts" / "shared" / "fonts"
RGB = Tuple[int, int, int]

VIOLET = (0x87, 0x7F, 0xC7)
VIOLET_LIGHT = (0xB4, 0xAE, 0xEA)
WHITE = (0xFE, 0xFD, 0xFD)
SILVER = (0xEC, 0xEB, 0xF0)
SILVER_LINE = (0xD0, 0xCD, 0xD2)
GRAPHITE = (0x17, 0x19, 0x23)
INK = (0x1B, 0x1C, 0x26)
SLATE = (0x3A, 0x3A, 0x47)
MUTED = (0x6E, 0x6C, 0x7C)

ENTER_MS, EXIT_MS, RISE_PX = 200, 120, 10


@dataclass(frozen=True)
class SubSpec:
    font_file: str
    font_name: str          # the family name inside the file (what libass is asked for)
    size: int               # px at 1080 wide, ASS font size
    ink: RGB
    active: RGB             # the word being spoken
    plate: Optional[RGB]    # rounded plate behind the line(s); None = no plate
    plate_alpha: int = 238
    shadow_alpha: int = 0   # soft text shadow (no plate)
    outline: float = 0.0
    outline_color: RGB = GRAPHITE
    pill: Optional[RGB] = None      # the spoken word sits on a pill of this colour (template 4)
    pill_ink: RGB = WHITE


@dataclass(frozen=True)
class Template:
    id: int
    slug: str
    name: str
    dark: bool
    sub: SubSpec
    theme: str              # overlays/themes.py key for the cards
    title_bg: RGB
    title_ink: RGB
    title_label: RGB
    title_font: str
    label_font: str
    stage: RGB = (0xF5, 0xF5, 0xF7)        # background around a slide / board shown whole (fit layout)
    sub_stage: Optional[SubSpec] = None     # subtitles when they sit on the stage, not on the picture


TEMPLATES: Dict[int, Template] = {
    1: Template(1, "white", "Белый лист", False,
                SubSpec("Onest-Bold.ttf", "Onest Bold", 76, INK, VIOLET, WHITE, 240),
                "edu-white", WHITE, INK, VIOLET, "Onest-Bold.ttf", "Onest-Bold.ttf",
                stage=(0xF2, 0xF2, 0xF5)),
    2: Template(2, "silver", "Серебро", False,
                SubSpec("Manrope-ExtraBold.ttf", "Manrope ExtraBold", 74, WHITE, VIOLET_LIGHT, None,
                        shadow_alpha=150, outline=2.2, outline_color=SLATE),
                "edu-silver", SILVER, SLATE, VIOLET, "Manrope-ExtraBold.ttf", "Manrope-ExtraBold.ttf",
                stage=(0xE4, 0xE3, 0xE9),
                sub_stage=SubSpec("Manrope-ExtraBold.ttf", "Manrope ExtraBold", 74, SLATE, VIOLET, None)),
    3: Template(3, "graphite", "Графит", True,
                SubSpec("GolosText-Bold.ttf", "Golos Text Bold", 76, (0xF4, 0xF3, 0xF8), VIOLET_LIGHT, GRAPHITE, 232),
                "edu-graphite", GRAPHITE, (0xF4, 0xF3, 0xF8), VIOLET_LIGHT, "GolosText-Bold.ttf", "GolosText-Bold.ttf",
                stage=(0x0E, 0x0F, 0x15),
                sub_stage=SubSpec("GolosText-Bold.ttf", "Golos Text Bold", 76, (0xF4, 0xF3, 0xF8), VIOLET_LIGHT, None)),
    4: Template(4, "violet", "Фиолет", False,
                SubSpec("DelaGothicOne-Regular.ttf", "Dela Gothic One", 70, WHITE, WHITE, None,
                        shadow_alpha=170, outline=2.0, outline_color=GRAPHITE, pill=VIOLET, pill_ink=WHITE),
                "edu-violet", VIOLET, WHITE, (0xE9, 0xE7, 0xF8), "Unbounded-SemiBold.ttf", "Onest-Bold.ttf",
                stage=(0xEE, 0xED, 0xF7),
                sub_stage=SubSpec("DelaGothicOne-Regular.ttf", "Dela Gothic One", 70, INK, WHITE, None,
                                  pill=VIOLET, pill_ink=WHITE)),
}
DEFAULT_TEMPLATE = 1


def get_template(value: Any) -> Template:
    """params["template"] may be 1–4, "1"…"4" or a slug; anything else → template 1 (never fails a job)."""
    if isinstance(value, Template):
        return value
    try:
        n = int(value)
        if n in TEMPLATES:
            return TEMPLATES[n]
    except (TypeError, ValueError):
        pass
    for t in TEMPLATES.values():
        if str(value or "").strip().lower() in (t.slug, t.name.lower()):
            return t
    return TEMPLATES[DEFAULT_TEMPLATE]


def catalog() -> List[Dict[str, Any]]:
    """For the site: id, name, light/dark, preview file name."""
    return [{"id": t.id, "slug": t.slug, "name": t.name, "dark": t.dark, "preview": f"edu_template_{t.id}.jpg"}
            for t in TEMPLATES.values()]


# ── measuring ────────────────────────────────────────────────────────────────

@lru_cache(maxsize=64)
def _font(file: str, px: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONTS / file), px)


@lru_cache(maxsize=64)
def _ass_to_pil(file: str) -> float:
    """libass sizes a font so that ascent+descent (OS/2 win metrics) equals the ASS size; Pillow sizes the em."""
    f = ImageFont.truetype(str(FONTS / file), 1000)
    asc, desc = f.getmetrics()
    return 1000.0 / max(1, asc + desc)


def text_width(text: str, file: str, ass_size: float) -> float:
    px = max(1, int(round(ass_size * _ass_to_pil(file))))
    return _font(file, px).getlength(text)


# ── subtitles ────────────────────────────────────────────────────────────────

def _c(rgb: RGB, alpha: int = 255) -> str:
    """ASS colour &HAABBGGRR (alpha 00 = opaque)."""
    return f"&H{255 - alpha:02X}{rgb[2]:02X}{rgb[1]:02X}{rgb[0]:02X}"


def _t(sec: float) -> str:
    cs = max(0, int(round(sec * 100)))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace("{", "(").replace("}", ")").replace("\n", " ")


def _rounded(w: float, h: float, r: float) -> str:
    """ASS vector drawing of a rounded rectangle, origin top-left (bezier corners)."""
    r = min(r, w / 2, h / 2)
    k = r * 0.4477
    return (f"m {r:.1f} 0 l {w - r:.1f} 0 b {w - k:.1f} 0 {w:.1f} {k:.1f} {w:.1f} {r:.1f} l {w:.1f} {h - r:.1f} "
            f"b {w:.1f} {h - k:.1f} {w - k:.1f} {h:.1f} {w - r:.1f} {h:.1f} l {r:.1f} {h:.1f} "
            f"b {k:.1f} {h:.1f} 0 {h - k:.1f} 0 {h - r:.1f} l 0 {r:.1f} b 0 {k:.1f} {k:.1f} 0 {r:.1f} 0")


@dataclass
class Cue:
    start: float
    end: float
    words: List[Tuple[str, float, float]]   # text, start, end (clip-local)
    lines: List[List[int]]                  # indexes into words per line
    size: float


@dataclass(frozen=True)
class Layout:
    width: int
    height: int
    on_stage: bool = False      # subtitles sit on the template's stage (fit layout), not on the picture

    @property
    def u(self) -> float:
        return self.width / 1080.0

    @property
    def safe_w(self) -> float:          # side margins 8.5 % each: clear of the right-hand icons in 9:16
        return self.width * 0.83

    @property
    def bottom(self) -> float:          # bottom of the subtitle block: above the platform caption zone (20 %)
        return self.height * (0.79 if self.height > self.width else 0.93)


def sub_spec(tpl: Template, lay: Layout) -> SubSpec:
    return tpl.sub_stage if (lay.on_stage and tpl.sub_stage) else tpl.sub


_WEAK = {"в", "во", "к", "ко", "с", "со", "и", "а", "но", "на", "по", "о", "об", "от", "до", "за", "из", "у", "не",
         "что", "как", "это", "же", "ли", "для", "при", "без", "над", "под", "то"}


def _wrap(words: Sequence[str], file: str, size: float, max_w: float) -> List[List[int]]:
    """Fewest lines that fit; two lines are balanced and a line never ends on a preposition / conjunction."""
    def w(a: int, b: int) -> float:
        return text_width(" ".join(words[a:b]), file, size)
    n = len(words)
    if n <= 1 or w(0, n) <= max_w:
        return [list(range(n))]
    best = None
    for i in range(1, n):
        widest = max(w(0, i), w(i, n))
        if widest > max_w:
            continue
        weak = words[i - 1].strip(",.").lower() in _WEAK
        key = (weak, widest)
        if best is None or key < best[0]:
            best = (key, i)
    if best is not None:
        i = best[1]
        return [list(range(i)), list(range(i, n))]
    lines: List[List[int]] = [[]]          # three or more lines (callers split the cue before this happens)
    for i in range(n):
        if lines[-1] and w(lines[-1][0], i + 1) > max_w:
            lines.append([i])
        else:
            lines[-1].append(i)
    return lines


def _line_width(words: Sequence[str], sp: SubSpec, size: float) -> float:
    if not sp.pill:
        return text_width(" ".join(words), sp.font_file, size)
    return sum(text_width(w, sp.font_file, size) for w in words) + \
        2 * text_width(" ", sp.font_file, size) * max(0, len(words) - 1)


def _sentence_tail(ws: Sequence[Tuple[str, float, float]], k: int, limit: int = 2) -> int:
    """How many words from k end the sentence (≤ limit), else 0."""
    for j in range(k, min(len(ws), k + limit)):
        if ws[j][0][-1:] in ".!?…":
            return j - k + 1
    return 0


def build_cues(words: Sequence[Dict[str, Any]], duration: float, tpl: Template, lay: Layout,
               max_lines: int = 2, max_sec: float = 3.4, gap_sec: float = 0.6) -> List[Cue]:
    """Short phrases on sentence / pause boundaries, at most two lines that fit the safe width."""
    sp = sub_spec(tpl, lay)
    size0 = sp.size * lay.u * (1.0 if lay.height > lay.width else 0.78)
    max_w = lay.safe_w - (2 * 34 * lay.u if sp.plate else 0) - (60 * lay.u if sp.pill else 0)
    ws = [(str(w.get("word", "")).strip(), float(w["start"]), float(w["end"])) for w in words
          if str(w.get("word", "")).strip() and float(w["end"]) > 0 and float(w["start"]) < duration]
    groups: List[List[Tuple[str, float, float]]] = []
    cur: List[Tuple[str, float, float]] = []
    for k, w in enumerate(ws):
        if cur:
            text = " ".join(x[0] for x in cur + [w])
            too_long = len(_wrap([x[0] for x in cur + [w]], sp.font_file, size0, max_w)) > max_lines
            tail = _sentence_tail(ws, k)
            if tail and not too_long and cur[-1][0][-1:] not in ".!?…" and w[1] - cur[-1][2] <= gap_sec:
                fits = len(_wrap([x[0] for x in cur + list(ws[k:k + tail])], sp.font_file, size0, max_w)) <= max_lines
                if fits and ws[k + tail - 1][2] - cur[0][1] <= max_sec + 1.6:
                    cur.append(w)               # 1–2 words to the sentence end: no orphan cue like «образом.»
                    continue
            if (cur[-1][0][-1:] in ".!?…" or w[1] - cur[-1][2] > gap_sec or w[2] - cur[0][1] > max_sec
                    or too_long or (cur[-1][0][-1:] in ",;:" and len(text) > 26)):
                carry = []
                if (len(cur) > 2 and cur[-1][0].strip(",.").lower() in _WEAK and cur[-1][0][-1:].isalpha()
                        and w[1] - cur[-1][2] <= gap_sec):
                    carry = [cur.pop()]             # «зациклен на» → «на» opens the next cue
                groups.append(cur)
                cur = carry
        cur.append(w)
    if cur:
        groups.append(cur)
    cues: List[Cue] = []
    for i, g in enumerate(groups):
        size = size0
        while True:
            lines = _wrap([x[0] for x in g], sp.font_file, size, max_w)
            widest = max(_line_width([g[j][0] for j in ln], sp, size) for ln in lines)
            if widest <= lay.safe_w - (2 * 34 * lay.u if sp.plate else 0) or size < size0 * 0.6:
                break
            size *= 0.92                    # one very long word: smaller, never outside the frame
        start = max(0.0, g[0][1] - 0.05)
        nxt = groups[i + 1][0][1] if i + 1 < len(groups) else duration
        end = min(duration, max(g[-1][2] + 0.35, start + 0.7), nxt - 0.02)
        cues.append(Cue(start, max(end, start + 0.3), list(g), lines[:max_lines + 1], size))
    return cues


def subtitle_block_top(tpl: Template, lay: Layout) -> int:
    """Top of the band the subtitles may occupy (two lines + plate): cards stay above it."""
    sp = sub_spec(tpl, lay)
    size = sp.size * lay.u * (1.0 if lay.height > lay.width else 0.78)
    return int(lay.bottom - 2 * size * 1.18 - (44 if sp.plate else 16) * lay.u)


def to_ass(cues: Sequence[Cue], tpl: Template, lay: Layout) -> str:
    sp = sub_spec(tpl, lay)
    u = lay.u
    head = ["[Script Info]", "ScriptType: v4.00+", f"PlayResX: {lay.width}", f"PlayResY: {lay.height}",
            "WrapStyle: 2", "ScaledBorderAndShadow: yes", "YCbCr Matrix: TV.709", "", "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
            "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
            "MarginR, MarginV, Encoding",
            f"Style: T,{sp.font_name},{int(sp.size * u)},{_c(sp.ink)},&H000000FF,{_c(sp.outline_color)},"
            f"{_c((0, 0, 0), sp.shadow_alpha)},0,0,0,0,100,100,0,0,1,{sp.outline * u:.1f},0,5,0,0,0,204",
            "Style: P,Arial,10,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,0",
            "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    ev: List[str] = []
    for c in cues:
        lh = c.size * 1.18
        n = len(c.lines)
        pad_x, pad_y = 30 * u, 16 * u
        block_h = n * lh + (2 * pad_y if sp.plate else 0)
        top = lay.bottom - block_h
        t0, t1 = _t(c.start), _t(c.end)
        fade = f"\\fad({ENTER_MS},{EXIT_MS})"
        widths = [_line_width([c.words[j][0] for j in ln], sp, c.size) for ln in c.lines]
        if sp.plate:
            pw = max(widths) + 2 * pad_x
            x0 = (lay.width - pw) / 2
            rise = RISE_PX * u
            # soft shadow first, then the plate; both rise with the text
            ev.append(f"Dialogue: 0,{t0},{t1},P,,0,0,0,,{{\\an7\\move({x0:.1f},{top + 8 * u + rise:.1f},{x0:.1f},"
                      f"{top + 8 * u:.1f},0,{ENTER_MS})\\bord0\\shad0\\blur{14 * u:.1f}\\1c{_c((0, 0, 0))}"
                      f"\\1a&H{255 - (60 if not tpl.dark else 110):02X}&{fade}\\p1}}{_rounded(pw, block_h, 26 * u)}{{\\p0}}")
            ev.append(f"Dialogue: 1,{t0},{t1},P,,0,0,0,,{{\\an7\\move({x0:.1f},{top + rise:.1f},{x0:.1f},{top:.1f},0,"
                      f"{ENTER_MS})\\bord0\\shad0\\blur0.6\\1c{_c(sp.plate)}\\1a&H{255 - sp.plate_alpha:02X}&{fade}\\p1}}"
                      f"{_rounded(pw, block_h, 26 * u)}{{\\p0}}")
        for li, ln in enumerate(c.lines):
            cy = top + (pad_y if sp.plate else 0) + lh * (li + 0.5)
            parts = []
            for j in ln:
                text, ws, we = c.words[j]
                a = int((ws - c.start) * 1000)
                b = int((we - c.start) * 1000)
                hi = sp.pill_ink if sp.pill else sp.active
                if hi == sp.ink:
                    tag = ""
                else:
                    tag = (f"{{\\1c{_c(sp.ink)}\\t({max(0, a - 40)},{max(0, a - 40) + 60},\\1c{_c(hi)})"
                           f"\\t({b + 80},{b + 160},\\1c{_c(sp.ink)})}}")
                parts.append(tag + _esc(text))
            shadow = ""
            if sp.shadow_alpha:
                shadow = f"\\shad{3 * u:.1f}\\4c{_c((0, 0, 0))}\\4a&H{255 - sp.shadow_alpha:02X}&\\blur{1.2 * u:.1f}"
            cx = lay.width / 2
            sep = " \\h" if sp.pill else " "      # Dela is tight: a wider word gap so a pill never touches a neighbour
            ev.append(f"Dialogue: 3,{t0},{t1},T,,0,0,0,,{{\\an5\\fs{c.size:.0f}\\move({cx:.1f},{cy + RISE_PX * u:.1f},"
                      f"{cx:.1f},{cy:.1f},0,{ENTER_MS})\\fscx96\\fscy96\\t(0,{ENTER_MS},0.5,\\fscx100\\fscy100)"
                      f"{shadow}{fade}}}" + sep.join(parts))
            if sp.pill:
                ev.extend(_pills(c, ln, li, cy, widths[li], tpl, lay))
    return "\n".join(head + ev) + "\n"


def _pills(c: Cue, ln: List[int], li: int, cy: float, line_w: float, tpl: Template, lay: Layout) -> List[str]:
    """Template 4: the spoken word gets a violet pill under it (drawn per word, its own short fade)."""
    sp, u = sub_spec(tpl, lay), lay.u
    out = []
    x = (lay.width - line_w) / 2
    space = 2 * text_width(" ", sp.font_file, c.size)
    h = c.size * 1.02
    padx = 9 * u
    for j in ln:
        text, ws, we = c.words[j]
        w = text_width(text, sp.font_file, c.size)
        a, b = max(c.start, ws - 0.04), min(c.end, max(we + 0.17, ws + 0.25))
        if b - a >= 0.08:
            px, py = x - padx, cy - h / 2
            fd = f"\\fad({min(90, int((b - a) * 300))},{min(70, int((b - a) * 250))})"
            out.append(f"Dialogue: 2,{_t(a)},{_t(b)},P,,0,0,0,,{{\\an7\\pos({px:.1f},{py:.1f})\\bord0\\shad0\\blur0.6"
                       f"\\1c{_c(sp.pill)}{fd}\\p1}}{_rounded(w + 2 * padx, h, 16 * u)}{{\\p0}}")
        x += w + space
    return out


def write_subtitles(words_local: Sequence[Dict[str, Any]], duration: float, tpl: Template, lay: Layout,
                    path: Path) -> Optional[str]:
    cues = build_cues(words_local, duration, tpl, lay)
    if not cues:
        return None
    path = Path(path)
    path.write_text(to_ass(cues, tpl, lay), encoding="utf-8")
    return str(path)


# ── title (the hook) ─────────────────────────────────────────────────────────

TITLE_SHOW = (0.15, 3.6)    # seconds of the clip: the title leads in, then gives the frame to the explanation


def title_image(title: str, tpl: Template, lay: Layout, part: Optional[Tuple[int, int]] = None
                ) -> Optional[Image.Image]:
    """A rounded plate with an optional «ЧАСТЬ 2 ИЗ 5» label and the title in up to two lines; None if it can't fit."""
    title = " ".join(str(title or "").split())
    if not title:
        return None
    u = lay.u
    max_w = int(lay.safe_w)
    pad = int(36 * u)
    sh = int(22 * u)
    tsize = int(64 * u)
    font = None
    lines: List[str] = []
    words = title.split()
    for k in (1.0, 0.9, 0.8):
        font = _font(tpl.title_font, int(tsize * k))
        room = max_w - 2 * pad
        if font.getlength(title) <= room:
            lines = [title]
            break
        # two balanced lines: no orphan word on the second line
        splits = [(max(font.getlength(" ".join(words[:i])), font.getlength(" ".join(words[i:]))), i)
                  for i in range(1, len(words))]
        if splits:
            widest, i = min(splits)
            if widest <= room:
                lines = [" ".join(words[:i]), " ".join(words[i:])]
                break
    else:
        return None
    lab_font = _font(tpl.label_font, int(26 * u))
    label = f"ЧАСТЬ {part[0]} ИЗ {part[1]}" if part and part[1] > 1 else ""
    lh = int(font.size * 1.16)
    lab_h = int(24 * u * 1.9) if label else 0
    w = int(max([font.getlength(l) for l in lines] + [lab_font.getlength(label) + 6 * u * len(label) * 0.25]) + 2 * pad)
    h = pad + lab_h + lh * len(lines) + int(pad * 0.85)
    im = Image.new("RGBA", (w + 2 * sh, h + 2 * sh), (0, 0, 0, 0))
    shadow = Image.new("RGBA", im.size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle((sh, sh + int(8 * u), sh + w, sh + h + int(8 * u)), int(30 * u),
                                             fill=(0, 0, 0, 70 if not tpl.dark else 120))
    im.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(sh / 2)))
    k = 3
    big = Image.new("RGBA", (im.width * k, im.height * k), (0, 0, 0, 0))
    bd = ImageDraw.Draw(big)
    bd.rounded_rectangle((sh * k, sh * k, (sh + w) * k, (sh + h) * k), int(30 * u) * k, fill=tpl.title_bg + (250,))
    if not tpl.dark and tpl.title_bg in (WHITE, SILVER):
        bd.rounded_rectangle((sh * k, sh * k, (sh + w) * k, (sh + h) * k), int(30 * u) * k, outline=SILVER_LINE + (255,),
                             width=max(1, int(1.5 * u * k)))
    im.alpha_composite(big.resize(im.size, Image.LANCZOS))
    d = ImageDraw.Draw(im)
    y = sh + pad
    if label:
        x = sh + pad
        for ch in label:                    # letter-spaced small caps label
            d.text((x, y), ch, font=lab_font, fill=tpl.title_label)
            x += lab_font.getlength(ch) + 1.5 * u
        y += lab_h
    for l in lines:
        d.text((sh + pad, y - int(font.size * 0.08)), l, font=font, fill=tpl.title_ink)
        y += lh
    return im


def title_position(img: Image.Image, lay: Layout) -> Tuple[int, int]:
    top = int(lay.height * (0.125 if lay.height > lay.width else 0.06))
    return max(0, (lay.width - img.width) // 2), top
