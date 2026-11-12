"""
cards.py — draws one validated overlay into transparent layers. Pillow only.

A card is a list of layers (full-card RGBA images, each with its own appear time): the panel with the title first,
then one layer per diagram step / chart value, so each element appears when the speaker reaches it.
Every text box is measured; a card that cannot fit raises CardOverflow and is dropped (text never leaves the panel,
and the panel's position is checked against the frame in placement.py).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from . import formula as fm
from .manifest import FACT_KINDS
from .themes import Theme, get_theme

# Part of the cache key: a change to the drawing code must never be answered with an old picture.
RENDERER_VERSION = "edu-cards-" + hashlib.sha256(
    b"".join(Path(__file__).with_name(n).read_bytes() for n in ("cards.py", "themes.py", "formula.py"))).hexdigest()[:12]
_FONTS = Path(__file__).resolve().parents[3] / "scripts" / "shared" / "fonts"
BOLD, SEMI = _FONTS / "Montserrat-ExtraBold.ttf", _FONTS / "Montserrat-SemiBold.ttf"
SHADOW = 14


def _B(t: Theme) -> Path:
    return _FONTS / t.bold_font


def _S(t: Theme) -> Path:
    return _FONTS / t.semi_font


class CardOverflow(RuntimeError):
    pass


@dataclass
class Layer:
    image: Image.Image
    at_ms: int


@dataclass
class Card:
    width: int
    height: int
    layers: List[Layer]


class _Box:
    """Drawing surface of fixed width that grows downward and refuses text wider than itself."""

    def __init__(self, width: int, u: float, theme: Theme, max_height: int):
        self.w, self.u, self.t = width, u, theme
        self.max_h = max_height
        self.pad = int(34 * u) + SHADOW
        self.inner = width - 2 * self.pad
        self.y = self.pad
        self.layers: List[Tuple[Image.Image, int]] = []
        self.new_layer(0)

    def new_layer(self, at_ms: int) -> None:
        self.im = Image.new("RGBA", (self.w, self.max_h), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.im)
        self.layers.append((self.im, at_ms))

    def font(self, path: Path, px: float) -> ImageFont.FreeTypeFont:
        return ImageFont.truetype(str(path), max(10, int(round(px * self.u))))

    def wrap(self, text: str, font, width: int, max_lines: int) -> List[str]:
        lines: List[str] = []
        cur = ""
        for word in str(text).split():
            if font.getlength(word) > width:
                raise CardOverflow("word_wider_than_card")
            trial = f"{cur} {word}".strip()
            if font.getlength(trial) <= width:
                cur = trial
            else:
                lines.append(cur)
                cur = word
        if cur:
            lines.append(cur)
        if len(lines) >= 2 and " " not in lines[-1] and " " in lines[-2]:
            head, last = lines[-2].rsplit(" ", 1)          # no orphan word on the last line («…, квантовый / бит»)
            if font.getlength(f"{last} {lines[-1]}") <= width:
                lines[-2], lines[-1] = head, f"{last} {lines[-1]}"
        if len(lines) > max_lines:
            raise CardOverflow("too_many_lines")
        return lines

    def text(self, text: str, path: Path, px: float, color, *, max_lines: int = 3, x: Optional[int] = None,
             width: Optional[int] = None, gap: float = 1.22, after: float = 10) -> None:
        f = self.font(path, px)
        left = self.pad if x is None else x
        w = self.inner if width is None else width
        for line in self.wrap(text, f, w, max_lines):
            self._check(left, self.y, f.getlength(line), px * self.u)
            self.d.text((left, self.y), line, font=f, fill=color)
            self.y += int(px * self.u * gap)
        self.y += int(after * self.u)

    def _check(self, x: float, y: float, w: float, h: float) -> None:
        if x < self.pad - 1 or x + w > self.w - self.pad + 1 or y + h > self.max_h - self.pad:
            raise CardOverflow("text_outside_panel")

    def finish(self) -> Card:
        h = self.y + self.pad - int(10 * self.u)
        if h > self.max_h:
            raise CardOverflow("card_too_tall")
        return _assemble(self.w, h, self.layers, self.t, self.u)


def _panel(width: int, height: int, theme: Theme, u: float) -> Image.Image:
    im = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    box = (SHADOW, SHADOW, width - SHADOW - 1, height - SHADOW - 1)
    radius = int(theme.radius * u)
    shadow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle((box[0], box[1] + int(4 * u), box[2], box[3] + int(4 * u)), radius,
                                             fill=(0, 0, 0, theme.shadow_alpha))
    im.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(SHADOW / 2)))
    # supersampled so the rounded corners are clean
    k = 3
    big = Image.new("RGBA", (width * k, height * k), (0, 0, 0, 0))
    ImageDraw.Draw(big).rounded_rectangle([v * k for v in box], radius * k, fill=theme.panel + (theme.panel_alpha,))
    if theme.border:
        ImageDraw.Draw(big).rounded_rectangle([v * k for v in box], radius * k, outline=theme.border + (255,),
                                              width=max(1, int((3 if theme.border == theme.accent else 1.5) * u * k)))
    im.alpha_composite(big.resize((width, height), Image.LANCZOS))
    return im


def _assemble(width: int, height: int, layers, theme: Theme, u: float) -> Card:
    out: List[Layer] = []
    for i, (im, at) in enumerate(layers):
        crop = im.crop((0, 0, width, height))
        if i == 0:
            base = _panel(width, height, theme, u)
            base.alpha_composite(crop)
            crop = base
        out.append(Layer(crop, at))
    return Card(width, height, out)


def _fmt(v: float) -> str:
    s = f"{v:,.0f}".replace(",", " ") if float(v).is_integer() else f"{v:g}".replace(".", ",")
    return s


def _cap(text: Any) -> str:
    t = str(text or "").strip()
    return t[:1].upper() + t[1:]


def formula_card(item: Dict[str, Any], width: int, u: float, theme: Theme, max_h: int, backend: Optional[str]) -> Card:
    c = item["content"]
    pad = int(34 * u) + SHADOW
    target = int((132 if "frac" in c["tex"] else 84) * u)
    im = fm.render_formula(c["tex"], target, theme.ink, backend)
    k = target / im.height      # backends differ in how tall a given size comes out: normalise to the target
    im = im.resize((max(1, int(im.width * k)), target), Image.LANCZOS)
    if im.width > width - 2 * pad:
        k = (width - 2 * pad) / im.width
        if k < 0.6:
            raise CardOverflow("formula_too_wide")
        im = im.resize((int(im.width * k), int(im.height * k)), Image.LANCZOS)
    cap_font = ImageFont.truetype(str(_S(theme)), int(29 * u))
    caption = _cap(c.get("caption"))
    cap_w = int(cap_font.getlength(caption)) if caption else 0
    w = min(width, max(im.width, cap_w) + 2 * pad)
    box = _Box(w, u, theme, max_h)
    if im.height + 2 * pad > max_h:
        raise CardOverflow("formula_too_tall")
    box.im.alpha_composite(im, (pad, box.y))
    box.y += im.height + int(14 * u)
    if caption:
        box.text(caption, _S(theme), 29, theme.muted, max_lines=2, after=0)
    return box.finish()


def fact_card(item: Dict[str, Any], width: int, u: float, theme: Theme, max_h: int) -> Card:
    c = item["content"]
    box = _Box(width, u, theme, max_h)
    box.text(FACT_KINDS[c["kind"]].upper(), _B(theme), 24, theme.accent, max_lines=1, after=8)
    if c.get("headline"):
        box.text(_cap(c["headline"]), _B(theme), 52, theme.ink, max_lines=2, after=6)
    if c.get("body"):
        box.text(_cap(c["body"]), _S(theme), 33, theme.ink, max_lines=3, after=4)
    if c.get("source_label"):
        box.text("Источник: " + c["source_label"], _S(theme), 24, theme.muted, max_lines=2, after=0)
    return box.finish()


def term_card(item: Dict[str, Any], width: int, u: float, theme: Theme, max_h: int) -> Card:
    c = item["content"]
    box = _Box(width, u, theme, max_h)
    box.text("ТЕРМИН", _B(theme), 24, theme.accent, max_lines=1, after=8)
    box.text(c["term"], _B(theme), 50, theme.ink, max_lines=2, after=8)
    box.text(_cap(c["definition"]), _S(theme), 33, theme.ink, max_lines=4, after=0)
    return box.finish()


def _chip(box: _Box, x: int, y: int, size: int, label: str, color) -> None:
    k = 3
    chip = Image.new("RGBA", (size * k, size * k), (0, 0, 0, 0))
    ImageDraw.Draw(chip).ellipse((0, 0, size * k - 1, size * k - 1), fill=color + (255,))
    box.im.alpha_composite(chip.resize((size, size), Image.LANCZOS), (x, y))
    f = box.font(_B(box.t), 28)
    tw = f.getlength(label)
    box.d.text((x + (size - tw) / 2, y + size / 2 - 28 * box.u * 0.62), label, font=f, fill=box.t.panel)


def diagram_card(item: Dict[str, Any], width: int, u: float, theme: Theme, max_h: int) -> Card:
    c = item["content"]
    kind, nodes = c["kind"], c["nodes"]
    box = _Box(width, u, theme, max_h)
    if c.get("title"):
        box.text(c["title"], _B(theme), 34, theme.ink, max_lines=2, after=14)
    chip = int(50 * u)
    tx = box.pad + chip + int(18 * u)
    tw = box.w - box.pad - tx
    if kind == "a_vs_b":
        gap = int(22 * u)
        cw = (box.inner - gap) // 2
        top = box.y
        bottom = top
        for i, n in enumerate(nodes):
            box.new_layer(int(n["at_ms"]))
            box.y = top
            x = box.pad + i * (cw + gap)
            box.d.rectangle((x, top, x + cw, top + int(6 * u)), fill=(theme.accent if i == 0 else theme.accent2) + (255,))
            box.y += int(16 * u)
            box.text(_cap(n["label"]), _S(theme), 32, theme.ink, max_lines=4, x=x, width=cw, after=0)
            bottom = max(bottom, box.y)
        box.y = bottom
        return box.finish()
    names = {"cause_effect": ["Причина", "Следствие"]}.get(kind)
    prev_bottom = None
    for i, n in enumerate(nodes):
        box.new_layer(int(n["at_ms"]))
        y0 = box.y
        if prev_bottom is not None:
            cx = box.pad + chip // 2
            box.d.line((cx, prev_bottom, cx, y0 + int(2 * u)), fill=theme.muted + (255,), width=max(2, int(3 * u)))
        _chip(box, box.pad, y0 + int(2 * u), chip, "→" if names and i else str(i + 1), theme.accent)
        if names:
            box.text(names[i], _S(theme), 23, theme.muted, max_lines=1, x=tx, width=tw, after=0)
        box.text(_cap(n["label"]), _S(theme), 33, theme.ink, max_lines=3, x=tx, width=tw, after=0)
        prev_bottom = y0 + chip + int(4 * u)
        box.y = max(box.y, prev_bottom) + int(16 * u)
    box.y -= int(8 * u)
    return box.finish()


def chart_card(item: Dict[str, Any], width: int, u: float, theme: Theme, max_h: int) -> Card:
    c = item["content"]
    kind, pts, unit = c["kind"], c["points"], str(c["unit"]).strip()
    box = _Box(width, u, theme, max_h)
    box.text(c.get("title") or "", _B(theme), 34, theme.ink, max_lines=2, after=4)
    box.text(f"Единицы: {unit}", _S(theme), 24, theme.muted, max_lines=1, after=12)
    values = [float(p["value"]) for p in pts]
    top = 100.0 if kind == "share" else max(values)
    if kind == "line":
        return _line_chart(box, pts, values, unit, theme, u)
    bar_h = int(30 * u)
    for p, v in zip(pts, values):
        box.new_layer(int(p["at_ms"]))
        f = box.font(_S(box.t), 27)
        vf = box.font(_B(box.t), 29)
        val = f"{_fmt(v)} {unit}"
        vw = int(vf.getlength(val))
        box._check(box.pad + box.inner - vw, box.y, vw, 29 * u)
        if f.getlength(str(p["label"])) + vw + 16 * u > box.inner:
            raise CardOverflow("chart_label_too_long")
        box.d.text((box.pad, box.y), str(p["label"]), font=f, fill=theme.ink)
        box.d.text((box.pad + box.inner - vw, box.y - int(1 * u)), val, font=vf, fill=theme.ink)
        box.y += int(38 * u)
        box.d.rounded_rectangle((box.pad, box.y, box.pad + box.inner, box.y + bar_h), bar_h // 2,
                                fill=theme.muted + (60,))
        bw = max(bar_h, int(box.inner * (v / top if top else 0)))    # the axis always starts at zero
        box.d.rounded_rectangle((box.pad, box.y, box.pad + bw, box.y + bar_h), bar_h // 2, fill=theme.accent + (255,))
        box.y += bar_h + int(18 * u)
    box.y -= int(8 * u)
    return box.finish()


def _line_chart(box: _Box, pts, values, unit: str, theme: Theme, u: float) -> Card:
    plot_h = int(165 * u)
    left, right = box.pad + int(10 * u), box.pad + box.inner - int(10 * u)
    y_top = box.y + int(44 * u)
    y_base = y_top + plot_h - int(44 * u)
    box.d.line((left, y_base, right, y_base), fill=theme.muted + (255,), width=max(2, int(2 * u)))
    n = len(pts)
    top = max(values) or 1.0        # zero-based, a line chart must not exaggerate a change
    xy = [(left + int((right - left) * (i + 0.5) / n), int(y_base - (y_base - y_top) * v / top)) for i, v in enumerate(values)]
    f, vf = box.font(_S(box.t), 24), box.font(_B(box.t), 26)
    for i, (p, v) in enumerate(zip(pts, values)):
        box.new_layer(int(p["at_ms"]))
        if i:
            box.d.line((xy[i - 1], xy[i]), fill=theme.accent + (255,), width=max(3, int(5 * u)))
        r = int(9 * u)
        box.d.ellipse((xy[i][0] - r, xy[i][1] - r, xy[i][0] + r, xy[i][1] + r), fill=theme.accent + (255,))
        val, lab = _fmt(v), str(p["label"])
        vw, lw = vf.getlength(val), f.getlength(lab)
        if max(vw, lw) > (right - left) / n:
            raise CardOverflow("chart_label_too_long")
        box.d.text((xy[i][0] - vw / 2, xy[i][1] - int(42 * u)), val, font=vf, fill=theme.ink)
        box.d.text((xy[i][0] - lw / 2, y_base + int(10 * u)), lab, font=f, fill=theme.muted)
    box.y = y_base + int(46 * u)
    return box.finish()


def render_card(item: Dict[str, Any], width: int, short_side: int, max_h: int, *, backend: Optional[str] = None,
                text_scale: float = 1.0) -> Card:
    style = item.get("style") or {}
    theme = get_theme(style.get("theme"))
    u = short_side / 1080.0 * float(style.get("size", 1.0)) * text_scale
    t = item["type"]
    if t == "formula":
        return formula_card(item, width, u, theme, max_h, backend)
    if t == "fact":
        return fact_card(item, width, u, theme, max_h)
    if t == "term":
        return term_card(item, width, u, theme, max_h)
    if t == "diagram":
        return diagram_card(item, width, u, theme, max_h)
    if t == "chart":
        return chart_card(item, width, u, theme, max_h)
    raise ValueError(t)


def cache_key(item: Dict[str, Any], width: int, short_side: int, max_h: int, backend: str, text_scale: float = 1.0) -> str:
    """Identical content + style + size + renderer → same layers (formula render and cards are skipped on re-runs)."""
    payload = {"type": item["type"], "content": item["content"], "style": item.get("style"), "w": width, "s": short_side,
               "h": max_h, "backend": backend, "scale": text_scale, "renderer": RENDERER_VERSION}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:24]


def render_cached(item: Dict[str, Any], width: int, short_side: int, max_h: int, cache_dir: Optional[str],
                  backend: Optional[str] = None, text_scale: float = 1.0) -> Tuple[Card, bool]:
    name = fm.backend_name(backend)
    if not cache_dir:
        return render_card(item, width, short_side, max_h, backend=name, text_scale=text_scale), False
    d = Path(cache_dir) / cache_key(item, width, short_side, max_h, name, text_scale)
    meta = d / "layers.json"
    if meta.exists():
        info = json.loads(meta.read_text(encoding="utf-8"))
        layers = []
        for l in info["layers"]:
            with Image.open(d / l["file"]) as im:
                layers.append(Layer(im.convert("RGBA"), l["at_ms"]))
        return Card(info["w"], info["h"], layers), True
    card = render_card(item, width, short_side, max_h, backend=name, text_scale=text_scale)
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, l in enumerate(card.layers):
        l.image.save(d / f"{i:02d}.png")
        rows.append({"file": f"{i:02d}.png", "at_ms": l.at_ms})
    meta.write_text(json.dumps({"w": card.width, "h": card.height, "layers": rows}), encoding="utf-8")
    return card, False
