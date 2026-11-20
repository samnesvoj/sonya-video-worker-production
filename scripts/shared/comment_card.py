"""
comment_card.py
===============
Pictures and effects of the «Завозный коммент» clip (scripts/shared/stream_comment.py), all for the 1080×1920 frame:

  card_png(...)   the viewer's comment as a Twitch / YouTube chat line on a dark rounded plate: a small badge,
                  the nick in the viewer's colour (bold), «: » and the text in white, wrapped — readable on a phone
                  (44 px text, ≤ 920 px wide). Nick hidden → «зритель» in grey.
  title_png(...)  the big two-line title («ЗЛОЙ УСТАНОВИЛ / МИРОВОЙ РЕКОРД В GTA6»): heavy caps, thick black
                  outline, accent words purple / yellow / pink, the rest white.
  glitch_vf(...)  ffmpeg filters for a ~0.4 s glitch transition (green tint, RGB split, scan lines, noise, a flash)
                  at a cut — every filter is timeline-enabled, so the rest of the clip is untouched (ffmpeg ≥ 4.3).
  layout(...)     where the title and the card go for a template (seam_y = the line the subtitles sit on).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

W, H = 1080, 1920
FONTS = Path(__file__).parent / "fonts"
TITLE_FONT = "Montserrat-Black.ttf"
CARD_BOLD, CARD_TEXT = "Rubik-Bold.ttf", "Rubik-Medium.ttf"
ACCENTS = ("#9B4DFF", "#FFE01A", "#FF3D8E")             # purple / yellow / pink, like the reference
TWITCH_DEFAULT = ("#FF4A4A", "#1E90FF", "#00C46A", "#B266FF", "#FF7F27", "#FF69B4", "#2EC4B6", "#DAA520")


def _font(name: str, size: int):
    from PIL import ImageFont
    return ImageFont.truetype(str(FONTS / name), size)


def _hex(c: str, a: int = 255) -> Tuple[int, int, int, int]:
    c = c.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16), a


def nick_color(author: str, color: Optional[str]) -> str:
    """The viewer's own chat colour when the replay has one (too dark ones lifted), else a stable Twitch-like pick."""
    if color:
        r, g, b, _ = _hex(color)
        if 0.299 * r + 0.587 * g + 0.114 * b < 110 and max(r, g, b) < 200:   # too dark on the black plate: lift
            r, g, b = (int(v + (255 - v) * 0.45) for v in (r, g, b))
        return f"#{r:02X}{g:02X}{b:02X}"
    h = int(hashlib.md5((author or "x").encode("utf-8")).hexdigest()[:6], 16)
    return TWITCH_DEFAULT[h % len(TWITCH_DEFAULT)]


def _wrap(draw, first_prefix_w: int, text: str, font, width: int) -> List[str]:
    """Greedy wrap; the first line is shorter by the nick in front of it."""
    lines, cur = [], ""
    avail = width - first_prefix_w
    for word in text.split():
        trial = (cur + " " + word).strip()
        if draw.textlength(trial, font=font) <= avail or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur, avail = word, width
    if cur:
        lines.append(cur)
    return lines


def card_png(out: str, author: str, text: str, color: Optional[str] = None, *, platform: str = "twitch",
             max_w: int = 920, size: int = 44, max_lines: int = 4) -> Dict[str, Any]:
    """-> {"path", "w", "h"}. The plate is semi-opaque black (#0E0E10 @ 88 %) with 22 px rounded corners."""
    from PIL import Image, ImageDraw
    hidden = not author or author == "зритель"
    nick = "зритель" if hidden else author
    fb, ft = _font(CARD_BOLD, size), _font(CARD_TEXT, size)
    pad, badge, gap = 30, int(size * 0.95), 14
    probe = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    nick_w = probe.textlength(nick + ":", font=fb)
    head = badge + gap + nick_w + probe.textlength(" ", font=ft)
    inner = max_w - 2 * pad
    lines = _wrap(probe, int(head), text, ft, inner)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(" ,.") + "…"
    lh = int(size * 1.32)
    widest = max([head + probe.textlength(lines[0], font=ft)] + [probe.textlength(x, font=ft) for x in lines[1:]])
    w = int(min(max_w, widest + 2 * pad))
    h = int(2 * pad + lh * len(lines) - (lh - size) + 6)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, w - 1, h - 1], radius=22, fill=(14, 14, 16, 225))
    top = pad
    by = top + (size - badge) // 2 + 4
    if platform == "youtube":                          # a round avatar with the first letter
        d.ellipse([pad, by, pad + badge, by + badge], fill=_hex(nick_color(author, color)))
        d.text((pad + badge / 2, by + badge / 2), (nick[:1] or "?").upper(), font=_font(CARD_BOLD, int(badge * 0.6)),
               fill=(255, 255, 255, 255), anchor="mm")
    else:                                              # Twitch-like square badge with a chat bubble
        d.rounded_rectangle([pad, by, pad + badge, by + badge], radius=8, fill=_hex("#9146FF"))
        bx, bw = pad + badge * 0.22, badge * 0.56
        d.rounded_rectangle([bx, by + badge * 0.24, bx + bw, by + badge * 0.62], radius=4, fill=(255, 255, 255, 255))
        d.polygon([(bx + bw * 0.25, by + badge * 0.6), (bx + bw * 0.25, by + badge * 0.8), (bx + bw * 0.5, by + badge * 0.6)],
                  fill=(255, 255, 255, 255))
    x = pad + badge + gap
    d.text((x, top), nick + ":", font=fb, fill=(173, 173, 184, 255) if hidden else _hex(nick_color(author, color)))
    x += nick_w + probe.textlength(" ", font=ft)
    for i, line in enumerate(lines):
        d.text((x if i == 0 else pad, top + i * lh), line, font=ft, fill=(239, 239, 241, 255))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return {"path": out, "w": w, "h": h}


def split_title(text: str) -> List[str]:
    """Two balanced lines (by characters); one line when it is short."""
    words = text.split()
    if len(text) <= 16 or len(words) < 2:
        return [text]
    best, cut = None, 1
    for k in range(1, len(words)):
        a, b = " ".join(words[:k]), " ".join(words[k:])
        d = abs(len(a) - len(b))
        if best is None or d < best:
            best, cut = d, k
    return [" ".join(words[:cut]), " ".join(words[cut:])]


def title_png(out: str, text: str, accent: Sequence[str] = (), *, max_w: int = 1000, size: int = 84) -> Dict[str, Any]:
    """Heavy caps, white with accent words coloured; 9 px black outline + a soft shadow. -> {"path", "w", "h"}"""
    from PIL import Image, ImageDraw, ImageFilter
    text = " ".join(text.upper().split())
    acc = [a.upper() for a in accent if a]
    if not acc:                                        # no accents given: the first word and the last one
        ws = text.split()
        acc = [ws[0]] + ([ws[-1]] if len(ws) > 2 else [])
    lines = split_title(text)
    probe = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    while size > 44:
        f = _font(TITLE_FONT, size)
        if max(probe.textlength(x, font=f) for x in lines) <= max_w - 30:
            break
        size -= 4
    f = _font(TITLE_FONT, size)
    stroke = max(5, size // 9)
    lh = int(size * 1.08)
    w = max_w
    h = lh * len(lines) + 2 * stroke + 16
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ds, d = ImageDraw.Draw(shadow), ImageDraw.Draw(img)
    colours: Dict[str, str] = {}
    k = 0
    for a in acc:
        for part in a.split():
            colours.setdefault(part, ACCENTS[k % len(ACCENTS)])
        k += 1
    space = probe.textlength(" ", font=f)
    for i, line in enumerate(lines):
        lw = probe.textlength(line, font=f)
        x = (w - lw) / 2
        y = stroke + 4 + i * lh
        for word in line.split():
            fill = colours.get(word, "#FFFFFF")
            ds.text((x + 4, y + 7), word, font=f, fill=(0, 0, 0, 160), stroke_width=stroke, stroke_fill=(0, 0, 0, 160))
            d.text((x, y), word, font=f, fill=_hex(fill), stroke_width=stroke, stroke_fill=(0, 0, 0, 255))
            x += probe.textlength(word, font=f) + space
    shadow = shadow.filter(ImageFilter.GaussianBlur(6))
    shadow.alpha_composite(img)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    shadow.save(out)
    return {"path": out, "w": w, "h": h}


def layout(seam_y: Optional[int], title_h: int, card_h: int) -> Tuple[int, int]:
    """(title_y, card_y) in the 1080×1920 frame. Template 1/2/3 (webcam on top, game below, subtitles just above
    the seam): the title sits in the lower part of the webcam above the subtitle band, the card at the top of the
    game. A low seam (template 4: full game, subtitles at ~1500): the title near the top, the card under it."""
    if seam_y and 600 <= seam_y <= 1150:
        title_y = max(40, seam_y - 250 - title_h)
        card_y = seam_y + 70
    else:
        title_y = 230
        card_y = title_y + title_h + 40
    return int(title_y), int(min(H - card_h - 360, card_y))


def glitch_vf(at: float, dur: float = 0.42) -> str:
    """Comma-joined filters for a glitch transition centred on ``at`` (output seconds). Every filter is timeline
    enabled, so outside the window the frame passes untouched. Flash = 2 frames of brightness in the middle."""
    a, b = max(0.0, at - dur / 2), at + dur / 2
    en = f"enable='between(t,{a:.3f},{b:.3f})'"
    mid = f"enable='between(t,{at - 0.035:.3f},{at + 0.035:.3f})'"
    return ",".join([
        f"rgbashift=rh=-22:bh=22:gv=6:edge=wrap:{en}",
        f"colorchannelmixer=rr=0.35:rg=0.25:gg=1.45:gb=0.15:bb=0.35:bg=0.2:{en}",
        f"eq=brightness=0.10:saturation=1.4:{en}",
        f"drawgrid=w=iw:h=14:t=6:c=black@0.45:{en}",
        f"noise=alls=38:allf=t:{en}",
        f"eq=brightness=0.32:contrast=1.3:{mid}",
    ])
