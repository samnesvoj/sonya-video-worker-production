"""
themes.py — the visual grammar of Educational overlays: one spacing and motion system, several colour themes.

Three v2 themes plus one per v4 template; each has one panel, ink, muted and two accents. No gold or beige.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

RGB = Tuple[int, int, int]


@dataclass(frozen=True)
class Theme:
    name: str
    panel: RGB
    panel_alpha: int
    ink: RGB
    muted: RGB
    accent: RGB
    accent2: RGB
    bold_font: str = "Montserrat-ExtraBold.ttf"
    semi_font: str = "Montserrat-SemiBold.ttf"
    radius: int = 20                # px at 1080
    border: Optional[RGB] = None    # hairline around a light panel (instead of a heavy shadow)
    shadow_alpha: int = 70


def _hex(s: str) -> RGB:
    return int(s[1:3], 16), int(s[3:5], 16), int(s[5:7], 16)


THEMES: Dict[str, Theme] = {
    "academic-minimal": Theme("academic-minimal", _hex("#F4F7FB"), 245, _hex("#0F172A"), _hex("#475569"),
                              _hex("#1D4ED8"), _hex("#0F766E")),
    "modern-tech": Theme("modern-tech", _hex("#0B1220"), 238, _hex("#F8FAFC"), _hex("#A9B8CC"),
                         _hex("#38BDF8"), _hex("#34D399")),
    "clean-whiteboard": Theme("clean-whiteboard", _hex("#FFFFFF"), 250, _hex("#111827"), _hex("#4B5563"),
                              _hex("#B91C1C"), _hex("#1D4ED8")),
    # Educational v4 templates (modes/educational/templates.py): four different looks, one card theme each
    "edu-clean": Theme("edu-clean", _hex("#FFFFFF"), 250, _hex("#111318"), _hex("#6B6F7A"), _hex("#0A6CFF"),
                       _hex("#2A2E38"), "Onest-Bold.ttf", "Onest-Medium.ttf", 26, _hex("#E3E5EA"), 46),
    "edu-night": Theme("edu-night", _hex("#0E0F12"), 244, _hex("#F5F5F2"), _hex("#9A9DA6"), _hex("#C6F135"),
                       _hex("#E6E7EA"), "GolosText-Bold.ttf", "GolosText-Medium.ttf", 14, _hex("#24262C"), 120),
    "edu-bright": Theme("edu-bright", _hex("#FFFFFF"), 252, _hex("#10123A"), _hex("#5A5D7A"), _hex("#FF4F64"),
                        _hex("#2430E0"), "Rubik-Bold.ttf", "Rubik-Medium.ttf", 34, None, 70),
    "edu-editorial": Theme("edu-editorial", _hex("#F1EFFA"), 250, _hex("#221D45"), _hex("#6A6585"), _hex("#6A5FC8"),
                           _hex("#3B3566"), "Lora-Bold.ttf", "Onest-Medium.ttf", 18, _hex("#DDD9F2"), 40),
}
DEFAULT_THEME = "modern-tech"

FADE_IN_SEC = 0.24             # enter with a strong ease-out, leave faster (Emil Kowalski motion rules)
FADE_OUT_SEC = 0.16
SLIDE_PX_1080 = 16              # entrance travel at 1080 short side; "soft" intensity halves it, "strong" adds half
INTENSITY = {"soft": 0.5, "normal": 1.0, "strong": 1.5}


def luminance(c: RGB) -> float:
    def f(v: int) -> float:
        x = v / 255.0
        return x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(c[0]) + 0.7152 * f(c[1]) + 0.0722 * f(c[2])


def contrast(a: RGB, b: RGB) -> float:
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def get_theme(name: str) -> Theme:
    return THEMES.get(name or DEFAULT_THEME, THEMES[DEFAULT_THEME])
