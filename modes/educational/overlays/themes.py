"""
themes.py — the visual grammar of Educational overlays: three themes, one spacing and motion system.

Only the three themes the task names exist; each has one panel, ink, muted and two accents. No gold or beige.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

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


def _hex(s: str) -> RGB:
    return int(s[1:3], 16), int(s[3:5], 16), int(s[5:7], 16)


THEMES: Dict[str, Theme] = {
    "academic-minimal": Theme("academic-minimal", _hex("#F4F7FB"), 245, _hex("#0F172A"), _hex("#475569"),
                              _hex("#1D4ED8"), _hex("#0F766E")),
    "modern-tech": Theme("modern-tech", _hex("#0B1220"), 238, _hex("#F8FAFC"), _hex("#A9B8CC"),
                         _hex("#38BDF8"), _hex("#34D399")),
    "clean-whiteboard": Theme("clean-whiteboard", _hex("#FFFFFF"), 250, _hex("#111827"), _hex("#4B5563"),
                              _hex("#B91C1C"), _hex("#1D4ED8")),
}
DEFAULT_THEME = "modern-tech"

FADE_IN_SEC = 0.28
FADE_OUT_SEC = 0.22
SLIDE_PX_1080 = 22              # entrance travel at 1080 short side; "soft" intensity halves it, "strong" adds half
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
