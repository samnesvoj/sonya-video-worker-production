"""
Telegram HTML formatter for every Vast cost message (daily, monthly, /vast, alert). Pure functions, no I/O, no LLM.

Layout rules (phones first: iPhone and Android, portrait):
  * tables are <pre> blocks no wider than WIDTH monospace columns -> no horizontal scrolling, no wrapping;
  * numbers are right-aligned in their column, labels are left-aligned and shortened with "…" when too long;
  * currency signs stay OUTSIDE <pre> (₽ is not in every monospace font); columns are titled USD / RUB instead;
  * a total row is separated by a dashed rule and written in capitals; block titles are <b>CAPS</b>;
  * every dynamic value goes through esc() (Telegram HTML: & < > only);
  * the whole message is kept under LIMIT characters (Telegram: 4096) by dropping optional blocks, never by cutting a tag.
Money: Decimal in, Decimal out. Rounding happens HERE, once, at display time (ROUND_HALF_UP), never before a sum.
"""
from __future__ import annotations

import html
import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Iterable, List, Optional, Sequence, Tuple, Union

WIDTH = 32                 # monospace columns that fit a portrait phone without wrapping (iPhone SE / small Android ≈ 34)
LIMIT = 3900               # Telegram hard limit is 4096; the bot also refuses > 3900
DASH = "—"                 # missing value
RULE = "-"

Num = Union[Decimal, int, float, str, None]
Cell = Union[str, Decimal, int, None]


def esc(v) -> str:
    return html.escape("" if v is None else str(v), quote=False)


def dec(v: Num) -> Optional[Decimal]:
    """Decimal or None for missing/invalid input. Floats go through str() so 0.1 stays 0.1."""
    if v is None or v == "":
        return None
    if isinstance(v, Decimal):
        return v if v.is_finite() else None
    try:
        d = Decimal(str(v))
    except (InvalidOperation, ValueError):
        return None
    return d if d.is_finite() else None


def num(v: Num, places: int = 2) -> str:
    """'1 234.56', '-0.26', '0.00' (never '-0.00'), '—' when missing. Thousands separated by a plain space."""
    d = dec(v)
    if d is None:
        return DASH
    q = d.quantize(Decimal(1).scaleb(-places), ROUND_HALF_UP)
    if q == 0:
        q = abs(q)
    s = f"{q:,.{places}f}".replace(",", " ")
    return s


def usd(v: Num, places: int = 2) -> str:
    s = num(v, places)
    if s == DASH:
        return DASH
    return "-$" + s[1:] if s.startswith("-") else "$" + s


def rub(v: Num, places: int = 2) -> str:
    s = num(v, places)
    return DASH if s == DASH else s + " ₽"


def to_rub(v: Num, rate: Num) -> Optional[Decimal]:
    a, r = dec(v), dec(rate)
    return None if a is None or r is None else a * r


def pair(usd_v: Num, rub_v: Num) -> str:
    """'$0.69 / 56.63 ₽' (both already exact; each rounded on its own)."""
    return f"{usd(usd_v)} / {rub(rub_v)}"


def hours(v: Num) -> str:
    s = num(v, 2)
    return DASH if s == DASH else s + " ч"


def shorten(label: str, width: int) -> str:
    label = re.sub(r"\s+", " ", str(label or "")).strip() or DASH
    if len(label) <= width:
        return label
    return label[: max(1, width - 1)].rstrip() + "…"


# ---------------------------------------------------------------- tables
class Row:
    __slots__ = ("label", "cells", "total", "rule_before")

    def __init__(self, label: str, *cells: str, total: bool = False, rule_before: bool = False):
        self.label, self.cells, self.total, self.rule_before = label, [str(c) for c in cells], total, rule_before or total


def table(rows: Sequence[Row], header: Sequence[str] = (), width: int = WIDTH) -> str:
    """<pre> table: first column left, the rest right-aligned; total rows get a rule and CAPS. Never wider than `width`."""
    ncols = max([len(r.cells) for r in rows] + [len(header)]) if rows or header else 0
    cols = [0] * ncols
    for r in rows:
        for i, c in enumerate(r.cells):
            cols[i] = max(cols[i], len(c))
    for i, h in enumerate(header):
        cols[i] = max(cols[i], len(h))
    gap = 2 if sum(cols) + 2 * ncols + 8 <= width else 1
    num_w = sum(cols) + gap * ncols
    label_w = max(4, width - num_w)
    longest = max([len(r.label.upper() if r.total else r.label) for r in rows] + [0])
    label_w = min(label_w, max(longest, 4))
    lines: List[str] = []
    if header:
        lines.append(" " * label_w + "".join(" " * gap + h.rjust(cols[i]) for i, h in enumerate(header)))
    for r in rows:
        if r.rule_before:
            lines.append(RULE * min(width, label_w + num_w))
        lab = shorten(r.label.upper() if r.total else r.label, label_w).ljust(label_w)
        lines.append(lab + "".join(" " * gap + c.rjust(cols[i]) for i, c in enumerate(r.cells)))
    return "<pre>" + esc("\n".join(line.rstrip() for line in lines)) + "</pre>"


def money_rows(items: Iterable[Tuple[str, Num, Num, dict]]) -> List[Row]:
    """[(label, usd, rub, {'total': bool, 'rule': bool})] -> Rows with USD/RUB cells."""
    out = []
    for label, u, r, opt in items:
        out.append(Row(label, num(u), num(r), total=opt.get("total", False), rule_before=opt.get("rule", False)))
    return out


def rounding_mismatch(parts: Sequence[Num], total: Num, places: int = 2) -> bool:
    """True when the displayed parts do not add up to the displayed total (only because each is rounded on its own)."""
    if dec(total) is None or any(dec(p) is None for p in parts):
        return False
    q = Decimal(1).scaleb(-places)
    shown = sum((dec(p).quantize(q, ROUND_HALF_UP) for p in parts), Decimal(0))
    return shown != dec(total).quantize(q, ROUND_HALF_UP)


# ---------------------------------------------------------------- message
def title(text: str) -> str:
    return f"<b>{esc(text)}</b>"


def block(head: str, *body: str, sub: str = "") -> str:
    h = title(head) + (f" · {esc(sub)}" if sub else "")
    return "\n".join([h] + [b for b in body if b])


def status_line(ok: bool, text_ok: str, text_bad: str = "Требует сверки") -> str:
    return f"Статус: {esc(text_ok)}" if ok else f"Статус: <b>{esc(text_bad)}</b>"


def bullets(items: Iterable[str]) -> str:
    return "\n".join("• " + esc(i) for i in items)


def plain(text: str) -> str:
    """Same message without HTML (fallback if Telegram ever rejects the markup)."""
    return html.unescape(re.sub(r"</?(b|i|u|s|code|pre)>", "", text))


def assemble(blocks: Sequence[Tuple[str, bool]], limit: int = LIMIT) -> str:
    """blocks: [(html, optional)]. Joins with blank lines; while too long, drops optional blocks from the END,
    then adds a one-line note. Required blocks are never cut (a tag is never split)."""
    keep = [list(b) for b in blocks if b[0]]
    dropped = 0

    def text() -> str:
        parts = [b[0] for b in keep]
        if dropped:
            parts.append(f"<i>Не поместилось в сообщение Telegram: {dropped} доп. блок(а). Полные данные — в журнале учёта.</i>")
        return "\n\n".join(parts)

    while len(text()) > limit:
        idx = next((i for i in range(len(keep) - 1, -1, -1) if keep[i][1]), None)
        if idx is None:
            break
        keep.pop(idx)
        dropped += 1
    out = text()
    if len(out) > limit:                       # only required blocks left and still too long: hard, tag-safe cut
        out = out[: limit - 40]
        out = out[: out.rfind("\n\n")] if "\n\n" in out else out[: out.rfind("\n")]
        out = _close_tags(out) + "\n\n<i>Сообщение сокращено.</i>"
    return out


def _close_tags(s: str) -> str:
    stack: List[str] = []
    for m in re.finditer(r"<(/?)(b|i|u|s|code|pre)>", s):
        if m.group(1):
            if stack and stack[-1] == m.group(2):
                stack.pop()
        else:
            stack.append(m.group(2))
    return s + "".join(f"</{t}>" for t in reversed(stack))

