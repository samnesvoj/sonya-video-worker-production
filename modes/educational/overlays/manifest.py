"""
manifest.py — the editable JSON between "what was said" and "what is drawn".

An overlay item: type, start_ms, end_ms (clip-local), content, source_segment_id, evidence, confidence, style,
position, review_required — plus status/reason, written by validate().

validate() is the gate. It accepts raw LLM proposals or a hand-edited manifest and trusts neither: quotes are looked
up in the ASR words, times are re-derived from those words, numbers and definitions are compared with the quote.
Result statuses:  ok (drawn) · review (not drawn until a person sets approved: true) · rejected (never drawn).
Absence of evidence never degrades into a softer card: no source → no card.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set

from . import evidence as ev
from . import tex as tx

SCHEMA_VERSION = 1
TYPES = ("formula", "fact", "term", "diagram", "chart")
FACT_KINDS = {"quote": "Цитата из видео", "author_data": "Данные автора", "verified": "Проверенный факт"}
DIAGRAM_KINDS = ("steps", "cause_effect", "a_vs_b", "timeline")
CHART_KINDS = ("bar", "line", "share")
PRESETS: Dict[str, Set[str]] = {
    "minimal": {"term", "fact"},
    "balanced": {"formula", "diagram"},
    "deep": {"formula", "fact", "term", "diagram", "chart"},
}
MIN_CONFIDENCE = 0.6
MIN_ASR_PROBABILITY = 0.45        # the speech recogniser itself was unsure → a person has to look
MAX_TEXT = {"title": 60, "label": 44, "body": 120, "term": 40, "definition": 110, "unit": 16, "source": 60, "caption": 64}
MIN_MS, MAX_MS = 2200, 12000


def reading_ms(chars: int) -> int:
    return int(min(9000, max(2800, 1600 + 55 * chars)))


def _reject(item: Dict[str, Any], reason: str) -> Dict[str, Any]:
    item["status"], item["reason"] = "rejected", reason
    item["review_required"] = False
    return item


def _review(item: Dict[str, Any], reason: str) -> Dict[str, Any]:
    if item.get("approved") is True:     # a person looked at it and took responsibility
        item["status"], item["reason"] = "ok", f"approved_over:{reason}"
        item["review_required"] = False
    else:
        item["status"], item["reason"] = "review", reason
        item["review_required"] = True
    return item


class _Ctx:
    def __init__(self, words: List[Dict[str, Any]], clip_duration: float, author_materials: Sequence[Dict[str, Any]],
                 verified_sources: Sequence[Dict[str, Any]]):
        self.words = words
        self.clip_duration = clip_duration
        self.segments = ev.build_segments(words)
        self.author = {str(m.get("id")): m for m in author_materials or []}
        self.verified = {str(s.get("id")): s for s in verified_sources or []}


def _text_len_ok(content: Dict[str, Any], fields: Dict[str, str]) -> Optional[str]:
    for key, kind in fields.items():
        v = content.get(key)
        if v is not None and len(str(v)) > MAX_TEXT[kind]:
            return f"text_too_long:{key}"
    return None


def _locate(item: Dict[str, Any], quote: str, ctx: _Ctx) -> Optional[ev.QuoteMatch]:
    m = ev.find_quote(ctx.words, quote, ctx.segments)
    if m:
        item["source_segment_id"] = m.segment_id
        item.setdefault("evidence", {})["matched_speech"] = m.text
        item["evidence"]["match_ratio"] = round(m.ratio, 3)
    return m


def _set_times(item: Dict[str, Any], start: float, last_end: float, chars: int, ctx: _Ctx) -> Optional[str]:
    start_ms = int(start * 1000)
    end_ms = max(int(last_end * 1000) + 700, start_ms + reading_ms(chars))
    end_ms = min(end_ms, start_ms + MAX_MS, int(ctx.clip_duration * 1000) - 150)
    if end_ms - start_ms < MIN_MS:
        return "too_close_to_clip_end"
    item["start_ms"], item["end_ms"] = start_ms, end_ms
    return None


def _validate_formula(item: Dict[str, Any], ctx: _Ctx) -> Dict[str, Any]:
    c = item["content"]
    try:
        c["tex"] = tx.check_tex(c.get("tex", ""))
    except tx.TexRejected as e:
        return _reject(item, f"tex:{e}")
    if len(str(c.get("caption", ""))) > MAX_TEXT["caption"]:
        c.pop("caption")        # the formula stands without its legend; a long legend is not a reason to lose it
    e = item["evidence"]
    m = _locate(item, e.get("quote", ""), ctx)
    if not m:
        return _reject(item, "quote_not_in_speech")
    if (err := _set_times(item, m.start, m.end, 55, ctx)):
        return _reject(item, err)
    author = ctx.author.get(str(e.get("author_material_id")))
    in_author = bool(author) and tx.canonical(c["tex"]) in {tx.canonical(f) for f in author.get("formulas", [])}
    if in_author:
        e["basis"] = "author_material"
        item.update(status="ok", review_required=False, reason=None)
        return item
    if ev.has_hedge(ev.context_around(ctx.words, m.start, m.end)):
        return _reject(item, "hedged_speech")
    if ev.coverage(str(c.get("caption", "")), m.text) < 0.75:
        return _reject(item, "caption_not_from_speech")
    support = tx.speech_support(c["tex"], m.text)
    if not support.ok:
        e["unsupported"] = support.missing
        return _reject(item, "formula_not_stated_in_speech")
    e["basis"] = "stated_in_speech"
    if m.probability is not None and m.probability < MIN_ASR_PROBABILITY:
        return _review(item, "asr_uncertain")
    if item.get("ambiguous"):
        return _review(item, "marked_ambiguous")
    item.update(status="ok", review_required=False, reason=None)
    return item


def _validate_fact(item: Dict[str, Any], ctx: _Ctx) -> Dict[str, Any]:
    c, e = item["content"], item["evidence"]
    kind = c.get("kind")
    if kind not in FACT_KINDS:
        return _reject(item, "fact_kind_unknown")
    if (err := _text_len_ok(c, {"headline": "label", "body": "body"})):
        return _reject(item, err)
    display = f"{c.get('headline', '')} {c.get('body', '')}".strip()
    if not display:
        return _reject(item, "empty")
    m = _locate(item, e.get("quote", ""), ctx)
    if not m:
        return _reject(item, "quote_not_in_speech")
    if ev.has_hedge(ev.context_around(ctx.words, m.start, m.end)):
        return _reject(item, "hedged_speech")
    spoken = ev.extract_numbers(m.text)
    for n in ev.numbers_in_display(display):
        if not ev.number_supported(n, spoken):
            return _reject(item, f"number_not_in_speech:{n:g}")
    if ev.coverage(display, m.text) < 0.6:
        return _reject(item, "text_not_from_speech")
    if kind == "verified":
        src = ctx.verified.get(str(e.get("source_id")))
        if not src:
            return _reject(item, "verified_fact_without_source")
        src_nums = ev.extract_numbers(str(src.get("text", "")))
        for n in ev.numbers_in_display(display):
            if not ev.number_supported(n, src_nums):
                return _reject(item, f"number_not_in_source:{n:g}")
        if ev.coverage(display, str(src.get("text", ""))) < 0.5:
            return _reject(item, "text_not_in_source")
        c["source_label"] = str(src.get("title", ""))[:MAX_TEXT["source"]]
    if (err := _set_times(item, m.start, m.end, len(display), ctx)):
        return _reject(item, err)
    if m.probability is not None and m.probability < MIN_ASR_PROBABILITY:
        return _review(item, "asr_uncertain")
    item.update(status="ok", review_required=False, reason=None)
    return item


def _validate_term(item: Dict[str, Any], ctx: _Ctx) -> Dict[str, Any]:
    c, e = item["content"], item["evidence"]
    if (err := _text_len_ok(c, {"term": "term", "definition": "definition"})):
        return _reject(item, err)
    term, definition = str(c.get("term", "")).strip(), str(c.get("definition", "")).strip()
    if not term or not definition:
        return _reject(item, "empty")
    m = _locate(item, e.get("quote", ""), ctx)
    if not m:
        return _reject(item, "quote_not_in_speech")
    if ev.has_hedge(ev.context_around(ctx.words, m.start, m.end)):
        return _reject(item, "hedged_speech")
    author = ctx.author.get(str(e.get("author_material_id")))
    source_text = m.text + " " + str((author or {}).get("text", ""))
    if ev.coverage(term, m.text) < 1.0:
        return _reject(item, "term_not_discussed")
    if ev.coverage(definition, source_text) < 0.75:
        return _reject(item, "definition_not_from_author")
    if (err := _set_times(item, m.start, m.end, len(term) + len(definition), ctx)):
        return _reject(item, err)
    if m.probability is not None and m.probability < MIN_ASR_PROBABILITY:
        return _review(item, "asr_uncertain")
    item.update(status="ok", review_required=False, reason=None)
    return item


def _validate_diagram(item: Dict[str, Any], ctx: _Ctx) -> Dict[str, Any]:
    c = item["content"]
    kind, nodes = c.get("kind"), c.get("nodes") or []
    if kind not in DIAGRAM_KINDS:
        return _reject(item, "diagram_kind_unknown")
    want = {"steps": (3, 3), "cause_effect": (2, 2), "a_vs_b": (2, 2), "timeline": (2, 4)}[kind]
    if not want[0] <= len(nodes) <= want[1]:
        return _reject(item, "diagram_node_count")
    if (err := _text_len_ok(c, {"title": "title"})):
        return _reject(item, err)
    last = -1.0
    last_end, chars = 0.0, 0
    first = None
    for n in nodes:
        label = str(n.get("label", "")).strip()
        if not label or len(label) > MAX_TEXT["label"]:
            return _reject(item, "node_label")
        m = ev.find_quote(ctx.words, n.get("quote", ""), ctx.segments)
        if not m:
            return _reject(item, "quote_not_in_speech")
        if ev.has_hedge(ev.context_around(ctx.words, m.start, m.end, 1.0)):
            return _reject(item, "hedged_speech")
        if ev.coverage(label, m.text) < 0.6:
            return _reject(item, "label_not_from_speech")
        if kind in ("steps", "timeline", "cause_effect") and m.start <= last:
            return _reject(item, "nodes_out_of_order")
        n["at_ms"] = int(m.start * 1000)
        last = m.start
        last_end = max(last_end, m.end)
        chars += len(label)
        first = m.start if first is None else min(first, m.start)
        item["source_segment_id"] = item.get("source_segment_id") or m.segment_id
    if (err := _set_times(item, first or 0.0, last_end + 0.8, chars, ctx)):
        return _reject(item, err)
    item.update(status="ok", review_required=False, reason=None)
    return item


def _validate_chart(item: Dict[str, Any], ctx: _Ctx) -> Dict[str, Any]:
    c = item["content"]
    kind, pts = c.get("kind"), c.get("points") or []
    if kind not in CHART_KINDS:
        return _reject(item, "chart_kind_unknown")
    if not str(c.get("unit", "")).strip():
        return _reject(item, "chart_without_unit")
    if (err := _text_len_ok(c, {"title": "title", "unit": "unit"})):
        return _reject(item, err)
    if not 2 <= len(pts) <= (6 if kind == "line" else 4):
        return _reject(item, "chart_point_count")
    values: List[float] = []
    first = None
    last_end = 0.0
    for p in pts:
        try:
            v = float(p.get("value"))
        except (TypeError, ValueError):
            return _reject(item, "value_not_a_number")
        if v < 0 or not str(p.get("label", "")).strip() or len(str(p.get("label"))) > MAX_TEXT["label"]:
            return _reject(item, "value_or_label_invalid")
        m = ev.find_quote(ctx.words, p.get("quote", ""), ctx.segments)
        if not m:
            return _reject(item, "quote_not_in_speech")
        if ev.has_hedge(ev.context_around(ctx.words, m.start, m.end)):
            return _reject(item, "hedged_speech")
        spoken = ev.extract_numbers(m.text)
        if not ev.number_supported(v, spoken):
            return _reject(item, f"value_not_in_speech:{v:g}")
        lab_nums = ev.numbers_in_display(str(p.get("label")))
        if any(not ev.number_supported(n, spoken) for n in lab_nums):
            return _reject(item, "label_number_not_in_speech")
        p["at_ms"] = int(m.start * 1000)
        values.append(v)
        first = m.start if first is None else min(first, m.start)
        last_end = max(last_end, m.end)
        item["source_segment_id"] = item.get("source_segment_id") or m.segment_id
    if kind == "share" and sum(values) > 100.5:
        return _reject(item, "shares_over_100")
    if (err := _set_times(item, first or 0.0, last_end + 1.2, 40, ctx)):
        return _reject(item, err)
    item.update(status="ok", review_required=False, reason=None)
    return item


_VALIDATORS = {"formula": _validate_formula, "fact": _validate_fact, "term": _validate_term,
               "diagram": _validate_diagram, "chart": _validate_chart}


def validate(items: Sequence[Dict[str, Any]], words: List[Dict[str, Any]], clip_duration: float, *,
             preset: str = "deep", enabled_types: Optional[Sequence[str]] = None,
             author_materials: Sequence[Dict[str, Any]] = (), verified_sources: Sequence[Dict[str, Any]] = (),
             min_confidence: float = MIN_CONFIDENCE) -> List[Dict[str, Any]]:
    """`words` are clip-local ASR words. Returns every item with status and reason (nothing is dropped silently)."""
    allowed = set(enabled_types) if enabled_types else PRESETS.get(preset, set())
    ctx = _Ctx(words, clip_duration, author_materials, verified_sources)
    out = []
    for i, raw in enumerate(items):
        item = dict(raw)
        item["id"] = item.get("id") or f"ov{i + 1:02d}"
        item["content"] = dict(item.get("content") or {})
        item["evidence"] = dict(item.get("evidence") or {})
        item.setdefault("style", {})
        item.setdefault("position", "auto")
        t = item.get("type")
        if t not in TYPES:
            out.append(_reject(item, "type_unknown"))
            continue
        if t not in allowed:
            out.append(_reject(item, "disabled_by_preset"))
            continue
        try:
            conf = float(item.get("confidence", 0))
        except (TypeError, ValueError):
            conf = 0.0
        item["confidence"] = conf
        if conf < min_confidence and not item.get("approved"):
            out.append(_reject(item, "low_confidence"))
            continue
        out.append(_VALIDATORS[t](item, ctx))
    return out


def apply_density(items: List[Dict[str, Any]], clip_duration: float, *, max_per_min: float = 2.0,
                  min_gap_ms: int = 2500) -> List[Dict[str, Any]]:
    """Keep the strongest items within the density limit; never two cards at once. Marks the rest `density_limit`."""
    ok = [i for i in items if i.get("status") == "ok"]
    limit = max(1, int(round(clip_duration / 60.0 * max_per_min)))
    kept: List[Dict[str, Any]] = []
    for it in sorted(ok, key=lambda x: (-float(x.get("confidence", 0)), x["start_ms"])):
        clash = any(it["start_ms"] < k["end_ms"] + min_gap_ms and k["start_ms"] < it["end_ms"] + min_gap_ms for k in kept)
        if clash or len(kept) >= limit:
            it["status"], it["reason"] = "skipped", "density_limit"
            continue
        kept.append(it)
    return items


def document(items: List[Dict[str, Any]], *, clip_duration: float, preset: str, aspect: str, **extra: Any) -> Dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "time_base": "clip_ms", "clip_duration_sec": round(clip_duration, 3),
            "preset": preset, "aspect": aspect, "overlays": items, **extra}
