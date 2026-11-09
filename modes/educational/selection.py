"""
selection.py — Educational v4: which parts of a long lecture become clips.

    ASR words ──► whole-sentence candidates (scripts/shared/cut_selection.build_story_candidates, the SAME rules
                  as Cut: a clip starts at a sentence start, ends at a statement end, never on a question or «…»)
              ──► score = Cut's educational story score
                        + v5 semantic layer on the candidate's own words (explanation quality, insight, structure,
                          takeaway / example / formula, confusion risk)
                        + v5 multimodal window score of the overlapping v5 moments (audio + semantics + topics)
                        + «complete thought»: a setup at the start, a conclusion at the end, no leaning on context
              ──► v5 topics (segment_topics_from_asr): one clip per topic first → a series that covers the lecture
              ──► optional Gemini editor (modes/educational/editor.py): only candidate ids + verdicts come back
              ──► chronological order, «part k of N»

Timestamps always come from ASR words. Exact render boundaries are done afterwards by scripts/shared/boundaries.
"""
from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from scripts.shared.boundaries import normalize_words
from scripts.shared.cut_selection import StoryConfig, build_story_candidates, is_promo, story_score

logger = logging.getLogger(__name__)

SELECTION_VERSION = "edu-select-v4"

_SETUP = re.compile(r"(?i)^(давайте|итак|что такое|почему|зачем|как\b|каким образом|представьте|рассмотрим|возьм[её]м|"
                    r"смотрите|вопрос|задача|сегодня|теперь|начн[её]м|первое|во-первых|главн|основн|есть такое)")
_CONCLUSION = re.compile(r"(?i)(таким образом|поэтому|значит|итак|в итоге|вывод|получается|следовательно|"
                         r"то есть|в результате|именно поэтому|вот почему|это и есть|так что)")
_NEEDS_CONTEXT = {"он", "она", "оно", "они", "его", "её", "ее", "их", "этот", "эта", "это", "эти", "этого", "такой",
                  "такая", "такие", "там", "тут", "здесь", "тогда", "тоже", "также", "кроме"}
_ALSO = {"тоже", "также", "ещё", "еще", "опять", "снова", "остальные", "второе", "второй", "третье", "третий"}
_NO_TITLE = {"первое", "второе", "третье", "это", "он", "она", "оно", "они", "который", "которая", "так", "вот"}
_BACK_REF = re.compile(r"(?i)(как я уже (говорил|сказал)|как мы уже|в прошлый раз|в прошлой лекции|выше мы|"
                       r"на прошлом занятии|как было сказано)")
_OPEN_REF = re.compile(r"(?i)(таким же образом|точно так же|так же, как|аналогично|обстоят .{0,20}так же)")


@dataclass
class EduConfig:
    max_clips: int = 6
    min_duration: float = 25.0
    max_duration: float = 90.0
    sweet_min: float = 30.0
    sweet_max: float = 75.0
    skip_intro_sec: float = 30.0         # lectures open with a greeting, not a teaser montage (Cut skips 90 s)
    min_gap_sec: float = 4.0             # two clips of a series never touch
    editor: bool = False                 # Gemini editor (needs a provider; off → local order)
    editor_pool: int = 24


def _first_word(text: str) -> str:
    m = re.match(r"[\W_]*([\w-]+)", text)
    return (m.group(1) if m else "").lower()


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?…])\s+", text) if s.strip()]


def completeness(text: str) -> Tuple[float, List[str]]:
    """Heuristic «complete thought»: a setup at the start, a conclusion at the end, no reference to what came before."""
    sents = _sentences(text)
    if not sents:
        return -1.0, ["empty"]
    score, why = 0.0, []
    first, last_two = sents[0], " ".join(sents[-2:])
    if _SETUP.search(first) or first.endswith("?"):
        score += 0.4
        why.append("opens with a setup")
    if _CONCLUSION.search(last_two):
        score += 0.4
        why.append("ends with a conclusion")
    if _first_word(first) in _NEEDS_CONTEXT:
        score -= 0.5
        why.append(f"opens leaning on context «{_first_word(first)}»")
    head = [re.sub(r"[^\w-]", "", w).lower() for w in first.split()[:5]]
    if any(w in _ALSO for w in head):
        score -= 0.4
        why.append("opens with «тоже / также / ещё» (continues something)")
    if _OPEN_REF.search(" ".join(sents[:2])):
        score -= 0.5
        why.append("opens comparing with something shown before")
    if _BACK_REF.search(text):
        score -= 0.6
        why.append("refers back to earlier parts")
    return score, why


def _semantic(text: str, topic_keywords: Optional[List[str]] = None) -> Tuple[float, Dict[str, Any]]:
    """The v5 semantic layer (rule-based, no LLM) on the candidate's own words."""
    from scripts.legacy_gpu import educational_mode_v5 as v5
    markers = v5.extract_semantic_markers(text, topic_keywords=topic_keywords or [])
    stype, conf = v5.classify_segment_type_semantic(markers, text)
    rel = v5._estimate_topic_relevance(text, topic_keywords or [], markers)
    sem = v5.score_semantic_window(markers, text, stype, rel)
    s = (1.2 * sem.explanation_quality + 1.0 * sem.insight_value + 0.8 * sem.structure_score
         + 0.4 * sem.has_takeaway + 0.25 * sem.has_example + 0.25 * sem.has_formula - 0.8 * sem.confusion_risk)
    w = v5.EDUCATIONAL_SEGMENT_TYPES.get(stype, {}).get("weight", 1.0)
    return round(s * w, 6), {"segment_type": stype, "type_confidence": round(conf, 3),
                             "explanation_quality": round(sem.explanation_quality, 3),
                             "insight": round(sem.insight_value, 3), "structure": round(sem.structure_score, 3),
                             "has_takeaway": sem.has_takeaway, "has_formula": sem.has_formula,
                             "has_example": sem.has_example}


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _topic_of(c: Dict[str, Any], topics: Sequence[Dict[str, Any]]) -> int:
    mid = (c["start"] + c["end"]) / 2
    for i, t in enumerate(topics):
        if float(t.get("start", 0)) <= mid < float(t.get("end", 0)):
            return int(t.get("topic_id", i))
    best = max(range(len(topics)), key=lambda i: _overlap(c["start"], c["end"], topics[i]["start"], topics[i]["end"]),
               default=None)
    return int(topics[best].get("topic_id", best)) if best is not None else -1


def _v5_bonus(c: Dict[str, Any], moments: Sequence[Dict[str, Any]]) -> float:
    dur = max(1e-6, c["end"] - c["start"])
    best = 0.0
    for m in moments:
        ov = _overlap(c["start"], c["end"], float(m.get("start", 0)), float(m.get("end", 0)))
        if ov / dur >= 0.4:
            best = max(best, float(m.get("score") or 0.0))
    return best


def local_title(text: str) -> str:
    """No Gemini: a title only when the speaker gives one in their own words (a definition or a short question)."""
    m = re.search(r"(?i)(?:^|[.!?]\s+)([А-ЯЁA-Z][\w-]{2,}(?:\s+[\w-]+){0,2})\s+[—-]\s+это\b", text)
    if m and m.group(1).split()[0].lower() not in _NO_TITLE:
        return f"Что такое {m.group(1).lower() if not m.group(1).isupper() else m.group(1)}"
    for s in _sentences(text)[:2]:
        n = len(s.split())
        if s.endswith("?") and 3 <= n <= 8:
            return s
    return ""


def score_candidates(words_raw: Sequence[Dict[str, Any]], cfg: EduConfig,
                     v5_result: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    words, _ = normalize_words(words_raw)
    scfg = StoryConfig(min_duration=cfg.min_duration, max_duration=cfg.max_duration, sweet_min=cfg.sweet_min,
                       sweet_max=cfg.sweet_max, skip_intro_sec=cfg.skip_intro_sec, strategy="educational")
    cands = [c for c in build_story_candidates(words, scfg) if not is_promo(c["text"])] if words else []
    topics = list((v5_result or {}).get("topic_segments") or [])
    moments = list((v5_result or {}).get("educational_moments") or [])
    for c in cands:
        base, why = story_score(c, scfg)
        tid = _topic_of(c, topics) if topics else -1
        kws = topics[tid].get("keywords") if 0 <= tid < len(topics) else None
        sem, sem_meta = _semantic(c["text"], kws)
        comp, comp_why = completeness(c["text"])
        v5b = _v5_bonus(c, moments)
        c.update(sem_meta)
        c["topic_id"] = tid
        c["score"] = round(base + sem + comp + 1.0 * v5b, 6)
        c["why"] = why + comp_why + [f"v5 semantic {sem:.2f}", f"v5 moment {v5b:.2f}", sem_meta["segment_type"]]
        c["complete_local"] = comp
    cands.sort(key=lambda c: (-c["score"], round(c["start"] * 1000), c["id"]))
    meta = {"selection_version": SELECTION_VERSION, "candidates": len(cands), "topics": len(topics),
            "v5_moments": len(moments), "params": asdict(cfg)}
    return cands, meta


def _free(c: Dict[str, Any], picked: Sequence[Dict[str, Any]], gap: float) -> bool:
    return all(c["end"] + gap <= k["start"] or c["start"] >= k["end"] + gap for k in picked)


def pick_series(ranked: Sequence[Dict[str, Any]], n: int, gap: float) -> List[Dict[str, Any]]:
    """Topic coverage first (v5 idea): the best clip of each topic, best topics first; then the best of the rest."""
    picked: List[Dict[str, Any]] = []
    seen_topics = set()
    for c in ranked:
        if len(picked) >= n:
            break
        if c.get("topic_id", -1) in seen_topics and c.get("topic_id", -1) != -1:
            continue
        if _free(c, picked, gap):
            picked.append(c)
            seen_topics.add(c.get("topic_id", -1))
    for c in ranked:
        if len(picked) >= n:
            break
        if c not in picked and _free(c, picked, gap):
            picked.append(c)
    return picked


def diverse_pool(ranked: Sequence[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for c in ranked:
        if _free(c, out, 0.0):
            out.append(c)
            if len(out) >= n:
                break
    return out


def select_lesson_clips(words_raw: Sequence[Dict[str, Any]], cfg: Optional[EduConfig] = None,
                        v5_result: Optional[Dict[str, Any]] = None, editor_fn: Optional[Callable] = None
                        ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """-> (segments in chronological order, meta). editor_fn(pool) -> (verdicts by id, meta) or None."""
    cfg = cfg or EduConfig()
    ranked, meta = score_candidates(words_raw, cfg, v5_result)
    meta["llm"] = {"status": "off"}
    if not ranked:
        meta["selection_status"] = "no_candidates" if words_raw else "no_words"
        return [], meta
    verdicts: Dict[str, Dict[str, Any]] = {}
    if cfg.editor and editor_fn is not None:
        pool = diverse_pool(ranked, cfg.editor_pool)
        verdicts, emeta = editor_fn(pool)
        meta["llm"] = emeta
        if verdicts:
            def tier(c):
                v = verdicts.get(c["id"])
                if not v:
                    return 0.0
                good = v["complete"] and v["self_contained"]
                return (2.0 if good and v["score"] >= 50 else 1.0 if v["score"] >= 60 else -1.0) + v["score"] / 100.0
            ranked = sorted(ranked, key=lambda c: (-(tier(c) * 10 + c["score"] * 0.1), round(c["start"] * 1000), c["id"]))
            ranked = [c for c in ranked if tier(c) >= 0] + [c for c in ranked if tier(c) < 0]
            # the editor's topics are better than the pause/marker topics of v5: use them for coverage
            for c in ranked:
                v = verdicts.get(c["id"])
                if v and v.get("topic"):
                    c["topic_title"] = v["topic"]
            topic_ids: Dict[str, int] = {}
            for c in ranked:
                if c.get("topic_title"):
                    c["topic_id"] = topic_ids.setdefault(c["topic_title"].lower(), 1000 + len(topic_ids))
    chosen = pick_series(ranked, cfg.max_clips, cfg.min_gap_sec)
    chosen.sort(key=lambda c: c["start"])
    segs = []
    for k, c in enumerate(chosen, 1):
        v = verdicts.get(c["id"])
        title = (v or {}).get("title") or local_title(c["text"])
        segs.append({"start": c["start"], "duration": c["end"] - c["start"], "score": c["score"],
                     "candidate_id": c["id"], "source": "edu_v4:" + ("editor" if v else "local"),
                     "topic_id": c["topic_id"], "topic_title": c.get("topic_title") or "",
                     "title": title, "title_source": ("gemini" if (v or {}).get("title") else "speaker" if title else "none"),
                     "segment_type": c["segment_type"], "series_index": k, "series_total": len(chosen),
                     "selection_reason": "; ".join(c["why"] + ([f"editor {v['score']:.0f}: {v['reason']}"] if v else [])),
                     "editor": v, "text": c["text"]})
    meta.update(selection_status="ok", selection_method="editor" if verdicts else "local",
                selected=[{k: s[k] for k in ("candidate_id", "start", "duration", "topic_id", "title", "segment_type",
                                             "selection_reason")} | {"text": s["text"][:300]} for s in segs])
    return segs, meta


def sentence_edges(words: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Deterministic check of a finished clip's own ASR: the first word starts a sentence (capital letter / digit),
    the last word ends a statement (. or !), never a question, «…» or a dangling comma."""
    ws = [str(w.get("word", "")).strip() for w in words if str(w.get("word", "")).strip()]
    if not ws:
        return {"ok": False, "start_ok": False, "end_ok": False, "first": "", "last": ""}
    first, last = ws[0], ws[-1]
    letter = re.sub(r"^[^\w]+", "", first)
    start_ok = bool(letter) and (letter[0].isupper() or letter[0].isdigit())
    end_ok = bool(re.search(r"[.!][\"'»”)]*$", last)) and not last.endswith("...")
    return {"ok": start_ok and end_ok, "start_ok": start_ok, "end_ok": end_ok, "first": first, "last": last}
