"""
stream_comment.py
=================
«Завозный коммент» (owner 2026-11-19, reference: TikTok «ЗЛОЙ УСТАНОВИЛ МИРОВОЙ РЕКОРД В GTA6»):
the streamer READS a viewer's comment aloud (funny / provocative / «завозный») and reacts. The clip shows the
comment as a chat card while it is read, the reaction, and — when the stream has it — a fast glitch cut to the
moment the comment is about (comment «ты реально первый в мире в стрипуху приехал?» → the drive to the club).

Pipeline (analysis only, no render here):
  1. reads   chat replay available (scripts/shared/stream_chat): every message is matched against what the streamer
             SAYS in the next ~2 min (ASR words, fuzzy stems, in order) — a match = he read it out, the card shows
             the real nick / colour / text. No chat: ASR cues («X пишет», «пишут», «спрашивают», «вопрос из чата»…)
             → the words after the cue are the comment, the nick is hidden («зритель»).
  2. local   reaction after the read (laugh / reaction words / loudness) → local score; reaction end = a sentence
             end 6–22 s after the read.
  3. payoff  «продолжение»: transcript windows anywhere else in the stream sharing the comment's content stems
             (+ game-event / excitement signal) → ≤ 3 options per read.
  4. editor  ONE Gemini call (analysis only, ≤ 1.5 ₽, forecast before the call): is it really a read-out comment,
             how «завозный» (0–100), the clean quote, where the reaction ends, which option (if any) shows the very
             thing the comment is about, a two-line title with accent words. Any failure → local order.

Output: stream segments (the runner's shape) with moment_type «comment» and
  "comment": {"author", "text", "color", "source": "chat"|"asr", "t0", "t1"}   (card window, source seconds)
  "payoff":  {"start", "end"}                                                    (optional, source seconds)
  "title":   {"text", "accent": [words]}                                        (big two-line title, start only)
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

VERSION = "stream-comment-v1"
PROMPT_VERSION = "stream-comment-editor-v1"
BUDGET_RUB = 1.5
PAYOFF_CONF = 70.0           # the editor's confidence that the option shows the very thing the comment is about
MIN_REACTION_SEC = 4.0       # the clip goes on at least this long after the comment is read
ASR_MIN_SCORE = 75.0         # a read found by an ASR cue only (no chat) must be clearly strong
VIEWER = "зритель"

_STOP = set("""а и в во на но не ни да нет что чтобы как так ты вы я он она оно они мы мне тебе тебя меня его ее её
их нам вам это этот эта эти тот та те то там тут здесь уже еще ещё же ли бы вот ну по за из от до для при про со
ко у о об над под без через или если когда где кто чем чего тоже типа просто очень вообще какой какая какие
был была были было будет есть все всё всех всем сам сама сейчас щас раз два""".split())

# «Вася пишет: …», «пишут…», «спрашивают…», «вопрос из чата…», «тут в чате…», «коммент…», «читаю…»
_CUE = re.compile(r"^(пишет|пишут|написал|написала|написали|спрашивает|спрашивают|спросил|спросила|вопрос|"
                  r"комментарий|коммент|читаю|цитирую|цитата|чат)$")
_CUE_AFTER_CHAT = re.compile(r"^(пишет|пишут|спрашивает|спрашивают|говорит|говорят|интересуется)$")
_LINK = re.compile(r"https?://|www\.|\.com\b|\.ru\b|!\w+")
# routine questions are never «завозный»: hardware / settings / schedule / money / greetings / donations
_ROUTINE = re.compile(r"(?i)\b(фпс|fps|4k|1080|60\s*кадр|настройк\w*|графи\w*|видеокарт\w*|процессор\w*|монитор\w*|"
                      r"пк\b|комп\w*|пинг\w*|лаг\w*|микрофон\w*|звук\w*|сколько\s+(стоит|часов|лет)|во\s+сколько|"
                      r"когда\s+стрим|расписани\w*|донат\w*|подписк\w*|спасибо|привет\w*|здравствуй\w*|"
                      r"ссылк\w*|промокод\w*|телеграм\w*|тг\b|дискорд\w*)")
_LAUGH = re.compile(r"^(а?ха(ха)+|хах\w*|хех\w*|ахах\w*|хи(хи)+|ржу|ору|смешно|угар\w*)$")


def _norm(t: str) -> str:
    return re.sub(r"[^\w]", "", str(t).lower().replace("ё", "е"))


def _stem(t: str) -> str:
    t = _norm(t)
    return t[:5] if len(t) > 5 else t


def _sim(a: str, b: str) -> bool:
    """Token match tolerant to ASR spelling: same stem, or one is a prefix of the other (≥ 4 chars)."""
    if not a or not b:
        return False
    if a == b or _stem(a) == _stem(b):
        return True
    n = min(len(a), len(b))
    return n >= 4 and a[:n] == b[:n]


def _content(tokens: Sequence[str]) -> List[str]:
    return [t for t in tokens if len(t) >= 3 and t not in _STOP and not t.isdigit()]


# ── 1a. reads from the chat replay ──────────────────────────────────────────

def _msg_ok(m: Dict[str, Any]) -> bool:
    txt = m.get("text") or ""
    if _LINK.search(txt) or len(txt) < 10:
        return False
    toks = [_norm(x) for x in txt.split()]
    toks = [t for t in toks if t]
    if len(toks) < 3 or len(toks) > 45:
        return False
    if len(set(toks)) < 0.5 * len(toks):              # spam: the same word again and again
        return False
    return len(_content(toks)) >= 2


def reads_from_chat(words, messages: Sequence[Dict[str, Any]], lag: Tuple[float, float] = (0.5, 150.0),
                    min_ratio: float = 0.6) -> List[Dict[str, Any]]:
    """Messages the streamer read out. words: normalized boundaries.Word list. -> reads (best match per span).
    He can only read a message after it was sent (lag ≥ 0.5 s): chat echoing what he / the game just said
    (message AFTER the words) is not a read."""
    if not words or not messages:
        return []
    toks = [_norm(w.text) for w in words]
    starts = np.array([w.start for w in words])
    by_stem: Dict[str, List[int]] = {}
    for i, t in enumerate(toks):
        if len(t) >= 3 and t not in _STOP:
            by_stem.setdefault(_stem(t), []).append(i)
    out: List[Dict[str, Any]] = []
    for m in messages:
        if not _msg_ok(m):
            continue
        mt = [t for t in (_norm(x) for x in m["text"].split()) if t]
        mc = _content(mt)
        stems = {_stem(t) for t in mc}
        lo, hi = np.searchsorted(starts, m["t"] + lag[0]), np.searchsorted(starts, m["t"] + lag[1])
        if hi <= lo:
            continue
        hits = sorted(i for s in stems for i in by_stem.get(s, ()) if lo <= i < hi)
        need = max(2 if len(mc) <= 3 else 3, int(np.ceil(min_ratio * len(mc))))
        if len(hits) < need:
            continue
        span = int(len(mt) * 1.8) + 3                  # the read may add «э-э», repeat a word, skip one
        best = None
        for k, i0 in enumerate(hits):
            got = {}
            for i in hits[k:]:
                if i - i0 > span:
                    break
                got.setdefault(_stem(toks[i]), i)
            if len(got) >= need and (best is None or len(got) > best[0]):
                best = (len(got), i0, max(got.values()))
        if not best:
            continue
        n, a, b = best
        if words[a].start - m["t"] > 60.0 and (n < 3 or n < 0.9 * len(stems)):
            continue                                   # a late, partial match is a coincidence, not a read
        # widen to the whole quote: stop words of the message right before / after the matched run
        while a > 0 and a - 1 >= lo and toks[a - 1] in mt and words[a].start - words[a - 1].end < 0.8:
            a -= 1
        while b + 1 < len(words) and toks[b + 1] in mt and words[b + 1].start - words[b].end < 0.8:
            b += 1
        out.append({"source": "chat", "author": m.get("author") or VIEWER, "text": m["text"], "color": m.get("color"),
                    "msg_t": m["t"], "first": a, "last": b, "t0": words[a].start, "t1": words[b].end,
                    "match": round(n / max(1, len(stems)), 3), "matched": n})
    return _dedupe(out)


def _dedupe(reads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One read per spoken span: the best match (more matched stems, then the closer message) wins."""
    reads.sort(key=lambda r: (-r["matched"], -r["match"], r["t0"] - r.get("msg_t", r["t0"])))
    kept: List[Dict[str, Any]] = []
    for r in reads:
        if all(r["t1"] < k["t0"] - 0.5 or r["t0"] > k["t1"] + 0.5 for k in kept):
            kept.append(r)
    return sorted(kept, key=lambda r: r["t0"])


# ── 1b. reads from ASR cues (no chat) ───────────────────────────────────────

def reads_from_asr(words, sents: Sequence[Tuple[int, int]]) -> List[Dict[str, Any]]:
    """«X пишет: …» / «пишут: …» / «вопрос из чата: …» → the words after the cue up to the sentence end
    (3–30 words, ≤ 10 s). The nick is never shown (ASR spells nicks badly)."""
    toks = [_norm(w.text) for w in words]
    sent_end = {}
    for a, b in sents:
        for i in range(a, b + 1):
            sent_end[i] = b
    out = []
    last = -10.0
    for i, t in enumerate(toks):
        if not _CUE.match(t) or words[i].start - last < 6.0:
            continue
        if t == "чат" and not (i + 1 < len(toks) and _CUE_AFTER_CHAT.match(toks[i + 1])):
            continue
        j = i + 1
        while j < len(toks) and toks[j] in ("что", "мне", "нам", "тут", "вот", "такой", "такое", "из", "чата", "в",
                                             "чате", "от", "пишет", "пишут", "спрашивает", "спрашивают"):
            j += 1
        if j >= len(toks):
            continue
        end = sent_end.get(j, j)
        if end - j < 2:                                # a cue at the very end of a sentence: the next sentence
            end = sent_end.get(end + 1, end)
        end = min(end, j + 29)
        while end > j and words[end].end - words[j].start > 10.0:
            end -= 1
        if end - j < 2:
            continue
        a = i - 1 if i > 0 and words[i].start - words[i - 1].end < 0.6 else i     # «Вася пишет»: the nick word
        out.append({"source": "asr", "author": VIEWER, "color": None,
                    "text": " ".join(w.text.strip() for w in words[j:end + 1]).strip(" ,.-"),
                    "first": a, "quote_first": j, "last": end, "t0": words[a].start, "t1": words[end].end,
                    "match": 0.0, "matched": 0, "cue": t})
        last = words[i].start
    return out


# ── 2. reaction + local score ───────────────────────────────────────────────

def _reaction(words, sents, r: Dict[str, Any], min_after: float = 8.0, max_after: float = 25.0) -> int:
    """Index of the last word of the reaction: a sentence end 8–25 s after the read (else the cap)."""
    end_t = r["t1"]
    best = r["last"]
    for a, b in sents:
        if b <= r["last"]:
            continue
        if words[b].end - end_t > max_after:
            break
        best = b
        if words[b].end - end_t >= min_after and (b + 1 >= len(words) or words[b + 1].start - words[b].end > 0.9):
            break
    return best


def local_score(r: Dict[str, Any], words, hits, loud: Optional[Sequence[float]]) -> Tuple[float, List[str]]:
    a, b = r["t1"], words[r["react_last"]].end
    why = []
    sc = 1.0 + (1.2 * r["match"] if r["source"] == "chat" else 0.4)
    lw = [w for w in words[r["last"] + 1:r["react_last"] + 1] if _LAUGH.match(_norm(w.text))]
    kinds = [h[1] for h in (hits or []) if a - 0.5 <= h[0] <= b]
    n_laugh = len(lw) + kinds.count("laugh")
    n_react = sum(kinds.count(k) for k in ("react", "shout", "mat", "phrase"))
    sc += min(1.5, 0.5 * n_laugh) + min(1.0, 0.25 * n_react)
    if n_laugh:
        why.append(f"laugh×{n_laugh}")
    if n_react:
        why.append(f"react×{n_react}")
    if loud is not None and len(loud) > int(b):
        lv = np.asarray(loud[int(a):int(b) + 1], dtype=np.float64)
        if len(lv) and float(lv.max()) > 1.5:
            sc += min(0.8, 0.3 * float(lv.max()))
            why.append(f"loud {float(lv.max()):.1f}")
    q = r["text"]
    if "?" in q:
        sc += 0.2
    if _ROUTINE.search(q):
        sc -= 1.5
        why.append("routine")
    return round(sc, 3), why


# ── 3. payoff options («продолжение») ───────────────────────────────────────

def payoff_options(words, sents, r: Dict[str, Any], excite: Optional[np.ndarray], n: int = 3,
                   win: float = 16.0, exclude: float = 90.0) -> List[Dict[str, Any]]:
    """Transcript windows elsewhere in the stream that share the comment's content stems. -> [{start, end, text,
    overlap, rel}] (rel = minutes from the read, negative = before it)."""
    stems = {_stem(t) for t in _content([_norm(x) for x in r["text"].split()])}
    if not stems or not sents:
        return []
    starts = [words[a].start for a, _ in sents]
    out = []
    for k, (a, b) in enumerate(sents):
        t0 = words[a].start
        if abs(t0 - r["t0"]) < exclude or (r["t0"] - exclude <= t0 <= words[r["react_last"]].end + 5):
            continue
        j = k
        while j + 1 < len(sents) and words[sents[j + 1][1]].end - t0 <= win:
            j += 1
        e = sents[j][1]
        got = {_stem(_norm(w.text)) for w in words[a:e + 1]} & stems
        if len(got) < max(1, min(2, len(stems) // 2)):
            continue
        ex = float(excite[int(t0):int(words[e].end) + 1].max()) if excite is not None and len(excite) > int(t0) else 0.0
        out.append({"start": round(t0, 2), "end": round(words[e].end, 2), "first": a, "last": e,
                    "text": " ".join(w.text.strip() for w in words[a:e + 1])[:260],
                    "overlap": len(got), "excite": round(ex, 2), "rel": round((t0 - r["t0"]) / 60.0, 1)})
    out.sort(key=lambda o: (-o["overlap"], -o["excite"]))
    picked: List[Dict[str, Any]] = []
    for o in out:
        if all(abs(o["start"] - p["start"]) > win for p in picked):
            picked.append(o)
        if len(picked) >= n:
            break
    del starts
    return picked


# ── 4. editor ───────────────────────────────────────────────────────────────

SYSTEM = ("Ты — строгий редактор вирусных вертикальных клипов со стримов (TikTok/Shorts). Формат «Завозный коммент»: "
          "стример ЧИТАЕТ ВСЛУХ комментарий зрителя из чата и ярко реагирует (ржёт, огрызается, в шоке, отвечает "
          "шуткой). Завозный коммент — смешной, дерзкий, провокационный, абсурдный, подкол стримеру, неожиданный "
          "вопрос («ты реально первый в мире в стрипуху приехал?»). НЕ завозный (score ≤ 35): технические вопросы "
          "(фпс, графика, настройки, железо, пинг, звук), расписание, цены, просьбы чата, приветы, благодарности за "
          "донаты и подписки, реклама; а ещё — если стример НЕ читает коммент (это реплика из игры / катсцены / его "
          "собственная речь) → is_comment=false. Реакция обязана быть: без смеха/ответа/эмоции — score ≤ 50. "
          "Шкала абсолютная: 85+ — разлетится, 70–84 — крепко, 50–69 — так себе. Большинство кандидатов — 30–60. "
          "«Продолжение» — фрагмент, где в игре/на стриме ПРОИСХОДИТ именно то, о чём коммент (то же действие, "
          "место, событие); общее слово (фпс, жопа, машина) — НЕ продолжение. Если не уверен — payoff=null. "
          "Отвечай только JSON.")


def editor_messages(items: List[Dict[str, Any]], context: str = "") -> List[Dict[str, str]]:
    user = ((f"Стрим: {context[:160]}\n" if context else "") +
            "Кандидаты: comment — текст комментария (source=chat: точный текст из чата; source=asr: распознанная речь "
            "стримера, в ней может быть и сам коммент, и его слова); said — что стример говорит, по фразам с номерами "
            "(время от начала чтения); options — фрагменты из других мест стрима (rel — минут от чтения), один из них "
            "может быть «продолжением».\n" + json.dumps(items, ensure_ascii=False) +
            "\n\nВерни JSON: {\"clips\": [{\"id\": str, \"is_comment\": bool (стример правда читает коммент зрителя), "
            "\"score\": 0-100 (насколько завозный коммент + реакция: 85+ разлетится, 70+ крепко, <40 скучно), "
            "\"quote\": \"текст коммента для карточки чата, как его написал зритель, до 140 знаков, без ника\", "
            "\"end\": int (номер фразы, где заканчивается реакция — дай реакции прозвучать, 5–20 с после коммента), "
            "\"payoff\": int|null (индекс option, который показывает ИМЕННО то, о чём коммент; null если нет), "
            "\"payoff_conf\": 0-100 (уверенность, что option — то самое), "
            "\"title\": \"заголовок 4–8 слов КАПСОМ про СУТЬ коммента, конкретно (кто/что), как в TikTok: «ЗЛОЙ "
            "УСТАНОВИЛ МИРОВОЙ РЕКОРД В GTA6», «ЧАТ СПРОСИЛ ПРО БЫВШУЮ — БУСТЕР В ШОКЕ»; НЕ общие слова вроде «стример "
            "реагирует на комментарий»; только правда из клипа\", \"accent\": [1–3 слова из title для цветного "
            "выделения: имя/главное слово], "
            "\"why\": \"до 10 слов\"}]} — для КАЖДОГО кандидата, по убыванию score.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def parse_editor(text: Optional[str], cands: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    try:
        data = json.loads(re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M))
    except ValueError:
        return {}
    out = {}
    for x in (data.get("clips") if isinstance(data, dict) else data) or []:
        if not isinstance(x, dict) or x.get("id") not in cands:
            continue
        c = cands[x["id"]]
        try:
            score = max(0.0, min(100.0, float(x.get("score", 0))))
        except (TypeError, ValueError):
            continue
        n_s = len(c["said_idx"])
        try:
            end = int(x.get("end"))
            end = end if 0 <= end < n_s else None
        except (TypeError, ValueError):
            end = None
        pay = x.get("payoff")
        pay = pay if isinstance(pay, int) and not isinstance(pay, bool) and 0 <= pay < len(c["options"]) else None
        try:
            conf = float(x.get("payoff_conf", 0) or 0)
        except (TypeError, ValueError):
            conf = 0.0
        if conf < PAYOFF_CONF:                         # «если не уверен — без продолжения»
            pay = None
        title = re.sub(r"\s+", " ", str(x.get("title") or "")).strip().upper()[:70]
        acc = [str(a).upper() for a in (x.get("accent") or [])[:3] if isinstance(a, str) and str(a).upper() in title]
        quote = re.sub(r"\s+", " ", str(x.get("quote") or "")).strip()[:160]
        out[x["id"]] = {"is_comment": x.get("is_comment") is True, "score": score, "end": end, "payoff": pay,
                        "title": title, "accent": acc, "quote": quote, "why": str(x.get("why") or "")[:120]}
    return out


def _quote_ok(quote: str, said: str) -> bool:
    """An ASR-sourced quote may be cleaned by the editor but must still be what the streamer said."""
    q = _content([_norm(x) for x in quote.split()])
    s = {_stem(_norm(x)) for x in said.split()}
    return bool(q) and sum(_stem(t) in s for t in q) >= 0.6 * len(q)


# ── selection ───────────────────────────────────────────────────────────────

def select(raw_words: Sequence[Dict[str, Any]], n_sec: int, signals: Dict[str, Sequence[float]], *,
           messages: Optional[Sequence[Dict[str, Any]]] = None, hits=None, ads: Sequence[Tuple[float, float]] = (),
           max_clips: int = 3, editor: bool = True, context: str = "", chat: Optional[Callable[..., Any]] = None,
           budget_rub: float = BUDGET_RUB, rub_per_usd: float = 100.0, min_score: float = 60.0,
           pool: int = 10) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """-> (segments in the runner's shape, meta)."""
    from .boundaries import normalize_words
    from . import stream_moments as sm
    words, _ = normalize_words(raw_words)
    meta: Dict[str, Any] = {"version": VERSION, "chat_messages": len(messages or []), "llm": {"status": "off"}}
    if not words:
        meta["status"] = "no_words"
        return [], meta
    sents = sm._sentences(words)
    reads = reads_from_chat(words, messages or []) if messages else []
    meta["reads_chat"] = len(reads)
    asr_reads = reads_from_asr(words, sents)
    meta["reads_asr"] = len(asr_reads)
    for r in asr_reads:                                # a chat match beats an ASR cue on the same span
        if all(r["t1"] < k["t0"] - 1.0 or r["t0"] > k["t1"] + 1.0 for k in reads):
            reads.append(r)
    from . import stream_ads
    reads = [r for r in reads if not stream_ads.overlaps(r["t0"], r["t1"] + 10.0, list(ads or []), 0.3)]
    if not reads:
        meta["status"] = "no_reads"
        return [], meta
    e = sm.excitement(signals, n_sec) if signals else None
    loud = signals.get("loud") if signals else None
    for r in reads:
        r["react_last"] = _reaction(words, sents, r)
        r["local"], r["why"] = local_score(r, words, hits, loud)
    reads.sort(key=lambda r: -r["local"])
    cand: List[Dict[str, Any]] = []
    for r in reads:                                    # one comment clip per ~40 s
        if all(abs(r["t0"] - c["t0"]) > 40.0 for c in cand):
            cand.append(r)
        if len(cand) >= pool:
            break
    for k, r in enumerate(cand):
        r["id"] = f"c{k}"
        r["options"] = payoff_options(words, sents, r, e)
        said, idx = [], []
        for a, b in sents:
            if b < r["first"] or a > r["react_last"]:
                continue
            said.append(f"{len(said)}) {words[max(a, r['first'])].start - r['t0']:+.0f}s "
                        + " ".join(w.text.strip() for w in words[max(a, r["first"]):b + 1])[:220])
            idx.append(b)
        r["said"], r["said_idx"] = said, idx
    meta["candidates"] = len(cand)
    verdicts: Dict[str, Dict[str, Any]] = {}
    if editor and cand:
        items = lambda cs: [{"id": r["id"], "source": r["source"], "comment": r["text"][:200], "said": r["said"],  # noqa: E731
                             "options": [{"i": i, "rel": o["rel"], "text": o["text"]} for i, o in enumerate(r["options"])]}
                            for r in cs]
        cs = list(cand)
        msgs = editor_messages(items(cs), context)
        fake = sm.MomentConfig(rub_per_usd=rub_per_usd)
        while len(cs) > 1 and sm.forecast_rub(msgs, len(cs), fake) > budget_rub:
            cs = cs[:-1]
            msgs = editor_messages(items(cs), context)
        fc = sm.forecast_rub(msgs, len(cs), fake)
        em: Dict[str, Any] = {"pool": len(cs), "forecast_rub": fc, "budget_rub": budget_rub}
        if fc <= budget_rub:
            from .stream_viral import _call
            txt, m = _call(chat, msgs, PROMPT_VERSION, fake, timeout=120.0)
            em.update(m)
            verdicts = parse_editor(txt, {r["id"]: r for r in cs})
            em["verdicts"] = len(verdicts)
        else:
            em["status"] = "skipped_budget"
        meta["llm"] = em
    picked: List[Tuple[Dict[str, Any], float, Dict[str, Any]]] = []
    if verdicts:
        for r in cand:
            v = verdicts.get(r["id"])
            bar = min_score if r["source"] == "chat" else max(min_score, ASR_MIN_SCORE)
            if _ROUTINE.search(r["text"]):
                bar = max(bar, 85.0)
            if v and v["is_comment"] and v["score"] >= bar:
                picked.append((r, v["score"] / 100.0 + 0.1 * r["local"], v))
        meta["selection_method"] = "editor"
    else:                                              # no editor: only real chat matches with a reaction
        for r in cand:
            if r["source"] == "chat" and r["local"] >= 2.4:
                picked.append((r, r["local"] / 4.0, {}))
        meta["selection_method"] = "local"
    picked.sort(key=lambda t: -t[1])
    segs = []
    for r, sc, v in picked[:max_clips]:
        last = r["react_last"]
        if v.get("end") is not None:
            last = max(r["last"], r["said_idx"][v["end"]])
        if words[last].end - r["t1"] < MIN_REACTION_SEC:     # the editor cut the reaction off: the local end
            last = max(last, r["react_last"])
        start = max(0.0, words[r["first"]].start - 0.25)
        end = words[last].end + 0.35
        quote = r["text"]
        if v.get("quote") and (r["source"] == "chat" or _quote_ok(v["quote"], r["text"])):
            quote = v["quote"] if r["source"] == "asr" else quote   # chat text stays exactly what the viewer wrote
        seg = {"start": round(start, 3), "duration": round(end - start, 3), "score": round(3.0 + sc, 4),
               "source": "stream_comment:" + ("editor" if v else "local"), "moment_type": "comment",
               "category": "comment", "candidate_id": r["id"],
               "selection_reason": "; ".join([f"{r['source']} read match {r['match']:.2f}"] + r["why"]
                                             + ([f"editor {v['score']:.0f}: {v['why']}"] if v else [])),
               "comment": {"author": r["author"] if r["source"] == "chat" else VIEWER, "text": quote,
                           "color": r.get("color"), "source": r["source"], "t0": round(r["t0"], 3),
                           "t1": round(r["t1"], 3), "msg_t": r.get("msg_t")},
               "peak_t": round(r["t1"], 3)}
        if v.get("title"):
            seg["hook"] = v["title"]
            seg["title"] = {"text": v["title"], "accent": v.get("accent") or []}
        if v.get("payoff") is not None:
            o = r["options"][v["payoff"]]
            seg["payoff"] = {"start": round(max(0.0, o["start"] - 0.2), 3), "end": round(o["end"] + 0.3, 3),
                             "text": o["text"][:200]}
        if v:
            seg["editor_score"] = v["score"]
        segs.append(seg)
    meta["selected"] = [{"t0": s["comment"]["t0"], "text": s["comment"]["text"][:120], "score": s["score"],
                         "payoff": bool(s.get("payoff")), "title": s.get("hook")} for s in segs]
    meta["status"] = "ok" if segs else "none_strong"
    return segs, meta


def merge(regular: List[Dict[str, Any]], comments: List[Dict[str, Any]], types: Sequence[str],
          max_clips: int, auto_max: int = 2) -> List[Dict[str, Any]]:
    """Comment clips among the others (runner segments: start_sec / duration_sec). «comment» chosen alone → only
    comment clips (the others fill up only when none was found); with other types or «Авто» → up to ``auto_max``
    comment clips (all requested ones when «comment» is in the list) go first, the rest keep their rank."""
    def span(s):
        a = float(s["start_sec"])
        return a, a + float(s["duration_sec"])

    def clash(s, taken):
        a, b = span(s)
        p = s.get("payoff")
        rng = [(a, b)] + ([(float(p["start"]), float(p["end"]))] if p else [])
        for t in taken:
            ta, tb = span(t)
            tp = t.get("payoff")
            trng = [(ta, tb)] + ([(float(tp["start"]), float(tp["end"]))] if tp else [])
            if any(x < y2 and x2 > y for x, x2 in rng for y, y2 in trng):
                return True
        return False
    only = list(types) == ["comment"]
    want_c = max_clips if only else (max_clips if "comment" in types else auto_max)
    out: List[Dict[str, Any]] = []
    for s in comments[:want_c]:
        if not clash(s, out):
            out.append(s)
    if only and out:
        return out[:max_clips]
    for s in regular:
        if len(out) >= max_clips:
            break
        if not clash(s, out):
            out.append(s)
    return out
