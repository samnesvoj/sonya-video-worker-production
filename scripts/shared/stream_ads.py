"""
stream_ads.py
=============
Sponsor / ad inserts inside a stream VOD (owner 2026-11-19: gumrod job 0c9c1be0 clip 3 ended on the streamer's
own ad read «Любители КС… площадку для обмена скинами… Liss Kins…»). A Streamer clip may not start in, end in or
contain one.

  1. local: every sentence gets an ad-cue weight (promo codes, «по ссылке», «спонсор», casino / betting / skins
     services, «закажите», «скидка», «работаем с ними», «баланс сайта»…); a 30 s window whose weight passes the bar
     is an ad candidate, grown by neighbouring sentences that still carry cues, merged;
  2. Gemini (analysis only, budget cap): each candidate region is shown as numbered phrases with ~25 s of context;
     the model answers ad yes / no and the first / last phrase of the insert. «No» drops a weak local region, a
     strong one (promo code + link, a casino / betting brand…) stays — excluding a few seconds is the safe side;
     any model failure keeps the local regions;
  3. `cut_out(segments, ranges)`: a clip that touches an insert is cut back to end before it (or start after it)
     on a finished phrase when ≥ min seconds remain, else dropped.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

VERSION = "stream-ads-v1"
PROMPT_VERSION = "stream-ads-check-v1"

# strong cue: alone it almost always means an ad read (weight 3)
_STRONG = re.compile(
    r"(?i)(промо-?код|по\s+(моей\s+|этой\s+)?ссылке|ссылк\w*\s+(\w+\s+){0,3}(в\s+описани|в\s+закреп|под\s+видео|в\s+чате)|"
    r"спонсор\w*|рекламн\w*|реклам[аеуы]\b|интеграци\w*|erid|партн[её]р\w*\s+(стрима|выпуска|канала)|"
    r"казино|казик|букмекер\w*|ставк[аиуе]\s+на\s+спорт|фрибет\w*|фриспин\w*|депозит\w*|кэшб[эе]к\w*|кешб[эе]к\w*|"
    r"1\s?x\s?bet|1\s?win|ван\s?вин|пин-?ап|pin-?up|мелбет|фонбет|винлайн|winline|лига\s+ставок|olimpbet|олимпбет|"
    r"betboom|бетбум|leon\s?bet|леон\s?бет|джойказино|joycasino|вулкан\w*|cs\s?money|кс\s?мани|"
    r"зарегистрир\w+\s+(по|на|у)|регистраци\w+\s+по|спешу\s+вас\s+обрадовать|работаем\s+с\s+ними|"
    r"переход\w*\s+по\s+ссылк\w*|скачива\w+\s+(приложени|по\s+ссылк)|первы\w+\s+(заказ|депозит|пополнени)|"
    r"баланс\w*\s+сайта|вывод\w*\s+средств|получени\w*\s+средств)")
# medium cue: an ad read is a dense run of these (weight 1)
_MEDIUM = re.compile(
    r"(?i)(скидк\w*|бонус\w*|закаж\w+|заказ\w*|доставк\w*|сервис\w*|площадк\w*|платформ\w*|приложени\w*|"
    r"сайт\w*|баланс\w*|пополн\w*|лимит\w*|выгодн\w*|безопасн\w*|удобн\w*|надёжн\w*|надежн\w*|быстр\w+\s+и\s+\w+|"
    r"рекомендую|советую\s+(вам|всем)|подписк\w*|маркетплейс\w*|магазин\w*|промо\w*|акци[яиюе]\w*|"
    r"скин\w*|обмен\w*|продат\w*\s+скин|трейд\w*|рубл\w*|процент\w*|%|бесплатн\w*|ссылк\w*|описани\w*|"
    r"средств\w*|сумм[аеуы]\w*|кошел[её]к\w*|карт[уыа]\s+(банка|сбер|тинькофф)|оформ\w+|покупк\w*|купит\w*|цен[аыу]\b|"
    r"покажу\s+вам|выбираем|указываем|вставляем|нажимаем|готово)")

WINDOW_SEC = 30.0
BAR = 6.0                 # window weight that makes an ad candidate (e.g. one strong + three medium)
STRONG_KEEP = 6.0         # strong weight in a region that keeps it even when the model says «no ad»


def sentence_weight(text: str) -> Tuple[float, float]:
    """(total, strong) cue weight of one sentence."""
    s = 3.0 * len(_STRONG.findall(text))
    m = 1.0 * len(_MEDIUM.findall(text))
    return s + m, s


def _sents(words) -> List[Tuple[int, int]]:
    from .boundaries import group_sentences
    return group_sentences(words, pause_sec=1.2)


def local_regions(words, window: float = WINDOW_SEC, bar: float = BAR) -> List[Dict[str, Any]]:
    """Ad candidate regions [{start, end, first, last, weight, strong, text}] from cue density (words = Word list)."""
    sents = _sents(words)
    if not sents:
        return []
    info = []
    for a, b in sents:
        txt = " ".join(w.text.strip() for w in words[a:b + 1])
        tot, strong = sentence_weight(txt)
        info.append((a, b, words[a].start, words[b].end, tot, strong, txt))
    hot = [False] * len(info)
    j0 = 0
    for i in range(len(info)):
        # window of sentences starting at i spanning ≤ window seconds
        tot = strong = 0.0
        j = i
        while j < len(info) and info[j][3] - info[i][2] <= window:
            tot += info[j][4]
            strong += info[j][5]
            j += 1
        if tot >= bar and (strong > 0 or tot >= bar + 3):
            for k in range(i, max(i + 1, j)):
                if info[k][4] > 0:
                    hot[k] = True
        j0 = j
    del j0
    # a region = hot sentences; gaps of ≤ 5 cue-less sentences (≤ 15 s) inside an ad read are bridged
    regions: List[Dict[str, Any]] = []
    i = 0
    while i < len(info):
        if not hot[i]:
            i += 1
            continue
        k = i
        while True:
            nxt = next((m for m in range(k + 1, min(len(info), k + 7)) if hot[m]
                        and info[m][2] - info[k][3] <= 15.0), None)
            if nxt is None:
                break
            k = nxt
        # neighbours that still carry cues belong to the read
        while k + 1 < len(info) and info[k + 1][4] > 0 and info[k + 1][2] - info[k][3] <= 3.0:
            k += 1
        s = i
        while s - 1 >= 0 and info[s - 1][4] > 0 and info[s][2] - info[s - 1][3] <= 3.0:
            s -= 1
        f = info[s][0]          # back to the start of the spoken sentence (a comma / short pause split it)
        while f > 0 and words[f].start - words[f - 1].end < 1.0 and words[info[s][0]].start - words[f - 1].start <= 6.0 \
                and not re.search(r"[.!?…]$", words[f - 1].text.strip()):
            f -= 1
        regions.append({"start": round(float(words[f].start), 3), "end": round(float(info[k][3]), 3),
                        "first": f, "last": info[k][1],
                        "weight": sum(x[4] for x in info[s:k + 1]), "strong": sum(x[5] for x in info[s:k + 1]),
                        "text": " ".join(x[6] for x in info[s:k + 1])[:1200]})
        i = k + 1
    return regions


# ── Gemini check (analysis only) ────────────────────────────────────────────

SYSTEM = ("Ты проверяешь расшифровку записи стрима: есть ли в фрагменте РЕКЛАМНАЯ или СПОНСОРСКАЯ вставка — "
          "стример (или ролик) рекламирует сервис, казино, ставки, обмен скинов, магазин, доставку, приложение, "
          "называет промокод, просит перейти по ссылке, рассказывает про скидку / бонус. Обычная игра, разговор о "
          "деньгах В ИГРЕ, донаты и общение с чатом — НЕ реклама. Ты не придумываешь время — только номера фраз. "
          "Отвечай только JSON.")


def check_messages(regions: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    items = [{"id": f"r{i}", "phrases": [f"{k}) {t[:220]}" for k, (_, t) in enumerate(r["phrases"])]}
             for i, r in enumerate(regions)]
    user = ("Фрагменты:\n" + json.dumps(items, ensure_ascii=False) +
            "\n\nВерни JSON: {\"regions\": [{\"id\": str, \"ad\": bool, \"first\": int (номер первой фразы рекламы), "
            "\"last\": int (номер последней фразы рекламы), \"brand\": \"что рекламируют, до 4 слов\"}]} — для КАЖДОГО.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def _context_phrases(words, sents, r, pad: float = 25.0) -> List[Tuple[Tuple[int, int], str]]:
    out = []
    for a, b in sents:
        if words[b].end < r["start"] - pad or words[a].start > r["end"] + pad:
            continue
        out.append(((a, b), " ".join(w.text.strip() for w in words[a:b + 1])))
    return out


def detect(raw_words: Sequence[Dict[str, Any]], *, editor: bool = False, chat: Optional[Callable[..., Any]] = None,
           budget_rub: float = 1.0, rub_per_usd: float = 100.0, words=None) -> Tuple[List[Tuple[float, float]], Dict[str, Any]]:
    """-> ([(start, end), ...] ad ranges in source seconds, meta)."""
    if words is None:
        from .boundaries import normalize_words
        words, _ = normalize_words(raw_words)
    meta: Dict[str, Any] = {"version": VERSION, "llm": {"status": "off"}}
    if not words:
        return [], dict(meta, regions=[], ranges=[])
    regs = local_regions(words)
    meta["local_regions"] = [{k: r[k] for k in ("start", "end", "weight", "strong")} for r in regs]
    if not regs:
        return [], dict(meta, regions=[], ranges=[])
    final = [dict(r, source="local") for r in regs]
    if editor and regs:
        sents = _sents(words)
        for r in regs[:12]:
            r["phrases"] = _context_phrases(words, sents, r)
        msgs = check_messages(regs[:12])
        chars = sum(len(m["content"]) for m in msgs)
        est = round((chars / 2.5 * 0.5 + (60 * len(regs) + 200) * 3.0) / 1e6 * rub_per_usd, 3)
        meta["llm"] = {"status": "skipped_budget", "forecast_rub": est, "prompt_version": PROMPT_VERSION}
        if est <= budget_rub:
            try:
                if chat is None:
                    from . import llm_provider
                    chat = llm_provider.chat_json
                res = chat(msgs, prompt_version=PROMPT_VERSION, opt_in=True, timeout=90.0,
                           extra={"reasoning": {"effort": "low"}})
                tel = res.telemetry() if hasattr(res, "telemetry") else {}
                meta["llm"] = dict(tel, status="ok", forecast_rub=est, prompt_version=PROMPT_VERSION)
                if tel.get("input_tokens") is not None:
                    meta["llm"]["cost_rub_estimate"] = round((float(tel["input_tokens"]) * 0.5 + float(
                        tel.get("output_tokens") or 0) * 3.0) / 1e6 * rub_per_usd, 4)
                verdicts = _parse(res.text, len(regs[:12]))
                final = []
                for i, r in enumerate(regs):
                    v = verdicts.get(f"r{i}") if i < 12 else None
                    if v is None:
                        final.append(dict(r, source="local (no verdict)"))
                    elif v["ad"]:
                        ph = r["phrases"]
                        a = ph[max(0, min(len(ph) - 1, v["first"]))][0][0]
                        b = ph[max(0, min(len(ph) - 1, v["last"]))][0][1]
                        if b < a:
                            a, b = r["first"], r["last"]
                        # the model's span, never smaller than the local cue run (cues are hard evidence)
                        a, b = min(a, r["first"]), max(b, r["last"])
                        final.append(dict(r, first=a, last=b, start=round(float(words[a].start), 3),
                                          end=round(float(words[b].end), 3), source="editor", brand=v["brand"]))
                    elif r["strong"] >= STRONG_KEEP:
                        final.append(dict(r, source="local strong (editor: no)"))
            except Exception as exc:                   # never let the check kill the run: keep the local regions
                meta["llm"] = {"status": "error", "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                               "prompt_version": PROMPT_VERSION}
                final = [dict(r, source="local (editor error)") for r in regs]
    ranges = merge([(r["start"], r["end"]) for r in final])
    meta["regions"] = [{"start": r["start"], "end": r["end"], "source": r["source"], "brand": r.get("brand", ""),
                        "text": r["text"][:200]} for r in final]
    meta["ranges"] = [[round(a, 3), round(b, 3)] for a, b in ranges]
    return ranges, meta


def _parse(text: str, n: int) -> Dict[str, Dict[str, Any]]:
    try:
        data = json.loads(re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M))
    except ValueError:
        return {}
    out = {}
    for x in (data.get("regions") if isinstance(data, dict) else data) or []:
        if not isinstance(x, dict) or not isinstance(x.get("id"), str):
            continue
        try:
            f, l = int(x.get("first", 0)), int(x.get("last", 0))
        except (TypeError, ValueError):
            f, l = 0, 0
        out[x["id"]] = {"ad": x.get("ad") is True, "first": f, "last": l, "brand": str(x.get("brand") or "")[:40]}
    return out


def merge(ranges: Sequence[Tuple[float, float]], gap: float = 2.0) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for a, b in sorted((float(a), float(b)) for a, b in ranges if b > a):
        if out and a <= out[-1][1] + gap:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def overlaps(a: float, b: float, ranges: Sequence[Tuple[float, float]], pad: float = 0.0) -> bool:
    return any(a < y + pad and b > x - pad for x, y in ranges)


def cut_out(segments: List[Dict[str, Any]], ranges: Sequence[Tuple[float, float]], words: Sequence[Dict[str, Any]],
            min_sec: float = 15.0, key_start: str = "start", key_dur: str = "duration") -> Tuple[List[Dict[str, Any]], List[str]]:
    """Clips touching an ad insert: keep the longer clean side (ending / starting on a finished phrase, ≥ min_sec),
    else drop. Returns (segments, notes)."""
    if not ranges:
        return list(segments), []
    notes: List[str] = []
    out = []
    for s in segments:
        st = float(s[key_start])
        en = st + float(s[key_dur])
        hit = [(a, b) for a, b in ranges if st < b + 0.3 and en > a - 0.3]
        lead = s.get("lead_in")
        if lead and overlaps(float(lead[0]), float(lead[1]), ranges, 0.3):
            s = {k: v for k, v in s.items() if k != "lead_in"}
        if not hit:
            out.append(s)
            continue
        a0, b0 = min(a for a, _ in hit), max(b for _, b in hit)
        best = None
        # side before the ad: the last word that ends ≥ 0.3 s before it
        pre = [w for w in words if float(w["start"]) >= st - 0.05 and float(w["end"]) <= a0 - 0.3]
        if pre:
            e1 = _clean_end(pre)
            if e1 is not None and e1 - st >= min_sec:
                best = (st, e1)
        post = [w for w in words if float(w["start"]) >= b0 + 0.3 and float(w["end"]) <= en + 0.05]
        if post:
            s1 = float(post[0]["start"])
            e1 = _clean_end(post)
            if e1 is not None and e1 - s1 >= min_sec and (best is None or e1 - s1 > best[1] - best[0] + 5.0):
                best = (s1, e1)
        if best is None:
            notes.append(f"ad_drop {st:.0f}-{en:.0f} (ad {a0:.0f}-{b0:.0f})")
            continue
        n = dict(s)
        n[key_start] = round(best[0], 3)
        n[key_dur] = round(best[1] - best[0], 3)
        if "end" in n:
            n["end"] = round(best[1], 3)
        for k in ("payoff_t", "punchline_t", "peak_t"):
            if n.get(k) is not None and not (best[0] <= float(n[k]) <= best[1]):
                n.pop(k)
        n["ad_trim"] = [round(a0, 3), round(b0, 3)]
        notes.append(f"ad_trim {st:.0f}-{en:.0f} → {best[0]:.0f}-{best[1]:.0f} (ad {a0:.0f}-{b0:.0f})")
        out.append(n)
    return out, notes


def _clean_end(ws: Sequence[Dict[str, Any]]) -> Optional[float]:
    """End of the last finished phrase in ws: terminal punctuation, else the last word before a ≥ 0.6 s pause."""
    from .cut_selection import _STATEMENT_END
    txt = lambda w: str(w.get("word", w.get("text", ""))).strip()   # noqa: E731
    for i in range(len(ws) - 1, -1, -1):
        if _STATEMENT_END.search(txt(ws[i])) or re.search(r"[?!]$", txt(ws[i])):
            return float(ws[i]["end"])
    for i in range(len(ws) - 2, -1, -1):
        if float(ws[i + 1]["start"]) - float(ws[i]["end"]) >= 0.6:
            return float(ws[i]["end"])
    return float(ws[-1]["end"]) if ws else None
