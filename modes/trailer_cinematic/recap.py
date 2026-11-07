"""
recap.py — LLM half of trailer_cinematic v2 "AI Cinematic Story Recap" (docs/TRAILER_V2_RECAP_SPEC.md).

    Story Analyst  map (reused from story.py: chunked, bounded memory) → recap_refine (global story map +
                   theme, causal links, emotional peaks, twists, opening/ending options)
    Writer         ONE continuous third-person script (segments), sized by the CALIBRATED voice, not by word count
    Visual Director  shots for the script from the analysed candidates (Visual Analyst) — it does NOT pick lines
    Creative Judge creative quality only (technical checks are validator.py's job — an LLM never grades them)

Every validate_* returns a clean structure or raises budget.SchemaError (one repair request by the LLM wrapper).
Prompts are SONYA's own text.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence

from .budget import SchemaError
from .story import _need, _num, _str, _str_list, fmt_t, validate_story_map

REFINE_VERSION = "recap-analyst-refine-v1"
WRITER_VERSION = "recap-writer-v1"
DIRECTOR_VERSION = "recap-visual-director-v2"
JUDGE_VERSION = "recap-judge-v1"

SEG_FUNCTIONS = ("hook", "setup", "development", "turn", "climax", "resolution")
PUNCHY_FUNCTIONS = SEG_FUNCTIONS + ("salvo", "punchline")
GAG_DEVICES = ("hyperbole", "absurd_comparison", "rhetorical_question", "false_twist")

# Two narration styles. "calm" = the v2 documentary recap (owner spec 2026-11-03). "punchy" = owner 2026-11-04:
# «пипец угар щас происходит» — fast, cheeky, funny, profanity allowed in place, a salvo of the wildest REAL moments,
# a punchline; jokes that are invented are allowed only as obvious devices and are tagged kind=invented_gag.
# The spec's "do not speed the voice up artificially" is lifted by the owner for this style: tempo > 1, short pauses.
STYLES: Dict[str, Dict[str, Any]] = {
    "calm": {"tempo": 1.0, "overhead_sec": 9.0, "segments": (6, 9),
             "pauses": {"sent": 0.30, "sent_q": 0.40, "after": {"short": 0.45, "normal": 0.65, "long": 0.95}}},
    "punchy": {"tempo": 1.8, "overhead_sec": 5.0, "segments": (8, 14),
               "pauses": {"sent": 0.07, "sent_q": 0.14, "after": {"short": 0.10, "normal": 0.18, "long": 0.40}}},
}
DEFAULT_STYLE = "punchy"
REFINE_VERSION_PUNCHY = "recap-analyst-refine-punchy-v1"
WRITER_VERSION_PUNCHY = "recap-writer-punchy-v10-grounded"
JUDGE_VERSION_PUNCHY = "recap-judge-punchy-v2-noname"
JUDGE_KEYS_PUNCHY = ("hook", "coherence", "factuality", "energy", "humor_surprise", "pace", "clarity", "visual_match",
                     "no_repetition", "ending", "no_names")
REWRITE_THRESHOLDS_PUNCHY = {"no_names": 10, "factuality": 9, "coherence": 7, "hook": 8, "visual_match": 7, "energy": 8,
                             "humor_surprise": 7, "pace": 8}


def style_cfg(style: str) -> Dict[str, Any]:
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}; known: {sorted(STYLES)}")
    return STYLES[style]


def versions(style: str) -> Dict[str, str]:
    if style == "punchy":
        return {"refine": REFINE_VERSION_PUNCHY, "writer": WRITER_VERSION_PUNCHY, "judge": JUDGE_VERSION_PUNCHY,
                "director": DIRECTOR_VERSION}
    return {"refine": REFINE_VERSION, "writer": WRITER_VERSION, "judge": JUDGE_VERSION, "director": DIRECTOR_VERSION}
JUDGE_KEYS = ("hook", "coherence", "factuality", "emotional_development", "clarity", "visual_match",
              "no_repetition", "ending", "no_unfounded_drama")
REWRITE_THRESHOLDS = {"factuality": 9, "coherence": 8, "hook": 8, "visual_match": 7, "no_unfounded_drama": 8}
PAUSE_AFTER = {"short": 0.45, "normal": 0.65, "long": 0.95}
SENT_PAUSE = 0.30
SENT_PAUSE_Q = 0.40
MIN_SHOT, MAX_SHOT = 1.5, 5.5


# ── 1. Story Analyst: global story map for a recap ─────────────────────────

REFINE_SYSTEM = (
    "Ты — аналитик сюжета. Перед тобой факты обо всём видео, собранные по частям. Собери из них одну "
    "глобальную карту истории для видеопересказа. Событий не выдумывай: используй только переданные id. "
    "Хронологию не искажай; не создавай причинно-следственных связей, которых нет в материале. Отвечай только JSON.")


REFINE_SYSTEM_PUNCHY = (
    "Ты — аналитик сюжета для дерзкого смешного трейлера-пересказа. Перед тобой факты обо всём фильме, собранные по "
    "частям, и куски реальной расшифровки. Собери глобальную карту истории И отдельно найди самое жёсткое, безумное, "
    "смешное и шокирующее, что РЕАЛЬНО есть в материале (конкретные поступки, суммы, вещества, выходки, унижения, "
    "абсурд), — это пойдёт в залп посреди ролика. Ничего не выдумывай: только переданные id и то, что сказано в "
    "событиях и расшифровке. Персонажи могут быть реальными людьми — им нельзя приписывать то, чего нет в материале. "
    "Хронологию не искажай. Отвечай только JSON.")


def refine_messages(parts: List[Dict[str, Any]], duration: float, language: str, style: str = "calm",
                    evidence: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    events = [{k: e[k] for k in ("id", "start", "end", "summary", "participants", "importance", "emotion")}
              for p in parts for e in p["events"]]
    if style == "punchy":
        for e in events:
            if (evidence or {}).get(e["id"]):
                e["transcript"] = evidence[e["id"]]
        chars = {}
        for p in parts:
            for c in p["characters"]:
                chars.setdefault(c["name"], c["description"])
        task = (
            f"Длина исходника: {fmt_t(duration)}. Частей: {len(parts)}. Язык: {language}.\n\n"
            "ПЕРСОНАЖИ:\n" + json.dumps(chars, ensure_ascii=False) +
            "\n\nСОБЫТИЯ ПО ПОРЯДКУ (transcript — кусок реальной расшифровки этого места):\n" + json.dumps(events, ensure_ascii=False) +
            "\n\nВерни JSON:\n{\"characters\": [{\"name\": str, \"description\": \"кто это; главный герой — первым\"}], "
            "\"premise\": \"о чём фильм, 1–2 предложения\", \"theme\": \"тема одной фразой\", \"conflict\": \"главный конфликт\", "
            "\"event_ids\": [18–30 id по порядку: сюжетный хребет + ВСЕ самые безумные моменты], "
            "\"causal_links\": [{\"cause\": id, \"effect\": id, \"why\": \"только если это ЕСТЬ в материале\"}], "
            "\"emotional_peaks\": [id], \"twists\": [id неожиданных поворотов], "
            "\"wild_moments\": [{\"event_id\": id, \"detail\": \"что именно дико/смешно/шокирует — конкретно, 4–14 слов, только из "
            "события и его transcript\"}] (8–14 штук, от самого жёсткого), "
            "\"opening_options\": [{\"event_id\": id, \"why\": \"чем бьёт с первой секунды\"}], "
            "\"ending_options\": [{\"event_id\": id, \"why\": \"почему годится для панчлайна\"}], "
            "\"open_questions\": [str], \"ending\": \"чем заканчивается фильм\", \"spoiler_sensitive_events\": []}")
        return [{"role": "system", "content": REFINE_SYSTEM_PUNCHY}, {"role": "user", "content": task}]
    chars: Dict[str, str] = {}
    for p in parts:
        for c in p["characters"]:
            chars.setdefault(c["name"], c["description"])
    task = (
        f"Длина исходника: {fmt_t(duration)}. Частей: {len(parts)}. Язык: {language}.\n\n"
        "ПЕРСОНАЖИ ИЗ ЧАСТЕЙ:\n" + json.dumps(chars, ensure_ascii=False) +
        "\n\nПОСЫЛКИ И КОНФЛИКТЫ ПО ЧАСТЯМ:\n" + json.dumps([{"premise": p["premise"], "conflict": p["conflict"]}
                                                          for p in parts], ensure_ascii=False) +
        "\n\nВСЕ СОБЫТИЯ ПО ПОРЯДКУ:\n" + json.dumps(events, ensure_ascii=False) +
        "\n\nВОПРОСЫ ИЗ ЧАСТЕЙ:\n" + json.dumps([q for p in parts for q in p["open_questions"]], ensure_ascii=False) +
        "\n\nВерни JSON:\n{\"characters\": [{\"name\": str, \"description\": \"кто это и чего хочет; главный герой — первым\"}], "
        "\"premise\": \"о чём вся история, 1–2 предложения\", \"theme\": \"тема одной фразой\", "
        "\"conflict\": \"главный конфликт\", "
        "\"event_ids\": [id значимых событий по порядку; повторы и лишнее убери], "
        "\"causal_links\": [{\"cause\": id, \"effect\": id, \"why\": \"как связаны, только если это ЕСТЬ в материале\"}], "
        "\"emotional_peaks\": [id самых эмоциональных событий], \"twists\": [id неожиданных поворотов], "
        "\"opening_options\": [{\"event_id\": id, \"why\": \"чем цепляет\"}], "
        "\"ending_options\": [{\"event_id\": id, \"why\": \"почему сильная концовка\"}], "
        "\"open_questions\": [\"что осталось нерешённым\"], \"ending\": \"чем заканчивается материал\", "
        "\"spoiler_sensitive_events\": []}")
    return [{"role": "system", "content": REFINE_SYSTEM}, {"role": "user", "content": task}]


def validate_recap_map(data: Any, parts: List[Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(data, dict):
        data = dict(data, spoiler_sensitive_events=data.get("spoiler_sensitive_events") or [])
    sm = validate_story_map(data, parts)
    known = {e["id"] for e in sm["events"]}

    def ids(x):
        return [i for i in _str_list(x, "ids", 40, 16) if i in known]
    links = []
    for l in data.get("causal_links") or []:
        if isinstance(l, dict) and l.get("cause") in known and l.get("effect") in known:
            links.append({"cause": l["cause"], "effect": l["effect"], "why": str(l.get("why") or "")[:200]})
    def opts(x):
        return [{"event_id": o["event_id"], "why": str(o.get("why") or "")[:160]} for o in (x or [])
                if isinstance(o, dict) and o.get("event_id") in known][:4]
    wild = [{"event_id": w["event_id"], "detail": str(w.get("detail") or "")[:160]}
            for w in (data.get("wild_moments") or []) if isinstance(w, dict) and w.get("event_id") in known and w.get("detail")]
    sm.update(theme=_str(data.get("theme") or sm["premise"], "theme", 200), causal_links=links[:20], wild_moments=wild[:16],
              emotional_peaks=ids(data.get("emotional_peaks")), twists=ids(data.get("twists")),
              opening_options=opts(data.get("opening_options")), ending_options=opts(data.get("ending_options")))
    return sm


# ── 2. Writer ───────────────────────────────────────────────────────────────

WRITER_SYSTEM = (
    "Ты — сценарист документальных мини-фильмов. Пишешь на естественном русском единый непрерывный закадровый "
    "рассказ от третьего лица — историю, которую рассказывает диктор поверх чужой картинки. Это не трейлер и не "
    "подборка цитат: связное повествование с яркой, но достоверной первой фразой, понятным развитием, причинно-"
    "следственными переходами, нарастанием интереса и сильной концовкой. Нельзя: сухая новостная сводка; пересказ "
    "как школьное сочинение; кликбейт; бессмысленные повторы и одинаковые зачины; шаблоны («но он ещё не знает», "
    "«в этом видео», «судьба распорядилась»); дешёвая драматизация; приписывать героям эмоции, мысли и мотивы, "
    "которых нет в материале; факты и числа вне карты истории. Цитат героев не используй: оригинальный звук в ролике "
    "отсутствует, говорит только диктор. Числа пиши словами. Без кавычек, эмодзи, тире в начале, скобок. "
    "Фразы должны звучать естественно вслух: средняя длина 9–16 слов, без длинных цепочек причастий. "
    "Отвечай только JSON.")


WRITER_SYSTEM_PUNCHY = (
    "Ты — автор дерзких вертикальных трейлеров-пересказов. Голос — друг, который взахлёб и с ухмылкой рассказывает "
    "самый безумный фильм, что он видел: быстро, нагло, смешно, на удивление — у зрителя должно быть чувство «пипец, "
    "что сейчас происходит». Рассказ ведёт ДИКТОР от третьего лица (он, этот парень, эти ребята) — никакого «я»: "
    "говорить голосом героя нельзя, это реальный человек. ИМЕНА: ни одного имени, фамилии, прозвища или ника "
    "человека в text — ни героя, ни ведущего, ни гостей, ни упомянутых людей (владелец так решил). Говори «этот "
    "парень», «этот мужик», «его кореш», «ведущий», «эта девушка», «его жена», «агент», «тот самый босс»; если "
    "двое — различай по роли и делу, а не по имени. Названия мест, компаний, брендов и фильмов можно. Имена можно "
    "держать только в visual_intent (для подбора кадров), в text — нет. Это НЕ документалка и НЕ школьный пересказ: никаких «он основал империю, нацеленную на…», "
    "никаких причастных оборотов и канцелярита. Подача: короткие рубленые фразы по 2–8 слов, много точек, ритм, "
    "нарастание. Первая фраза — удар: самый дикий факт или провокация. В середине — ЗАЛП (function=salvo): через "
    "запятую 6–9 самых жёстких и безумных вещей, которые РЕАЛЬНО есть в фильме (wild_moments, transcript), конкретно, "
    "без обобщений; каждый пункт — с тем, кто это сделал, если это не главный герой (не приписывай герою чужие "
    "поступки). Дальше — неожиданный поворот, ирония, сарказм. В конце — панчлайн (function=punchline): короткая "
    "дерзкая добивка. Мат разрешён: к месту, два–четыре раза за ролик, не через слово, без звёздочек. Провокация — да. "
    "Оскорбления по национальности, полу, ориентации, внешности — нет. ВЫДУМКА разрешена только как приём, по которому "
    "сразу ясно, что это шутка: явная гипербола, абсурдное сравнение, риторический вопрос, ложный поворот, который "
    "тут же обыгрывается. Каждую такую фразу перечисли в gags своего сегмента (точная цитата из text). Нельзя: "
    "выдумывать события сюжета, преступления, поступки и цифры; подавать шутку как факт фильма. Героев можно "
    "жёстко и смешно, но: никаких шуток и подробностей про суицид, самоповреждение, слепоту и болезни, про "
    "смерть и убийства, зависимости реальных людей (факт можно назвать одним словом, без деталей и без "
    "юмора); если в рассказе упомянуто чужое преступление или убийство, ясно говори, что его совершил кто-то "
    "другой, а не этот человек. Слова «суицид», «вскрыть вены», «член» и шутки про слепоту ЗАПРЕЩЕНЫ (код проверяет и отклоняет). Пункты залпа "
    "не повторяй в других сегментах. Это важно ещё и потому, что модерация отклоняет такие подробности. Герои основаны на "
    "реальных людях — про них только то, что есть в карте. Факты — только из карты. Цитат героев нет: оригинального "
    "звука в ролике нет, говорит только диктор. Числа словами. Без кавычек, эмодзи, скобок, тире в начале. Без "
    "«в этом видео», «смотрите до конца», «судьба распорядилась». Отвечай только JSON.")


def story_for_writer(sm: Dict[str, Any], evidence: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
    out = {k: sm.get(k) for k in ("characters", "premise", "theme", "conflict", "causal_links", "emotional_peaks",
                                  "twists", "opening_options", "ending_options", "open_questions", "ending")}
    if sm.get("wild_moments"):
        out["wild_moments"] = sm["wild_moments"]
    if SENSITIVE_PUNCHY_INPUT:       # the writer must not even see suicide / self-harm material (it then keeps coming back to it)
        for k in ("wild_moments", "emotional_peaks", "twists", "opening_options", "ending_options", "causal_links", "open_questions"):
            if isinstance(out.get(k), list):
                out[k] = [x for x in out[k] if not _SENS_INPUT.search(json.dumps(x, ensure_ascii=False))]
        out["characters"] = [dict(c, description="") if isinstance(c, dict) and _SENS_INPUT.search(str(c.get("description") or "")) else c
                             for c in (out.get("characters") or [])]
    out["events"] = [{"id": e["id"], "when": f"{fmt_t(e['start'])}", "summary": e["summary"],
                      "participants": e["participants"], "emotion": e["emotion"], "importance": e["importance"],
                      **({"evidence": (evidence or {}).get(e["id"], [])[:1]} if evidence else {})}
                     for e in sm["events"] if not (SENSITIVE_PUNCHY_INPUT and _SENS_INPUT.search(e["summary"]))]
    return out


def writer_messages(sm: Dict[str, Any], target_sec: float, chars_lo: int, chars_hi: int, rate: Dict[str, Any],
                    previous: Optional[Dict[str, Any]] = None, instructions: Optional[str] = None,
                    style: str = "calm", evidence: Optional[Dict[str, List[str]]] = None) -> List[Dict[str, Any]]:
    if style == "punchy":
        lo_n, hi_n = STYLES["punchy"]["segments"]
        task = (
            f"ЗАДАЧА: трейлер-пересказ ≈ {target_sec:.0f} секунд БЫСТРОЙ речи. Голос измерен: ≈ {rate['chars_per_sec']:.1f} "
            f"знаков в секунду. ВЕСЬ текст (сумма text, с пробелами) — от {chars_lo} до {chars_hi} знаков (это примерно "
            f"{chars_lo // 7}–{chars_hi // 7} слов: текста МНОГО, речь быстрая). Считай знаки.\n\n"
            "КАРТА ИСТОРИИ (единственный источник фактов; evidence — реальная расшифровка):\n"
            + json.dumps(story_for_writer(sm, evidence), ensure_ascii=False) +
            "\n\nПорядок подачи можно менять ради ритма, но событие, которое было ПОСЛЕ, нельзя подавать как причину. "
            f"Сценарий — {lo_n}–{hi_n} сегментов: hook (удар), дальше разгон истории, ровно один salvo (залп безумия), "
            "turn/climax (как всё летит к чертям), последний — punchline. Каждый сегмент — 1–5 коротких предложений; "
            "в среднем не больше 7 слов на предложение (залп — одно длинное перечисление через запятую). "
            "Для каждого сегмента: event_ids (только id из карты), visual_intent (что должно быть в кадре: кто, где, "
            "что делает — конкретно, 5–12 слов, по смыслу ЭТОГО текста), pause_after (short|normal|long), "
            "gags — список выдуманных приёмов этого сегмента [{\"text\": \"точная цитата из text\", \"device\": \""
            + "|".join(GAG_DEVICES) + "\"}] или [] (всё остальное в тексте — факты фильма).\n"
            "Верни JSON: {\"title\": \"рабочее название\", \"segments\": [{\"id\": \"S1\", \"function\": \""
            + "|".join(PUNCHY_FUNCTIONS) + "\", \"text\": \"…\", \"event_ids\": [\"E…\"], \"visual_intent\": \"…\", "
            "\"pause_after\": \"short|normal|long\", \"gags\": []}]}.")
        if sm.get("forbid_names"):
            task += ("\n\nЗАПРЕЩЁННЫЕ СЛОВА (имена людей, в любом падеже; в text их быть не должно, даже если слово похоже на "
                     "обычное): " + ", ".join(x.rstrip("$") for x in sm["forbid_names"]) + ".")
        if sm.get("source_crime_stems") is not None:
            task += ("\n\nСЛОВА О НАСИЛИИ И ПРЕСТУПЛЕНИЯХ: можно ТОЛЬКО слова с корнями, которые реально звучат в источнике: "
                     + (", ".join(sm["source_crime_stems"]) or "(нет таких)") + ". Любые другие (зарезали, вломились, "
                     "застрелили, избили…) — нельзя, даже как шутку: пиши нейтрально (приходили, приехали, забрали).")
        if previous is not None and instructions:
            task += ("\n\nПРЕДЫДУЩИЙ ВАРИАНТ:\n" + json.dumps(previous, ensure_ascii=False) +
                     "\n\nПРАВКИ (выполни все, тон и темп сохрани):\n" + instructions)
        return [{"role": "system", "content": WRITER_SYSTEM_PUNCHY}, {"role": "user", "content": task}]
    task = (
        f"ЗАДАЧА: единый рассказ ≈ {target_sec:.0f} секунд речи. Голос измерен: ≈ {rate['chars_per_sec']:.1f} знаков в "
        f"секунду, пауза между фразами добавляется отдельно. Поэтому ВЕСЬ текст (сумма text по сегментам, с пробелами) — "
        f"от {chars_lo} до {chars_hi} знаков. Считай знаки.\n\nКАРТА ИСТОРИИ (единственный источник фактов):\n"
        + json.dumps(story_for_writer(sm), ensure_ascii=False) +
        "\n\nПорядок подачи можно менять ради рассказа, если это не создаёт ложных причинно-следственных связей: "
        "событие, которое было ПОСЛЕ другого, нельзя подавать как его причину. Начни с самого цепляющего факта "
        "(см. opening_options), а закончи сильной, но честной мыслью (ending_options) — без морали и без выдумок.\n"
        "Сценарий — 6–9 сегментов. Каждый сегмент — 1–3 законченных предложения. Для каждого верни event_ids "
        "(только id из карты — события, о которых сегмент), visual_intent (что в кадре помогло бы зрителю: человек, "
        "место, действие; конкретно, 6–14 слов), pause_after (short|normal|long — смысловая пауза после сегмента).\n"
        "Верни JSON: {\"title\": \"рабочее название\", \"segments\": [{\"id\": \"S1\", \"function\": \""
        + "|".join(SEG_FUNCTIONS) + "\", \"text\": \"…\", \"event_ids\": [\"E…\"], \"visual_intent\": \"…\", "
        "\"pause_after\": \"short|normal|long\"}]}. Первый сегмент — hook, последний — resolution.")
    if previous is not None and instructions:
        task += ("\n\nПРЕДЫДУЩИЙ ВАРИАНТ:\n" + json.dumps(previous, ensure_ascii=False) +
                 "\n\nПРАВКИ (выполни все, остальное сохрани по возможности):\n" + instructions)
    return [{"role": "system", "content": WRITER_SYSTEM}, {"role": "user", "content": task}]


def sentences_of(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?…])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


_BAD_START = re.compile(r"^[\s\-—–«»\"']+")


def _norm_txt(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t.lower().replace("ё", "е"))).strip()


# owner rule (crimes, illness, addiction, relatives of real people) + the provider's content filter: no self-harm /
# suicide wording, no sexual-organ gags, no jokes about blindness in the narration. Deterministic, like the names.
SENSITIVE = re.compile(r"суицид\w*|самоубий\w*|самоповреж\w*|вскры(?:л|ть|вал)\w*\s*(?:вен\w*|себя|ся)?|вскрыва\w+|"
                       r"перереза\w+|повес\w+ себ\w+|\bчлен(?:а|у|ом|е)?\b|\bпенис\w*|"
                       r"(?:причин|нанос|нанес|навред)\w*\s+(?:себе\s+)?вред\w*|вред\w*\s+себе|себе\s+вред\w*|"
                       r"над\w*\s+собой|покончи\w+|полов\w+\s+орган\w*|гениталь\w*|интимн\w+", re.I)
SENSITIVE_PUNCHY_INPUT = True
_SENS_INPUT = re.compile(SENSITIVE.pattern + r"|самоубий\w*|вскры\w*|вены|вен\b|петл\w+|угрож\w+\s+(?:себе|собой)", re.I)
SENSITIVE_GAG = re.compile(r"глаз\w*|слеп\w*|ослеп\w*|слепот\w*|рак\b|болезн\w*", re.I)

# owner rule: never invent crimes / violence of real people. A crime or violence word is allowed in the narration only
# if the same stem is really said in the source transcript («убили» yes, «зарезали» no unless someone said it).
CRIME_STEMS = ("зареза", "убил", "убийств", "убит", "застрел", "задуш", "ограб", "украл", "краж", "вломил", "взлом", "поджег",
               "подожг", "драк", "подрал", "дрался", "дерущ", "наручник", "изнасил", "избил", "избив", "пырнул", "расстрел", "повязал", "арестов", "посадил", "тюрьм")


def crime_stems_in(text: str) -> List[str]:
    t = text.lower().replace("ё", "е")
    return [s for s in CRIME_STEMS if s in t]


FIRST_PERSON = re.compile(r"(?<![\w-])(я|меня|мне|мной|мой|моя|моё|мое|мои|моих|мы|нас|нам|нами|наш|наша|наши)(?![\w-])", re.I)


# ── owner rule 2026-11-04: the narrator never names people (no first names, surnames, nicknames) ──────────────
_NAME_STOP = {"агент", "агента", "отец", "мать", "жена", "муж", "сын", "дочь", "друг", "брат", "сестра", "ведущий",
              "ведущая", "гость", "герой", "героя", "безумный", "босс", "мистер", "господин", "доктор", "дядя",
              "фбр", "сша", "мама", "папа", "мать", "девушка", "парень", "сосед", "соседка", "полицейский", "полицейские",
              "охранник", "врач", "следователь", "адвокат", "судья", "тренер", "учитель", "менеджер", "продюсер", "the", "and", "mr", "mrs", "dr"}


def _name_stem(tok: str) -> str:
    """Stem of a name for case-form matching. A short name that ends in a consonant («Реш») matches exactly only
    (marked with a trailing $), otherwise «реш» would hit «решил»."""
    t = tok.lower().replace("ё", "е")
    if len(t) >= 3 and t[-1] in "ая" or len(t) >= 6 and t[-1] in "уюыиеоьй":
        return t[:-1]
    return t + "$" if len(t) <= 4 else t


def forbidden_names(sm: Dict[str, Any], extra: Sequence[str] = ()) -> List[str]:
    """Stems of every personal name that the story map knows (characters, event participants) + `extra` (CLI)."""
    raw: List[str] = list(extra)
    for c in sm.get("characters") or []:
        if isinstance(c, dict):
            raw.append(str(c.get("name") or ""))
    for e in sm.get("events") or []:
        for p in e.get("participants") or []:
            raw.append(str(p))
    stems: List[str] = []
    for chunk in raw:
        for tok in re.findall(r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё'-]{2,}", chunk):
            if tok.lower().replace("ё", "е") in _NAME_STOP or not tok[0].isupper():
                continue
            st = _name_stem(tok)
            if st not in stems:
                stems.append(st)
    return stems


def find_names(text: str, stems: Sequence[str]) -> List[str]:
    """Deterministic: names (any case form) found in `text`."""
    low = text.lower().replace("ё", "е")
    hits = []
    for st in stems:
        exact = st.endswith("$")
        m = re.search(r"(?<![\w])" + re.escape(st.rstrip("$")) + ("" if exact else (r"\w{0,2}" if len(st) <= 4 else r"\w{0,3}")) + r"(?![\w])", low)
        if m:
            hits.append(m.group(0))
    return hits


def words_per_sentence(text: str) -> float:
    ss = sentences_of(text)
    return sum(len(x.split()) for x in ss) / max(1, len(ss))


def validate_script(data: Any, sm: Dict[str, Any], chars_lo: int, chars_hi: int,
                    length_slack: float = 0.0, style: str = "calm") -> Dict[str, Any]:
    _need(data, ("segments",), "writer")
    segs = data["segments"]
    punchy = style == "punchy"
    n_lo, n_hi = (STYLES["punchy"]["segments"][0] - 1, STYLES["punchy"]["segments"][1] + 1) if punchy else (5, 10)
    if not isinstance(segs, list) or not n_lo <= len(segs) <= n_hi:
        raise SchemaError(f"segments: {n_lo + 1}–{n_hi - 1} segments expected")
    functions = PUNCHY_FUNCTIONS if punchy else SEG_FUNCTIONS
    known = {e["id"] for e in sm["events"]}
    out, total = [], 0
    for i, s in enumerate(segs):
        w = f"segments[{i}]"
        _need(s, ("text", "event_ids", "visual_intent"), w)
        text = _BAD_START.sub("", _str(s["text"], w + ".text", 500)).replace("«", "").replace("»", "").replace('"', "")
        if re.search(r"\d", text):
            raise SchemaError(f"{w}.text: digits are not allowed, spell numbers out in words")
        if not re.search(r"[.!?…]$", text):
            text += "."
        ids = _str_list(s["event_ids"], w + ".event_ids", 4, 16)
        bad = [x for x in ids if x not in known]
        if not ids or bad:
            raise SchemaError(f"{w}.event_ids: {bad or 'empty'} — only ids from the story map")
        fn = s.get("function") if s.get("function") in functions else ("hook" if i == 0 else "development")
        pa = s.get("pause_after") if s.get("pause_after") in PAUSE_AFTER else "normal"
        total += len(text)
        rec = {"id": f"S{i + 1}", "function": fn, "text": text, "event_ids": ids,
               "visual_intent": _str(s["visual_intent"], w + ".visual_intent", 200), "pause_after": pa}
        if punchy:
            gags = []
            for k, g in enumerate(s.get("gags") or []):
                gt = str((g or {}).get("text") if isinstance(g, dict) else g or "").strip()
                if not gt:
                    continue
                if _norm_txt(gt) not in _norm_txt(text):
                    raise SchemaError(f"{w}.gags[{k}]: {gt[:60]!r} is not an exact quote of the segment text")
                dev = g.get("device") if isinstance(g, dict) and g.get("device") in GAG_DEVICES else "hyperbole"
                gags.append({"kind": "invented_gag", "text": gt[:200], "device": dev})
            rec["gags"] = gags
            nm = find_names(text, sm.get("forbid_names") or [])
            if nm:
                raise SchemaError(f"{w}.text: names of people ({', '.join(nm[:4])}) — owner rule: never name anyone in the "
                                  f"narration, say «этот парень», «этот мужик», «его кореш», «ведущий», «эта девушка»")
            bad_t = [m.group(0) for m in SENSITIVE.finditer(text)] + [m.group(0) for g in gags for m in SENSITIVE_GAG.finditer(g["text"])]
            if bad_t:
                raise SchemaError(f"{w}.text: forbidden topics/wording ({', '.join(bad_t[:3])}) — no suicide / self-harm / "
                                  f"sexual-organ / blindness jokes about real people; drop that item and use another real wild "
                                  f"moment (drunk stunts, police, theft, the murder next door, Bali stage injury, etc.)")
            if sm.get("source_crime_stems") is not None:
                inv = [c for c in crime_stems_in(text) if c not in sm["source_crime_stems"]]
                if inv:
                    raise SchemaError(f"{w}.text: crime/violence wording {inv} is not in the source transcript — never invent "
                                      f"crimes or violence of real people; use only the allowed roots from the task, or neutral words "
                                      f"(приходили, приехали, забрали, заметили)")
            fp = FIRST_PERSON.findall(text)
            if fp:
                raise SchemaError(f"{w}.text: first person ({', '.join(fp[:3])}) — the narrator tells it in the third "
                                  f"person, never in the voice of the (real) hero")
            if fn != "salvo" and words_per_sentence(text) > 9.5:
                raise SchemaError(f"{w}.text: {words_per_sentence(text):.1f} words per sentence — chop it into short "
                                  f"phrases (≤ 7 words on average, only the salvo is one long list)")
        out.append(rec)
    if out[0]["function"] != "hook":
        out[0]["function"] = "hook"
    if punchy:
        if not any(o["function"] == "salvo" for o in out):
            raise SchemaError("no salvo segment: one segment must be the rapid-fire list of the wildest real moments")
        out[-1]["function"] = "punchline"
    lo, hi = chars_lo * (1 - length_slack), chars_hi * (1 + length_slack)
    if not lo <= total <= hi:
        raise SchemaError(f"total text length {total} chars is outside {chars_lo}–{chars_hi}: "
                          f"{'shorten' if total > hi else 'lengthen'} it by about "
                          f"{abs(total - (chars_lo + chars_hi) // 2)} chars")
    firsts = [re.sub(r"[^\w]", "", o["text"].split()[0].lower()) for o in out]
    for a, b in zip(firsts, firsts[1:]):
        if a == b and len(a) > 2:
            raise SchemaError(f"two consecutive segments start with the same word {a!r}")
    return {"title": str(data.get("title") or "")[:120], "segments": out, "style": style}


def fit_instructions(measured_sec: float, lo: float, hi: float, per_seg: Dict[str, float], rate: Dict[str, Any]) -> str:
    mid = (lo + hi) / 2
    delta = measured_sec - mid
    dc = int(abs(delta) * rate["chars_per_sec"])
    segs = ", ".join(f"{k}: {v:.1f} с" for k, v in per_seg.items())
    return (f"Реальная озвучка занимает {measured_sec:.1f} с (с паузами), нужно {lo:.0f}–{hi:.0f} с, лучше {mid:.0f} с. "
            f"{'Сократи' if delta > 0 else 'Добавь'} примерно {dc} знаков суммарно, затрагивая в основном самые "
            f"длинные/короткие сегменты, не ломая историю и не добавляя фактов. Длительность сегментов сейчас: {segs}.")


def shot_cap(c: Dict[str, Any]) -> float:
    """Longest a muted shot of this window may stay on screen: a person who may be talking while the narrator speaks
    reads as out-of-sync; a wide shot (lips unreadable) or a listener is safe."""
    st, shot = c["speaking"]["state"], c["shot_type"]
    if st == "likely_speaking":
        return 2.5 if shot in ("closeup", "medium") else 3.5
    if st == "uncertain":
        return 3.0 if shot == "closeup" else 4.0
    return MAX_SHOT


# ── 3. Visual Director ──────────────────────────────────────────────────────

DIRECTOR_SYSTEM = (
    "Ты — режиссёр монтажа документального видеопересказа. Закадровый текст уже написан и озвучен; оригинального "
    "звука в ролике НЕТ. Твоя работа — подобрать к каждому сегменту видеоряд из проанализированных окон исходника. "
    "Используй только окна из списка (usable=true). Кадр должен иллюстрировать произносимое по смыслу или быть "
    "честной нейтральной перебивкой (не выдавай нейтральные кадры за съёмку события). Избегай долгих планов, где "
    "человек говорит (speaking ≠ not_speaking/likely_listening): пока его голоса нет, это выглядит как рассинхрон. "
    "Предпочитай реакции, выразительные крупные планы, перебивки. Не мельтеши: смена кадра — на смысловой границе. "
    "Изображения не генерируются. Отвечай только JSON.")


def candidate_card(c: Dict[str, Any]) -> Dict[str, Any]:
    return {"id": c["id"], "t": f"{c['start']:.0f}-{c['end']:.0f}", "len": round(c["end"] - c["start"], 1),
            "type": c["shot_type"], "kind": c["kind"], "who": [p["who"] for p in c["people"]],
            "action": c["action"], "setting": c["setting"], "speaking": c["speaking"]["state"],
            "interest": c["visual_interest"], "good_for": c["good_for"], "usable": c["usable"],
            "events": c.get("event_ids", []), "problems": c["problems"], "max_seconds": shot_cap(c)}


def director_messages(script: Dict[str, Any], seg_sec: Dict[str, float], cards: List[Dict[str, Any]],
                      hero: str, previous: Optional[Dict[str, Any]] = None,
                      instructions: Optional[str] = None) -> List[Dict[str, Any]]:
    segs = [{"segment_id": s["id"], "function": s["function"], "text": s["text"], "visual_intent": s["visual_intent"],
             "event_ids": s["event_ids"], "seconds": round(seg_sec[s["id"]], 1),
             "windows_of_these_events": [c["id"] for c in cards if set(c.get("events") or []) & set(s["event_ids"])][:12]}
            for s in script["segments"]]
    task = (
        f"ГЛАВНЫЙ ГЕРОЙ: {hero}.\n\nСЦЕНАРИЙ (seconds — реальная длительность озвучки сегмента с паузой):\n"
        + json.dumps(segs, ensure_ascii=False) + "\n\nДОСТУПНЫЕ ОКНА ИСХОДНИКА (len — сколько секунд материала в окне):\n"
        + json.dumps(cards, ensure_ascii=False) +
        f"\n\nДля КАЖДОГО сегмента верни план: 1–4 кадра, seconds каждого {MIN_SHOT}–{MAX_SHOT} с, сумма ≈ seconds сегмента "
        "(±25%). Правила: окно целиком использовать не нужно, но seconds ≤ len окна; одно окно — не дважды (повторное "
        "использование — только если в доступных окнах нет иного и укажи reuse_ok=true); первый кадр ролика — главный "
        "герой в кадре (не случайный человек; окна с hero_in_frame=true — проверено кодом); вместе кадры не должны скакать по хронологии без причины; "
        "seconds ≤ max_seconds окна (говорящие — коротко; слушающие, реакции и общие планы — дольше). В alternates — до 2 запасных окон на сегмент (для замены, если материала не хватит).\n"
        "СМЫСЛ ВАЖНЕЕ КРАСОТЫ: кадр показывает тех, о ком и о чём говорит диктор в этот момент (кто в кадре — поле who). "
        "Если сегмент про агентов ФБР — в кадре агенты, а не жена героя; про деньги — деньги/офис, а не случайный "
        "крупный план. windows_of_these_events — окна, снятые в моменты событий сегмента: хотя бы один кадр бери из них, "
        "если они ещё свободны. Быстрый текст — быстрый монтаж: 1.5–3 с на кадр, в залпе (salvo) по кадру на пункт.\n"
        "Верни JSON: {\"segments\": [{\"segment_id\": \"S1\", \"shots\": [{\"candidate_id\": \"V5\", \"seconds\": 3.0, "
        "\"framing\": \"closeup|medium|wide\", \"why\": \"≤ 12 слов, как кадр соответствует фразе\", \"reuse_ok\": false}], "
        "\"alternates\": [\"V9\"]}]}")
    if previous is not None and instructions:
        task += ("\n\nПРЕДЫДУЩИЙ ПЛАН:\n" + json.dumps(previous, ensure_ascii=False) +
                 "\n\nПРАВКИ (выполни все):\n" + instructions)
    return [{"role": "system", "content": DIRECTOR_SYSTEM}, {"role": "user", "content": task}]


def hero_visible(card: Dict[str, Any], hero: str) -> bool:
    if card.get("faces") == 0:                       # local detector saw no face in any of the window's frames: a veto
        return False
    toks = [t.lower() for t in re.split(r"[\s()«»,]+", hero) if len(t) > 2]
    return any(any(t in p["who"].lower() for t in toks) and p["confidence"] >= 0.5 for p in card["people_raw"])


def validate_director(data: Any, script: Dict[str, Any], seg_sec: Dict[str, float],
                      cands: Dict[str, Dict[str, Any]], hero: str) -> Dict[str, Any]:
    _need(data, ("segments",), "director")
    by = {s["segment_id"]: s for s in data["segments"] if isinstance(s, dict) and "segment_id" in s}
    out, used, dropped = [], {}, []
    hero_possible = any(hero_visible(dict(c, people_raw=c["people"]), hero) for c in cands.values() if c["usable"])
    for si, seg in enumerate(script["segments"]):
        d = by.get(seg["id"])
        if not d or not isinstance(d.get("shots"), list) or not 1 <= len(d["shots"]) <= 4:
            raise SchemaError(f"{seg['id']}: 1–4 shots expected")
        shots, tot, all_capped = [], 0.0, True
        for k, sh in enumerate(d["shots"]):
            w = f"{seg['id']}.shots[{k}]"
            _need(sh, ("candidate_id", "seconds"), w)
            cid = sh["candidate_id"]
            c = cands.get(cid)
            if c is None:
                raise SchemaError(f"{w}: unknown candidate {cid!r}")
            if not c["usable"]:
                raise SchemaError(f"{w}: {cid} is not usable (advertisement / graphic / on-screen text)")
            sec = max(MIN_SHOT, min(MAX_SHOT, _num(sh["seconds"], w + ".seconds")))
            cap = min(shot_cap(c), max(MIN_SHOT, c["end"] - c["start"]))
            all_capped = all_capped and sec >= cap - 1e-6
            sec = min(sec, cap)
            if cid in used and not sh.get("reuse_ok"):
                # observed twice in a row (answer + repair) on a 3-hour film: one stray repeat must not kill the run.
                # Another shot in this segment → drop the repeat (the composer fills the time); the only shot → keep
                # it as an explicit, reported reuse.
                if len(d["shots"]) > 1:
                    dropped.append(f"{w}: {cid} already used in {used[cid]} — dropped")
                    continue
                dropped.append(f"{w}: {cid} already used in {used[cid]} — kept as reuse")
                sh = dict(sh, reuse_ok=True)
            used.setdefault(cid, seg["id"])
            tot += sec
            shots.append({"candidate_id": cid, "seconds": round(sec, 2), "framing": str(sh.get("framing") or "")[:12],
                          "why": str(sh.get("why") or "")[:160], "reuse_ok": bool(sh.get("reuse_ok"))})
        # meaning: if windows filmed during THIS segment's events are still free, at least one shot must come from them
        # (or show the same people) — owner: «федеральные агенты» over the hero's wife on the yacht must not pass
        own = [c for c in cands.values() if c["usable"] and set(c.get("event_ids") or []) & set(seg["event_ids"])
               and c["id"] not in used]
        if own and not any(set(cands[x["candidate_id"]].get("event_ids") or []) & set(seg["event_ids"]) for x in shots):
            raise SchemaError(f"{seg['id']}: no shot is from the events the voice talks about ({', '.join(seg['event_ids'])}); "
                              f"free windows of these events: {', '.join(c['id'] for c in own[:8])}")
        if not shots:
            raise SchemaError(f"{seg['id']}: every shot repeats a window used earlier — pick free windows")
        want = seg_sec[seg["id"]]
        # a shortfall is fine when every shot already sits at its cap (talking heads ≤ 2.5 s, short windows): the model
        # cannot do better with ≤ 4 shots and the composer fills the rest from alternates / neutral windows
        if not (0.7 * want <= tot <= 1.3 * want or (tot < 0.7 * want and all_capped)):
            raise SchemaError(f"{seg['id']}: shots total {tot:.1f}s, the segment lasts {want:.1f}s (±30%)")
        if si == 0 and hero_possible and not hero_visible(dict(cands[shots[0]["candidate_id"]],
                                                              people_raw=cands[shots[0]["candidate_id"]]["people"]), hero):
            raise SchemaError(f"the first shot must show the main character ({hero})")
        alts = [a for a in _str_list(d.get("alternates"), "alternates", 3, 8) if a in cands and cands[a]["usable"]]
        out.append({"segment_id": seg["id"], "shots": shots, "alternates": alts[:2]})
    return {"segments": out, **({"reuse_fixes": dropped} if dropped else {})}


# ── 4. Creative Judge ───────────────────────────────────────────────────────

JUDGE_SYSTEM = (
    "Ты — строгий редактор документального кино, отдельный от автора. Оцени ТОЛЬКО творческую сторону пересказа "
    "по шкале 0–10 (технические параметры — длительность, звук, файл — проверяет код, их не оценивай). Ты видишь "
    "сценарий, карту фактов и кадры, подобранные к каждому сегменту (контактные листы). hook — цепляет ли первая "
    "фраза и первый кадр; coherence — связный рассказ; factuality — каждая фраза подтверждена картой (10 = ничего не "
    "выдумано, хронология не искажена); emotional_development — интерес и эмоция нарастают; clarity — понятно "
    "человеку, не видевшему исходник; visual_match — кадры соответствуют тому, что произносится; no_repetition — "
    "10 = нет повторов мыслей, зачинов и кадров; ending — сила и честность концовки; no_unfounded_drama — 10 = нет "
    "приписанных эмоций/мотивов и дешёвой драматизации. Замечания конкретные: какой сегмент, что не так, как "
    "исправить. В weak_beats — каждый слабый сегмент со степенью severity (minor|major). Отвечай только JSON.")


JUDGE_SYSTEM_PUNCHY = (
    "Ты — строгий редактор дерзких вертикальных трейлеров, отдельный от автора. Цель ролика: быстро, нагло, смешно, "
    "на удивление — зритель должен почувствовать «пипец, что сейчас происходит». Оцени ТОЛЬКО творческую сторону по "
    "шкале 0–10 (длительность, звук, файл, субтитры проверяет код). hook — бьёт ли первая фраза и первый кадр; "
    "coherence — история всё равно понятна и связна; factuality — каждое СЮЖЕТНОЕ утверждение подтверждено картой "
    "(10 = ни одного выдуманного события, поступка, цифры; хронология не искажена). Выдумки разрешены только как "
    "явные приёмы (они перечислены в gags): гипербола, абсурдное сравнение, риторический вопрос, ложный поворот, "
    "который тут же обыгрывается; если шутку можно принять за факт фильма или она приписывает реальному человеку "
    "поступок, которого нет в карте, — это factual_error. Шутки вне gags, которые читаются как факт, — тоже ошибка. "
    "energy — драйв, наглость, нарастание (школьный пересказ = 3); humor_surprise — смешно ли, есть ли повороты и "
    "«чего?!»-моменты, залп самого жёсткого из фильма; pace — короткие рубленые фразы, нет канцелярита и длинных "
    "оборотов; clarity — понятно человеку, не видевшему фильм; visual_match — в кадре те, о ком говорит диктор "
    "(агенты, когда речь про агентов; деньги, когда про деньги); no_repetition — 10 = нет повторов; ending — сила "
    "панчлайна; no_names — 10 = в тексте диктора нет ни одного имени, фамилии, прозвища человека (герой, ведущий, "
    "гости, любые люди; допустимы «этот парень», «ведущий», «его кореш»), любое имя = не выше 3 и major text weak_beat. "
    "Названия мест, компаний и фильмов — не имена людей. Мат к месту допустим; оскорбления по национальности/полу/ориентации — ошибка. Замечания конкретные. "
    "В weak_beats — каждый слабый сегмент: severity (minor|major) и kind (text — проблема текста; visual — кадр не "
    "соответствует тексту). Кадр, где показан не тот человек или не то событие, — всегда visual major. "
    "Отвечай только JSON.")


def judge_keys(style: str = "calm") -> tuple:
    return JUDGE_KEYS_PUNCHY if style == "punchy" else JUDGE_KEYS


def judge_messages(sm: Dict[str, Any], script: Dict[str, Any], plan: Dict[str, Any], cands: Dict[str, Dict[str, Any]],
                   sheets: Sequence[Dict[str, Any]], style: str = "calm") -> List[Dict[str, Any]]:
    storyboard = []
    for seg in script["segments"]:
        p = next((x for x in plan["segments"] if x["segment_id"] == seg["id"]), {"shots": []})
        storyboard.append({"segment_id": seg["id"], "function": seg["function"], "voice": seg["text"],
                           "event_ids": seg["event_ids"], **({"gags": seg.get("gags", [])} if style == "punchy" else {}),
                           "shots": [{"candidate_id": s["candidate_id"], "seconds": s["seconds"],
                                      "who": [x["who"] for x in cands[s["candidate_id"]]["people"]],
                                      "analysis": cands[s["candidate_id"]]["action"]} for s in p["shots"]]})
    if style == "punchy":
        text = ("КАРТА ФАКТОВ (с концовкой и wild_moments — для проверки достоверности):\n"
                + json.dumps(story_for_writer(sm), ensure_ascii=False)
                + "\n\nРАСКАДРОВКА (сегмент: текст диктора, помеченные выдумки gags, кадры по порядку):\n"
                + json.dumps(storyboard, ensure_ascii=False)
                + "\n\nНиже контактные листы реально смонтированного видео: один кадр на план, подпись — сегмент и номер "
                "плана. Сверь, кто в кадре, с тем, о ком говорит диктор.\n"
                "Верни JSON: {\"scores\": {" + ", ".join(f"\"{k}\": 0-10" for k in JUDGE_KEYS_PUNCHY) +
                "}, \"factual_errors\": [str], \"gag_check\": [{\"text\": str, \"reads_as_joke\": true|false, "
                "\"problem\": str}], \"clichés\": [str], \"weak_beats\": [{\"segment_id\": \"S…\", \"severity\": "
                "\"minor|major\", \"kind\": \"text|visual\", \"problem\": str, \"fix\": str}], \"visual_problems\": [str], "
                "\"rewrite_instructions\": \"конкретные правки для автора и режиссёра, одним текстом\"}")
        return [{"role": "system", "content": JUDGE_SYSTEM_PUNCHY},
                {"role": "user", "content": [{"type": "text", "text": text}] + list(sheets)}]
    text = ("КАРТА ФАКТОВ (с концовкой — для проверки достоверности):\n" + json.dumps(story_for_writer(sm), ensure_ascii=False)
            + "\n\nРАСКАДРОВКА (сегмент: текст диктора + кадры по порядку):\n" + json.dumps(storyboard, ensure_ascii=False)
            + "\n\nНиже контактные листы реально смонтированного видео: один кадр на план, подпись — сегмент и номер плана.\n"
            "Верни JSON: {\"scores\": {" + ", ".join(f"\"{k}\": 0-10" for k in JUDGE_KEYS) +
            "}, \"factual_errors\": [str], \"clichés\": [str], \"weak_beats\": [{\"segment_id\": \"S…\", "
            "\"severity\": \"minor|major\", \"kind\": \"text|visual\", \"problem\": str, \"fix\": str}], \"visual_problems\": [str], "
            "\"rewrite_instructions\": \"конкретные правки для автора и режиссёра, одним текстом\"}")
    return [{"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": [{"type": "text", "text": text}] + list(sheets)}]


def validate_judge(data: Any, style: str = "calm") -> Dict[str, Any]:
    _need(data, ("scores", "factual_errors", "rewrite_instructions"), "judge")
    keys = judge_keys(style)
    sc = _need(data["scores"], keys, "judge.scores")
    scores = {k: max(0.0, min(10.0, _num(sc[k], f"scores.{k}"))) for k in keys}
    weak = []
    for w in data.get("weak_beats") or []:
        if isinstance(w, dict):
            weak.append({"segment_id": str(w.get("segment_id") or "")[:8],
                         "severity": "major" if w.get("severity") == "major" else "minor",
                         "kind": "visual" if w.get("kind") == "visual" else "text",
                         "problem": str(w.get("problem") or "")[:300], "fix": str(w.get("fix") or "")[:300]})
        elif isinstance(w, str):
            weak.append({"segment_id": w.split(":")[0][:8], "severity": "minor", "kind": "text", "problem": w[:300], "fix": ""})
    cl = data.get("clichés", data.get("cliches"))
    fe = _str_list(data["factual_errors"], "factual_errors", 10, 300)
    gc = [{"text": str(g.get("text") or "")[:200], "reads_as_joke": g.get("reads_as_joke") is not False,
           "problem": str(g.get("problem") or "")[:300]} for g in (data.get("gag_check") or []) if isinstance(g, dict)]
    # an invented gag that may be taken for a fact of the film is a factual error, whatever the scores say
    fe += [f"выдумка читается как факт: {g['text']} — {g['problem']}" for g in gc if not g["reads_as_joke"]]
    return {"scores": scores, "factual_errors": fe[:12], "gag_check": gc[:20], "style": style,
            "clichés": _str_list(cl, "clichés", 10, 200), "weak_beats": weak[:10],
            "visual_problems": _str_list(data.get("visual_problems"), "visual_problems", 10, 300),
            "rewrite_instructions": str(data.get("rewrite_instructions") or "")[:2500]}


def thresholds(verdict: Dict[str, Any]) -> Dict[str, float]:
    return REWRITE_THRESHOLDS_PUNCHY if verdict.get("style") == "punchy" else REWRITE_THRESHOLDS


def needs_rewrite(verdict: Dict[str, Any]) -> List[str]:
    """Thresholds, factual errors AND weak beats: a major weak beat triggers the (single) rewrite even when
    every score is above its threshold (spec: weak_beats are never ignored). Punchy: a VISUAL weak beat of any
    severity counts too (owner: a frame that does not match the words must not pass)."""
    th = thresholds(verdict)
    why = [f"{k} {verdict['scores'][k]:g} < {v}" for k, v in th.items() if verdict["scores"][k] < v]
    if verdict["factual_errors"]:
        why.append(f"{len(verdict['factual_errors'])} factual error(s)")
    why += [f"weak beat {w['segment_id']} ({w['severity']}, {w.get('kind', 'text')}): {w['problem'][:80]}" for w in verdict["weak_beats"]
            if w["severity"] == "major" or (verdict.get("style") == "punchy" and w.get("kind") == "visual")]
    return why


def rewrite_scope(verdict: Dict[str, Any]) -> Optional[str]:
    """What the single allowed rewrite should touch: "text" (script + montage) if anything textual is wrong — factual
    errors, low hook / coherence / factuality / no_unfounded_drama, a major text weak beat; "visual" (montage only,
    the voice and script stay) if only picture problems are reported; None = no rewrite."""
    if not needs_rewrite(verdict):
        return None
    sc, th = verdict["scores"], thresholds(verdict)
    text_low = [k for k in ("factuality", "coherence", "hook", "no_unfounded_drama", "energy", "humor_surprise", "pace")
                if k in th and sc[k] < th[k]]
    major_text = [w for w in verdict["weak_beats"] if w["severity"] == "major" and w["kind"] == "text"]
    return "text" if (verdict["factual_errors"] or text_low or major_text) else "visual"
