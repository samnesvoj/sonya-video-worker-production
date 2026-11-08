"""cut-local-rank-v3: whole stories / question + answer from ASR words."""
from scripts.shared.cut_selection import StoryConfig, build_story_candidates, select_story_clips, true_sentences
from scripts.shared.boundaries import normalize_words


def words_from(sentences, t0=100.0, rate=0.4, pause_between=0.3, pauses_inside=None):
    """sentences: list of strings; returns ASR-like word dicts. pauses_inside: {(sent_idx, word_idx): sec}."""
    out, t = [], t0
    for si, s in enumerate(sentences):
        for wi, w in enumerate(s.split()):
            t += (pauses_inside or {}).get((si, wi), 0.0)
            out.append({"word": w, "start": round(t, 3), "end": round(t + rate * 0.9, 3)})
            t += rate
        t += pause_between
    return out


ANSWER = ("Я вырос в Нью-Джерси рядом с Нью-Йорком и всё детство провёл у бабушки на Брайтоне. "
          "Там все говорили по-русски и я думал что так живёт вся Америка. "
          "Потом в школе выяснилось что это совсем не так и мне пришлось переучиваться. ")


def test_sentences_ignore_pauses_inside_a_sentence():
    w, _ = normalize_words(words_from(["Он сказал что придёт завтра утром.", "Да."], pauses_inside={(0, 3): 1.5}))
    assert true_sentences(w) == [(0, 5), (6, 6)]


def test_never_starts_mid_sentence_or_ends_on_a_question():
    sents = ["Это было в девяностом году когда мы переехали в другой город и всё изменилось."] * 3 + \
            ["А ты сам как думаешь почему так получилось?"] + ANSWER.split(". ")[:3]
    raw = words_from([s if s.endswith((".", "?")) else s + "." for s in sents], pauses_inside={(1, 5): 1.2})
    w, _ = normalize_words(raw)
    starts = {w[s0].start for s0, _ in true_sentences(w)}
    for c in build_story_candidates(w, StoryConfig(min_duration=5)):
        assert c["start"] in starts
        assert not c["text"].rstrip().endswith("?")


def test_question_with_its_answer_beats_a_monologue_fragment():
    filler = ["Ну и вот мы там стояли и разговаривали о чём-то своём довольно долго."] * 8
    qa = ["Скажи честно как ты оказался в Америке?"] + [s + "." for s in ANSWER.split(". ") if s.strip()]
    nextq = ["Следующий вопрос про футбол и про твоего любимого игрока?"]
    raw = words_from(filler + qa + qa[1:] + nextq + ["Ну и вот мы там стояли."] * 2 + nextq)
    segs, meta = select_story_clips(raw, 1, StoryConfig(min_duration=15))
    sel = meta["selected"][0]
    q_next = next(w for w in raw if w["word"] == "Следующий")
    before = max(w["end"] for w in raw if w["end"] <= q_next["start"])
    assert sel["text"].startswith("Скажи честно") and abs(sel["end"] - before) < 1e-3   # the whole answer
    assert "question + answer" in sel["reason"] and "ends at a closure" in sel["reason"]


def test_long_answer_may_run_past_60_seconds():
    long_answer = [s + "." for s in (ANSWER * 2).split(". ") if s.strip()]
    raw = words_from(["Расскажи как тебя арестовали и что было потом?"] + long_answer + ["Хорошо а дальше что?"],
                     rate=0.75)
    segs, meta = select_story_clips(raw, 1)
    assert 60 < segs[0]["duration"] <= 90 and meta["selected"][0]["text"].startswith("Расскажи")


def test_teaser_intro_is_skipped_and_result_is_deterministic():
    qa = ["Скажи честно как ты оказался в Америке?"] + [s + "." for s in ANSWER.split(". ") if s.strip()]
    raw = words_from(qa, t0=5.0) + words_from(qa, t0=200.0)
    a = select_story_clips(raw, 2)
    b = select_story_clips(list(reversed(raw)), 2)
    assert all(s["start"] >= 90 for s in a[0]) and a[0] == b[0]


def test_short_source_keeps_its_start_and_exclude_gives_new_moments():
    qa = ["Скажи честно как ты оказался в Америке?"] + [s + "." for s in ANSWER.split(". ") if s.strip()]
    raw = words_from(qa, t0=1.0, rate=0.5)
    assert select_story_clips(raw, 1)[0][0]["start"] < 2.0            # 1-minute source: no intro skip
    long = words_from(qa, t0=200.0, rate=0.5) + words_from(qa, t0=400.0, rate=0.5) + words_from(["Конец."], t0=700.0)
    first = select_story_clips(long, 1)[0][0]
    again = select_story_clips(long, 1, StoryConfig(exclude=[[first["start"], first["start"] + first["duration"]]]))[0][0]
    assert abs(again["start"] - first["start"]) > 100


def test_qa_format_opens_with_the_question_and_has_only_the_answer_after_it():
    qa = ["Каким ты себя видишь через десять лет?"] + [s + "." for s in ANSWER.split(". ") if s.strip()]
    other = ["Мы долго гуляли по набережной и смотрели на океан и корабли."] * 6
    followup = ["А почему?"]
    raw = words_from(other + qa + ["Следующий большой вопрос про твою работу в газете?"] + other + qa[:2]
                     + followup + qa[2:] + ["Конец разговора на сегодня?"], rate=0.5)
    segs, meta = select_story_clips(raw, 3, StoryConfig(qa_only=True, min_duration=20, max_duration=62))
    texts = [x["text"] for x in meta["selected"] if "story fill" not in x["reason"]]   # the Q→A picks
    assert any(t.startswith("Каким ты себя видишь") for t in texts)
    assert not any("А почему?" in t for t in texts)               # a follow-up question = not one Q + A


def test_qa_format_needs_a_question_to_the_guest_not_a_rhetorical_one():
    rhet = ["Как я оцениваю это всё сейчас спустя столько лет?"] + [s + "." for s in ANSWER.split(". ") if s.strip()]
    asked = ["А как ты сам оцениваешь свою жизнь в Америке сейчас?"] + [s + "." for s in ANSWER.split(". ") if s.strip()]
    raw = words_from(rhet + ["Хорошо а что было дальше с тобой?"] + asked + ["Последний вопрос про твоих родителей?"],
                     rate=0.5)
    segs, meta = select_story_clips(raw, 3, StoryConfig(qa_only=True, min_duration=20))
    qa = [x for x in meta["selected"] if "story fill" not in x["reason"]]
    assert qa and all(not x["text"].startswith("Как я") for x in qa)


def test_never_ends_on_a_trailing_ellipsis():
    sents = ["Расскажи как ты оказался в Москве?"] + [s + "." for s in ANSWER.split(". ") if s.strip()][:2] + \
            ["А сама страна как бы..."] + [s + "." for s in ANSWER.split(". ") if s.strip()][2:]
    w, _ = normalize_words(words_from(sents, rate=0.5))
    assert all(not c["text"].rstrip().endswith(("...", "…")) for c in build_story_candidates(w, StoryConfig(min_duration=5)))


def test_promo_and_reported_speech_are_not_clips():
    from scripts.shared.cut_selection import is_promo
    assert is_promo("Условия будут у нас в Telegram-каналах. Не пропустите.") and not is_promo("Мы долго гуляли.")
    # sponsor read inside the kishlak_15m control (picked as clip 1 before): never a clip
    assert is_promo("Фоновое прослушивание доступно лишь с подпиской Премиум, которую невозможно оплатить напрямую.")
    assert is_promo("Но маркетплейс GJS легко помогает нам решить эту проблему.")
    assert is_promo("Все ссылки оставили в описании.") and is_promo("Заходим, выбираем YouTube Premium.")
    assert not is_promo("Я подписал контракт и уехал в Москву, там всё описание работы было другое.")
    ans = [s + "." for s in ANSWER.split(". ") if s.strip()]
    quoted = ["И значит, мне пишут: «Ну, как ты можешь сравнивать такую музыку?»"] + ans
    promo = ["А что это у тебя за футболка такая?"] + ans[:2] + ["Условия розыгрыша будут у нас в Telegram-каналах."]
    real = ["А как ты сам относишься к своей музыке?"] + ans
    raw = words_from(quoted + promo + real + ["Последний вопрос про твоих родителей?"], rate=0.5)
    segs, meta = select_story_clips(raw, 3, StoryConfig(qa_only=True, min_duration=20))
    qa = [x["text"] for x in meta["selected"] if "story fill" not in x["reason"]]
    assert qa and all(t.startswith("А как ты сам") for t in qa)
    assert not any("Telegram" in x["text"] for x in meta["selected"])            # promo never, not even as fill


def test_qa_first_then_story_fill_up_to_the_requested_count():
    ans = [s + "." for s in ANSWER.split(". ") if s.strip()]
    story = ["Однажды мы поехали на дачу к бабушке и там всё пошло не по плану."] * 3 + ans
    raw = words_from(["А как ты сам относишься к своей музыке?"] + ans + ["Следующий вопрос про работу твою?"] + story
                     + ["Конец разговора на сегодня?"], rate=0.5)
    segs, meta = select_story_clips(raw, 2, StoryConfig(qa_only=True, min_duration=15))
    assert len(segs) == 2 and meta["story_fill"] == 1
    assert meta["selected"][0]["text"].startswith("А как ты сам") and "story fill" in meta["selected"][1]["reason"]


def test_opening_question_must_be_self_contained():
    from scripts.shared.cut_selection import self_contained_question as ok
    assert not ok("У тебя есть ещё?")
    assert not ok("Просто прийти в детский дом и там, типа, чё у тебя родителей нету, да?")
    assert not ok("А это тоже было там?")
    assert ok("Сколько у вас стоит билет на концерт в Москве?")
    assert ok("Ты же раньше в стонике ходил, а сейчас волосы красишь?")
    assert ok("Каким ты себя видишь через десять лет?")


class _Res:
    def __init__(self, text):
        self.text = text

    def telemetry(self):
        return {"model": "fake"}


def _qa_raw():
    qa = ["Каким ты себя видишь через десять лет?"] + [s + "." for s in ANSWER.split(". ") if s.strip()]
    qa2 = ["А как ты сам оцениваешь свою жизнь в Америке сейчас?"] + [s + "." for s in ANSWER.split(". ") if s.strip()]
    other = ["Мы долго гуляли по набережной и смотрели на океан и корабли."] * 6
    return words_from(other + qa + other + qa2 + other, rate=0.5)


def test_editor_reorders_and_adds_hook_and_only_spoken_keywords():
    import json as _json
    seen = {}

    def chat(messages, **kw):
        items = _json.loads(messages[1]["content"].split("Фрагменты:\n", 1)[1].split("\n\nВерни", 1)[0])
        seen["ids"] = [x["id"] for x in items]
        clips = [{"id": x["id"], "score": 90 - i * 10, "self_contained": i != 0, "qa": True,
                  "hook": f"Хук {i}", "keywords": ["Брайтоне", "марс"], "reason": "r"} for i, x in enumerate(items)]
        return _Res(_json.dumps({"clips": clips}, ensure_ascii=False))

    segs, meta = select_story_clips(_qa_raw(), 1, StoryConfig(qa_only=True, min_duration=15, editor=True), chat=chat)
    assert meta["selection_method"] == "editor" and meta["llm"]["model"] == "fake"
    s = segs[0]
    assert s["candidate_id"] == seen["ids"][1]             # #0 was not self-contained → skipped
    assert s["hook"] == "Хук 1" and s["keywords"] == ["Брайтоне"] and s["editor_score"] == 80
    two, _ = select_story_clips(_qa_raw(), 2, StoryConfig(qa_only=True, min_duration=15, editor=True), chat=chat)
    assert [x["candidate_id"] for x in two] == [seen["ids"][1], seen["ids"][0]]   # strong but strict → 2nd tier


def test_editor_failure_falls_back_to_local():
    def chat(messages, **kw):
        raise RuntimeError("no key")

    a, _ = select_story_clips(_qa_raw(), 2, StoryConfig(qa_only=True, min_duration=15))
    b, meta = select_story_clips(_qa_raw(), 2, StoryConfig(qa_only=True, min_duration=15, editor=True), chat=chat)
    assert [x["start"] for x in a] == [x["start"] for x in b] and meta["selection_method"] == "local"
    assert "no key" in meta["llm"]["error"] and "hook" not in b[0]


def test_strategy_goes_to_the_editor_and_changes_local_ranking():
    import json as _json
    seen = {}

    def chat(messages, **kw):
        seen["u"] = messages[1]["content"]
        return _Res(_json.dumps({"clips": []}))

    select_story_clips(_qa_raw(), 1, StoryConfig(min_duration=15, editor=True, strategy="educational"), chat=chat)
    assert "ЗАДАЧА: ищем законченное объяснение" in seen["u"]
    edu = ["Главное правило монтажа: нужно резать на движении, потому что так склейку не видно."] * 3
    talk = ["Мы долго гуляли по набережной и смотрели на океан и корабли вечером."] * 3
    raw = words_from(talk + ["Ну."] + edu + ["Ну."] + talk, rate=0.5, pause_between=0.9)
    segs, meta = select_story_clips(raw, 1, StoryConfig(min_duration=10, strategy="educational"))
    assert "explains" in meta["selected"][0]["reason"] and "Главное правило" in meta["selected"][0]["text"]
    assert "energy" not in meta["params"]


def test_viral_strategy_prefers_the_loud_moment():
    talk = ["Мы долго гуляли по набережной и смотрели на океан и корабли вечером."] * 3
    raw = words_from(talk + talk + talk, rate=0.5, pause_between=0.9)
    end = raw[-1]["end"]
    energy = [0.0] * int(end + 2)
    for t in range(int(raw[len(raw) // 2]["start"]), int(raw[len(raw) // 2]["start"]) + 4):
        energy[t] = 3.0
    segs, meta = select_story_clips(raw, 1, StoryConfig(min_duration=10, strategy="viral", energy=energy))
    s = meta["selected"][0]
    assert s["start"] <= raw[len(raw) // 2]["start"] <= s["end"] and "energy peak" in s["reason"]
