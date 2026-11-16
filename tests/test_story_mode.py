"""scripts/shared/story_mode.py: SUPER Storytelling — Gemini finds stories on numbered sentences, every story is checked
on the ASR words (never model times), jump cuts never clip a word, teaser / title / «Часть 1/2», ≤ 1.5 ₽, fallbacks."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts.shared import jumpcut as jc
from scripts.shared import story_mode as sm


def talk(n_sent: int = 120, words_per: int = 6, gap: float = 0.6, start: float = 0.0, text=None):
    """n sentences of ``words_per`` 0.5 s words (0.1 s apart), ``gap`` s between sentences."""
    out, t = [], start
    for k in range(n_sent):
        for j in range(words_per):
            last = j == words_per - 1
            w = (text(k) if text and j == 0 else f"слово{k}_{j}") + ("." if last else "")
            out.append({"word": w, "start": round(t, 3), "end": round(t + 0.5, 3)})
            t += 0.6
        t += gap - 0.1
    return out


class FakeChat:
    def __init__(self, answer, tokens=(4000, 500)):
        self.answer, self.tokens, self.calls = answer, tokens, []

    def __call__(self, msgs, **kw):
        self.calls.append((msgs, kw))
        text = self.answer(msgs) if callable(self.answer) else self.answer
        return SimpleNamespace(text=text, input_tokens=self.tokens[0], output_tokens=self.tokens[1], latency_ms=5,
                               model="fake", provider="fake", telemetry=lambda: {})


def story(s, e, **kw):
    d = {"start": s, "end": e, "cut": [], "hook": -1, "title": "Как я чуть не опоздал", "who": "я и друг",
         "conflict": "поезд уходит", "twist": "поезд мой", "moral": "не спеши", "watch": 85, "series": "",
         "keywords": []}
    d.update(kw)
    return d


def ans(*stories):
    return json.dumps({"stories": list(stories)}, ensure_ascii=False)


def words_of(seg, words):
    return [w for w in words if any(a - 1e-3 <= w["start"] and w["end"] <= a + d + 1e-3 for a, d in seg["pieces"])]


def test_story_is_cut_on_words_and_within_30_90():
    words = talk()
    segs, meta = sm.select_stories(words, 10, chat=FakeChat(ans(story(10, 20))))
    assert meta["selection_method"] == "gemini_stories" and len(segs) == 1
    seg = segs[0]
    sents = sm.sentences(sm.normalize_words(words)[0])
    assert seg["start"] <= sents[10]["start"] and seg["end"] >= sents[20]["end"]
    assert 30 <= seg["body_sec"] <= 90
    # no word is clipped by a jump cut: every word of the story lies whole inside one piece
    inside = [w for w in words if sents[10]["start"] - 1e-3 <= w["start"] and w["end"] <= sents[20]["end"] + 1e-3]
    assert len(words_of(seg, words)) == len(inside)
    assert seg["title"] == "Как я чуть не опоздал" and seg["story"]["conflict"] == "поезд уходит"


def test_cut_sentences_are_removed_and_pauses_shortened():
    words = talk(gap=1.5)                                          # long pauses between sentences
    segs, _ = sm.select_stories(words, 10, chat=FakeChat(ans(story(10, 22, cut=[14, 15]))))
    seg = segs[0]
    got = {w["word"] for w in words_of(seg, words)}
    assert not any(w.startswith("слово14_") or w.startswith("слово15_") for w in got)
    assert any(w.startswith("слово16_") for w in got)
    assert seg["jump_cuts"] >= 10 and seg["cut_sec"] > 10
    assert seg["story"]["cut_sentences"] == [14, 15]


def test_model_cutting_too_much_is_ignored():
    words = talk()
    segs, _ = sm.select_stories(words, 10, chat=FakeChat(ans(story(10, 22, cut=list(range(11, 21))))))
    assert segs[0]["story"]["cut_sentences"] == []


def test_story_never_ends_on_a_question():
    q = lambda k: "Почему?" if k == 20 else f"с{k}"              # noqa: E731
    words = talk(text=None)
    for w in words:                                               # sentence 20 ends with «?»
        if w["word"].startswith("слово20_") and w["word"].endswith("."):
            w["word"] = w["word"][:-1] + "?"
    segs, _ = sm.select_stories(words, 10, chat=FakeChat(ans(story(10, 20))))
    assert segs and segs[0]["story"]["sentences"][1] == 21       # extended to the answer
    del q


def test_hum_never_opens_and_is_cut_inside():
    words = talk()
    for w in words:
        if w["word"].startswith("слово10_") or w["word"].startswith("слово15_"):
            w["word"] = "Угу."
    # a sentence of six «Угу.» is not a 1-2 word hum; make sentence 10/15 single words instead
    words = [w for w in words if not (w["word"] == "Угу." and w is not None)] + \
        [{"word": "Угу.", "start": 999.0, "end": 999.3}]
    segs, _ = sm.select_stories(talk(), 10, chat=FakeChat(ans(story(10, 20))))
    assert segs


def test_teaser_is_the_hook_line_from_the_middle():
    words = talk()
    segs, _ = sm.select_stories(words, 10, chat=FakeChat(ans(story(10, 22, hook=17))))
    tz = segs[0]["teaser"]
    sents = sm.sentences(sm.normalize_words(words)[0])
    assert tz["start"] <= sents[17]["start"] and tz["end"] >= sents[17]["end"]
    assert 1.2 <= tz["duration"] <= 6.5
    assert segs[0]["body_sec"] + tz["duration"] + 0.85 <= 90


def test_no_teaser_when_the_story_opens_with_it():
    segs, _ = sm.select_stories(talk(), 10, chat=FakeChat(ans(story(10, 22, hook=10))))
    assert "teaser" not in segs[0] and segs[0]["story"]["teaser_skipped"] == "already_opens_with_it"


NOINTRO = sm.StoryModeConfig(skip_intro_sec=0.0)        # talk(200+) is > 10 min: the intro skip would hide ids < 22


def test_long_source_intro_montage_is_not_a_story():
    segs, meta = sm.select_stories(talk(200), 10, chat=FakeChat(ans(story(10, 20))))
    assert meta["stories_rejected"].get("crosses_gap") and all(s["source"] == "story:local" for s in segs)


def test_only_good_stories_best_first_no_overlap():
    words = talk(200)
    chat = FakeChat(ans(story(10, 20, watch=70), story(15, 25, watch=90), story(40, 50, watch=88),
                        story(60, 70, watch=40), story(80, 90, watch=62)))
    segs, meta = sm.select_stories(words, 10, chat=chat, cfg=NOINTRO)
    assert [s["editor_score"] for s in segs] == [90, 88, 62]       # 70 overlaps the 90, 40 is junk
    assert all(not sm._overlaps(a, b) for i, a in enumerate(segs) for b in segs[i + 1:])


def test_fewer_than_three_good_takes_fair_never_junk():
    chat = FakeChat(ans(story(10, 20, watch=80), story(40, 50, watch=55), story(60, 70, watch=30)))
    segs, _ = sm.select_stories(talk(200), 10, chat=chat, cfg=NOINTRO)
    assert [s["editor_score"] for s in segs] == [80, 55]
    assert segs[1]["story"].get("fair_fill")


def test_model_series_gets_part_numbers_in_time_order():
    chat = FakeChat(ans(story(60, 70, watch=90, series="армия", title="Как я попал в армию"),
                        story(10, 20, watch=70, series="армия", title="Первый день"),
                        story(100, 110, watch=80)))
    segs, _ = sm.select_stories(talk(200), 10, chat=chat, cfg=NOINTRO)
    assert [s["title"] for s in segs][:2] == ["Часть 1/2. Первый день", "Часть 2/2. Как я попал в армию"]
    assert segs[0]["series_index"] == 1 and segs[1]["series_total"] == 2


def test_too_long_story_is_split_into_two_parts_kept_together():
    words = talk(300)
    segs, _ = sm.select_stories(words, 10, chat=FakeChat(ans(story(10, 45, hook=40))), cfg=NOINTRO)
    assert len(segs) == 2
    a, b = segs
    assert a["title"].startswith("Часть 1/2") and b["title"].startswith("Часть 2/2")
    assert a["end"] <= b["start"] and all(30 <= s["body_sec"] <= 90 for s in segs)
    assert a.get("teaser") and "teaser" not in b              # part 1 teases with a line of part 2
    # with room for only one clip the split story is not half-delivered
    one, _ = sm.select_stories(words, 1, chat=FakeChat(ans(story(10, 45), story(100, 112, watch=70))), cfg=NOINTRO)
    assert len(one) == 1 and one[0]["editor_score"] == 70


def test_short_strong_story_gets_more_setup():
    segs, _ = sm.select_stories(talk(), 10, chat=FakeChat(ans(story(20, 25, watch=85))))
    assert segs and segs[0]["story"]["setup_added"] >= 1 and segs[0]["body_sec"] >= 30
    weak, _ = sm.select_stories(talk(), 10, chat=FakeChat(ans(story(20, 25, watch=65))))
    assert weak[0]["source"] == "story:local"                # a weak short one is not stretched; local fallback


def test_model_failure_falls_back_to_local_storytelling():
    def boom(msgs, **kw):
        raise RuntimeError("network")
    segs, meta = sm.select_stories(talk(200), 5, chat=boom)
    assert meta["llm"]["status"] == "error" and meta["selection_method"] == "local_story"
    assert segs and all(s["source"] == "story:local" and s["pieces"] for s in segs)
    segs2, meta2 = sm.select_stories(talk(200), 5, use_llm=False)
    assert meta2["selection_method"] == "local_story" and len(segs2) == len(segs)


def test_garbage_and_cut_off_answers():
    assert sm.parse("not json") == []
    full = ans(story(10, 20), story(40, 50))
    cut_off = full[: full.rindex("{") + 30]                         # max_tokens cut the second story
    got = sm.parse(cut_off)
    assert len(got) == 1 and got[0]["start"] == 10
    segs, _ = sm.select_stories(talk(), 10, chat=FakeChat(ans(story(5, 999), story(30, 10), {"start": "x"})))
    assert all(s["source"] == "story:local" for s in segs)


def test_times_never_come_from_the_model():
    chat = FakeChat(json.dumps({"stories": [dict(story(10, 20), start_time=1.0, end_time=2.0)]}))
    segs, _ = sm.select_stories(talk(), 10, chat=chat)
    sents = sm.sentences(sm.normalize_words(talk())[0])
    assert abs(segs[0]["start"] - sents[10]["start"]) < 0.2


def test_budget_long_transcript_is_sent_as_its_densest_blocks():
    words = talk(2500, words_per=10)                                # ~6 h of speech
    chat = FakeChat(ans())
    sm.select_stories(words, 10, chat=chat)
    sents = sm.sentences(sm.normalize_words(words)[0])
    cfg = sm.StoryModeConfig()
    sent_total = sum(len(m[0][1]["content"]) for m in chat.calls)
    assert sent_total < sum(len(sm.line(s)) for s in sents) / 2
    fixed = sum(len(m["content"]) for m in sm.build_messages([], cfg.per_window, cfg))
    est = sum(sm._rub(len(m[0][0]["content"]) + len(m[0][1]["content"]), cfg.per_window * cfg.out_tokens_per_story)
              for m in chat.calls)
    assert est <= 1.5 and fixed > 0
    for _, kw in chat.calls:
        assert kw["extra"]["max_tokens"] <= 2000                    # the answer is capped too


def test_windows_overlap_and_never_cross_a_gap():
    sents = sm.sentences(sm.normalize_words(talk(400))[0])
    ws = sm.windows(sents[:150] + sents[200:], sm.StoryModeConfig())
    for w in ws:
        ids = [s["id"] for s in w]
        assert ids == list(range(ids[0], ids[-1] + 1))             # contiguous
    for a, b in zip(ws, ws[1:]):
        if b[0]["id"] <= a[-1]["id"]:
            assert a[-1]["end"] - b[0]["start"] >= 30              # overlapping neighbours


def test_story_crossing_an_unsent_block_is_rejected():
    words = talk(200)
    W = sm.normalize_words(words)[0]
    sents = sm.sentences(W)
    got, why = sm.plan(story(10, 20), W, sents, sm.StoryModeConfig(), sent_ids=set(range(0, 15)))
    assert got == [] and why == "crosses_gap"


def test_kill_switch(monkeypatch):
    monkeypatch.setenv("SONYA_STORY_SUPER", "off")
    assert not sm.enabled()
    monkeypatch.delenv("SONYA_STORY_SUPER")
    assert sm.enabled()


def test_prompt_is_data_safe_and_sentence_numbered():
    sents = sm.sentences(sm.normalize_words(talk(30))[0])
    msgs = sm.build_messages(sents, 5, sm.StoryModeConfig())
    assert "игнорируй" in msgs[0]["content"]
    assert "\n0|0:00|" in msgs[1]["content"] and "\n1||" in msgs[1]["content"]


def test_pieces_survive_json_and_cover_body():
    segs, _ = sm.select_stories(talk(), 10, chat=FakeChat(ans(story(10, 20))))
    seg = json.loads(json.dumps(segs[0]))
    assert jc.total(seg["pieces"]) == pytest.approx(seg["body_sec"], abs=1e-3)


def _w(texts, start=0.0, dur=0.4, gap=0.1):
    out, t = [], start
    for x in texts:
        out.append({"word": x, "start": round(t, 3), "end": round(t + dur, 3)})
        t += dur + gap
    return out


def test_hook_words_short_line_joins_next_long_line_keeps_its_best_clause():
    cfg = sm.StoryModeConfig()
    words = sm.normalize_words(_w(["Кто", "это", "был?", "Это", "я.", "Я", "сам", "спел.", "Дальше", "было", "так."]))[0]
    sents = sm.sentences(words)
    assert sm.hook_words(1, [0, 1, 2, 3], set(), words, sents, cfg) == (3, 7)     # «Это я. Я сам спел.»
    long = ["И", "мы", "его", "закрыли,", "потому", "что", "разные", "были", "ситуации,", "вдруг", "в", "зале",
            "начинали", "драться", "люди."]
    words = sm.normalize_words(_w(long, dur=0.5))[0]
    sents = sm.sentences(words)
    b, e = sm.hook_words(0, [0], set(), words, sents, cfg)
    assert e == len(long) - 1 and b in (4, 9) and words[e].end - words[b].start <= 6.0


def test_topic_series_joins_stories_about_the_same_thing():
    sts = [{"title": "Неожиданный рекорд онлайна на стриме", "series": ""},
           {"title": "Исторический рекорд онлайна", "series": ""},
           {"title": "Курьер узнал меня на стриме", "series": ""},
           {"title": "Первый стрим после бана", "series": ""}]
    sm.topic_series(sts)
    assert sts[0]["series"] and sts[0]["series"] == sts[1]["series"]
    assert not sts[2]["series"] and not sts[3]["series"]
