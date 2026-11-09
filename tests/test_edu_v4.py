"""Educational v4: templates, selection on whole sentences, Gemini editor parsing/budget, runner wiring."""
import colorsys
import json
import re

import pytest

from modes.educational import editor as ed
from modes.educational import selection as sl
from modes.educational import stage
from modes.educational import templates as tp
from modes.educational.overlays import budget as bg
from modes.educational.overlays import themes


def _words(text: str, t0: float = 0.0, step: float = 0.4):
    out, t = [], t0
    for w in text.split():
        out.append({"word": w, "start": round(t, 3), "end": round(t + step * 0.8, 3)})
        t += step
    return out


# ── templates ───────────────────────────────────────────────────────────────

def test_four_templates_one_dark_owner_palette():
    assert sorted(tp.TEMPLATES) == [1, 2, 3, 4]
    assert any(t.dark for t in tp.TEMPLATES.values())
    for t in tp.TEMPLATES.values():
        for c in (t.title_bg, t.title_ink, t.stage, t.sub.ink, t.sub.active) + ((t.sub.plate,) if t.sub.plate else ()):
            h, s, v = colorsys.rgb_to_hsv(*(x / 255 for x in c))
            assert not (15 <= h * 360 <= 65 and s > 0.12), (t.name, c)       # no gold / beige / brown
        for f in (t.sub.font_file, t.title_font, t.label_font, themes.THEMES[t.theme].bold_font):
            assert "Montserrat" not in f and (tp.FONTS / f).is_file()


def test_get_template_never_fails():
    assert tp.get_template(3).id == 3 and tp.get_template("2").id == 2 and tp.get_template("graphite").id == 3
    for bad in (None, "", 9, "glass", {"x": 1}):
        assert tp.get_template(bad).id == 1
    assert [c["id"] for c in tp.catalog()] == [1, 2, 3, 4]


@pytest.mark.parametrize("k", [1, 2, 3, 4])
@pytest.mark.parametrize("on_stage", [False, True])
def test_subtitle_lines_never_leave_the_frame(k, on_stage):
    tpl, lay = tp.get_template(k), tp.Layout(1080, 1920, on_stage)
    sp = tp.sub_spec(tpl, lay)
    text = ("Электрогидродинамический преобразователь превышает рентгеноструктурный "
            "предел, а значит тут нужна осторожность. Короткая фраза.")
    cues = tp.build_cues(_words(text), 30.0, tpl, lay)
    assert cues
    for c in cues:
        assert len(c.lines) <= 2
        for ln in c.lines:
            assert tp._line_width([c.words[j][0] for j in ln], sp, c.size) <= lay.safe_w + 1
    ass = tp.to_ass(cues, tpl, lay)
    assert f"\\fad({tp.ENTER_MS},{tp.EXIT_MS})" in ass and tp.EXIT_MS < tp.ENTER_MS <= 250
    assert "\\fscx96" in ass                     # enters from 96 %, never from zero


def test_no_orphan_word_cue_before_sentence_end():
    text = "Дела с велосипедом обстоят в точности таким же образом. Колесо толкает дорогу назад."
    cues = tp.build_cues(_words(text, step=0.5), 20.0, tp.get_template(1), tp.Layout(1080, 1920))
    assert all(len(c.words) > 1 for c in cues), [[w[0] for w in c.words] for c in cues]


def test_line_never_ends_on_a_preposition():
    lines = tp._wrap("Скорость велосипеда относительно земли направлена в другую сторону".split(),
                     "Onest-Bold.ttf", 76, 700)
    words = "Скорость велосипеда относительно земли направлена в другую сторону".split()
    assert words[lines[0][-1]] not in tp._WEAK


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_title_fits_and_is_balanced(k):
    lay = tp.Layout(1080, 1920)
    im = tp.title_image("Почему велосипед быстрее ног", tp.get_template(k), lay, (2, 6))
    assert im is not None and im.width <= 1080
    x, y = tp.title_position(im, lay)
    assert x >= 0 and x + im.width <= 1080 and y >= 1920 * 0.115
    assert tp.title_image("", tp.get_template(k), lay) is None


def test_stage_rect_inside_frame():
    for w, h in ((1920, 1080), (1280, 720), (1440, 1080), (720, 1280)):
        x, y, rw, rh = stage.video_rect(w, h)
        assert x >= 0 and y >= 0 and x + rw <= 1080 and y + rh <= 1920 and rw % 2 == 0 and rh % 2 == 0


# ── selection: whole sentences ──────────────────────────────────────────────

LECTURE = (
    "Добрый день, сегодня поговорим о трении. "
    "Что такое сила трения? Сила трения — это сила, которая мешает телу скользить по поверхности. "
    "Она возникает потому, что поверхности шероховатые, и неровности цепляются друг за друга. "
    "Например, брусок на столе не сдвинется, пока мы не приложим достаточно большую силу. "
    "Именно поэтому зимой посыпают дороги песком: песок увеличивает трение. "
    "Таким образом, трение бывает полезным и вредным, и инженеры учитывают его всегда. "
) * 6


def test_selected_clips_are_whole_sentences_and_the_real_selection_runs():
    words = _words(LECTURE, step=0.45)
    segs, meta = sl.select_lesson_clips(words, sl.EduConfig(max_clips=3, min_duration=15, skip_intro_sec=0))
    assert meta["selection_status"] == "ok" and meta["selection_version"] == sl.SELECTION_VERSION and segs
    for s in segs:
        inside = [w for w in words if s["start"] - 1e-6 <= w["start"] and w["end"] <= s["start"] + s["duration"] + 1e-6]
        chk = sl.sentence_edges(inside)
        assert chk["ok"], chk
        assert not s["text"].rstrip().endswith("?")
    starts = sorted(s["start"] for s in segs)
    assert starts == [s["start"] for s in segs]                    # series in lecture order


def test_sentence_edges_check():
    assert sl.sentence_edges(_words("Сила трения мешает скольжению."))["ok"]
    assert not sl.sentence_edges(_words("трения мешает скольжению."))["start_ok"]
    assert not sl.sentence_edges(_words("Почему так происходит?"))["end_ok"]
    assert not sl.sentence_edges(_words("И тогда мы видим,"))["end_ok"]


def test_completeness_penalises_context_openers():
    good, _ = sl.completeness("Что такое трение? Это сила. Поэтому дороги посыпают песком.")
    bad, why = sl.completeness("Дела с велосипедом обстоят точно так же. Колесо толкает дорогу.")
    assert good > bad and any("comparing" in w for w in why)


# ── Gemini editor: analysis only, ids only, budget first ────────────────────

def test_editor_parse_accepts_wrapped_answers_and_drops_unknown_ids():
    valid = {"c1": {}, "c2": {}}
    wrapped = json.dumps([{"clips": [{"id": "c1", "score": 80, "complete": True, "self_contained": True,
                                      "title": "Что такое трение", "topic": "трение"},
                                     {"id": "zzz", "score": 99}]}], ensure_ascii=False)
    out = ed.parse("```json\n" + wrapped + "\n```", valid)
    assert set(out) == {"c1"} and out["c1"]["title"] == "Что такое трение"
    assert ed.parse("not json", valid) == {}
    long_title = json.dumps({"clips": [{"id": "c2", "score": 50, "title": "очень длинный заголовок из девяти слов без всякого смысла тут"}]})
    assert ed.parse(long_title, valid)["c2"]["title"] == ""


def test_editor_budget_stop_makes_no_request(tmp_path):
    from scripts.shared import llm_provider as lp
    calls = []

    def transport(*a, **k):
        calls.append(1)
        raise AssertionError("no request over budget")
    prov = lp.ProviderConfig(name="t", base_url="https://example.invalid/v1", api_key="x", model="gemini/gemini-3.1-flash-lite")
    led = bg.Ledger(str(tmp_path / "l.json"), video_limit_rub=0.01)
    fn = ed.make_editor(video_id="v", ledger=led, want=3, providers=[prov], transport=transport)
    verdicts, meta = fn([{"id": "c1", "start": 0, "end": 30, "text": "Сила трения."}])
    assert verdicts == {} and meta["status"] == "budget_stop" and not calls


def test_editor_prompt_is_analysis_only():
    msgs = ed.build_messages([{"id": "c1", "start": 0, "end": 30, "text": "x"}], 3)
    text = " ".join(m["content"] for m in msgs)
    assert "Отвечай только JSON" in text and "ВСЕ 1" in text
    assert not re.search(r"(?i)generate (an )?image|сгенерируй (картинку|видео)", text)


# ── runner wiring ───────────────────────────────────────────────────────────

def test_look_options_from_site_settings():
    from modes.educational import runner
    assert runner._look_options({"settings": {"edu_template": "3"}})["template"] == 3
    assert runner._look_options({"template": 4})["template"] == 4
    assert runner._look_options({})["template"] == 1
    assert runner._look_options({"settings": {"subtitleStyle": "none"}})["subtitles"] is False


def test_llm_switch(monkeypatch):
    from modes.educational import runner
    monkeypatch.setenv("SONYA_LLM_EDITOR", "off")
    assert runner._llm_on({"llm": "on"}) and not runner._llm_on({})
    assert not runner._llm_on({"llm": "off"})
    opts = runner._overlay_options({"llm": "on"}, {"template": 2, "subtitles": True, "title": True},
                                   {"title": "T", "series_index": 1, "series_total": 3}, 3)
    assert opts["cards"] and opts["llm_opt_in"] and opts["template"] == 2 and opts["part"] == (1, 3)
    assert not runner._overlay_options({"llm": "on", "edu": {"cards": False}}, {"template": 1, "subtitles": True,
                                       "title": True}, {}, 1)["cards"]
    assert not runner._llm_on({"llm": "on", "edu": {"editor": False}})


def test_cue_never_ends_on_a_preposition():
    text = ("Цикл почти никогда не может быть зациклен на бесконечное число шагов потому что "
            "у него всегда есть граница и счётчик который растёт с каждым шагом.")
    cues = tp.build_cues(_words(text, step=0.42), 30.0, tp.get_template(1), tp.Layout(1080, 1920))
    for c in cues[:-1]:
        assert c.words[-1][0].strip(",.").lower() not in tp._WEAK, [w[0] for w in c.words]


def test_a_clip_never_opens_on_a_continuation():
    assert not sl._opens_cleanly("...за доходами Церкви, контроля за политикой.")
    assert not sl._opens_cleanly("и поэтому так вышло.")
    assert sl._opens_cleanly("Что же это были за реформы?") and sl._opens_cleanly("«Квант» — это порция.")


def test_filler_openers_are_penalised():
    a, _ = sl.completeness("Ой, ребята, да тут, а, нет, тут есть вариация. Поэтому так.")
    b, _ = sl.completeness("Тут есть вариация начального состояния. Поэтому так.")
    assert a < b


def test_editor_approved_pieces_come_first():
    words = _words(LECTURE, step=0.45)
    ranked, _ = sl.score_candidates(words, sl.EduConfig(min_duration=15, skip_intro_sec=0))
    last = ranked[-1]                                   # the locally worst candidate

    def editor(pool):
        return {last["id"]: {"score": 90.0, "complete": True, "self_contained": True, "title": "Трение",
                             "topic": "трение", "reason": "ok"}}, {"status": "ok"}
    cfg = sl.EduConfig(max_clips=1, min_duration=15, skip_intro_sec=0, editor=True, editor_pool=10_000)
    segs, meta = sl.select_lesson_clips(words, cfg, None, editor)
    assert segs[0]["candidate_id"] == last["id"] and segs[0]["title"] == "Трение"
