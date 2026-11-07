"""trailer_cinematic economy path (lite.py): squeezed transcript, local frame choice, forecast + hard stop at 5 ₽,
one Story+Writer call, one text-only fix, local shot plan. No network, no models, no ffmpeg."""
import pytest

from modes.trailer_cinematic import budget, compose, lite, recap, recap_runner
from modes.trailer_cinematic.budget import SchemaError


def _words(spec):
    """[(text, start)] → ASR word dicts, 0.3 s each."""
    return [{"word": " " + t, "start": s, "end": s + 0.3} for t, s in spec]


# ── transcript squeeze ──────────────────────────────────────────────────────

def test_clean_text_drops_fillers_and_stutters():
    assert lite.clean_text("ну вот да да да он короче это самое пошёл пошёл домой") == "да он пошёл домой"


def test_digest_is_short_ordered_and_has_timecodes():
    w, t = [], 100.0
    for i in range(60):
        for tok in ("ну", "царь", "кричит", "громко", f"слово{i}", "!"[:0] or "дальше."):
            w.append((tok, t))
            t += 0.4
        t += 1.0
    ev = lite.digest(_words(w), 100.0, t, target_chars=900, event_sec=40)
    assert ev[0]["id"] == "E1" and all(a["start"] < b["start"] for a, b in zip(ev, ev[1:]))
    assert 0 < sum(len(e["text"]) for e in ev) <= 900 + 30 * len(ev)
    assert "[0:" in ev[0]["text"] and " ну " not in ev[0]["text"]
    assert lite.digest_text(ev, 100.0).startswith("E1 (0:00–")


def test_digest_keeps_the_strong_line_when_space_is_short():
    raw = _words([("тихий", 0), ("скучный", 0.4), ("текст", 0.8), ("тут.", 1.2),
                  ("царь", 5), ("орёт", 5.4), ("пиздец", 5.8), ("что", 6.2), ("творится!", 6.6)])
    ev = lite.digest(raw, 0.0, 10.0, target_chars=110, event_sec=10)
    assert "пиздец" in ev[0]["text"]


# ── frames ──────────────────────────────────────────────────────────────────

def test_select_frames_one_per_bin_in_order_and_skips_black():
    times = [i * 0.5 for i in range(40)]                       # 0..19.5 s
    diff = [0.1] * 40
    diff[7] = 5.0                                              # a cut in bin 1 (of 4: 0-5, 5-10, ...)
    bright = [100.0] * 40
    bright[7] = 5.0                                            # but that frame is black → skipped
    picks = lite.select_frames(times, diff, bright, {12: 2}, [], 4, 0.0, 20.0)
    assert len(picks) == 4 and picks == sorted(picks)
    assert 7 not in picks
    assert 12 in picks                                         # faces win inside bin 2 (10-15)


def test_select_frames_rewards_strong_words():
    times = [i * 1.0 for i in range(10)]
    picks = lite.select_frames(times, [1.0] * 10, [90.0] * 10, {}, [7.2], 1, 0.0, 10.0)
    assert picks[0] in (6, 7, 8)


# ── forecast and the 5 ₽ stop ───────────────────────────────────────────────

def _msgs(n_img):
    parts = [{"type": "text", "text": "я" * 14000}]
    for _ in range(n_img):
        parts.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}})
    return [{"role": "system", "content": "s" * 6000}, {"role": "user", "content": parts}]


def test_forecast_of_the_real_request_is_inside_the_owner_range():
    fc = lite.forecast_rub(_msgs(16), "gemini/gemini-3.1-flash-lite")
    assert 1.0 < fc["rub"] < 5.0, fc                           # owner: 2.5–4.5 ₽ for the whole trailer


def test_gate_stops_before_the_call_when_over_5_rub():
    fc = lite.gate(_msgs(16), "gemini/gemini-3.1-flash-lite", 0.0)
    assert fc["rub"] < lite.LITE_STOP_RUB
    with pytest.raises(budget.BudgetExceeded):
        lite.gate(_msgs(16), "gemini/gemini-3.1-flash-lite", 4.9)
    with pytest.raises(budget.BudgetExceeded):                 # 60 stills would not fit either
        lite.gate(_msgs(60), "gemini/gemini-3.1-flash-lite", 3.0)


def test_llm_wrapper_stops_before_sending_with_the_5_rub_budget():
    sent = []
    llm = budget.LLM(lambda *a, **k: sent.append(1), lite.LITE_STOP_RUB / budget.RUB_PER_USD, "gemini/gemini-3.1-flash-lite")
    with pytest.raises(budget.BudgetExceeded):
        llm.ask("x", _msgs(200), "v", lambda d: d, max_out=lite.MAX_OUT_TOKENS)
    assert sent == []


# ── script checks ───────────────────────────────────────────────────────────

EVENTS = [{"id": f"E{i}", "start": 100.0 + 60 * (i - 1), "end": 100.0 + 60 * i, "text": ""} for i in range(1, 9)]
FRAMES = [{"id": f"F{i}", "t": 100.0 + 30 * i, "rel": "0:30", "path": ""} for i in range(1, 11)]
SM = {"events": EVENTS, "forbid_names": [], "source_crime_stems": []}


def _seg(i, fn, text, ev, fr):
    return {"id": f"S{i}", "function": fn, "text": text, "event_ids": ev, "frame_ids": fr, "visual_intent": "мужик в кадре кричит",
            "pause_after": "short", "gags": []}


def _script():
    return {"title": "t", "characters": [{"role": "царь", "name": "Горох"}], "segments": [
        _seg(1, "hook", "Этот парень вышел на сцену.", ["E1"], ["F1", "F2"]),
        _seg(2, "setup", "Ему шепчут в ухо. Он слушается.", ["E1"], ["F2"]),
        _seg(3, "development", "Дальше хуже. Мужик чешет жопу.", ["E3"], ["F4", "F5"]),
        _seg(4, "salvo", "Нюхает бороду, дерётся мечом, орёт на всех, падает, встаёт, снова орёт.", ["E5"], ["F6", "F7", "F8"]),
        _seg(5, "turn", "Партнёры в шоке. Сцена горит.", ["E6"], ["F7", "F8"]),
        _seg(6, "climax", "Всё разваливается. Мат летит.", ["E7"], ["F9"]),
        _seg(7, "development", "Его уводят. Тихо, тихо.", ["E8"], ["F10"]),
        _seg(8, "punchline", "Оскар отменяется.", ["E8"], ["F10"]),
    ]}


def test_validate_lite_keeps_frame_ids_and_rejects_unknown_ones():
    d = _script()
    d["segments"][0]["frame_ids"] = ["F1", "F99"]
    out = lite.validate_lite(d, SM, FRAMES, 10, 800, 1.0)
    assert out["segments"][0]["frame_ids"] == ["F1"]
    assert out["characters"][0]["name"] == "Горох"


def test_validate_lite_rejects_a_name():
    d = _script()
    d["segments"][2]["text"] = "Горох чешет жопу."
    sm = dict(SM, forbid_names=recap.forbidden_names({"characters": d["characters"]}))
    with pytest.raises(SchemaError):
        lite.validate_lite(d, sm, FRAMES, 10, 800, 1.0)


def test_free_checks_catch_a_story_that_runs_backwards_and_missing_frames():
    s = lite.validate_lite(_script(), SM, FRAMES, 10, 800, 1.0)
    assert lite.free_checks(s, FRAMES, EVENTS) == []
    bad = _script()
    bad["segments"][2]["event_ids"] = ["E8"]                   # the body jumps to the end ...
    bad["segments"][5]["event_ids"] = ["E1"]                   # ... and back to the start
    s2 = lite.validate_lite(bad, SM, FRAMES, 10, 800, 1.0)
    assert any("back in time" in x for x in lite.free_checks(s2, FRAMES, EVENTS))
    bad2 = _script()
    for seg in bad2["segments"]:
        seg["frame_ids"] = []
    s3 = lite.validate_lite(bad2, SM, FRAMES, 10, 800, 1.0)
    assert any("no valid frame_ids" in x for x in lite.free_checks(s3, FRAMES, EVENTS))


# ── shot plan ───────────────────────────────────────────────────────────────

def test_plan_shots_is_contiguous_and_avoids_hidden_cuts():
    script = lite.validate_lite(_script(), SM, FRAMES, 10, 800, 1.0)
    secs = {s["id"]: [2.2 for _ in recap.sentences_of(s["text"])] for s in script["segments"]}
    vt = compose.voice_timeline(script, secs, pauses=recap.STYLES["punchy"]["pauses"])
    cuts = lambda lo, hi: [c for c in (131.0, 250.0, 333.3) if lo < c < hi]          # noqa: E731
    tl_ = lite.plan_shots(script, vt, FRAMES, EVENTS, 100.0, 580.0, cuts)
    assert compose.check_plan(tl_) == []
    for sh in tl_["shots"]:
        assert 100.0 <= sh["src_start"] < sh["src_end"] <= 580.0 + 1e-6
        assert not any(sh["src_start"] < c < sh["src_end"] for c in (131.0, 250.0, 333.3)), sh
        assert abs((sh["src_end"] - sh["src_start"]) - sh["duration"]) < 1e-2
    assert len({round(s["src_start"]) for s in tl_["shots"]}) >= len(tl_["shots"]) * 0.7      # no machine-gun repeats


# ── CLI wiring ──────────────────────────────────────────────────────────────

def test_parse_window_and_economy_default():
    assert recap_runner.parse_window("1:22:25-1:43:08") == (4945.0, 6188.0)
    assert recap_runner.parse_window("4945-6188") == (4945.0, 6188.0)
    with pytest.raises(SystemExit):
        recap_runner.parse_window("5-1")
    assert recap_runner.economy_enabled("auto", "punchy", "1:00-2:00") is True      # default for the energetic style + fragment
    assert recap_runner.economy_enabled("auto", "punchy", None) is False            # a whole film keeps the full pipeline
    assert recap_runner.economy_enabled("auto", "calm", "1:00-2:00") is False
    assert recap_runner.economy_enabled("off", "punchy", "1:00-2:00") is False
    with pytest.raises(SystemExit):
        recap_runner.economy_enabled("on", "punchy", None)


# ── the whole call with a fake chat: one call, one text fix, cost booked ───────

class _Res:
    def __init__(self, text, inp, out):
        self.text, self.raw = text, {}
        self._t = {"input_tokens": inp, "output_tokens": out, "provider": "fake", "model": "m"}

    def telemetry(self):
        return dict(self._t)


def test_llm_ask_books_cost_of_one_call():
    import json
    calls = []

    def chat(msgs, **kw):
        calls.append(1)
        return _Res(json.dumps({"segments": []}), 20000, 2000)
    llm = budget.LLM(chat, 0.05, "gemini/gemini-3.1-flash-lite")
    llm.ask("lite_story_writer", _msgs(16), "v", lambda d: d, max_out=lite.MAX_OUT_TOKENS)
    assert len(calls) == 1
    assert 0.5 < budget.rub(llm.spent) < 3.0                   # 20k in × 0.5 + 2k out × 3 per M → ≈ 1.6 ₽


# ── cheap defect list for the single text fix ─────────────────────────────────

def test_quick_defects_reports_count_length_and_names_at_once():
    d = {"segments": [{"text": "Этот мужик вышел. Саня чешет зад."}, {"text": "Леший требует вертолёт."}]}
    bad = lite.quick_defects(d, 1400, 1600, (8, 14))
    assert any("сегментов 2" in x for x in bad)
    assert any("знаков" in x and "допиши" in x for x in bad)
    assert any("Саня" in x for x in bad) and not any("Леший" in x for x in bad)       # a fairy-tale role is fine
    ok = {"segments": [{"text": "Этот парень идёт. Царь орёт."}] * 10}
    assert lite.quick_defects(ok, 100, 400, (8, 14)) == []
