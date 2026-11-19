"""scripts/shared/stream_viral.py: Streamer moments on the Viral peak engine (stream-moments-v3)."""
from __future__ import annotations

import json

from scripts.shared import stream_viral as sv
from tests.test_stream_moments import _signals, _words


class _Res:
    def __init__(self, text):
        self.text = text

    def telemetry(self):
        return {"input_tokens": 1000, "output_tokens": 200}


def _chat(verdict, check=None, calls=None):
    """Editor: verdict(item) for every pool item; final check: check(item) (default clear 80)."""
    def chat(msgs, prompt_version="", **k):
        if calls is not None:
            calls.append(prompt_version)
        data = json.loads(msgs[1]["content"].split("\n", 1)[1].rsplit("\n\n", 1)[0]) \
            if prompt_version == sv.CHECK_PROMPT_VERSION else None
        if prompt_version == sv.CHECK_PROMPT_VERSION:
            return _Res(json.dumps({"clips": [dict({"id": it["id"], "clear": True, "chat_talk": False, "ad": False,
                                                    "score": 80, "title": "т"}, **((check or (lambda it: {}))(it)))
                                              for it in data]}))
        body = msgs[1]["content"]
        items = json.loads(body[body.index("Фрагменты:\n") + 11:body.index("\n\nВерни JSON")])
        return _Res(json.dumps({"clips": [dict({"id": it["id"], "score": 80, "type": "эмоция", "self_contained": True,
                                                "chat_talk": False, "ad": False, "start": it["clip_starts_at"],
                                                "end": len(it["phrases"]) - 1, "title": "Заголовок"}, **verdict(it))
                                          for it in items]}))
    return chat


def test_chat_talk_penalty():
    pen, n = sv.chat_talk("Антоха, здорово! Спасибо тебе. Нильфгард, спасибо за поддержку, хорошего просмотра. "
                          "Спасибо большое за тёплые слова! Спасибо за стрим, братуха.", 25.0)
    assert pen > 0 and n >= 4
    assert sv.chat_talk("Менты приехали! На тебе! Дай застрелить его.", 25.0) == (0.0, 0)


def test_floor_is_three_for_a_long_stream():
    assert sv.floor_for(3600, 12) == 3 and sv.floor_for(900, 12) == 2 and sv.floor_for(200, 12) == 1


def test_editor_picks_strong_only_and_keeps_the_floor():
    n = 1900
    calls = []
    weak = lambda it: {} if it["id"] in first else {"score": 40, "self_contained": False}   # noqa: E731
    first = set()
    # pick two pool ids as strong: the rest weak → 2 strong + 1 floor fill (n ≥ 1800 → floor 3)
    from scripts.shared.boundaries import normalize_words  # noqa: F401

    def verdict(it):
        if len(first) < 2:
            first.add(it["id"])
        return weak(it)
    segs, meta = sv.select_moments(_words(n), n, _signals(n), 10, sv.sm.MomentConfig(editor=True),
                                   chat=_chat(verdict, calls=calls))
    assert meta["selection_method"] == "editor"
    assert len(segs) == 3 and sum(1 for s in segs if s["tier"] == 0) == 2
    assert calls == [sv.PROMPT_VERSION, sv.CHECK_PROMPT_VERSION]


def test_ads_and_chat_talk_never_picked_and_ads_ranges_excluded():
    n = 1900
    segs, meta = sv.select_moments(_words(n), n, _signals(n), 10, sv.sm.MomentConfig(editor=True),
                                   chat=_chat(lambda it: {"ad": True}), ads=[(500.0, 560.0)])
    assert all(not (s["start"] < 560 and s["start"] + s["duration"] > 500) for s in segs)
    assert all(s["source"] != "stream_moment:editor" for s in segs)       # every verdict said «ad»
    assert segs                                                           # speech → never NO_CLIPS


def test_final_check_drops_unclear_but_keeps_the_floor():
    n = 1900
    segs, meta = sv.select_moments(_words(n), n, _signals(n), 10, sv.sm.MomentConfig(editor=True),
                                   chat=_chat(lambda it: {}, check=lambda it: {"clear": False, "score": 40}))
    assert len(segs) == 3 and meta["final_check"]["dropped_weak"]


def test_editor_may_start_on_the_setup_before_the_moment():
    n = 1900
    segs, _ = sv.select_moments(_words(n), n, _signals(n), 3, sv.sm.MomentConfig(editor=True),
                                chat=_chat(lambda it: {"start": 0, "end": it["clip_starts_at"] + 3}))
    assert any("moment at" in s["selection_reason"] for s in segs)


def test_runner_engine_switch(monkeypatch):
    from modes.streamer import runner
    monkeypatch.delenv("STREAMER_SELECT", raising=False)
    assert runner.select_engine({}) == "viral"
    monkeypatch.setenv("STREAMER_SELECT", "classic")
    assert runner.select_engine({}) == "classic"
    assert runner.select_engine({"moments_engine": "viral"}) == "viral"
    monkeypatch.setenv("STREAMER_SELECT", "junk")
    assert runner.select_engine({}) == "viral"
