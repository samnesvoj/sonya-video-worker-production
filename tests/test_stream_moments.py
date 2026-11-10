"""scripts/shared/stream_moments.py: candidates from whole sentences, peak/hook scoring, editor with a ₽ cap."""
from __future__ import annotations

import json

from scripts.shared import stream_moments as sm


def _words(n_sec=600, every=3.0):
    """A sentence every 3 s: «Слово раз два три.»; a few exclamations around 300 s."""
    out = []
    t = 0.0
    i = 0
    while t < n_sec - 3:
        txt = ["Я", "иду", "на", "точку", "сейчас."]
        if 296 <= t <= 304:
            txt = ["Да", "ладно", "что", "это", "было?!"]
        for k, w in enumerate(txt):
            out.append({"word": w, "start": round(t + k * 0.4, 2), "end": round(t + k * 0.4 + 0.35, 2)})
        t += every
        i += 1
    return out


def _signals(n, peak_at=300):
    loud = [0.0] * n
    for k in range(peak_at - 2, peak_at + 4):
        loud[k] = 4.0
    return {"loud": loud, "pitch": [0.0] * n, "react": [0.0] * n}


def test_candidates_start_and_end_on_sentences_within_20_45():
    from scripts.shared.boundaries import normalize_words
    words, _ = normalize_words(_words())
    cands = sm.build_candidates(words, sm.MomentConfig())
    assert cands
    for c in cands:
        assert 20.0 - 1e-6 <= c["tight"] <= 45.0 + 1e-6      # length after the jump cuts
        assert c["start"] >= 120.0                       # intro skipped
        assert words[c["last"]].text.endswith((".", "!", "?", "?!")) or c["next_gap"] >= 1.2


def test_selection_finds_the_peak_with_room_for_the_reaction():
    n = 600
    segs, meta = sm.select_moments(_words(n), n, _signals(n), 3)
    assert segs and meta["selection_method"] == "local"
    best = segs[0]
    assert best["start"] < 300 < best["start"] + best["duration"] - sm.MomentConfig().tail_sec
    assert 20 <= best["duration"] <= 60
    # the selected clips never overlap
    iv = sorted((s["start"], s["start"] + s["duration"]) for s in segs)
    assert all(a[1] <= b[0] for a, b in zip(iv, iv[1:]))


def test_hook_prefers_a_lively_opening():
    n = 600
    sig = _signals(n)
    e = sm.excitement(sig, n)
    assert e[300] > e[200]


def test_no_words_no_clips():
    segs, meta = sm.select_moments([], 100, {}, 3)
    assert segs == [] and meta["selection_status"] == "no_words"


class _Res:
    def __init__(self, text, cost=0.0004):
        self.text = text
        self._cost = cost

    def telemetry(self):
        return {"provider": "fake", "model": "gemini-fake", "cost_usd": self._cost}


def test_editor_reorders_and_reports_cost():
    n = 900
    words = _words(n)
    sig = _signals(n, peak_at=300)
    seen = {}

    def chat(msgs, **kw):
        seen["msgs"] = msgs
        items = json.loads(msgs[1]["content"].split("Фрагменты:\n", 1)[1].split("\n\nВерни", 1)[0])
        seen["n"] = len(items)
        assert all("signals" in it for it in items)
        # the editor likes the LAST candidate best
        assert all(it["phrases"] and it["phrases"][0].startswith("0) +0s") for it in items)
        clips = [{"id": it["id"], "score": 90 if k == len(items) - 1 else 70, "clear_3s": True, "type": "эмоция",
                  "start": 0, "end": len(it["phrases"]) - 1,
                  "title": "Хук", "keywords": ["точку"], "why": "ок"} for k, it in enumerate(items)]
        return _Res(json.dumps({"clips": clips}))

    segs, meta = sm.select_moments(words, n, sig, 3, sm.MomentConfig(editor=True), chat=chat)
    assert meta["llm"]["status"] == "ok" and meta["selection_method"] == "editor"
    assert meta["llm"]["cost_rub"] == 0.04
    assert meta["llm"]["forecast_rub"] <= 10.0
    assert segs[0]["editor_score"] == 90 and segs[0]["hook"] == "Хук" and segs[0]["keywords"] == ["точку"]


def test_editor_failure_falls_back_to_local_order():
    n = 600

    def chat(msgs, **kw):
        raise TimeoutError("provider down")

    segs, meta = sm.select_moments(_words(n), n, _signals(n), 2, sm.MomentConfig(editor=True), chat=chat)
    assert segs and meta["llm"]["status"] == "error" and meta["selection_method"] == "local"


def test_editor_budget_cap_shrinks_or_skips():
    n = 600
    called = []
    cfg = sm.MomentConfig(editor=True, budget_rub=0.0001)
    segs, meta = sm.select_moments(_words(n), n, _signals(n), 2, cfg, chat=lambda *a, **k: called.append(1))
    assert not called and meta["llm"]["status"] == "skipped_budget" and segs


def test_forecast_is_far_below_cap_for_a_full_pool():
    pool = [{"id": f"c{i}", "start": 0, "end": 40, "text": "слово " * 140, "tags": "громко z4, смех×2"}
            for i in range(30)]
    msgs = sm.editor_messages(pool, 8)
    assert sm.forecast_rub(msgs, 30, sm.MomentConfig()) < 10.0


def _strict_chat(boring_ids=(), trim_first=0):
    def chat(msgs, **kw):
        items = json.loads(msgs[1]["content"].split("Фрагменты:\n", 1)[1].split("\n\nВерни", 1)[0])
        clips = []
        for k, it in enumerate(items):
            boring = k in boring_ids
            clips.append({"id": it["id"], "score": 20 if boring else 80 - k, "clear_3s": not boring,
                          "type": "скучно" if boring else "смешное", "start": trim_first,
                          "end": len(it["phrases"]) - 1, "title": "Т", "why": "w"})
        return _Res(json.dumps({"clips": clips}))
    return chat


def test_editor_drops_boring_but_keeps_min_keep():
    n = 900
    words = _words(n)
    cfg = sm.MomentConfig(editor=True, min_keep=2)
    # every candidate is «скучно» → only min_keep clips survive, the rest are rejected
    chat = _strict_chat(boring_ids=range(100))
    segs, meta = sm.select_moments(words, n, _signals(n), 5, cfg, chat=chat)
    assert len(segs) == 2 and meta["editor_rejected"] >= 1


def test_editor_trim_moves_the_start_to_a_later_phrase_and_keeps_length():
    n = 900
    words = _words(n)
    cfg = sm.MomentConfig(editor=True)
    base, _ = sm.select_moments(words, n, _signals(n), 1, cfg, chat=_strict_chat())
    trimmed, _ = sm.select_moments(words, n, _signals(n), 1, cfg, chat=_strict_chat(trim_first=2))
    assert trimmed[0]["start"] > base[0]["start"]
    assert 18.0 <= trimmed[0]["tight_sec"] <= 48.0
    assert "editor trim" in trimmed[0]["selection_reason"]


def test_parse_editor_is_fail_closed():
    pool = {"a": {"text": "привет мир", "sentences": [(0, 1), (2, 3)]}}
    v = sm.parse_editor(json.dumps({"clips": [{"id": "a", "score": "x"}, {"id": "zz", "score": 99},
                                              {"id": "a", "score": 77, "start": 9, "end": -3,
                                               "keywords": ["мир", "чужое"], "clear_3s": "true"}]}), pool)
    assert len(v) == 1 and v[0]["start"] == 1 and v[0]["end"] == 0 or v[0]["start"] <= v[0]["end"]
    assert v[0]["keywords"] == ["мир"] and v[0]["clear_3s"] is False
