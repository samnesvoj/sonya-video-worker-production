"""Moment types: philosophy / zavoz / chaos (scripts/shared/stream_types + stream_moments integration)."""
import numpy as np

from scripts.shared import stream_types as st
from scripts.shared.stream_moments import MomentConfig, select_moments


def W(text, t0, step=0.4):
    return [{"word": w, "start": round(t0 + i * step, 2), "end": round(t0 + i * step + 0.3, 2)}
            for i, w in enumerate(text.split())]


PHIL = ("Мне никто не верил. Все смеялись и говорили что у меня ничего не получится. Но я не сдавался каждый день. "
        "А потом все поверили. Поэтому верьте в себя, это главное. Помните что путь к цели начинается с первого шага. "
        "Никогда не поздно начать заново и идти к своей мечте.")


def test_normalize_types():
    assert st.normalize_types("хаос, philosophy, bogus") == ["philosophy", "chaos"]
    assert st.normalize_types(None) == [] and st.normalize_types(["Zavoz"]) == ["zavoz"]


def test_type_hits():
    kinds = {k for _, k, _ in st.type_hits(W("верьте в себя друзья встречайте нашего гостя они всё ломают", 0))}
    assert kinds == {"wisdom", "zavoz_lead", "chaos_word"}


def test_music_runs_needs_sound_without_speech():
    loud = [0.0] * 30 + [1.0] * 6 + [0.0] * 10
    assert st.music_runs(W("привет", 0), loud, 46) == [(30, 36)]
    assert st.music_runs([], [-1.0] * 46, 46) == []                # silence is not music


def _cand(a, b):
    return {"start": a, "end": b, "tight": b - a, "silent": 0.05}


def test_philosophy_calm_monologue_only():
    words = W(PHIL, 0, 0.5)
    sig = {"loud": [0.1] * 100}
    ctx = st.Context(words, 100, sig, [])
    assert st.classify(_cand(0, 40), ctx)["philosophy"] > 0
    shouting = st.Context(words, 100, {"loud": [3.5] * 100}, [(5.0, "laugh", "ха"), (6.0, "joke", "x"), (7.0, "laugh", "ха")])
    assert st.classify(_cand(0, 40), shouting)["philosophy"] == 0


def test_chaos_needs_volume_and_motion():
    words = W("он всё ломает ломай ломай аааа нет", 10, 0.5)
    hits = [(10.5 + i, "shout", "аа") for i in range(6)]
    loud = [0.0] * 30
    for i in range(10, 18):
        loud[i] = 3.8
    assert st.classify(_cand(10, 20), st.Context(words, 30, {"loud": loud, "flash": [3.0] * 30}, hits))["chaos"] > 0
    assert st.classify(_cand(10, 20), st.Context(words, 30, {"loud": [0.0] * 30}, hits[:2]))["chaos"] == 0


def test_zavoz_lead_music_reaction():
    words = W("встречайте нашего гостя", 10, 0.5) + W("привет ребята всем привет как дела ха", 20, 0.5)
    loud = [0.0] * 40
    for i in range(13, 19):
        loud[i] = 1.2
    ctx = st.Context(words, 40, {"loud": loud}, [(21.0, "guest_in", "привет"), (23.0, "laugh", "ха")])
    assert st.classify(_cand(10, 26), ctx)["zavoz"] > 0
    assert st.classify(_cand(10, 26), st.Context(words, 40, {"loud": [0.0] * 40}, []))["zavoz"] == 0


def test_config_philosophy_allows_60s_and_editor_budget():
    cfg = MomentConfig(moment_types="philosophy")
    assert cfg.moment_types == ["philosophy"] and cfg.max_duration == 60.0 and cfg.budget_rub <= 5.0


def test_select_only_requested_type():
    words = W(PHIL, 130, 0.5) + W("а вот обычная болтовня про настройки видеокарты и драйверов без событий ", 300, 0.5)
    n = 400
    sig = {"loud": [0.1] * n}
    segs, meta = select_moments(words, n, sig, 5, MomentConfig(moment_types=["philosophy"], skip_intro_sec=0.0))
    assert segs and all(s.get("category") == "philosophy" and s.get("music") == "none" for s in segs)
    none, _ = select_moments(words, n, sig, 5, MomentConfig(moment_types=["chaos"], skip_intro_sec=0.0))
    assert none == []


def test_editor_prompt_lists_requested_types():
    from scripts.shared.stream_moments import editor_messages
    msg = editor_messages([], 3, "", ["philosophy", "chaos"], 60.0)[1]["content"]
    assert "философия —" in msg and "хаос —" in msg and "20–60" in msg and "завоз —" not in msg
