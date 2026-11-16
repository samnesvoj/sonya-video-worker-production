"""A clip never opens on «Угу.» / «Уда.» (production 2026-11-15, job 920396ed clips 2 and 3)."""
from scripts.shared.cut_selection import StoryConfig, build_story_candidates, is_backchannel
from scripts.shared.boundaries import Word


def _words(text, t0=0.0, step=0.4, pause_after=None):
    out, t = [], t0
    for w in text.split():
        out.append(Word(text=w, start=t, end=t + step * 0.8))
        t += step + (pause_after.get(w, 0.0) if pause_after else 0.0)
    return out


def test_is_backchannel():
    for t in ("Угу.", "Уда.", "Ага, да.", "Ну.", "Мм...", "Yeah.", "Нет."):
        assert is_backchannel(t), t
    for t in ("Котлеты как у нас?", "Кто?", "Потрясающая!", "Я не знаю.", "Зачем?"):
        assert not is_backchannel(t), t


def test_candidates_skip_backchannel_start():
    body = " ".join(["Мы приехали в Париж и купили шубу за огромные деньги, это было смешно."] * 8)
    ws = _words("Угу. " + body, pause_after={"Угу.": 2.5})
    cands = build_story_candidates(ws, StoryConfig(min_duration=5, max_duration=90))
    assert cands
    assert all(not c["text"].startswith("Угу") for c in cands)
