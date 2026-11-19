"""scripts/shared/stream_ads.py: sponsor / ad inserts in a stream (owner 2026-11-19, gumrod 0c9c1be0 clip 3)."""
from __future__ import annotations

import json

from scripts.shared import stream_ads as sa

AD = ("Любители КС и те, кто давно искал удобную площадку для быстрого и безопасного обмена скинами, спешу вас "
      "обрадовать. Ведь Лисскинс в этом деле одни из лучших, и мы работаем с ними уже не первый год. Вставляем "
      "ссылку на трейд, выбираем скины и удобный способ получения средств. Готово! А можно зачислить всю сумму на "
      "баланс сайта. Переходи по ссылке в закреплённом комментарии.")
GAME = "Ебать, менты быстро приехали! На тебе, сука! Дай застрелить его. Ладно, пусть живёт. Да уходи ты уже."


def _words(chunks):
    """[(start, text)] → words, 0.4 s per word, sentence-final punctuation kept."""
    out = []
    for t, txt in chunks:
        for k, w in enumerate(txt.split()):
            out.append({"word": w, "start": round(t + k * 0.4, 2), "end": round(t + k * 0.4 + 0.35, 2)})
    return out


def _stream():
    chunks = []
    t, done = 0.0, False
    while t < 600:
        if t >= 300 and not done:
            done = True
            chunks.append((t, AD))
            t += len(AD.split()) * 0.4 + 1.0
            continue
        chunks.append((t, GAME))
        t += len(GAME.split()) * 0.4 + 1.0
    return _words(chunks)


class _Res:
    def __init__(self, text):
        self.text = text

    def telemetry(self):
        return {"input_tokens": 500, "output_tokens": 50}


def test_local_finds_the_skins_ad_and_nothing_in_game_talk():
    w = _stream()
    rng, meta = sa.detect(w)
    t_ad = next(x["start"] for x in w if x["word"] == "Любители")
    assert len(rng) == 1
    a, b = rng[0]
    assert a == t_ad and b >= t_ad + len(AD.split()) * 0.4 - 2
    assert meta["llm"]["status"] == "off"
    rng2, _ = sa.detect(_words([(t, GAME) for t in range(0, 600, 6)]))
    assert rng2 == []


def test_editor_no_drops_a_weak_region_and_errors_keep_it():
    w = _words([(0, GAME), (10, "Тут удобная площадка и быстрый сервис, бонус на баланс сайта. Выгодно и безопасно."),
                (30, GAME)])
    local, _ = sa.detect(w)
    assert local
    no = lambda *a, **k: _Res(json.dumps({"regions": [{"id": "r0", "ad": False, "first": 0, "last": 0}]}))  # noqa
    rng, meta = sa.detect(w, editor=True, chat=no)
    assert rng == [] and meta["llm"]["status"] == "ok"

    def boom(*a, **k):
        raise RuntimeError("down")
    rng, meta = sa.detect(w, editor=True, chat=boom)
    assert rng == local and meta["llm"]["status"] == "error"


def test_cut_out_trims_a_clip_before_the_ad_or_drops_it():
    w = _stream()
    rng, _ = sa.detect(w)
    a0 = rng[0][0]
    segs = [{"start": a0 - 30.0, "duration": 36.0, "peak_t": a0 + 2}, {"start": a0 - 4.0, "duration": 10.0}]
    out, notes = sa.cut_out(segs, rng, w, min_sec=15.0)
    assert len(out) == 1 and len(notes) == 2
    s = out[0]
    assert s["start"] + s["duration"] <= a0 and s["duration"] >= 15.0 and "peak_t" not in s
    assert s["ad_trim"]
