"""«Завозный коммент»: chat replay loading / Twitch fetch, read detection, editor parsing, merge, card pictures,
render graph (overlays + glitch) and the runner hook."""
import json

import numpy as np
import pytest

from scripts.shared import comment_card as cc
from scripts.shared import stream_chat as sch
from scripts.shared import stream_comment as scm
from scripts.shared import stream_moments as sm
from scripts.shared.boundaries import normalize_words


def W(text, t0, step=0.4):
    return [{"word": w, "start": round(t0 + i * step, 2), "end": round(t0 + i * step + 0.3, 2)}
            for i, w in enumerate(text.split())]


def _words(*parts):
    out = []
    for text, t0 in parts:
        out += W(text, t0)
    return out


# ── stream_chat ─────────────────────────────────────────────────────────────

def test_load_twitch_rechat(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"comments": [
        {"content_offset_seconds": 12.5, "commenter": {"display_name": "tumca52"},
         "message": {"body": "Вась а ты реально первый?", "user_color": "#8A2BE2"}},
        {"content_offset_seconds": 3, "commenter": {"display_name": "x"}, "message": {"body": "  "}}]}),
        encoding="utf-8")
    m = sch.load(str(p))
    assert m == [{"t": 12.5, "author": "tumca52", "text": "Вась а ты реально первый?", "color": "#8A2BE2"}]


def test_load_youtube_live_chat(tmp_path):
    line = {"replayChatItemAction": {"videoOffsetTimeMsec": "61500", "actions": [{"addChatItemAction": {"item": {
        "liveChatTextMessageRenderer": {"authorName": {"simpleText": "Петя"},
                                        "message": {"runs": [{"text": "ты где "}, {"text": "права купил"}]}}}}}]}}
    p = tmp_path / "v.live_chat.json"
    p.write_text(json.dumps(line, ensure_ascii=False) + "\n{bad\n", encoding="utf-8")
    assert sch.load(str(p)) == [{"t": 61.5, "author": "Петя", "text": "ты где права купил", "color": None}]


def test_twitch_id():
    assert sch.twitch_video_id("https://www.twitch.tv/videos/2932185478?t=1h") == "2932185478"
    assert sch.twitch_video_id("https://twitch.tv/zloyn/v/123") == "123"
    assert sch.twitch_video_id("https://twitch.tv/zloyn") is None


def test_fetch_twitch_offset_paging(tmp_path):
    # fake GQL: 3 messages per second, pages of 4 starting at the requested offset
    allm = [(t, f"u{t}{k}", f"msg {t} {k}") for t in range(100, 130) for k in range(3)]
    calls = []

    def gql(body):
        off = body["variables"]["off"]
        calls.append(off)
        page = [m for m in allm if m[0] >= off][:4]
        return {"data": {"video": {"comments": {
            "edges": [{"node": {"contentOffsetSeconds": t, "commenter": {"displayName": u},
                                "message": {"fragments": [{"text": x}], "userColor": None}}} for t, u, x in page],
            "pageInfo": {"hasNextPage": bool(page) and page[-1] != allm[-1]}}}}}
    out = tmp_path / "c.json"
    n = sch.fetch_twitch("1", str(out), 100, 130, parallel=1, gql=gql)
    assert n == len(allm)
    m = sch.load(str(out))
    assert m[0]["t"] == 0.0 and m[-1]["t"] == 29.0                     # rebased to the downloaded piece
    assert all(b >= a for a, b in zip(calls, calls[1:]))                 # moves forward only


def test_fetch_for_source_never_raises(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("network")
    monkeypatch.setattr(sch, "fetch_twitch", boom)
    r = sch.fetch_for_source("https://www.twitch.tv/videos/5", "twitch", str(tmp_path / "in.mp4"))
    assert r["status"] == "error"
    monkeypatch.setenv("STREAMER_CHAT_FETCH", "off")
    assert sch.fetch_for_source("https://www.twitch.tv/videos/5", "twitch", str(tmp_path / "in.mp4"))["status"] == "off"
    monkeypatch.delenv("STREAMER_CHAT_FETCH")
    assert sch.fetch_for_source("https://vk.com/video1", "vk", str(tmp_path / "in.mp4"))["status"] == "none"


# ── reads ───────────────────────────────────────────────────────────────────

SAID = _words(("короче едем дальше по карте смотрим что тут есть.", 100.0),
              ("о вась а ты шо реально первый кто в мире в стрипуху приехал ахахаха ахаха да я первый в мире.", 110.0),
              ("ладно поехали отсюда быстрее давай.", 125.0))


def test_reads_from_chat_matches_the_read_out_message():
    words, _ = normalize_words(SAID)
    msgs = [{"t": 104.0, "author": "tumca52", "text": "Вась а ты шо реально первый кто в мире в стрипуху приехал?",
             "color": "#8A2BE2"},
            {"t": 105.0, "author": "spam", "text": "https://t.me/x купи"},
            {"t": 106.0, "author": "other", "text": "какой у тебя фпс сейчас в игре"}]
    r = scm.reads_from_chat(words, msgs)
    assert len(r) == 1 and r[0]["author"] == "tumca52"
    assert 110.0 <= r[0]["t0"] <= 110.5 and r[0]["match"] >= 0.8


def test_reads_from_chat_needs_the_streamer_to_say_it():
    words, _ = normalize_words(SAID)
    assert scm.reads_from_chat(words, [{"t": 104.0, "author": "a", "text": "лучший стример планеты играет отлично"}]) == []
    # a message sent AFTER the words were said was not read out
    assert scm.reads_from_chat(words, [{"t": 140.0, "author": "a", "text": "первый кто в мире в стрипуху приехал"}]) == []


def test_reads_from_asr_cue():
    words, _ = normalize_words(_words(("так вася пишет ты реально первый в мире кто туда приехал.", 50.0),
                                      ("ахахах ну да.", 56.0)))
    r = scm.reads_from_asr(words, sm._sentences(words))
    assert len(r) == 1 and r[0]["author"] == scm.VIEWER and r[0]["text"].startswith("ты реально первый")
    assert r[0]["t0"] <= words[1].start                     # starts at the nick word before «пишет»


def test_payoff_options_find_the_moment_elsewhere():
    said = SAID + _words(("смотри я заезжаю в стрипуху первый в мире вот это рекорд.", 400.0))
    words, _ = normalize_words(said)
    sents = sm._sentences(words)
    r = scm.reads_from_chat(words, [{"t": 104.0, "author": "t", "text": "ты шо реально первый кто в мире в стрипуху приехал"}])[0]
    r["react_last"] = scm._reaction(words, sents, r)
    ops = scm.payoff_options(words, sents, r, None)
    assert ops and ops[0]["start"] == 400.0 and ops[0]["rel"] > 0


def _fake_chat(answer):
    class R:
        text = json.dumps(answer, ensure_ascii=False)

        def telemetry(self):
            return {"cost_usd": 0.004, "input_tokens": 3000, "output_tokens": 300}
    calls = []

    def chat(msgs, **kw):
        calls.append(msgs)
        return R()
    return chat, calls


def test_select_with_editor_payoff_and_title():
    said = SAID + _words(("смотри я заезжаю в стрипуху первый в мире вот это рекорд.", 400.0))
    msgs = [{"t": 104.0, "author": "tumca52", "text": "Вась а ты шо реально первый кто в мире в стрипуху приехал?",
             "color": "#8A2BE2"}]
    chat, calls = _fake_chat({"clips": [{"id": "c0", "is_comment": True, "score": 88, "quote": "x", "end": 1,
                                         "payoff": 0, "payoff_conf": 90, "title": "Злой установил мировой рекорд в GTA6",
                                         "accent": ["злой", "GTA6", "нет такого"], "why": "смешно"}]})
    segs, meta = scm.select(said, 500, {"loud": [0.0] * 500}, messages=msgs, chat=chat)
    assert len(calls) == 1 and meta["llm"]["forecast_rub"] <= scm.BUDGET_RUB
    s = segs[0]
    assert s["moment_type"] == "comment" and s["comment"]["text"].startswith("Вась")     # chat text kept verbatim
    assert s["comment"]["author"] == "tumca52" and s["payoff"]["start"] < 400.0 < s["payoff"]["end"]
    assert s["title"] == {"text": "ЗЛОЙ УСТАНОВИЛ МИРОВОЙ РЕКОРД В GTA6", "accent": ["ЗЛОЙ", "GTA6"]}
    assert s["start"] <= 110.4 and s["start"] + s["duration"] >= 120.0


def test_select_editor_rejects_and_no_editor_fallback():
    msgs = [{"t": 104.0, "author": "t", "text": "Вась а ты шо реально первый кто в мире в стрипуху приехал?"}]
    chat, _ = _fake_chat({"clips": [{"id": "c0", "is_comment": True, "score": 30}]})
    segs, meta = scm.select(SAID, 200, {}, messages=msgs, chat=chat)
    assert segs == [] and meta["status"] == "none_strong"
    segs, meta = scm.select(SAID, 200, {}, messages=msgs, editor=False)
    assert meta["selection_method"] == "local" and len(segs) == 1 and "payoff" not in segs[0]


def test_select_budget_trims_pool():
    said, msgs = [], []
    for k in range(10):
        t = 100.0 + 60 * k
        said += _words((f"о вась а ты шо реально первый кто в мире в стрипуху приехал номер {k} ахаха.", t))
        msgs.append({"t": t - 5, "author": f"u{k}", "text": f"Вась а ты шо реально первый кто в мире в стрипуху приехал номер {k}"})
    chat, calls = _fake_chat({"clips": []})
    _, meta = scm.select(said, 800, {}, messages=msgs, chat=chat, budget_rub=0.05)
    assert meta["llm"]["pool"] < meta["candidates"]


def test_quote_must_be_what_was_said():
    assert scm._quote_ok("ты реально первый в мире", "ну ты реально первый в мире кто туда")
    assert not scm._quote_ok("купи скины на сайте", "ну ты реально первый в мире кто туда")


def _seg(a, d, **kw):
    return dict({"start_sec": a, "duration_sec": d}, **kw)


def test_merge():
    reg = [_seg(0, 30), _seg(100, 30), _seg(200, 30)]
    com = [_seg(105, 20, payoff={"start": 300, "end": 310}), _seg(500, 20)]
    assert scm.merge(reg, com, ["comment"], 10) == com                       # chosen alone: only comment clips
    assert scm.merge(reg, [], ["comment"], 10) == reg                        # none found: the regular ones
    auto = scm.merge(reg, com, [], 10, auto_max=1)
    assert auto[0] is com[0] and _seg(100, 30) not in auto and len(auto) == 3
    mixed = scm.merge(reg, com, ["chaos", "comment"], 4)
    assert mixed[:2] == com and len(mixed) == 4


# ── pictures / render graph ─────────────────────────────────────────────────

def test_card_and_title_png(tmp_path):
    c = cc.card_png(str(tmp_path / "c.png"), "tumca52", "Вась а ты шо реально первый кто в мире в стрипуху приехал?",
                    "#8A2BE2")
    assert c["w"] <= 920 and 120 <= c["h"] <= 330
    long = cc.card_png(str(tmp_path / "l.png"), "", "слово " * 80)
    assert long["h"] <= 2 * 30 + 4 * 59 + 10                                 # ≤ 4 lines
    t = cc.title_png(str(tmp_path / "t.png"), "злой установил мировой рекорд в gta6", ["злой", "gta6"])
    assert t["w"] == 1000 and t["h"] < 260
    assert cc.split_title("ЗЛОЙ УСТАНОВИЛ МИРОВОЙ РЕКОРД В GTA6") == ["ЗЛОЙ УСТАНОВИЛ МИРОВОЙ", "РЕКОРД В GTA6"] or \
        len(cc.split_title("ЗЛОЙ УСТАНОВИЛ МИРОВОЙ РЕКОРД В GTA6")) == 2


def test_nick_color():
    assert cc.nick_color("a", "#FF0000") == "#FF0000"
    dark = cc.nick_color("a", "#000080")
    assert dark != "#000080" and dark.startswith("#")
    assert cc.nick_color("same", None) == cc.nick_color("same", None)


def test_layout_positions():
    ty, cy = cc.layout(1180, 170, 220)
    assert ty + 170 <= 1180 - 200 and cy > 1180
    ty, cy = cc.layout(960, 170, 220)
    assert ty + 170 <= 960 - 200 and cy > 960                                 # title above the subtitle band, card in the game
    ty, cy = cc.layout(1500, 170, 220)
    assert ty < 400 and cy > ty + 170


def test_glitch_filters_are_timeline_enabled():
    vf = cc.glitch_vf(10.0)
    assert vf.count("enable=") == 6 and "between(t,9.790,10.210)" in vf


def test_render_pieces_graph_has_overlays_and_vfx(monkeypatch, tmp_path):
    from scripts.shared.crop import smart_crop_adapter as sca
    got = {}

    def fake_run(make, enc, timeout=None, cwd=None, what=None):
        got["cmd"] = make(enc)

        class R:
            returncode = 0
            stderr = ""
        return R(), "libx264", 0.1
    import scripts.shared.cut_profile as cp
    monkeypatch.setattr(cp, "run_encode", fake_run)
    track = {"layout": "game", "crop_version": "x", "crop_status": "ok"}
    monkeypatch.setattr(sca, "track_graph", lambda t: "scale=1080:1920")
    ovs = [{"path": str(tmp_path / "a.png"), "x": 10, "y": 20, "t0": 0.0, "t1": 3.0, "fade_out": 0.3},
           {"path": str(tmp_path / "b.png"), "x": 5, "y": 900, "t0": 1.0, "t1": 4.0, "fade_in": 0.1}]
    sca.render_pieces("in.mp4", str(tmp_path / "o.mp4"), [(10.0, 5.0), (50.0, 4.0)], track, 10.0,
                      overlays=ovs, vfx=cc.glitch_vf(5.0))
    cmd = got["cmd"]
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert cmd.count("-loop") == 2 and "[vcat][ol0]overlay=x=10:y=20" in graph and "[vo0][ol1]overlay" in graph
    assert "[vo1]rgbashift" in graph and "alpha=1" in graph


# ── runner hook ─────────────────────────────────────────────────────────────

def test_runner_with_comments_explicit_and_auto(monkeypatch, tmp_path):
    from modes.streamer import runner
    seg = {"start": 109.75, "duration": 20.0, "score": 3.9, "source": "stream_comment:editor", "moment_type": "comment",
           "category": "comment", "editor_score": 65.0, "payoff": {"start": 400.0, "end": 410.0},
           "comment": {"author": "t", "text": "q", "color": None, "source": "chat", "t0": 110.0, "t1": 115.0}}
    monkeypatch.setattr(scm, "select", lambda *a, **k: ([dict(seg)], {"status": "ok"}))
    reg = [{"start_sec": 0.0, "duration_sec": 30.0}, {"start_sec": 112.0, "duration_sec": 30.0}]
    words = SAID + _words(("смотри я заезжаю в стрипуху.", 400.0))
    chatf = tmp_path / "x.live_chat.json"
    chatf.write_text(json.dumps({"comments": []}), encoding="utf-8")
    sel = {}
    out = runner._with_comments(reg, words, 500, {"loud": [0.0] * 500}, [], [], str(chatf), ["comment"], 10,
                                {"editor": True}, sel)
    assert len(out) == 1 and out[0]["comment"]["platform"] == "twitch"
    assert any(w["start"] >= 400.0 for w in out[0]["words"])               # payoff words for the subtitles
    out = runner._with_comments(reg, words, 500, {}, [], [], None, [], 10, {"editor": True}, sel)
    assert out == reg                                                         # «Авто»: 65 < 70 → not added
    seg["editor_score"] = 80.0
    out = runner._with_comments(reg, words, 500, {}, [], [], None, [], 10, {"editor": True}, sel)
    assert out[0]["moment_type"] == "comment" and len(out) == 2               # the overlapping regular one gives way


def test_types_know_comment():
    from scripts.shared.stream_types import normalize_types
    assert normalize_types("коммент, хаос") == ["chaos", "comment"]
    assert normalize_types(["comment"]) == ["comment"]


def test_background_fetch_marker(tmp_path, monkeypatch):
    import time as _t
    video = tmp_path / "input.mp4"
    seen = {}

    def fake(url, plat, path, start, end, yt_opts=None):
        seen.update(url=url, plat=plat, start=start, end=end)
        _t.sleep(0.3)
        return {"status": "ok"}
    monkeypatch.setattr(sch, "fetch_for_source", fake)
    th = sch.start_background("https://www.twitch.tv/videos/9", "twitch", str(video),
                              {"trim_start_sec": 600, "trim_end_sec": 4200})
    assert video.with_suffix(sch.PENDING_SUFFIX).exists()
    sch.wait_pending(str(video), timeout=5)
    th.join()
    assert not video.with_suffix(sch.PENDING_SUFFIX).exists()
    assert seen == {"url": "https://www.twitch.tv/videos/9", "plat": "twitch", "start": 600.0, "end": 4200.0}
    assert sch.start_background("https://vk.com/video1", "vk", str(video)) is None


def test_routine_and_unsure_payoff():
    msgs = [{"t": 104.0, "author": "t", "text": "Вась а ты шо реально первый кто в мире в стрипуху приехал?"}]
    said = SAID + _words(("смотри я заезжаю в стрипуху первый в мире вот это рекорд.", 400.0))
    chat, _ = _fake_chat({"clips": [{"id": "c0", "is_comment": True, "score": 80, "end": 0, "payoff": 0,
                                     "payoff_conf": 40, "title": "T"}]})
    segs, _ = scm.select(said, 500, {}, messages=msgs, chat=chat)
    assert "payoff" not in segs[0]                                          # not sure → no continuation
    assert segs[0]["start"] + segs[0]["duration"] - 110.0 >= scm.MIN_REACTION_SEC   # reaction never cut to nothing
    assert scm._ROUTINE.search("какой у тебя фпс в 4K") and not scm._ROUTINE.search("ты реально первый в стрипухе")


def test_card_drops_emoji(tmp_path):
    assert cc.clean_text("ПЛЯЖ 🏝️ ДАВАЙ 😂😂") == "ПЛЯЖ ДАВАЙ"
    assert cc.card_png(str(tmp_path / "e.png"), "x", "🏝️")["w"] > 0
