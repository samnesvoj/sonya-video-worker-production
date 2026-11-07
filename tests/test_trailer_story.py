"""Style «story» (storystyle.py): pure logic only — no network, no ffmpeg render."""
import pytest

from modes.trailer_cinematic import compose, recap, storyfx as fx, storystyle as S
from modes.trailer_cinematic.budget import SchemaError


def words(spec):
    return [{"word": w, "start": t, "end": t + 0.3} for w, t in spec]


def talk(a, b, text="деньги нужны мне сейчас", step=0.5):
    out, t = [], a
    while t < b:
        for w in text.split():
            out.append({"word": " " + w, "start": t, "end": t + 0.3})
            t += step
    return out


def test_candidates_length_overlap_and_order():
    w = talk(0, 7000)
    c = S.candidates(w, 7200, skip_head=300, skip_tail=300)
    assert 1 <= len(c) <= S.N_CANDIDATES
    for x in c:
        assert S.EP_MIN <= x["b"] - x["a"] <= S.EP_MAX
    for x, y in zip(c, c[1:]):
        assert y["a"] >= x["a"] and min(x["b"], y["b"]) - max(x["a"], y["a"]) <= 0.25 * (x["b"] - x["a"]) + 1


def test_candidates_prefer_stakes_over_silence():
    w = talk(0, 1500, "ну да ладно") + talk(1500, 3000, "миллион долларов фбр арест") + talk(3000, 4500, "ну да ладно")
    c = S.candidates(w, 4600, win=900, skip_head=0, skip_tail=0)
    best = max(c, key=lambda x: x["score"])
    assert 1400 <= best["a"] <= 2100


def test_candidates_skip_damaged_stretch():
    w = talk(0, 7000)
    c0 = S.candidates(w, 7200, skip_head=300, skip_tail=300)
    bad = [(c0[0]["a"] + c0[0]["b"]) / 2]
    c1 = S.candidates(w, 7200, skip_head=300, skip_tail=300, bad=bad)
    assert all(not (x["a"] - 20 <= bad[0] <= x["b"] + 20) for x in c1) and c1


def test_pick_parse():
    c = [{"id": "C1"}, {"id": "C2"}]
    assert S.parse_pick({"choice": "c2", "why": "x"}, c)[0]["id"] == "C2"
    with pytest.raises(SchemaError):
        S.parse_pick({"choice": "C9"}, c)


def test_fit_tempo_hits_range():
    r = S.fit_tempo(170.0, 8.0, 300)
    assert r["ok"] and 85 <= r["expected_total"] <= 105 and 2.9 <= r["expected_pace"] <= 3.4
    assert S.fit_tempo(400.0, 8.0, 300)["ok"] is False          # tempo clipped at the ceiling → honest failure


def test_shot_durs_rhythm_by_position():
    for prog, (lo, hi) in ((0.1, (2.0, 3.0)), (0.5, (1.6, 2.4)), (0.9, (1.2, 1.8))):
        assert fx.shot_range(prog) == (lo, hi)
        for span in (2.5, 4.0, 5.6, 8.7, 11.0):
            d = fx.shot_durs(span, prog)
            assert abs(sum(d) - span) < 1e-3
            if span >= 2 * hi:
                assert all(lo - 0.05 <= x <= hi + 0.05 for x in d), (prog, span, d)
    d = fx.shot_durs(9.0, 0.0, first=True)
    assert d[0] == 1.9 and abs(sum(d) - 9.0) < 1e-3                        # the strongest picture first, short


def test_crop_geometry_fills_the_frame():
    assert fx.canvas_size("1:1") == (1080, 1080) and fx.canvas_size("4:5") == (1080, 1350) and fx.canvas_size("3:4") == (1080, 1440)
    assert fx.crop_window(854, 356, "1:1") == (356, 356)
    cw, ch = fx.crop_window(854, 356, "3:4")
    assert ch == 356 and abs(cw / ch - 0.75) < 0.01 and cw % 2 == 0
    assert fx.crop_window(1920, 1080, "4:5") == (864, 1080)
    with pytest.raises(ValueError):
        fx.canvas_size("9:16")


def _scan(samples, W=854, H=356):
    return fx.Scan({"meta": {"width": W, "height": H}, "samples": samples})


def _smp(t, faces=(), motion=1.0, cut=False):
    return {"t": t, "motion": motion, "cut": cut, "faces": [list(f) for f in faces]}


def test_scan_prefers_big_moving_faces_over_empty_static_rooms():
    big = [(500, 40, 120, 130, 0.9)]
    s = _scan([_smp(i * 0.2, big if 10 <= i * 0.2 < 14 else (), 2.5 if 10 <= i * 0.2 < 14 else 0.1) for i in range(100)])
    face, empty = s.window(10.2, 12.2), s.window(2.0, 4.0)
    assert s.score(face) > s.score(empty) + 2
    best = fx.choose_window(s, (0.0, 20.0), 2.0, [], [])
    assert 10.0 <= best["src_start"] <= 12.0
    assert fx.choose_window(s, (0.0, 20.0), 2.0, [11.0], [])["src_start"] != 10.5          # never across a cut


def test_only_faces_inside_the_crop_window_count():
    s = _scan([_smp(i * 0.2, [(700, 60, 100, 120, 0.9)], 2.0) for i in range(30)])
    assert s.window(1.0, 3.0)["presence"] == 1.0                                  # somewhere in the wide frame
    assert s.window(1.0, 3.0, (0, 0, 356))["presence"] == 0.0                     # …but not in a window at the left edge
    assert s.window(1.0, 3.0, (498, 498, 356))["presence"] == 1.0
    assert s.score(s.window(1.0, 3.0, (0, 0, 356))) < s.score(s.window(1.0, 3.0, (498, 498, 356))) - 2


def test_choose_window_skips_used_source():
    s = _scan([_smp(i * 0.2, [(300, 40, 100, 120, 0.9)]) for i in range(100)])
    w1 = fx.choose_window(s, (0.0, 20.0), 2.0, [], [])
    w2 = fx.choose_window(s, (0.0, 20.0), 2.0, [], [(w1["src_start"], w1["src_start"] + 2.0)])
    assert w2["src_start"] + 2.0 <= w1["src_start"] or w2["src_start"] >= w1["src_start"] + 2.0


def test_crop_follows_the_face_and_stays_inside():
    s = _scan([_smp(i * 0.2, [(650, 60, 100, 120, 0.9)]) for i in range(30)])
    x0, x1 = s.crop_xs(1.0, 3.0, 356)
    assert x0 == 854 - 356 and x0 % 2 == 0                                  # face at the right edge → window clamped inside
    mid = _scan([_smp(i * 0.2, [(400, 60, 100, 120, 0.9)]) for i in range(30)]).crop_xs(1.0, 3.0, 356)[0]
    assert abs(mid - (450 - 178)) <= 4
    nof = _scan([_smp(i * 0.2) for i in range(30)])
    assert nof.crop_xs(1.0, 3.0, 356)[0] == (854 - 356) // 2 // 2 * 2


def test_push_in_only_on_static_shots_and_filter_is_valid():
    sh = {"src_start": 10.0, "src_end": 12.4, "crop_x0": 300, "crop_x1": 320, "push_in": 0.05, "fx": 0.5, "n": 1}
    f = fx.shot_filter(sh, 854, 356, 356, 1080, 1080)
    assert "zoompan" in f and "crop=1068:1068" in f
    assert "zoompan" not in fx.shot_filter(dict(sh, push_in=0.0), 854, 356, 356, 1080, 1080)


def test_subtitles_bold_on_picture_one_or_two_lines():
    ws, t = [], 0.0
    for w in ("Я", "сам", "посадил", "себя", "в", "камеру", "и", "выбросил", "ключ.", "Мой", "напарник", "заметил", "мой", "взгляд,", "но", "я",
              "всё", "равно", "сказал", "единственное", "запрещённое."):
        ws.append((w, t, t + 0.3))
        t += 0.33
    cues = S.sub_cues(ws)
    assert all(1 <= len(g) <= 6 for g in cues)
    assert sum(len(g) for g in cues) == len(ws)
    assert all(S.cue_fits([x[0] for x in g]) for g in cues)
    assert sum(len(g) >= 3 for g in cues) >= len(cues) - 2
    for geo in (S.canvas_geometry("1:1"), S.canvas_geometry("4:5"), S.canvas_geometry("3:4")):
        ass = S.story_ass(ws, 10.0, geo)
        ck = S.check_story_ass(ass, geo)
        assert ck["ok"], ck
        assert f"{S.SUB_FONT_NAME},{S.SUB_SIZE}" in ass and f",8," in ass
    assert "\\N" in S.story_ass(ws, 10.0, S.canvas_geometry("1:1"))                     # long cues wrap to a second line
    bad = "Dialogue: 0,0:00:00.00,0:00:02.00,Story,,0,0,0,,Невероятно длинная строка без переносов вообще никаких"
    assert not S.check_story_ass(bad, S.canvas_geometry("1:1"))["ok"]


def test_cue_does_not_end_on_a_short_word():
    ws, t = [], 0.0
    for w in "Мы начали торги с четырёх с половиной долларов за акцию и цена выросла в разы.".split():
        ws.append((w, t, t + 0.3))
        t += 0.33
    for g in S.sub_cues(ws)[:-1]:
        last = g[-1][0]
        assert len(last.strip(".,")) > 2 or len(g) <= 2 or last.endswith((",", ".")), g


def _events(n=10):
    return [{"id": f"E{i + 1}", "start": i * 20.0, "end": (i + 1) * 20.0, "text": "x"} for i in range(n)]


def _frames(n=6):
    return [{"id": f"F{i + 1}", "t": i * 30.0 + 5, "rel": "0:00", "path": "x"} for i in range(n)]


def _seg(i, fn, text, ev, fr="F1"):
    return {"id": f"S{i}", "function": fn, "text": text, "event_ids": [ev], "frame_ids": [fr], "visual_intent": "v", "pause_after": "normal"}


def _good(nseg=12, w_each=24):
    h = (w_each - 3) // 2
    txt = " ".join(["слово"] * h) + ". " + " ".join(["слово"] * (w_each - 3 - h)) + " я знаю это."
    fns = ["hook"] + ["origin"] * 3 + ["rising"] * (nseg - 6) + ["twist", "cliffhanger"]
    segs = [_seg(i + 1, fns[i], txt, f"E{min(10, i + 1)}") for i in range(nseg)]
    segs[0]["event_ids"] = ["E8"]
    segs[0]["text"] = "Я сам запер себя внутри и выбросил ключ далеко."
    return {"segments": segs}


def test_defects_clean_and_each_defect():
    ev, fr = _events(), _frames()
    assert S.story_defects(_good(), ev, fr, 270, 330, [], []) == []
    d = _good()
    d["segments"][3]["text"] = "Я сказал хуй."
    assert any("мат" in x for x in S.story_defects(d, ev, fr, 270, 330, [], []))
    d = _good()
    d["segments"][-1]["function"] = "twist"
    assert any("cliffhanger" in x for x in S.story_defects(d, ev, fr, 270, 330, [], []))
    d = _good()
    for s in d["segments"]:
        s["text"] = "Это было давно и слова слова слова слова слова слова слова слова слова слова слова слова."
    assert any("первого лица" in x for x in S.story_defects(d, ev, fr, 0, 999, [], []))
    d = _good()
    d["segments"][2]["text"] += " Джордан пришёл."
    assert any("имена" in x for x in S.story_defects(d, ev, fr, 270, 330, ["джордан"], []))
    d = _good()
    d["segments"][2]["text"] += " Он убил там пятерых."
    assert S.story_defects(d, ev, fr, 270, 330, [], [])
    d = _good(nseg=8)
    assert any("сегментов" in x for x in S.story_defects(d, ev, fr, 0, 999, [], []))
    d = _good()
    d["segments"][5]["event_ids"] = ["E1"]
    d["segments"][6]["event_ids"] = ["E9"]
    d["segments"][7]["event_ids"] = ["E1"]
    d["segments"][8]["event_ids"] = ["E1"]
    assert any("назад" in x for x in S.story_defects(d, ev, fr, 270, 330, [], []) )


def test_clean_script_forces_hook_and_cliffhanger():
    sc = S.clean_script(_good(), _events(), _frames())
    assert sc["segments"][0]["function"] == "hook" and sc["segments"][-1]["function"] == "cliffhanger"
    assert all(s["text"].endswith((".", "!", "?", "…")) for s in sc["segments"])


def test_plan_shots_v2_contiguous_faster_climax_and_unique_sources():
    sc = S.clean_script(_good(), _events(), _frames())
    secs = {s["id"]: [3.6] * len(recap.sentences_of(s["text"])) for s in sc["segments"]}
    vt = compose.voice_timeline(sc, secs, pauses=S.PAUSES)
    big = [(300, 40, 110, 130, 0.9)]
    scan = _scan([_smp(i * 0.2, big, 1.0 + (i % 5) * 0.1) for i in range(1100)])
    tl = fx.plan_shots_v2(sc, vt, _events(), 0.0, 200.0, lambda lo, hi: [], scan, 356)
    shots = tl["shots"]
    assert compose.check_plan(tl) == []
    assert abs(shots[0]["t0"]) < 1e-6 and abs(shots[-1]["t1"] - vt["total"]) < 1e-3
    st = fx.cut_stats_v2(tl)
    assert st["min"] >= 1.15 and st["max"] <= 3.45 and st["avg_last_third"] < st["avg_first_third"], st
    assert shots[0]["duration"] <= 2.0
    srcs = sorted((x["src_start"], x["src_end"]) for x in shots)
    assert all(b[0] >= a[1] - 1e-6 for a, b in zip(srcs, srcs[1:])) or tl["warnings"]
    assert all(0 <= x["crop_x0"] <= 854 - 356 for x in shots)


def test_accents_land_on_the_turns_and_the_break(tmp_path):
    sc = S.clean_script(_good(), _events(), _frames())
    secs = {s["id"]: [3.6] * len(recap.sentences_of(s["text"])) for s in sc["segments"]}
    vt = compose.voice_timeline(sc, secs, pauses=S.PAUSES)
    scan = _scan([_smp(i * 0.2, [(300, 40, 110, 130, 0.9)]) for i in range(1100)])
    tl = fx.plan_shots_v2(sc, vt, _events(), 0.0, 200.0, lambda lo, hi: [], scan, 356)
    for rel in [*fx.SFX_WHOOSH, *fx.SFX_IMPACT.values(), fx.SFX_RISER]:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(b"x")
    acc = fx.plan_accents(tl, sc, vt, str(tmp_path))
    kinds = [a["kind"] for a in acc]
    assert kinds.count("riser") == 1 and kinds.count("impact") == 2 and 1 <= kinds.count("whoosh") <= 7
    last_end = vt["total"] - compose.TAIL                                    # the last word ends before the tail
    assert abs([a for a in acc if a["kind"] == "riser"][0]["at"] - last_end) < 0.2
    assert all(str(tmp_path) in a["path"] for a in acc)                    # only the owner's pack folder
    assert fx.plan_accents(tl, sc, vt, str(tmp_path / "nope")) == []        # no pack → no accents, no crash


def test_v2_hook_check():
    ev, fr = _events(), _frames()
    d = _good()
    d["segments"][0]["text"] = " ".join(["слово"] * 24) + " я."
    assert any("hook слишком длинный" in x for x in S.story_defects(d, ev, fr, 0, 999, [], []))
    assert S.story_messages(ev, [], 0.0, 900.0, 300, 340)[1]["content"].count("КАДРЫ") == 0     # text-only call (no stills)


def test_story_is_a_separate_style():
    assert "story" not in recap.STYLES and set(recap.STYLES) == {"calm", "punchy"}
    from modes.trailer_cinematic import recap_runner
    with pytest.raises(SystemExit):
        recap_runner.main(["--input", "x", "--style", "nonsense"])
