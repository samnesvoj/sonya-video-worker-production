"""Trailer production (scripts/shared/trailer): no-voice narration timing, timeline, script flow with a fake Gemini,
edit plan on a synthetic scan, runner dispatch. No network, no models, no video."""
import json

import pytest

from scripts.shared.trailer import edit, narration, pipeline, script
from scripts.shared.trailer.shots import Scan


# ── narration (the voice slot, subtitles only) ─────────────────────────────────────────────────────────────────

def test_reading_pace_and_monotonic_words():
    txt = "Я вернулся домой после службы. Мечтал об одном, а нашёл пустоту и чужих людей."
    r = narration.segment_timing(txt)
    ws = r["words"]
    assert [w for w, _, _ in ws][:3] == ["Я", "вернулся", "домой"]
    assert all(a < b for _, a, b in ws)
    assert all(ws[i][2] <= ws[i + 1][1] for i in range(len(ws) - 1))
    pace = len(ws) / r["sec"]
    assert 2.3 <= pace <= narration.READ_PACE + 0.05
    # the sentence end gets a beat: the gap after «службы.» is the biggest
    gaps = [ws[i + 1][1] - ws[i][2] for i in range(len(ws) - 1)]
    assert gaps.index(max(gaps)) == 4
    assert r["wav"] is None


def test_reading_short_segment_floor_and_shape():
    r = narration.reading([{"n": 1, "text": "Я сорвал шлем."}, {"n": 2, "text": ""}])
    assert r["segments"][1]["sec"] >= narration.MIN_SEG - 1e-6
    assert r["segments"][2]["words"] == []
    assert r["tts"]["engine"] == "none"


def test_timeline_from_reading_has_hook_beat_and_end_card():
    segs = [{"n": 1, "fn": "hook", "text": "Я сам посадил себя в камеру и выкинул ключ."},
            {"n": 2, "fn": "origin", "text": "Всё началось с одного звонка. Мне предложили работу мечты."},
            {"n": 3, "fn": "cliffhanger", "text": "И тут дверь открылась."}]
    v = narration.reading(segs)["segments"]
    tl = edit.timeline(segs, v)
    s = tl["segments"]
    assert s[1]["t0"] - s[0]["t1"] == pytest.approx(edit.HOOK_BEAT, abs=0.04)
    assert s[2]["t0"] - s[1]["t1"] == pytest.approx(edit.GAP, abs=0.04)
    assert tl["total"] == pytest.approx(tl["end_card"][0] + edit.END_CARD, abs=0.04)
    assert all(seg["wav"] is None for seg in s)
    # word times are absolute and inside their segment
    for seg in s:
        assert all(seg["t0"] - 1e-3 <= a < b <= seg["t1"] + 1e-3 for _, a, b in seg["words"])


def test_voice_mode_default_off():
    assert pipeline.voice_mode({}) == "off"
    assert pipeline.voice_mode({"SONYA_TRAILER_VOICE": "qwen"}) == "qwen"
    assert pipeline.voice_mode({"SONYA_TRAILER_VOICE": "garbage"}) == "off"
    assert pipeline.enabled({}) and not pipeline.enabled({"SONYA_TRAILER": "off"})


def test_voice_choice():
    assert pipeline.voice_choice({}, "f") == "female"
    assert pipeline.voice_choice({"settings": {"trailerVoice": "male"}}, "f") == "male"


# ── script flow with a fake Gemini ─────────────────────────────────────────────────────────────────────────────

def _words(n_sent=80, per=7, step=0.4):
    out, t = [], 0.0
    for i in range(n_sent):
        for k in range(per):
            w = f"слово{i}x{k}" + ("." if k == per - 1 else "")
            out.append({"word": w, "start": round(t, 2), "end": round(t + step * 0.8, 2)})
            t += step
        t += 1.0
    return out


class _Res:
    def __init__(self, text):
        self.text, self.input_tokens, self.output_tokens, self.model, self.latency_ms = text, 3000, 800, "fake", 1


def _fake_chat(seen):
    def chat(msgs, prompt_version=None, **kw):
        seen.append(prompt_version)
        if prompt_version == script.PICK_VERSION:
            return _Res(json.dumps({"start": 10, "end": 60, "hero": "парень", "hero_gender": "m", "line": "линия",
                                    "hook": 40, "peak": 58, "names": ["Вася"], "genre": "драма", "why": "x"}))
        if prompt_version in (script.WRITE_VERSION, script.FIX_VERSION):
            segs = [{"n": 1, "fn": "hook", "text": "Я сам посадил себя в камеру и выкинул ключ.", "src": [40], "shots": ["K1"]}]
            for i in range(2, 13):
                segs.append({"n": i, "fn": "rising", "text": ("Я пошёл туда снова. Меня там ждали трое. Мой кореш "
                                                               "молчал и смотрел в пол. Я понял, что попал."),
                             "src": [10 + i * 4], "shots": ["K2"]})
            segs.append({"n": 13, "fn": "cliffhanger", "text": "И тут Вася открыл дверь.", "src": [58], "shots": []})
            return _Res(json.dumps({"title": "Он вернулся не туда", "segments": segs, "shorts": [[1, 2, 3, 13], [4, 5, 13]],
                                    "short_titles": ["А", "Б"]}, ensure_ascii=False))
        if prompt_version == script.CHECK_VERSION:
            return _Res(json.dumps({"bad": [{"n": 2, "why": "нет в источнике", "fixed": "Я пошёл туда снова. Меня ждали."}]},
                                   ensure_ascii=False))
        raise AssertionError(prompt_version)
    return chat


def test_build_script_flow_budget_names_and_fact_fix():
    seen = []
    r = script.build_script(_words(), 400.0, frames_fn=lambda a, b: (["K1", "K2"], []), chat=_fake_chat(seen),
                            log=lambda m: None)
    assert seen[0] == script.PICK_VERSION and script.CHECK_VERSION in seen
    sc = r["script"]
    assert sc["segments"][0]["fn"] == "hook" and sc["segments"][-1]["fn"] == "cliffhanger"
    assert "Вася" not in sc["segments"][-1]["text"]            # names never ship
    assert sc["segments"][1].get("fact_fixed")
    assert r["llm"]["rub"] <= 1.5
    assert r["episode"][0] < r["episode"][1]


# ── edit plan on a synthetic scan ──────────────────────────────────────────────────────────────────────────────

def _scan(a=0.0, b=120.0, cut_every=4.0):
    samples, t, k = [], a, 0
    while t < b:
        cut = k > 0 and abs((t - a) % cut_every) < 0.1
        samples.append({"t": round(t, 2), "motion": 0.3 + 0.1 * (k % 3), "cut": cut,
                        "faces": [[800, 200, 300, 360, 0.9]] if (k // 10) % 2 == 0 else []})
        t += 0.2
        k += 1
    return Scan({"meta": {"width": 1920, "height": 1080, "fps": 5.0, "version": "x", "start": a, "end": b},
                 "samples": samples})


def test_plan_covers_timeline_without_reuse_or_cut_crossing():
    segs = [{"n": i, "fn": "hook" if i == 1 else ("cliffhanger" if i == 6 else "rising"),
             "text": "Я пошёл туда снова. Меня там ждали трое.", "src": [i], "shots": []} for i in range(1, 7)]
    tl = edit.timeline(segs, narration.reading(segs)["segments"])
    sc = _scan()
    by_id = {i: {"id": i, "start": 10.0 * i, "end": 10.0 * i + 3} for i in range(1, 7)}
    p = edit.plan(tl, sc, by_id, {}, (0.0, 120.0), 810, hook_t=60.0)
    shots = p["shots"]
    assert shots[0]["t0"] == 0.0 and shots[-1]["t1"] == pytest.approx(tl["end_card"][0], abs=1e-3)
    assert all(abs(x["t1"] - y["t0"]) < 1e-3 for x, y in zip(shots, shots[1:]))
    for x in shots:
        assert not sc.cuts_in(x["src_start"] + 0.01, x["src_end"] - 0.01)
        assert 0.9 <= x["t1"] - x["t0"] <= 3.2
    spans = sorted((x["src_start"], x["src_end"]) for x in shots)
    assert all(b0 <= a1 for (_, b0), (a1, _) in zip(spans, spans[1:]))


def test_sub_timeline_keeps_segment_order_and_fills():
    segs = [{"n": i, "fn": "rising", "text": "Я пошёл туда снова. Меня там ждали трое.", "src": [i], "shots": []}
            for i in range(1, 7)]
    tl = edit.timeline(segs, narration.reading(segs)["segments"])
    sc = _scan()
    by_id = {i: {"id": i, "start": 10.0 * i, "end": 10.0 * i + 3} for i in range(1, 7)}
    p = edit.plan(tl, sc, by_id, {}, (0.0, 120.0), 810)
    sub, sh = edit.sub_timeline(tl, [5, 2, 6], p["shots"])
    assert [s["n"] for s in sub["segments"]] == [5, 2, 6]
    assert sh[0]["t0"] == 0.0 and sh[-1]["t1"] == pytest.approx(sub["end_card"][0], abs=1e-3)


# ── worker dispatch ────────────────────────────────────────────────────────────────────────────────────────────

def test_runner_refuses_when_switched_off(monkeypatch, tmp_path):
    from modes.virality import runner
    monkeypatch.setenv("SONYA_TRAILER", "off")
    with pytest.raises(RuntimeError, match="TRAILER_DISABLED"):
        runner._trailer("x.mp4", {"word_timestamps": [{"word": "a", "start": 0, "end": 1}]}, {}, tmp_path / "out")


# ── subtitles never outside the frame, without shrinking everything ────────────────────────────────────────────

def test_fit_ass_shrinks_only_the_too_wide_cues():
    from scripts.shared.trailer import render
    segs = [{"n": 1, "fn": "hook", "text": "Мои кореша постоянно упарывались всякой дрянью."},
            {"n": 2, "fn": "cliffhanger", "text": "Я пришёл один на их поляну."}]
    tl = edit.timeline(segs, narration.reading(segs)["segments"])
    txt, info = render.fit_ass(render.ass(tl, "Он вернулся не туда", "highlight", "dela"))
    assert info["widest_px"] <= render.SAFE_W
    cues = [l for l in txt.splitlines() if l.startswith("Dialogue: 0")]
    plain = [c for c in cues if "\\fs" not in c.split(",", 9)[9].replace("\\fscx", "").replace("\\fscy", "")]
    assert plain, "short lines keep the style size"
    for row, _ in render.ass_widths(txt):
        assert len(row.replace("\\N", "")) <= 2 * render.SUB_MAX_CHARS + 12
