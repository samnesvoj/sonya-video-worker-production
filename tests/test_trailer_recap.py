"""trailer_cinematic v2 (AI Story Recap): budget with images + LLM cache, visual analyst (local stage, speaking fusion,
vision schema), recap schemas (script / director / judge), timeline composer, deterministic validator pieces.
No network, no models; ffmpeg-based checks are skipped when ffmpeg is missing."""
import json
import os
import shutil
import subprocess
import wave
from types import SimpleNamespace

import numpy as np
import pytest

from modes.trailer_cinematic import budget, compose, recap, validator, visual
from modes.trailer_cinematic.budget import SchemaError

FFMPEG = os.environ.get("SONYA_FFMPEG") or shutil.which("ffmpeg")


# ── budget: images, cache ───────────────────────────────────────────────────

def test_image_parts_are_counted_as_tokens_not_base64_text():
    big = "A" * 400_000
    msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"},
                                         {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + big}}]}]
    assert budget.estimate_input_tokens(msgs) < budget.IMAGE_TOKENS + 200


def test_llm_cache_second_call_is_free(tmp_path):
    calls = []

    def chat(msgs, **kw):
        calls.append(1)
        return SimpleNamespace(text='{"a": 1}', telemetry=lambda: {"input_tokens": 10, "output_tokens": 5, "cost_usd": 0.01})
    llm = budget.LLM(chat, 1.0, "m", cache_dir=str(tmp_path))
    m = [{"role": "user", "content": "x"}]
    assert llm.ask("s", m, "v", lambda d: d) == {"a": 1}
    assert llm.ask("s", m, "v", lambda d: d) == {"a": 1}
    assert len(calls) == 1 and llm.spent == 0.01 and llm.summary()["cached_calls"] == 1


# ── visual analyst ──────────────────────────────────────────────────────────

W = lambda s, e: SimpleNamespace(start=s, end=e)        # noqa: E731


def test_speech_ratio_and_windows_never_cross_a_cut():
    assert visual.speech_ratio([W(0, 2)], 0, 4) > 0.4 and visual.speech_ratio([], 0, 4) == 0.0
    shots = [{"id": "C1", "start": 0, "end": 3}, {"id": "C2", "start": 3, "end": 40}]
    ws = visual.candidate_windows(shots, 40)
    assert all(w["start"] >= w["shot_start"] and w["end"] <= w["shot_end"] for w in ws)
    assert len([w for w in ws if w["shot_id"] == "C2"]) == 3


def test_select_candidates_drops_near_duplicates_and_prefers_story_events():
    c = lambda s, h: {"id": "x", "start": s, "end": s + 4, "dhash": h, "shot_id": "C", "shot_start": 0, "shot_end": 99}  # noqa: E731
    cands = [c(0, 0b1010), c(10, 0b1010), c(20, 0xFFFFFF00)]
    ev = [{"start": 18, "end": 30, "importance": 5}]
    out = visual.select_candidates(cands, ev, [], max_n=5)
    assert len(out) == 2 and [o["id"] for o in out] == ["V1", "V2"]
    assert any(o["start"] == 20 for o in out)


@pytest.mark.parametrize("mouths,verdict,ratio,state", [
    (["open_changing"] * 3, "likely_speaking", 0.8, "likely_speaking"),
    (["closed"] * 3, "likely_listening", 0.8, "likely_listening"),
    (["open_changing", "closed", "closed"], "likely_speaking", 0.8, "uncertain"),      # evidence conflicts
    (["open_changing"], "likely_speaking", 0.9, "uncertain"),                          # one frame never decides
    (["unclear"] * 3, "likely_speaking", 0.9, "uncertain"),
    (["open_changing"] * 3, "likely_speaking", 0.05, "not_speaking"),                  # ASR silence wins
])
def test_speaking_fusion_is_honest(mouths, verdict, ratio, state):
    assert visual.fuse_speaking(mouths, verdict, ratio)["state"] == state


def _vision_ok(ids):
    return {"windows": [{"id": i, "shot_type": "closeup", "kind": "reaction", "people": [
        {"who": "ведущий", "confidence": 0.8, "mouth": "closed"}], "action": "слушает", "visual_interest": 4,
        "on_screen_text": {"present": False}} for i in ids]}


def test_vision_schema_requires_every_window():
    out = visual.validate_vision(_vision_ok(["V1", "V2"]), ["V1", "V2"])
    assert out["V1"]["shot_type"] == "closeup" and out["V2"]["people"][0]["mouth"] == "closed"
    with pytest.raises(SchemaError):
        visual.validate_vision(_vision_ok(["V1"]), ["V1", "V2"])
    with pytest.raises(SchemaError):
        visual.validate_vision({"nope": 1}, ["V1"])


def test_ads_and_text_are_not_usable():
    c = {"id": "V1", "start": 0, "end": 4, "shot_id": "C1", "shot_start": 0, "shot_end": 9, "event_weight": 0,
         "speech_ratio": 0.0, "motion": 1, "sharpness": 1, "dhash": 1}
    v = visual.validate_vision(_vision_ok(["V1"]), ["V1"])["V1"]
    assert visual.finalize(c, v)["usable"] is True
    v["kind"] = "advertisement"
    assert visual.finalize(c, v)["usable"] is False
    v["kind"], v["on_screen_text"]["present"] = "main_interview", True
    assert visual.finalize(c, v)["usable"] is True                      # a neon sign in the room is set dressing
    v["on_screen_text"]["cut_by_vertical_center_crop"] = True
    assert visual.finalize(c, v)["usable"] is False                     # text the 9:16 crop would cut
    # the model said "not cut" for a caption across the 16:9 frame (observed «October 19 1987» → «ctober 19 198»)
    v["on_screen_text"] = {"present": True, "text": "October 19 1987", "cut_by_vertical_center_crop": False}
    assert visual.finalize(c, v)["usable"] is False
    v["on_screen_text"] = {"present": True, "text": "EXIT", "cut_by_vertical_center_crop": False}
    assert visual.finalize(c, v)["usable"] is True                      # a short sign fits the crop


# ── recap schemas ───────────────────────────────────────────────────────────

SM = {"events": [{"id": "E1.1"}, {"id": "E1.2"}]}


def _script(n=6, text="Он приехал в город и снял комнату."):
    return {"segments": [{"text": f"{text[:-1]} номер{'а' * i}.", "event_ids": ["E1.1"], "visual_intent": "герой в комнате",
                          "function": "development", "pause_after": "normal"} for i in range(n)]}


def test_script_validation():
    s = recap.validate_script(_script(), SM, 100, 400)
    assert s["segments"][0]["function"] == "hook" and len(s["segments"]) == 6
    with pytest.raises(SchemaError, match="digits"):
        d = _script()
        d["segments"][1]["text"] = "Прошло 40 лет."
        recap.validate_script(d, SM, 100, 400)
    with pytest.raises(SchemaError, match="outside"):
        recap.validate_script(_script(), SM, 1000, 2000)
    with pytest.raises(SchemaError, match="only ids"):
        d = _script()
        d["segments"][0]["event_ids"] = ["E9.9"]
        recap.validate_script(d, SM, 100, 400)
    with pytest.raises(SchemaError):
        recap.validate_script({"segments": _script()["segments"][:2]}, SM, 1, 900)


def _cand(i, speaking="not_speaking", who="Максим (Кишлак)", usable=True, ln=5.0):
    return {"id": i, "start": 10.0, "end": 10.0 + ln, "shot_id": "C1", "shot_start": 9.0, "shot_end": 40.0, "usable": usable,
            "speaking": {"state": speaking}, "people": [{"who": who, "confidence": 0.9, "mouth": "closed"}],
            "visual_interest": 4, "event_weight": 3, "shot_type": "closeup", "kind": "reaction", "action": "a", "setting": "",
            "good_for": [], "problems": []}


def test_director_rules():
    script = {"segments": [{"id": "S1", "function": "hook", "text": "т", "visual_intent": "", "event_ids": []}]}
    cands = {"V1": _cand("V1"), "V2": _cand("V2", who="ведущий"), "V3": _cand("V3", "likely_speaking"),
             "V4": _cand("V4", usable=False)}
    ok = {"segments": [{"segment_id": "S1", "shots": [{"candidate_id": "V1", "seconds": 4.0}]}]}
    assert recap.validate_director(ok, script, {"S1": 4.0}, cands, "Максим (Кишлак)")["segments"][0]["shots"][0]["seconds"] == 4.0
    bad = lambda sh, sec=4.0: recap.validate_director({"segments": [{"segment_id": "S1", "shots": sh}]}, script,    # noqa: E731
                                                       {"S1": sec}, cands, "Максим (Кишлак)")
    with pytest.raises(SchemaError, match="main character"):
        bad([{"candidate_id": "V2", "seconds": 4.0}])
    clamped = recap.validate_director({"segments": [{"segment_id": "S1", "shots": [{"candidate_id": "V3", "seconds": 4.0}]}]},
                                      script, {"S1": 3.0}, cands, "Максим (Кишлак)")       # a talker is capped, not shown long
    assert clamped["segments"][0]["shots"][0]["seconds"] == recap.shot_cap(cands["V3"]) == 2.5
    with pytest.raises(SchemaError, match="not usable"):
        bad([{"candidate_id": "V4", "seconds": 4.0}])
    with pytest.raises(SchemaError, match="unknown"):
        bad([{"candidate_id": "V9", "seconds": 4.0}])
    # a stray repeat (observed twice in a row on a 3-hour film) is dropped and reported instead of failing the run
    fixed = bad([{"candidate_id": "V1", "seconds": 2.0}, {"candidate_id": "V1", "seconds": 2.0}], 2.0)
    assert [x["candidate_id"] for x in fixed["segments"][0]["shots"]] == ["V1"] and "dropped" in fixed["reuse_fixes"][0]
    with pytest.raises(SchemaError, match="shots total"):
        bad([{"candidate_id": "V1", "seconds": 1.5}], 8.0)


def _verdict(**kw):
    v = {"scores": {k: 9.0 for k in recap.JUDGE_KEYS}, "factual_errors": [], "weak_beats": [], "clichés": [],
         "visual_problems": [], "rewrite_instructions": ""}
    v.update(kw)
    return v


def test_judge_rewrite_triggers_and_weak_beats_are_never_ignored():
    assert recap.needs_rewrite(_verdict()) == []
    assert recap.needs_rewrite(_verdict(scores=dict(_verdict()["scores"], hook=6)))
    assert recap.needs_rewrite(_verdict(factual_errors=["x"]))
    major = _verdict(weak_beats=[{"segment_id": "S3", "severity": "major", "problem": "повтор", "fix": ""}])
    assert any("S3" in r for r in recap.needs_rewrite(major))                  # all scores 9, still rewrites
    assert recap.needs_rewrite(_verdict(weak_beats=[{"segment_id": "S3", "severity": "minor", "problem": "", "fix": ""}])) == []
    assert recap.rewrite_scope(_verdict()) is None
    vis = _verdict(weak_beats=[{"segment_id": "S6", "severity": "major", "kind": "visual", "problem": "p", "fix": "f"}])
    assert recap.rewrite_scope(vis) == "visual"                                  # only the montage is redone
    assert recap.rewrite_scope(_verdict(scores=dict(_verdict()["scores"], factuality=7))) == "text"
    assert recap.rewrite_scope(_verdict(weak_beats=[dict(vis["weak_beats"][0], kind="text")])) == "text"
    parsed = recap.validate_judge({"scores": _verdict()["scores"], "factual_errors": [], "rewrite_instructions": "",
                                   "weak_beats": ["S2: слабый переход"]})
    assert parsed["weak_beats"][0]["segment_id"] == "S2"
    assert "duration_fit" not in parsed["scores"]                              # the LLM does not grade technical things


# ── composer ────────────────────────────────────────────────────────────────

def _vt(n=3, sec=(5.0, 4.0)):
    script = {"segments": [{"id": f"S{i + 1}", "text": "Первая фраза. Вторая фраза.", "pause_after": "normal"} for i in range(n)]}
    return script, compose.voice_timeline(script, {f"S{i + 1}": list(sec) for i in range(n)})


def test_voice_timeline_is_built_from_measured_seconds():
    _, vt = _vt()
    assert vt["voice_sec"] == 27.0 and vt["segments"][0]["t0"] == compose.LEAD_IN
    assert vt["total"] == compose.q(vt["voice_end"] + compose.TAIL)
    spans = compose.segment_spans(vt)
    assert spans["S1"][0] == 0.0 and spans["S3"][1] == vt["total"]
    assert abs(sum(b - a for a, b in spans.values()) - vt["total"]) < 0.1


def test_compose_is_contiguous_covers_voice_and_stays_inside_scene_shots():
    _, vt = _vt()
    cands = {f"V{i}": dict(_cand(f"V{i}"), shot_id=f"C{i}", start=100.0 * i, end=100.0 * i + 6, shot_start=100.0 * i - 1,
                           shot_end=100.0 * i + 30, event_weight=2) for i in range(1, 12)}
    plan = {"segments": [{"segment_id": f"S{i}", "shots": [{"candidate_id": f"V{2 * i - 1}", "seconds": 3.0},
                                                          {"candidate_id": f"V{2 * i}", "seconds": 3.0}],
                          "alternates": ["V9"]} for i in (1, 2, 3)]}
    tl = compose.compose(plan, cands, vt)
    assert compose.check_plan(tl) == [] and tl["shots"][0]["t0"] == 0 and tl["shots"][-1]["t1"] == vt["total"]
    for s in tl["shots"]:
        c = cands[s["candidate_id"]]
        assert c["shot_start"] < s["src_start"] and s["src_end"] <= c["shot_end"]
        assert compose.MIN_SHOT - 0.05 <= s["duration"] <= compose.MAX_SHOT + 0.05
        assert abs(s["t0"] * 30 - round(s["t0"] * 30)) < 0.01                  # frame grid


def test_compose_replaces_material_that_runs_out():
    _, vt = _vt(1, (6.0, 6.0))
    short = dict(_cand("V1", ln=2.0), shot_start=9.0, shot_end=12.5)          # only ~2.2 s of material in its scene shot
    spare = dict(_cand("V2"), shot_id="C2", shot_end=60.0, event_weight=1)
    plan = {"segments": [{"segment_id": "S1", "shots": [{"candidate_id": "V1", "seconds": 5.0}]}]}
    pool = {f"V{i}": dict(spare, id=f"V{i}", shot_id=f"C{i}") for i in range(2, 7)}
    tl = compose.compose(plan, dict(pool, V1=short), vt)
    assert all(s["candidate_id"] != "V1" or s["duration"] <= 2.3 for s in tl["shots"])
    assert tl["warnings"] and compose.check_plan(tl) == []


def test_cut_free_window_and_verification_of_hidden_cuts():
    assert compose.cut_free_window([], 0, 10, 2.0, 3.0) == (2.0, 5.0)
    w = compose.cut_free_window([4.0], 0, 10, 2.0, 3.0)               # a hidden cut at 4.0 → the window moves clear of it
    assert w is not None and not (w[0] - 0.12 < 4.0 < w[1] + 0.12)
    assert compose.cut_free_window([2.0, 4.0, 6.0, 8.0], 0, 10, 2.0, 3.0) is None
    tl = {"shots": [{"n": 1, "candidate_id": "V1", "src_start": 12.0, "src_end": 15.0, "duration": 3.0, "t0": 0, "t1": 3.0},
                    {"n": 2, "candidate_id": "V2", "src_start": 50.0, "src_end": 53.0, "duration": 3.0, "t0": 3.0, "t1": 6.0}]}
    cands = {"V1": _cand("V1"), "V2": _cand("V2"), "V3": dict(_cand("V3"), shot_id="C3", start=100.0, end=106.0,
                                                               shot_start=99.0, shot_end=140.0)}
    cuts = lambda a, b: [c for c in (13.5, 50.8, 51.5, 52.2) if a < c < b]           # V2 is full of cuts, V1 has one   # noqa: E731
    out = compose.verify_sources(tl, cands, cuts, 200.0, spare=["V3"])
    assert out["shots"][1]["candidate_id"] == "V3" and out["shots"][0]["candidate_id"] == "V1"
    s0, e0 = out["shots"][0]["src_start"], out["shots"][0]["src_end"]
    assert not (s0 < 13.5 < e0) and e0 - s0 == 3.0 and out["warnings"]
    with pytest.raises(RuntimeError):
        compose.verify_sources({"shots": [{"n": 2, "candidate_id": "V2", "src_start": 50.0, "src_end": 53.0, "duration": 3.0,
                                           "t0": 0.0, "t1": 3.0}]}, cands, lambda a, b: [x / 5 for x in range(int(a * 5), int(b * 5)) if x / 5 > a],
                               200.0, spare=[])


def test_write_voice_places_sentences_and_stability(tmp_path):
    script, vt = _vt(1, (1.0, 1.0))
    wavs = {}
    for j in range(2):
        p = str(tmp_path / f"s{j}.wav")
        with wave.open(p, "wb") as w:
            w.setnchannels(1), w.setsampwidth(2), w.setframerate(48000)
            w.writeframes((np.ones(48000) * 8000).astype("<i2").tobytes())
        wavs[("S1", j)] = p
    out = compose.write_voice(vt, wavs, str(tmp_path / "v.wav"))
    with wave.open(out, "rb") as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    assert abs(x[int(0.25 * 48000)]) == 0 and x[int(1.0 * 48000)] > 7000
    assert compose.stability(vt, out)["spread_db"] < 0.5


# ── validator ───────────────────────────────────────────────────────────────

def test_duration_check_range():
    assert validator.check_duration(60.0)["ok"] and validator.check_duration(54.0)["ok"] and validator.check_duration(66.0)["ok"]
    assert not validator.check_duration(52.7)["ok"] and not validator.check_duration(70)["ok"]


def test_ncc_finds_lag_and_rejects_noise():
    rng = np.random.default_rng(0)
    a = rng.standard_normal(16000)
    v, lag = validator.ncc(a[100:], a[:-100], max_lag=200)
    assert v > 0.99 and abs(lag) == 100
    assert abs(validator.ncc(a, rng.standard_normal(16000), max_lag=0)[0]) < 0.1


@pytest.mark.skipif(not FFMPEG, reason="ffmpeg not available")
def test_no_original_audio_check_catches_a_mixed_in_source(tmp_path, monkeypatch):
    """Positive: narrator-only audio passes. Negative: the same video with the SOURCE sound mixed in at −14 dB fails."""
    monkeypatch.setenv("SONYA_FFMPEG", FFMPEG)
    sr = 16000
    rng = np.random.default_rng(1)
    t = np.arange(sr * 4) / sr
    voice = (np.sin(2 * np.pi * 180 * t) * (np.sin(2 * np.pi * 3 * t) > 0) * 0.3).astype("float32")
    voice[: sr // 2] = 0
    voice[-sr // 2:] = 0
    src_audio = (rng.standard_normal(sr * 4) * 0.2).astype("float32")

    def wav(path, x):
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1), w.setsampwidth(2), w.setframerate(sr)
            w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    wav(tmp_path / "voice.wav", voice)
    wav(tmp_path / "src.wav", src_audio)
    wav(tmp_path / "mixed.wav", voice + src_audio * 0.2)

    def mk(audio, name):
        out = str(tmp_path / name)
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=gray:s=64x64:d=4:r=30", "-i", str(audio),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", out], check=True)
        return out
    good, bad = mk(tmp_path / "voice.wav", "good.mp4"), mk(tmp_path / "mixed.wav", "bad.mp4")
    srcmp4 = mk(tmp_path / "src.wav", "src.mp4")
    vt = {"segments": [{"id": "S1", "sentences": [{"text": "x", "t0": 0.5, "t1": 3.5}]}], "total": 4.0, "voice_end": 3.5}
    shots = [{"n": 1, "t0": 0.0, "t1": 4.0, "src_start": 0.0, "src_end": 4.0}]
    run = lambda f: validator.check_no_original_audio(f, str(tmp_path / "voice.wav"), 0.0, vt, [], ["narration"], srcmp4, shots)  # noqa: E731
    assert run(good)["ok"], run(good)
    r = run(bad)
    assert not r["ok"] and "residual" in r["detail"] or "correlates" in r["detail"]
    assert not validator.check_no_original_audio(good, str(tmp_path / "voice.wav"), 0.0, vt, [srcmp4], ["narration"], srcmp4,
                                                 shots)["ok"]                    # an input shot WITH audio is a failure


def test_recap_is_cli_only_and_v1_untouched():
    from scripts.mode_registry import list_modes
    assert "trailer_cinematic" not in list_modes()
    from modes.trailer_cinematic import runner, story
    assert story.DIRECTOR_VERSION == "trailer-director-v1" and callable(runner.run)


@pytest.mark.skipif(not FFMPEG, reason="ffmpeg missing")
def test_assemble_recap_shot_frames_makes_every_shot_frame_exact(tmp_path, monkeypatch):
    """25 fps source clips used to lose a frame each → cuts drifted earlier than the plan. With shot_frames the output
    has exactly the planned number of 30 fps frames."""
    from modes.trailer_cinematic import media
    monkeypatch.setattr(media, "ffmpeg_bin", lambda: FFMPEG)
    monkeypatch.setattr(media, "_env_encoder_args", lambda: None)
    files = []
    for i, n25 in enumerate((46, 54, 62)):                       # 1.84 s, 2.16 s, 2.48 s at 25 fps (all a bit short)
        f = str(tmp_path / f"s{i}.mp4")
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=96x160:rate=25",
                        "-frames:v", str(n25), "-pix_fmt", "yuv420p", f], check=True)
        files.append(f)
    voice = str(tmp_path / "v.wav")
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono", "-t", "7", voice], check=True)
    want = [55, 66, 75]                                          # 1.833 s, 2.2 s, 2.5 s on the 30 fps grid
    out = str(tmp_path / "o.mp4")
    media.assemble_recap(files, voice, 0.0, 196 / 30, None, out, fade_out=0.1, shot_frames=want)
    r = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                        "stream=nb_read_frames", "-of", "csv=p=0", out], capture_output=True, text=True)
    assert int(r.stdout.strip()) == sum(want)


def test_prefilter_spreads_over_a_long_film():
    """3-hour input: the quota must not be filled from the first minutes (all events equally important)."""
    dur = 10800.0
    raw = [{"shot_id": i, "start": i * 4.0, "end": i * 4.0 + 3.0, "shot_start": i * 4.0, "shot_end": i * 4.0 + 4.0}
           for i in range(int(dur // 4))]
    events = [{"start": 0, "end": dur, "importance": 5}]
    pre = visual.prefilter(raw, events, dur, 168)
    assert len(pre) == 168
    starts = [c["start"] for c in pre]
    assert starts == sorted(starts)
    for k in range(18):                                   # every 10-minute part of the film is represented
        assert any(k * 600 <= s < (k + 1) * 600 for s in starts), k
    assert max(starts) > dur * 0.9


def test_prefilter_prefers_important_events_inside_a_bucket():
    dur = 1200.0
    raw = [{"shot_id": i, "start": i * 10.0, "end": i * 10.0 + 5.0} for i in range(120)]
    events = [{"start": 300, "end": 330, "importance": 5}]
    pre = visual.prefilter(raw, events, dur, 6)
    assert any(300 <= c["start"] < 330 for c in pre)
    assert len(visual.prefilter(raw[:4], events, dur, 6)) == 4


def test_prepare_candidates_skips_unreadable_window(tmp_path, monkeypatch):
    """A damaged frame in a long film must drop that window, not abort the whole run."""
    def fake(src, t, out, width=640):
        if t > 100:
            raise RuntimeError("decode error")
        open(out, "wb").write(b"x")
        return out
    monkeypatch.setattr(visual, "extract_frame", fake)
    monkeypatch.setattr(visual, "local_features", lambda ps: {"dhash": 1, "motion": 0.0, "sharpness": 1.0, "brightness": 1.0})
    cands = [{"start": 10.0, "end": 14.0}, {"start": 120.0, "end": 124.0}]
    got = visual.prepare_candidates("x.mp4", cands, tmp_path, "abc", workers=1)
    assert [c["start"] for c in got] == [10.0]


def test_extract_frame_retries_other_seek(tmp_path, monkeypatch):
    calls = []
    def fake_run(cmd, **kw):
        calls.append(cmd)
        ok = len(calls) == 3
        if ok:
            open(cmd[-1], "wb").write(b"jpg")
        return SimpleNamespace(returncode=0 if ok else 1, stderr="boom")
    monkeypatch.setattr(visual.subprocess, "run", fake_run)
    out = visual.extract_frame("x.mp4", 50.0, str(tmp_path / "f.jpg"))
    assert out.endswith("f.jpg") and len(calls) == 3


def test_ask_writer_falls_back_to_loose_length():
    """The model misses the strict length window twice → one more ask with a loose length check, not a dead run."""
    from modes.trailer_cinematic import recap_runner
    sm = {"events": [{"id": f"E{i}", "start": 0.0, "end": 5.0, "summary": "s", "participants": [], "emotion": "n", "importance": 3}
                     for i in range(1, 6)], "characters": []}
    seg = lambda n: {"text": ("Слово " * n).strip() + ".", "event_ids": ["E1", "E2", "E3", "E4"], "visual_intent": "x"}  # noqa: E731
    data = {"title": "t", "segments": [dict(seg(30), text="Раз" + "абвгде"[i] + " " + seg(30)["text"]) for i in range(6)]}
    steps = []

    class FakeLLM:
        def ask(self, step, msgs, ver, validate, **kw):
            steps.append(step)
            try:
                return validate(data)
            except SchemaError as e:
                raise RuntimeError(str(e))
    out = recap_runner.ask_writer(FakeLLM(), sm, 60, 700, 800, {"chars_per_sec": 15.0})
    assert steps == ["writer", "writer_loose"] and len(out["segments"]) == 6


def test_compose_fills_from_uncertain_windows_when_no_silent_ones_exist():
    """A long film is mostly talking heads: with no silent/listening filler the composer must still finish
    (uncertain windows are used after silent ones; probably-speaking ones only as the last resort for a short shot)."""
    _, vt = _vt(1, (6.0, 6.0))
    short = dict(_cand("V1", ln=2.0), shot_start=9.0, shot_end=12.5)
    cands = {"V1": short}
    for i in range(2, 8):
        cands[f"V{i}"] = dict(_cand(f"V{i}", "uncertain"), shot_id=f"C{i}", shot_end=60.0)
    plan = {"segments": [{"segment_id": "S1", "shots": [{"candidate_id": "V1", "seconds": 5.0}]}]}
    tl = compose.compose(plan, cands, vt)
    assert compose.check_plan(tl) == [] and any(s["candidate_id"] != "V1" for s in tl["shots"])


def test_director_shortfall_ok_only_when_every_shot_is_at_its_cap():
    """Talking heads are capped at 2.5 s, so 4 of them can never fill a 9.6 s segment: the composer fills the rest."""
    script = {"segments": [{"id": "S1", "function": "hook", "text": "т", "visual_intent": "", "event_ids": []}]}
    cands = {f"V{i}": _cand(f"V{i}", "likely_speaking") for i in range(1, 5)}
    cands["V5"] = _cand("V5")
    mk = lambda sh: {"segments": [{"segment_id": "S1", "shots": sh}]}                 # noqa: E731
    hero = "Максим (Кишлак)"
    capped = [{"candidate_id": f"V{i}", "seconds": 5.0, "reuse_ok": True} for i in range(1, 4)]
    assert recap.validate_director(mk(capped), script, {"S1": 12.0}, cands, hero)             # 3 × 2.5 = 7.5 < 8.4, all capped
    with pytest.raises(SchemaError, match="shots total"):
        recap.validate_director(mk([{"candidate_id": "V5", "seconds": 2.0}]), script, {"S1": 12.0}, cands, hero)  # could be longer


def test_verify_sources_splits_a_slot_when_no_clean_long_window_exists():
    """Fast-cut film: no clean 4.6 s stretch, but two clean 2.3 s ones exist → two shots, slot boundaries unchanged."""
    sh = {"n": 1, "segment_id": "S1", "candidate_id": "V1", "src_start": 100.0, "src_end": 104.6, "duration": 4.6,
          "t0": 10.0, "t1": 14.6, "why": "", "fallback": False, "scene_shot": "C1", "speaking": "uncertain"}
    cands = {"V1": dict(_cand("V1", "uncertain"), start=100.0, end=104.5, shot_start=99.0, shot_end=108.0),
             "V2": dict(_cand("V2", "uncertain"), shot_id="C2", start=200.0, end=204.5, shot_start=199.0, shot_end=208.0)}
    cuts = lambda a, b: [c for c in (102.4, 202.4) if a < c < b]                                # a cut every ~2.4 s   # noqa: E731
    tl = compose.verify_sources({"shots": [sh], "total": 14.6, "warnings": []}, cands, cuts, 400.0, spare=["V2"])
    assert len(tl["shots"]) == 2 and [x["n"] for x in tl["shots"]] == [1, 2]
    assert tl["shots"][0]["t0"] == 10.0 and tl["shots"][1]["t1"] == 14.6 and tl["shots"][0]["t1"] == tl["shots"][1]["t0"]
    assert abs(sum(x["duration"] for x in tl["shots"]) - 4.6) < 1e-6
    assert all(compose.MIN_SHOT - 1e-6 <= x["duration"] for x in tl["shots"])
    for x in tl["shots"]:
        assert not any(x["src_start"] < c < x["src_end"] for c in (102.4, 202.4))


def test_unplanned_cuts_need_a_hard_scene_score():
    """Fast motion (confetti, camera whip) scores 0.15–0.35 and is not a cut; a real cut scores high."""
    txt = ("frame:1 pts:100 pts_time:16.73\nlavfi.scene_score=0.221\nframe:2 pts:200 pts_time:17.13\nlavfi.scene_score=0.302\n"
           "frame:3 pts:300 pts_time:20.37\nlavfi.scene_score=0.612\n")
    sc = validator.parse_scene_scores(txt)
    assert sc == [(16.73, 0.221), (17.13, 0.302), (20.37, 0.612)]
    assert [t for t, s in sc if s >= validator.HARD_CUT] == [20.37]


def test_hero_needs_a_detected_face_when_the_detector_ran():
    """The vision model named the hero on a plain elevator door; the local face detector vetoes it."""
    card = {"people_raw": [{"who": "Джордан Белфорд", "confidence": 0.9}], "faces": 0}
    assert recap.hero_visible(card, "Джордан Белфорд") is False
    assert recap.hero_visible(dict(card, faces=1), "Джордан Белфорд") is True
    assert recap.hero_visible({"people_raw": card["people_raw"]}, "Джордан Белфорд") is True      # no detector data → as before


@pytest.mark.skipif(not os.path.isfile(os.path.join(os.path.dirname(__import__("cv2").__file__), "data",
                                                    "haarcascade_frontalface_default.xml")), reason="no cascade")
def test_count_faces_is_zero_on_a_plain_frame():
    assert visual.count_faces([np.full((360, 640), 90, dtype=np.uint8)]) == 0
