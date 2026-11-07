"""trailer_cinematic v1: JSON schemas (Analyst / Director / Writer / Judge), boundary snapping, mix plan,
budget guard. No network, no ffmpeg."""
import numpy as np
import pytest

from modes.trailer_cinematic import budget, story, timeline as tl


def _words():
    # S1 "Привет всем." 1.0–2.0 ; S2 "Он ушёл из дома и не вернулся." 5.0–8.0 ; S3 "Почему?" 12.0–12.6
    ws = [("Привет", 1.0, 1.4), ("всем.", 1.5, 2.0),
          ("Он", 5.0, 5.2), ("ушёл", 5.3, 5.7), ("из", 5.8, 5.9), ("дома", 6.0, 6.4), ("и", 6.5, 6.6),
          ("не", 6.7, 6.9), ("вернулся.", 7.0, 8.0), ("Почему?", 12.0, 12.6)]
    return [{"word": w, "start": s, "end": e} for w, s, e in ws]


def _beat(**kw):
    b = {"beat_id": "B1", "event_ids": ["E1.1"], "story_function": "hook", "start_candidate": 0.0,
         "end_candidate": 4.0, "audio_owner": "narration", "emotion": "suspense", "spoiler_level": 0.1,
         "priority": 3}
    b.update(kw)
    return b


def _frag(**kw):
    return tl.beat_to_fragment(_beat(**kw), tl.sentences(_words())[1])


def _event(i, s, e, **kw):
    d = {"id": f"x{i}", "start": s, "end": e, "summary": f"событие {i}", "participants": ["Он"],
         "importance": 3, "evidence": ["Он ушёл из дома"], "emotion": "tension"}
    d.update(kw)
    return d


CHUNK = {"index": 0, "total": 1, "start": 0.0, "end": 100.0, "sentences": [], "shots": []}


def _map_ok():
    return {"characters": [{"name": "Он", "description": "герой"}], "premise": "p", "conflict": "c",
            "events": [_event(1, 5, 8), _event(2, 20, 30), _event(3, 40, 50)], "open_questions": ["почему?"]}


# ── schemas ─────────────────────────────────────────────────────────────────

def test_analyst_map_schema_valid_and_reids_events():
    part = story.validate_map(_map_ok(), CHUNK)
    assert [e["id"] for e in part["events"]] == ["E1.1", "E1.2", "E1.3"]
    assert set(part["events"][0]) == {"id", "start", "end", "summary", "participants", "importance", "evidence",
                                      "emotion"}


@pytest.mark.parametrize("mutate", [
    lambda d: d.pop("events"),
    lambda d: d.update(events=[]),
    lambda d: d["events"][0].pop("evidence"),
    lambda d: d["events"][0].update(evidence=[]),
    lambda d: d["events"][0].update(emotion="joyful"),
    lambda d: d["events"][0].update(start=500, end=600),          # outside the chunk
    lambda d: d["events"][0].update(start="abc"),
])
def test_analyst_map_schema_rejects(mutate):
    d = _map_ok()
    mutate(d)
    with pytest.raises(budget.SchemaError):
        story.validate_map(d, CHUNK)


def _parts():
    return [story.validate_map(_map_ok(), CHUNK)]


def _sm_ok():
    return {"characters": [{"name": "Он", "description": "d"}], "premise": "p", "conflict": "c",
            "event_ids": ["E1.3", "E1.1", "E1.2", "E9.9"], "open_questions": [], "ending": "e",
            "spoiler_sensitive_events": ["E1.3", "zz"]}


def test_story_map_keeps_only_known_events_in_time_order():
    sm = story.validate_story_map(_sm_ok(), _parts())
    assert [e["id"] for e in sm["events"]] == ["E1.1", "E1.2", "E1.3"]
    assert sm["spoiler_sensitive_events"] == ["E1.3"]
    for k in ("characters", "premise", "conflict", "events", "open_questions", "ending", "spoiler_sensitive_events"):
        assert k in sm
    bad = dict(_sm_ok(), event_ids=["E9.9"])
    with pytest.raises(budget.SchemaError):
        story.validate_story_map(bad, _parts())


def _sm():
    return story.validate_story_map(_sm_ok(), _parts())


def test_director_beats_schema():
    sm = _sm()
    ok = {"beats": [_beat(beat_id="B1", event_ids=["E1.1"], start_candidate=5, end_candidate=8),
                    _beat(beat_id="B4", event_ids=["E1.2"], start_candidate=27, end_candidate=30),
                    _beat(beat_id="B2", event_ids=["E1.2"], start_candidate=21, end_candidate=26,
                          audio_owner="dialogue", story_function="escalation"),
                    _beat(beat_id="B3", event_ids=["E1.3"], start_candidate=41, end_candidate=45,
                          story_function="tease", emotion="dread")]}
    beats = story.validate_beats(ok, sm, 100.0)
    assert [b["beat_id"] for b in beats] == ["B1", "B4", "B2", "B3"]
    dialogue_only = {"beats": [dict(b, audio_owner="dialogue") for b in ok["beats"]]}
    with pytest.raises(budget.SchemaError, match="narration"):
        story.validate_beats(dialogue_only, sm, 100.0)
    for bad in (dict(event_ids=["E7"]), dict(story_function="intro"), dict(audio_owner="music"),
                dict(start_candidate=80, end_candidate=90),             # not inside its event
                dict(start_candidate=8, end_candidate=5)):
        d = {"beats": [dict(ok["beats"][0], **bad)] + ok["beats"][1:]}
        with pytest.raises(budget.SchemaError):
            story.validate_beats(d, sm, 100.0)


def test_writer_phrases_only_for_voiced_beats_and_known_events():
    sm = _sm()
    beats = [dict(_beat(beat_id="B1"), voice_room_sec=3.0), dict(_beat(beat_id="B2", audio_owner="dialogue"))]
    ok = {"phrases": [{"text": "— «Он не вернулся.»", "beat_id": "B1", "event_ids": ["E1.1"], "target_seconds": 2.5}]}
    ph = story.validate_phrases(ok, beats, sm)
    assert ph[0]["text"] == "Он не вернулся."
    for bad in ({"beat_id": "B2"}, {"event_ids": ["E5.5"]}, {"event_ids": []}, {"text": ""}):
        d = {"phrases": [dict(ok["phrases"][0], **bad)]}
        with pytest.raises(budget.SchemaError):
            story.validate_phrases(d, beats, sm)


def test_writer_prompt_has_no_transcript_and_no_ending():
    sm = dict(_sm(), ending="СЕКРЕТНЫЙ ФИНАЛ")
    sm["events"][0] = dict(sm["events"][0], evidence=["ДОСЛОВНАЯ ЦИТАТА"])
    m = story.writer_messages(sm, [dict(_beat(), voice_room_sec=3, max_chars=40)], 60)
    text = m[1]["content"]
    assert "СЕКРЕТНЫЙ ФИНАЛ" not in text and "ДОСЛОВНАЯ ЦИТАТА" not in text


def test_judge_schema_and_rewrite_thresholds():
    good = {"scores": {k: 9 for k in story.JUDGE_KEYS}, "factual_errors": [], "clichés": ["x"],
            "weak_beats": [], "rewrite_instructions": ""}
    v = story.validate_judge(good)
    assert v["clichés"] == ["x"] and not story.needs_rewrite(v)
    assert story.needs_rewrite(story.validate_judge(dict(good, scores=dict(good["scores"], factuality=8.5))))
    assert story.needs_rewrite(story.validate_judge(dict(good, scores=dict(good["scores"], coherence=7))))
    assert story.needs_rewrite(story.validate_judge(dict(good, scores=dict(good["scores"], hook=7.9))))
    assert story.needs_rewrite(story.validate_judge(dict(good, scores=dict(good["scores"], suspense=6))))
    assert not story.needs_rewrite(story.validate_judge(dict(good, scores=dict(good["scores"], suspense=7))))
    with pytest.raises(budget.SchemaError):
        story.validate_judge({"scores": {"hook": 9}, "factual_errors": [], "rewrite_instructions": ""})


def test_memory_is_bounded_and_carries_no_transcript():
    mem = {}
    part = story.validate_map(_map_ok(), CHUNK)
    for _ in range(100):
        mem = story.update_memory(mem, part)
    import json
    assert len(json.dumps(mem, ensure_ascii=False)) <= story.MEMORY_CHARS
    m = story.map_messages(dict(CHUNK, sentences=tl.sentences(_words())[1]), mem, "ru")
    assert "[5.0-8.0] Он ушёл из дома и не вернулся." in m[1]["content"]
    assert isinstance(m[1]["content"], str)                       # text only: no frames in v1


# ── boundaries ──────────────────────────────────────────────────────────────

def test_dialogue_beat_snaps_to_whole_sentence():
    words, sents = tl.sentences(_words())
    sid = {s["id"]: s for s in sents}
    f = tl.refine(_frag(audio_owner="dialogue", start_candidate=5.5, end_candidate=7.5), words, sid, [], 20.0)
    assert f["lines"] == ["S2"]
    assert f["render_start"] <= 5.0 and f["render_end"] >= 8.0
    assert f["render_start"] > 2.0 and f["render_end"] < 12.0
    assert not f["cuts_inside_word"]


def test_dialogue_beat_without_speech_becomes_narration():
    f = _frag(audio_owner="dialogue", start_candidate=14.0, end_candidate=18.0)
    assert f["audio"] == "narration" and f["downgraded"]


def test_narration_never_cuts_inside_a_word_and_snaps_to_cut():
    words, sents = tl.sentences(_words())
    sid = {s["id"]: s for s in sents}
    f = tl.refine(_frag(start_candidate=5.5, end_candidate=7.5), words, sid, [], 20.0)
    assert not f["cuts_inside_word"]
    f = tl.refine(_frag(start_candidate=13.6, end_candidate=17.0), words, sid, [14.0, 16.8], 20.0)
    assert f["render_start"] == 14.0 and f["render_end"] < 16.8


def test_both_needs_a_real_pause_else_dialogue():
    words, sents = tl.sentences(_words())
    sid = {s["id"]: s for s in sents}
    wide = tl.refine(_frag(audio_owner="both", start_candidate=0.5, end_candidate=8.5), words, sid, [], 20.0)
    narrow = tl.refine(_frag(beat_id="B2", audio_owner="both", start_candidate=5.2, end_candidate=7.8),
                       words, sid, [], 20.0)
    notes = tl.voice_rooms([wide, narrow], words)
    assert wide["audio"] == "both" and wide["voice_gap"] == (2.0, 5.0)
    assert narrow["audio"] == "dialogue" and any("both→dialogue" in n for n in notes)
    # a voice longer than the pause is dropped, never laid over the words
    issues = tl.place_narration([wide], {"B1": 3.5}, words, [], 20.0)
    assert wide.get("voice_dropped") and issues
    wide.pop("voice_dropped")
    tl.place_narration([wide], {"B1": 2.0}, words, [], 20.0)
    a = wide["render_start"] + wide["voice_offset"]
    assert 2.0 <= a and a + 2.0 <= 5.0


def test_dedupe_and_fit_keep_hook_and_tease():
    fr = [dict(_frag(beat_id=f"B{i}", event_ids=[f"E{i}"], priority=p), render_start=i * 10.0,
               render_end=i * 10.0 + 8) for i, p in enumerate([5, 1, 4, 2, 5])]
    dup = dict(fr[1], id="Bdup", render_start=11.0, render_end=18.0)
    kept, dropped = tl.dedupe(fr + [dup])
    assert "Bdup" not in [f["id"] for f in kept] and dropped
    fitted, _ = tl.fit_to_target(kept, 24.0)
    assert fitted[0]["id"] == "B0" and fitted[-1]["id"] == "B4" and tl.total(fitted) <= 24 * 1.12


def test_chunks_overlap_and_do_not_split_sentences():
    _, sents = tl.sentences(_words())
    ch = tl.chunks(13.0, sents, tl.shots([3.0, 9.0], 13.0), chunk_sec=6.0, overlap=0.0)
    ids = [s["id"] for c in ch for s in c["sentences"]]
    assert sorted(ids) == ["S1", "S2", "S3"]
    ch = tl.chunks(2400.0, [], [], chunk_sec=1200.0, overlap=15.0)
    assert len(ch) == 2 and ch[1]["start"] == 1185.0
    assert len(tl.chunks(900.0, [], [])) == 1                    # 15-min fast run: one chunk


# ── voice placement + mix plan ──────────────────────────────────────────────

def test_mix_plan_dialogue_untouched_narration_ducked_in_spec_range():
    words, _ = tl.sentences([])
    fr = [dict(_frag(beat_id="B1"), render_start=10.0, render_end=13.0),
          dict(_frag(beat_id="B2", audio_owner="dialogue"), audio="dialogue", render_start=30.0, render_end=34.0)]
    tl.voice_rooms(fr, words)
    assert fr[0]["voice_room_sec"] > 0 and "voice_room_sec" not in fr[1]
    assert not tl.place_narration(fr, {"B1": 3.5}, words, [], 100.0)
    assert fr[0]["render_end"] - fr[0]["render_start"] >= 0.3 + 3.5 + 0.35 - 1e-6
    tl.layout(fr)
    plan = tl.mix_plan(fr)
    s1, s2 = plan["segments"]
    assert s2["voice"] is None and s2["original_db"] == 0 and s2["original_under_voice_db"] is None
    assert s1["voice"] == [0.3, 3.8] and -16 <= s1["original_under_voice_db"] <= -12
    assert 0 < plan["voice_share"] < 1
    expr = tl.duck_expr(tl.voice_windows(fr))
    g = 1 - float(expr.split("-")[1].split("*")[0])
    assert abs(20 * np.log10(g) - tl.DUCK_DB) < 0.05 and tl.duck_expr([]) == "1"


def test_subtitles_are_narration_only():
    words, _ = tl.sentences(_words())
    fr = [dict(_frag(beat_id="B1"), render_start=20.0, render_end=24.0, voice_offset=0.3, voice_sec=2.0),
          dict(_frag(beat_id="B2", audio_owner="dialogue"), render_start=4.9, render_end=8.3)]
    tl.layout(fr)
    sw = tl.subtitle_words(fr, {"B1": "Всё изменилось в одну ночь"})
    assert [w.text for w in sw] == ["Всё", "изменилось", "в", "одну", "ночь"]
    assert abs(sw[0].start - 0.3) < 1e-6 and sw[-1].end <= 2.3 + 1e-6


# ── budget guard ────────────────────────────────────────────────────────────

class _Res:
    def __init__(self, text, cost=0.01):
        self.text, self._c = text, cost

    def telemetry(self):
        return {"provider": "fake", "model": "m", "prompt_version": "v", "input_tokens": 10,
                "output_tokens": 5, "cost_usd": self._c, "latency_ms": 1, "attempts": [], "raw": {}}


def test_budget_stops_before_the_request_and_repairs_schema_once():
    calls = []

    def chat(msgs, **kw):
        calls.append(msgs)
        return _Res('```json\n{"x": 1}\n```' if len(calls) > 1 else "not json", cost=0.002)

    llm = budget.LLM(chat, 1.0, "google/gemini-3.8-flash")
    assert llm.ask("a", [{"role": "user", "content": "hi"}], "v", lambda d: d, max_out=100) == {"x": 1}
    assert len(calls) == 2 and "проверку схемы" in calls[1][-1]["content"]
    big = [{"role": "user", "content": "я" * 4_000_000}]           # ~1.6 M tokens > $1
    with pytest.raises(budget.BudgetExceeded):
        llm.ask("b", big, "v", lambda d: d)
    assert len(calls) == 2 and llm.calls[-1]["stopped"] == "budget"
