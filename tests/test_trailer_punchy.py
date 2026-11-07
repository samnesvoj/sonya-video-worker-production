"""trailer_cinematic v2.1 — owner feedback 2026-11-04 on «Волк с Уолл-стрит»: subtitles never leave the frame (measured
with the real font), the punchy style (salvo, tagged invented gags, short phrases), semantic montage (the right people
for the words), the Judge never lets a visual mismatch pass, faster narration, a rouble limit per video.
No network, no models; ffmpeg-based checks are skipped when ffmpeg is missing."""
import os
import shutil
import subprocess
import wave
from types import SimpleNamespace

import numpy as np
import pytest

from modes.trailer_cinematic import budget, compose, recap, subs, validator
from modes.trailer_cinematic.budget import SchemaError

FFMPEG = os.environ.get("SONYA_FFMPEG") or shutil.which("ffmpeg")
HAS_FONT = (subs._fonts_dir() / subs.FONT_FILE).is_file()


# ── subtitles: measured, wrapped, in frame ─────────────────────────────────

def _words(text, t=0.0, step=0.3):
    out = []
    for w in text.split():
        out.append((w, round(t, 3), round(t + step, 3)))
        t += step
    return out


@pytest.mark.skipif(not HAS_FONT, reason="Dela Gothic One font file missing")
def test_subtitles_long_russian_words_stay_inside_the_safe_width():
    # the draft's overflowing cues («ЧЕРНЫЙ ПОНЕДЕЛЬНИК ОБРУШИЛ РЫНОК», «НАЦЕЛЕННУЮ НА АГРЕССИВНЫЕ») + monsters
    text = ("Черный понедельник обрушил рынок. Окман, нацеленную на агрессивные продажи и богатых инвесторов. "
            "Высокопоставленные федеральные человеконенавистнические достопримечательности.")
    ass = subs.to_ass(_words(text), 30.0)
    r = subs.check_ass(ass)
    assert r["ok"], r["problems"]
    assert r["worst_px"] <= subs.SAFE_W
    assert "\\fscx" not in ass                          # the spoken word is recoloured, never scaled wider
    assert "Dela Gothic One" in ass and subs.PRIMARY in ass and subs.ACTIVE in ass


@pytest.mark.skipif(not HAS_FONT, reason="Dela Gothic One font file missing")
def test_subtitle_checker_catches_an_overflowing_line_and_three_lines():
    bad = ("[V4+ Styles]\nStyle: Recap,Dela Gothic One,74,&H0,&H0,&H0,&H0,0,0,0,0,100,100,0,0,1,5,2,2,96,96,400,204\n"
           "[Events]\nDialogue: 0,0:00:00.00,0:00:01.00,Recap,,0,0,0,,{\\fs74}ЧЕРНЫЙ ПОНЕДЕЛЬНИК ОБРУШИЛ РЫНОК\n"
           "Dialogue: 0,0:00:01.00,0:00:02.00,Recap,,0,0,0,,{\\fs50}А\\NБ\\NВ\n")
    r = subs.check_ass(bad)
    assert not r["ok"] and any("px >" in p for p in r["problems"]) and any("3 lines" in p for p in r["problems"])
    assert not validator.check_subtitles_in_frame.__name__ == ""


@pytest.mark.skipif(not HAS_FONT, reason="Dela Gothic One font file missing")
def test_subtitle_size_steps_down_before_breaking_a_word():
    size, rows = subs.fit(["ЧЕЛОВЕКОНЕНАВИСТНИЧЕСКИЙ"])
    assert subs.MIN_SIZE <= size < subs.BASE_SIZE and len(rows) == 1
    assert subs.line_px("ЧЕЛОВЕКОНЕНАВИСТНИЧЕСКИЙ", size) <= subs.SAFE_W


# ── punchy script ──────────────────────────────────────────────────────────

SM = {"events": [{"id": "E1"}, {"id": "E2"}, {"id": "E3"}]}


def _punchy(n=10):
    segs = []
    for i in range(n):
        first = ["Парень", "Потом", "Дальше", "Вдруг", "Итак", "Короче", "Ну", "Затем", "Опа", "Блин"][i % 10]
        segs.append({"text": f"{first} лезет наверх. Быстро. Нагло. Шаг номер{'а' * i}.", "event_ids": ["E1"],
                     "visual_intent": "герой в офисе кричит в трубку", "function": "development", "pause_after": "short",
                     "gags": []})
    segs[4] = dict(segs[4], function="salvo",
                   text="Кокаин, проститутки, яхта, вертолёт, швейцарский банк, ползком к машине на таблетках.")
    segs[2]["text"] = "Он богаче Бога? Почти. Ну ладно, богаче соседа."
    segs[2]["gags"] = [{"text": "Он богаче Бога?", "device": "rhetorical_question"}]
    return {"segments": segs}


def test_punchy_script_keeps_tagged_gags_salvo_and_punchline():
    s = recap.validate_script(_punchy(), SM, 100, 2000, style="punchy")
    assert s["style"] == "punchy" and s["segments"][-1]["function"] == "punchline"
    assert s["segments"][2]["gags"] == [{"kind": "invented_gag", "text": "Он богаче Бога?", "device": "rhetorical_question"}]
    assert any(x["function"] == "salvo" for x in s["segments"])


def test_punchy_script_rejects_untraceable_gags_no_salvo_and_school_essay_sentences():
    d = _punchy()
    d["segments"][2]["gags"] = [{"text": "Он купил Луну", "device": "hyperbole"}]
    with pytest.raises(SchemaError, match="exact quote"):
        recap.validate_script(d, SM, 100, 2000, style="punchy")
    d = _punchy()
    d["segments"][4]["function"] = "development"
    with pytest.raises(SchemaError, match="salvo"):
        recap.validate_script(d, SM, 100, 2000, style="punchy")
    d = _punchy()
    d["segments"][1]["text"] = ("Вместе с партнером Донни он основал империю Стрэттон Окман, нацеленную на агрессивные "
                                "продажи и богатых инвесторов из разных городов.")
    with pytest.raises(SchemaError, match="words per sentence"):
        recap.validate_script(d, SM, 100, 2000, style="punchy")
    d = _punchy()
    d["segments"][1]["text"] = "Я начал клерком. Мне было скучно."        # the real hero's voice: not allowed
    with pytest.raises(SchemaError, match="first person"):
        recap.validate_script(d, SM, 100, 2000, style="punchy")


def test_calm_style_is_unchanged_and_prompts_differ():
    sm = {"events": [{"id": "E1", "start": 0, "end": 1, "summary": "s", "participants": [], "emotion": "", "importance": 3}],
          "characters": [], "premise": "", "conflict": ""}
    rate = {"chars_per_sec": 27.0}
    calm = recap.writer_messages(sm, 60, 100, 200, rate)
    punchy = recap.writer_messages(sm, 60, 100, 200, rate, style="punchy")
    assert "документальных" in calm[0]["content"] and "Мат разрешён" in punchy[0]["content"]
    assert "salvo" in punchy[1]["content"] and "invented" not in calm[1]["content"]
    assert recap.versions("calm")["writer"] == recap.WRITER_VERSION != recap.versions("punchy")["writer"]


# ── judge: gags must read as jokes, a visual mismatch never passes ────────

def _jv(**kw):
    v = {"scores": {k: 9.0 for k in recap.JUDGE_KEYS_PUNCHY}, "factual_errors": [], "rewrite_instructions": ""}
    v.update(kw)
    return recap.validate_judge(v, "punchy")


def test_punchy_judge_gag_taken_for_fact_is_a_factual_error():
    v = _jv(gag_check=[{"text": "Он купил Луну", "reads_as_joke": False, "problem": "звучит как сюжет"}])
    assert v["factual_errors"] and recap.rewrite_scope(v) == "text"


def test_punchy_judge_minor_visual_weak_beat_still_triggers_a_montage_rewrite():
    v = _jv(weak_beats=[{"segment_id": "S5", "severity": "minor", "kind": "visual", "problem": "жена вместо агентов"}])
    assert recap.needs_rewrite(v) and recap.rewrite_scope(v) == "visual"
    assert recap.rewrite_scope(_jv(scores=dict({k: 9.0 for k in recap.JUDGE_KEYS_PUNCHY}, energy=5))) == "text"
    with pytest.raises(SchemaError):
        recap.validate_judge({"scores": {k: 9 for k in recap.JUDGE_KEYS}, "factual_errors": [], "rewrite_instructions": ""},
                             "punchy")


# ── director / composer: the right people for the words ───────────────────

def _cand(i, who, ev, start=100.0, speaking="not_speaking"):
    return {"id": i, "start": start, "end": start + 5, "shot_id": f"C{i}", "shot_start": start, "shot_end": start + 5,
            "usable": True, "speaking": {"state": speaking}, "people": [{"who": who, "confidence": 0.9, "mouth": "closed"}],
            "visual_interest": 4, "event_weight": 3, "shot_type": "medium", "kind": "reaction", "action": "a", "setting": "",
            "good_for": [], "problems": [], "event_ids": ev}


def test_director_must_use_a_window_of_the_segment_events_when_one_is_free():
    script = {"segments": [{"id": "S1", "function": "hook", "text": "агенты на яхте", "visual_intent": "", "event_ids": ["E5"]}]}
    cands = {"V1": _cand("V1", "Наоми", ["E9"]), "V2": _cand("V2", "агент Деннем", ["E5"])}
    plan = lambda cid: {"segments": [{"segment_id": "S1", "shots": [{"candidate_id": cid, "seconds": 4.0}]}]}  # noqa: E731
    with pytest.raises(SchemaError, match="V2"):
        recap.validate_director(plan("V1"), script, {"S1": 4.0}, cands, "никто")
    assert recap.validate_director(plan("V2"), script, {"S1": 4.0}, cands, "никто")


def test_filler_prefers_the_same_people_over_the_most_interesting_window():
    refs = [_cand("V71", "агент Деннем", ["E6"], 5358)]
    naomi = dict(_cand("V75", "Наоми", [], 5382), visual_interest=5)
    agent = _cand("V63", "агент Деннем", ["E6"], 5305)
    assert compose.relevance(agent, refs) > compose.relevance(naomi, refs)


# ── budget: roubles per video ──────────────────────────────────────────────

def test_video_ledger_in_roubles_stops_before_the_call(tmp_path):
    led = budget.VideoLedger(str(tmp_path / "v.json"), limit_rub=100, seed_usd=0.95)       # 95 ₽ already spent
    assert budget.rub(led.spent()) == pytest.approx(95.0)

    def chat(msgs, **kw):
        raise AssertionError("must not be called")
    llm = budget.LLM(chat, 10.0, "gemini/gemini-3.1-flash-lite", ledger=[None, led])
    with pytest.raises(budget.BudgetExceeded, match="₽"):
        llm.ask("writer", [{"role": "user", "content": "x" * 10}], "v", lambda d: d, max_out=20000)
    # seed is booked once: a second ledger on the same file does not add it again
    assert budget.rub(budget.VideoLedger(str(tmp_path / "v.json"), 100, seed_usd=0.95).spent()) == pytest.approx(95.0)


# ── faster narration and pace ─────────────────────────────────────────────

@pytest.mark.skipif(not FFMPEG, reason="ffmpeg missing")
def test_speed_up_shortens_without_changing_pitch(tmp_path, monkeypatch):
    from modes.trailer_cinematic import recap_runner
    sr = 48000
    t = np.arange(int(sr * 2.0)) / sr
    x = np.concatenate([np.zeros(sr // 10), 0.3 * np.sin(2 * np.pi * 220 * t), np.zeros(sr // 10)])
    src = tmp_path / "a.wav"
    with wave.open(str(src), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(sr)
        w.writeframes((x * 32767).astype("<i2").tobytes())
    out = recap_runner.speed_up(str(src), str(tmp_path / "b.wav"), 1.8)
    with wave.open(out) as w:
        n, rate = w.getnframes(), w.getframerate()
        y = np.frombuffer(w.readframes(n), "<i2").astype(float)
    assert 2.0 / 1.8 - 0.1 < n / rate - 0.08 < 2.0 / 1.8 + 0.12           # + 2 × 40 ms of edge silence
    assert np.abs(y[:int(0.04 * rate)]).max() == 0 and np.abs(y[-int(0.04 * rate):]).max() == 0
    spec = np.abs(np.fft.rfft(y))
    peak = np.argmax(spec) * rate / len(y)
    assert abs(peak - 220) < 6                                # time-stretch, not a chipmunk


def test_pace_check_and_style_pauses():
    assert validator.check_pace("раз два три четыре", 1.0, 3.3, 4.4)["ok"]
    assert not validator.check_pace("раз два", 1.0, 3.3, 4.4)["ok"]
    script = {"segments": [{"id": "S1", "text": "Раз. Два.", "pause_after": "normal"},
                           {"id": "S2", "text": "Три.", "pause_after": "normal"}]}
    calm = compose.voice_timeline(script, {"S1": [1.0, 1.0], "S2": [1.0]})
    fast = compose.voice_timeline(script, {"S1": [1.0, 1.0], "S2": [1.0]}, pauses=recap.STYLES["punchy"]["pauses"])
    assert fast["total"] < calm["total"] - 0.6


def test_composer_replaces_a_repeat_of_shown_footage_with_the_closest_free_window():
    cands = {"V86": dict(_cand("V86", "Джордан", ["E7"], 7000), shot_cap=None),
             "V87": _cand("V87", "Джордан", ["E7"], 7010), "V88": _cand("V88", "Джордан", ["E7"], 7020),
             "V1": _cand("V1", "Наоми", [], 100)}
    for c in cands.values():
        c.pop("shot_cap", None)
        c["end"] = c["start"] + 6
        c["shot_end"] = c["end"]
    script = {"segments": [{"id": "S1", "text": "Раз.", "pause_after": "short"}, {"id": "S2", "text": "Два.", "pause_after": "short"}]}
    vt = compose.voice_timeline(script, {"S1": [3.0], "S2": [3.0]}, pauses=recap.STYLES["punchy"]["pauses"])
    plan = {"segments": [{"segment_id": "S1", "shots": [{"candidate_id": "V86", "seconds": 3.0}], "alternates": []},
                         {"segment_id": "S2", "shots": [{"candidate_id": "V86", "seconds": 3.0, "reuse_ok": True}],
                          "alternates": ["V88"]}]}
    tl = compose.compose(plan, cands, vt)
    ids = [s["candidate_id"] for s in tl["shots"]]
    assert ids.count("V86") == 1 and "V88" in ids and any("repeat of V86" in w for w in tl["warnings"])


def test_steady_crop_removes_jumps_inside_a_shot():
    from modes.trailer_cinematic import media
    tr = {"keyframes": [[0.0, 100], [2.3, 100], [2.31, 400], [3.0, 400]]}      # a face switch at 2.3 s
    x = media.steady_track(tr, 3.0)
    assert x == 100 and tr["keyframes"] == [[0.0, 100.0], [3.0, 100.0]]
    assert media.steady_track({"layout": "fit"}, 3.0) is None


def test_whip_pan_in_a_steady_shot_is_not_an_unplanned_cut(monkeypatch):
    shots = [{"n": 1, "t0": 0.0, "t1": 3.0, "src_start": 100.0, "crop_steady": 200.0},
             {"n": 2, "t0": 3.0, "t1": 6.0, "src_start": 500.0}]                  # moving window: never excused
    monkeypatch.setattr(validator, "hard_cuts_in", lambda src, a, b, threshold=0.4: [101.9] if a < 101.9 < b else [])
    assert validator.camera_motion([1.0, 4.0], shots, "src.mp4") == [1.0]       # no source cut near 101.0
    assert validator.camera_motion([1.9], shots, "src.mp4") == []                # a real cut in the source at 101.9


def test_forecast_is_calibrated_and_production_limit_is_20_rub():
    from modes.trailer_cinematic import budget, recap_runner as rr
    assert budget.PROD_VIDEO_LIMIT_RUB == 20.0 and budget.DEBUG_VIDEO_LIMIT_RUB == 100.0
    m = "gemini/gemini-3.1-flash-lite"
    fc = rr.forecast_creative(m, 40, 86, "punchy", vision_batches=5)
    # a tail-only gate on a cached film (what the owner measured as ≈ 2–3 ₽ per run) must stay in single roubles
    tail = rr.forecast_creative(m, 40, 86, "punchy", stage="tail", writer_cached=True, director_cached=True, rewrite=False)
    assert budget.rub(tail["usd"]) < 3.0 and fc["usd"] < 0.30
    assert budget.forecast_out(4000) == 1600


# ── owner rule: no personal names in narration / subtitles ─────────────────

_NSM = {"characters": [{"name": "Джордан Белфорд", "description": "x"}, {"name": "Агент Деннем", "description": "y"},
                       {"name": "Макс (Безумный Макс)", "description": "z"}],
        "events": [{"id": "E1", "participants": ["Донни", "Марк Хан"]}]}


def test_forbidden_names_cover_case_forms_and_skip_roles():
    st = recap.forbidden_names(_NSM, extra=["Кишлак"])
    for txt in ("Белфорд купил яхту", "У Белфорда яхта", "Деннема это бесит", "Донни снова ширяется", "Хан орёт",
                "Кишлаку всё равно", "Джордан жрёт таблетки"):
        assert recap.find_names(txt, st), txt
    for txt in ("Этот парень купил яхту. Агент приехал. Его кореш орёт.", "Он сорвался в Швейцарию."):
        assert not recap.find_names(txt, st), txt


def test_writer_script_with_a_name_is_rejected():
    sm = {"events": [{"id": "E1"}], "forbid_names": recap.forbidden_names(_NSM)}
    seg = lambda t, f: {"text": t, "event_ids": ["E1"], "visual_intent": "кто-то что-то делает", "function": f}  # noqa: E731
    ok = {"segments": [seg("Этот парень жрёт таблетки.", "hook")] * 7 + [seg("Этот мужик, его кореш, ведущий, агент.", "salvo")]
          + [seg("Конец, ублюдки.", "punchline")]}
    ok["segments"] = [dict(s, text=s["text"] + (" " * i)) for i, s in enumerate(ok["segments"])]
    with pytest.raises(SchemaError, match="names of people"):
        bad = {"segments": [dict(s) for s in ok["segments"]]}
        bad["segments"][3]["text"] = "Белфорд сошёл с ума."
        recap.validate_script(bad, sm, 0, 10000, 0.0, style="punchy")


def test_validator_no_names_checks_narration_and_subtitles(tmp_path):
    st = recap.forbidden_names(_NSM)
    ass = tmp_path / "n.ass"
    ass.write_text("[Events]\nDialogue: 0,0:00:00.00,0:00:01.00,Recap,,0,0,0,,{\\fs60}ЭТОТ ПАРЕНЬ\\NБЕЛФОРД\n", encoding="utf-8")
    r = validator.check_no_names("Этот парень орёт.", str(ass), st)
    assert not r["ok"] and r["value"]["found"]["subtitles"] and not r["value"]["found"]["narration"]
    ass.write_text("[Events]\nDialogue: 0,0:00:00.00,0:00:01.00,Recap,,0,0,0,,{\\fs60}ЭТОТ ПАРЕНЬ\n", encoding="utf-8")
    assert validator.check_no_names("Этот парень орёт.", str(ass), st)["ok"]


def test_judge_has_a_no_names_criterion():
    assert "no_names" in recap.JUDGE_KEYS_PUNCHY and recap.REWRITE_THRESHOLDS_PUNCHY["no_names"] == 10
    assert "ИМЕНА" in recap.WRITER_SYSTEM_PUNCHY


def test_sensitive_wording_is_rejected_in_the_script():
    sm = {"events": [{"id": "E1"}]}
    seg = lambda t, f="development", g=None: {"text": t, "event_ids": ["E1"], "visual_intent": "кто-то что-то делает",  # noqa: E731
                                              "function": f, "gags": g or []}
    base = [seg("Этот парень орёт."), seg("Тот мужик бухает."), seg("Менты приехали."), seg("Он сбежал."), seg("Дальше хуже."),
            seg("Орёт снова."), seg("Список, ещё список, пьёт, бьёт.", "salvo"), seg("Финал, ублюдки.", "punchline")]
    recap.validate_script({"segments": base}, sm, 0, 10000, 0.0, style="punchy")
    for bad in ("Он вскрывался в полиции.", "Угрожал суицидом ради водки.", "Намазал член мылом."):
        segs = [dict(s) for s in base]
        segs[2] = seg(bad)
        with pytest.raises(SchemaError, match="forbidden topics"):
            recap.validate_script({"segments": segs}, sm, 0, 10000, 0.0, style="punchy")
    segs = [dict(s) for s in base]
    segs[1] = seg("Он глупо шутит.", g=[{"text": "Как слепой котёнок.", "device": "hyperbole"}])
    segs[1]["text"] = "Он глупо шутит. Как слепой котёнок."
    with pytest.raises(SchemaError):
        recap.validate_script({"segments": segs}, sm, 0, 10000, 0.0, style="punchy")


def test_invented_crime_wording_is_rejected_unless_said_in_the_source():
    sm = {"events": [{"id": "E1"}], "source_crime_stems": ["убил"]}
    seg = lambda t, f="development": {"text": t, "event_ids": ["E1"], "visual_intent": "кто-то что-то делает", "function": f}  # noqa: E731
    base = [seg(t) for t in ("Этот парень орёт.", "Тот мужик бухает.", "Менты приехали.", "Он сбежал.", "Дальше хуже.", "Орёт снова.")]
    base += [seg("Список, ещё список, пьёт, бьёт.", "salvo"), seg("Финал, ублюдки.", "punchline")]

    def run(t):
        segs = [dict(s) for s in base]
        segs[2] = seg(t)
        return recap.validate_script({"segments": segs}, sm, 0, 10000, 0.0, style="punchy")
    run("В доме убили человека.")
    with pytest.raises(SchemaError, match="not in the source"):
        run("В подъезде зарезали человека.")
