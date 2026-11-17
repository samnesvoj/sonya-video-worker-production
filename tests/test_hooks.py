"""Hooks mode (scripts/shared/hooks.py): virality clips + a 1.5–3 s hook in front of each."""
import json
import shutil
import subprocess

import pytest

from scripts.shared import hooks
from scripts.shared.boundaries import Word

FF = shutil.which("ffmpeg")


def _words(spec):
    """[(text, start, end)] → Word list."""
    return [Word(t, s, e) for t, s, e in spec]


def test_enabled_by_job_param_and_switched_off_on_the_worker():
    assert hooks.enabled({"teaser": True}, env={})
    assert hooks.enabled({"teaser": {"enabled": True}}, env={})
    assert not hooks.enabled({}, env={})
    assert not hooks.enabled({"teaser": True}, env={"SONYA_HOOKS_TEASER": "off"})


def test_windows_never_cut_a_word_and_stay_in_bounds():
    w = _words([("Ты", 10.0, 10.2), ("что", 10.25, 10.5), ("сделал?!", 10.55, 11.1), ("Стой!", 12.0, 12.4),
                ("Я", 15.0, 15.1), ("ухожу.", 15.2, 15.7)])
    cfg = hooks.HookConfig()
    out = hooks.windows(w, 9.0, 20.0, cfg)
    assert out
    for s, e, i, j in out:
        assert 9.0 <= s < e <= 20.0
        assert cfg.min_sec - 1e-6 <= e - s <= cfg.max_sec + 0.5
        for x in w:                                  # no word is cut by an edge
            assert x.end <= s + 1e-6 or x.start >= e - 1e-6 or (x.start >= s - 1e-6 and x.end <= e + 1e-6)
    assert any(i == -1 for _, _, i, _ in out)        # silent windows exist (fights / kisses without words)


def test_text_cues_find_conflict_romance_and_flag_explicit():
    c = hooks.text_cues(_words([("Заткнись!", 0, 1), ("Я", 1, 1.1), ("тебя", 1.1, 1.3), ("люблю", 1.3, 1.6)]))
    assert "conflict" in c["kinds"] and "romance" in c["kinds"] and "phrase" in c["kinds"] and c["excl"] == 1
    assert not c["unsafe"]
    assert hooks.text_cues(_words([("порно", 0, 1)]))["unsafe"]


def _sig(n=120, loud_at=None):
    loud = [0.0] * n
    pitch = [0.0] * n
    voiced = [0.5] * n
    if loud_at is not None:
        loud[loud_at] = loud[loud_at + 1] = 3.5
        pitch[loud_at] = 2.5
    return {"loud": loud, "pitch": pitch, "voiced": voiced}


def test_clip_candidates_skip_the_clip_start_and_prefer_the_shout():
    words = _words([(f"слово{k}", 20 + k * 0.9, 20 + k * 0.9 + 0.6) for k in range(40)])
    clip = {"start": 20.0, "duration": 36.0}
    sig = dict(_sig(loud_at=40), motion={}, analysis=None, clip_start=20.0, motion_ref=2.0)
    c = hooks.clip_candidates(words, clip, sig, hooks.HookConfig(), "1")
    assert c and all(x["start"] >= 24.0 - 1e-6 for x in c)            # never the clip's own first seconds
    assert c[0]["start"] <= 41.0 and c[0]["end"] >= 40.0                # the loud + high-pitched second
    assert "крик" in c[0]["tags"] and c[0]["id"] == "1a"


def test_face_cues_two_faces_close_is_a_kiss_like_cue():
    an = {"meta": {"height": 1080}, "samples": [{"t": 1.0, "faces": [[800, 300, 200, 220, .9, .1, 0],
                                                                     [1010, 310, 190, 210, .9, .1, 0]]}]}
    assert hooks.face_cues(an, 0.5, 1.5)["close"] >= 0.6
    far = {"meta": {"height": 1080}, "samples": [{"t": 1.0, "faces": [[100, 300, 200, 220, .9, .1, 0],
                                                                      [1500, 310, 190, 210, .9, .1, 0]]}]}
    assert hooks.face_cues(far, 0.5, 1.5)["close"] == 0.0


def _view():
    clips = [{"n": 1, "about": "a", "cands": [{"id": "1a"}, {"id": "1b"}]},
             {"n": 2, "about": "b", "cands": [{"id": "2a"}]}]
    return clips, [{"id": "G1"}]


def test_parse_validates_ids_and_gives_a_global_moment_to_one_clip_only():
    clips, g = _view()
    txt = json.dumps({"hooks": [{"clip": 1, "id": "G1", "safety": "edgy"}, {"clip": 2, "id": "G1"},
                                {"clip": 3, "id": "1a"}, {"clip": 2, "id": "1a"}], "unsafe_ids": ["1b"]})
    out, unsafe = hooks.parse(txt, clips, g)
    assert out == {1: {"id": "G1", "safety": "edgy", "why": ""}} and unsafe == {"1b"}
    assert hooks.parse("not json", clips, g) == ({}, set())


def test_budget_drops_images_first():
    clips, g = _view()
    for c in clips:
        for k in c["cands"]:
            k.update(text="x", tags=[], start=0, end=2)
    for x in g:
        x.update(text="y", tags=[])
    with_img = hooks.build_messages(clips, g, ["AAAA", "BBBB"])
    no_img = hooks.build_messages(clips, g, [])
    assert hooks.estimate_rub(with_img, 2) - hooks.estimate_rub(no_img, 2) > 0.08   # ≈ 1100 tokens per image


def test_merge_ass_shifts_main_cues_and_puts_the_hook_first(tmp_path):
    head = "[Script Info]\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    main = tmp_path / "m.ass"
    main.write_text(head + "Dialogue: 0,0:00:00.50,0:00:01.20,Default,,0,0,0,,Привет\n", encoding="utf-8")
    t = tmp_path / "t.ass"
    t.write_text(head + "Dialogue: 0,0:00:00.10,0:00:02.00,Default,,0,0,0,,Стой!\n", encoding="utf-8")
    hooks.merge_ass(str(t), str(main), 2.5)
    lines = [ln for ln in main.read_text(encoding="utf-8").splitlines() if ln.startswith("Dialogue")]
    assert lines == ["Dialogue: 0,0:00:00.10,0:00:02.00,Default,,0,0,0,,Стой!",
                     "Dialogue: 0,0:00:03.00,0:00:03.70,Default,,0,0,0,,Привет"]


def _clip(path, dur, color):
    subprocess.run([FF, "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c={color}:s=270x480:r=25:d={dur}",
                    "-f", "lavfi", "-i", f"sine=f=440:d={dur}:sample_rate=48000", "-shortest",
                    "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "48000",
                    str(path)], check=True)


def _dur(p):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                 str(p)], capture_output=True, text=True).stdout)


@pytest.mark.skipif(not FF, reason="ffmpeg")
def test_finish_and_join_hook_plus_clip(tmp_path, monkeypatch):
    monkeypatch.setenv("SONYA_RENDER_ARGS", json.dumps(["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"]))
    raw, main = tmp_path / "h.mp4", tmp_path / "m.mp4"
    _clip(raw, 2.4, "red")
    _clip(main, 6.0, "blue")
    fin = tmp_path / "hf.mp4"
    hooks.finish_teaser(str(raw), str(fin), hooks.HookConfig(), FF)
    assert abs(_dur(fin) - (2.4 + hooks.HookConfig().white_hold_sec)) < 0.15   # + the white hold
    out = tmp_path / "o.mp4"
    how = hooks.join(str(fin), str(main), str(out), FF)
    assert how in ("copy", "reencode")
    assert abs(_dur(out) - (8.4 + hooks.HookConfig().white_hold_sec)) < 0.3
    # the last hook frame is the white flash
    png = tmp_path / "f.png"
    subprocess.run([FF, "-v", "error", "-y", "-sseof", "-0.04", "-i", str(fin), "-frames:v", "1", str(png)], check=True)
    import cv2
    assert cv2.imread(str(png)).mean() > 200


@pytest.mark.skipif(not FF, reason="ffmpeg")
def test_hook_matching_a_bt709_clip_joins_by_stream_copy(tmp_path, monkeypatch):
    """Vast 2026-11-17: the flash filters dropped the transfer tag -> the hook's SPS differed from the clip's in one
    byte and every join re-encoded the whole clip (25-56 s each). The hook now carries the clip's colour tags."""
    enc = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p"]
    monkeypatch.setenv("SONYA_RENDER_ARGS", json.dumps(enc))
    tags = ["-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv"]

    def clip(path, dur, color, tg):
        subprocess.run([FF, "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c={color}:s=270x480:r=30000/1001:d={dur}",
                        "-f", "lavfi", "-i", f"sine=f=440:d={dur}:sample_rate=44100", "-shortest", *enc, *tg,
                        "-c:a", "aac", "-b:a", "192k", str(path)], check=True)
    raw, main, fin, out = (tmp_path / n for n in ("h.mp4", "m.mp4", "hf.mp4", "o.mp4"))
    clip(raw, 2.0, "red", [])           # the hook piece lost its tags on the way (ffmpeg 5.1 filter chain)
    clip(main, 5.0, "blue", tags)
    hooks.finish_teaser(str(raw), str(fin), hooks.HookConfig(), FF, match=str(main))
    assert hooks._sig(hooks._probe(str(fin))) == hooks._sig(hooks._probe(str(main)))
    assert hooks.join(str(fin), str(main), str(out), FF) == "copy"
    err = subprocess.run([FF, "-v", "error", "-i", str(out), "-f", "null", "-"], capture_output=True, text=True).stderr
    assert not err.strip()


def test_sig_ignores_the_aac_profile_string_but_not_its_config():
    a = {"streams": [{"codec_type": "audio", "codec_name": "aac", "profile": "LC", "extradata": "1210"}]}
    b = {"streams": [{"codec_type": "audio", "codec_name": "aac", "extradata": "1210"}]}
    c = {"streams": [{"codec_type": "audio", "codec_name": "aac", "profile": "LC", "extradata": "1190"}]}
    assert hooks._sig(a) == hooks._sig(b) and hooks._sig(a) != hooks._sig(c)
    assert hooks._color_args({"color_transfer": "bt709", "color_space": "unknown"}) == ["-color_trc", "bt709"]


@pytest.mark.skipif(not FF, reason="ffmpeg")
def test_attach_teasers_local_and_editor(tmp_path):
    src = tmp_path / "src.mp4"
    _clip(src, 40.0, "gray")
    words = [{"word": f"слово{k}", "start": 1 + k * 0.8, "end": 1 + k * 0.8 + 0.5} for k in range(45)]
    words[20]["word"] = "Заткнись!"
    sel = [{"start": 2.0, "duration": 30.0}]
    meta = hooks.attach_teasers(str(src), [dict(s) for s in sel], words, duration=40.0, ffmpeg=FF,
                                cfg=hooks.HookConfig(pose=False))
    assert meta["hooks"] == 1

    seen = {}

    class R:
        input_tokens, output_tokens, model = 1000, 50, "fake"
        text = json.dumps({"hooks": [{"clip": 1, "id": "1b", "safety": "ok", "why": "крик"}]})

    def chat(msgs, **kw):
        seen["msgs"] = msgs
        return R()
    s2 = [dict(s) for s in sel]
    meta = hooks.attach_teasers(str(src), s2, words, duration=40.0, ffmpeg=FF, chat=chat, editor=True,
                                spent_rub=0.7, cfg=hooks.HookConfig(pose=False))
    assert meta["editor"]["status"] == "ok" and s2[0]["teaser"]["id"] == "1b"
    assert s2[0]["teaser"]["picked_by"] == "editor" and 1.5 <= s2[0]["teaser"]["duration"] <= 3.6
    user = seen["msgs"][1]["content"]
    assert isinstance(user, list) and any(p["type"] == "image_url" for p in user)     # frames went as a collage
    # an unsafe pick is never used
    R.text = json.dumps({"hooks": [{"clip": 1, "id": "1a", "safety": "unsafe"}], "unsafe_ids": ["1a"]})
    s3 = [dict(s) for s in sel]
    hooks.attach_teasers(str(src), s3, words, duration=40.0, ffmpeg=FF, chat=chat, editor=True,
                         cfg=hooks.HookConfig(pose=False))
    assert s3[0]["teaser"]["id"] != "1a" and s3[0]["teaser"]["picked_by"] == "local"


def test_phrase_shape_prefers_a_line_that_lands_over_a_fragment():
    w = _words([("и", 0.0, 0.2), ("оттуда", 0.3, 0.7), ("вываливался", 0.8, 1.4), ("труп.", 1.5, 1.9),
                ("Минуту,", 3.0, 3.4), ("на", 3.5, 3.6), ("каждом", 3.7, 4.0), ("шагу,", 4.1, 4.5)])
    assert hooks.phrase_shape(w, 1, 3) > 0.3          # the punchy end of a line
    assert hooks.phrase_shape(w, 4, 7) < -0.3         # stops on a comma


def test_hook_is_never_longer_than_3_s_even_with_pads():
    w = _words([("Понял", 0.5, 0.8), ("меня?", 0.85, 1.3), ("Грязный", 1.6, 2.2), ("бандюга!", 2.3, 3.45)])
    for s, e, i, j in hooks.windows(w, 0.0, 10.0, hooks.HookConfig()):
        assert e - s <= 3.0 + 1e-6


def test_teaser_report_is_plain_json_even_from_numpy_maths(tmp_path):
    import numpy as np
    from scripts.shared.mode_output import write_selected_segments
    assert json.dumps(hooks._plain({"a": np.float32(1.5), "b": [np.int64(2)]})) == '{"a": 1.5, "b": [2]}'
    seg = {"start": 1.0, "duration": 20.0, "teaser": {"score": np.float32(2.5)}}
    write_selected_segments(str(tmp_path), "virality", [seg], [str(tmp_path / "viral_clip_01.mp4")])
    d = json.loads((tmp_path / "selected_segments.json").read_text(encoding="utf-8"))
    assert d["segments"][0]["teaser"]["score"] == 2.5


@pytest.mark.skipif(not FF, reason="ffmpeg")
def test_white_hold_is_white_and_configurable(tmp_path, monkeypatch):
    monkeypatch.setenv("SONYA_RENDER_ARGS", json.dumps(["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"]))
    raw = tmp_path / "h.mp4"
    _clip(raw, 2.0, "red")
    monkeypatch.setenv("SONYA_HOOK_WHITE_HOLD", "1.0")
    fin = tmp_path / "hf.mp4"
    hooks.finish_teaser(str(raw), str(fin), hooks.HookConfig(), FF)
    assert abs(_dur(fin) - 3.0) < 0.15
    png = tmp_path / "w.png"   # middle of the hold: fully white
    subprocess.run([FF, "-v", "error", "-y", "-ss", "2.5", "-i", str(fin), "-frames:v", "1", str(png)], check=True)
    import cv2
    assert cv2.imread(str(png)).mean() > 245
