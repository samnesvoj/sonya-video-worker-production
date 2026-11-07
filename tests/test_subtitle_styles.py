"""Trendy subtitle presets: grouping, per-word highlight, karaoke timing, caps, fonts bundled."""
import re

from scripts.shared.subtitles import Word
from scripts.shared.subtitle_styles import FONTS_DIR, PRESETS, groups, to_ass

WORDS = [Word(t, i * 0.4, i * 0.4 + 0.35) for i, t in
         enumerate("Чувак, где ты берёшь столько энергии в таком возрасте? Ну это просто.".split())]


def events(ass):
    return [l for l in ass.splitlines() if l.startswith("Dialogue")]


def test_fonts_are_bundled_with_cyrillic_names_resolvable():
    names = {p.name for p in FONTS_DIR.glob("*.ttf")}
    assert {"Montserrat-Black.ttf", "Montserrat-ExtraBold.ttf", "Montserrat-SemiBold.ttf"} <= names
    assert not list(FONTS_DIR.glob("*.txt"))            # libass tries to load every file there


def test_highlight_one_event_per_word_with_the_spoken_word_coloured():
    ev = events(to_ass(WORDS, "highlight", 6.0))
    assert len(ev) == len(WORDS)
    assert "{\\c&H0000E5FF&" in ev[1] and "ГДЕ" in ev[1] and "ЧУВАК" in ev[1]   # caps, group kept on screen
    assert all(len(g) <= PRESETS["highlight"].max_words for g in groups(WORDS, PRESETS["highlight"], 6.0))


def test_groups_break_at_sentence_end():
    gs = groups(WORDS, PRESETS["minimal"], 6.0)
    assert any(g[-1].text.endswith("?") for g in gs)
    assert not any(g[-1].text.endswith("?") and g is not gs[-1] and False for g in gs)


def test_karaoke_uses_kf_and_durations_cover_the_words():
    ev = events(to_ass(WORDS, "karaoke", 6.0))
    ks = [int(x) for x in re.findall(r"\\kf(\d+)", " ".join(ev))]
    assert len(ks) == len(WORDS) and all(k >= 1 for k in ks)


def test_static_presets_one_event_per_group_and_no_lower_case_in_caps_styles():
    ev = events(to_ass(WORDS, "mrbeast", 6.0))
    assert len(ev) == len(groups(WORDS, PRESETS["mrbeast"], 6.0))
    assert "ЭНЕРГИИ" in " ".join(ev) and "энергии" not in " ".join(ev)


def test_unknown_preset_falls_back_and_avoid_text_still_moves_cues():
    ass = to_ass(WORDS, "nope", 6.0, avoid=[[0.0, 6.0, 1420, 1640]])
    assert "Montserrat Black,82" in ass and "\\an2\\pos(540," in ass


def test_noble_presets_exist_and_glass_pills_cover_each_group():
    from scripts.shared.subtitle_styles import glass_pills
    for pid in ("champagne", "lavender", "silver", "glass"):
        assert pid in PRESETS
    ass = to_ass(WORDS, "glass", 6.0)
    pills = glass_pills(ass, "glass")
    gs = groups(WORDS, PRESETS["glass"], 6.0)
    assert len(pills) == len(gs)
    for (a, b, x, y, w, h), g in zip(pills, gs):
        assert a <= g[0].start + 1e-3 and b >= g[-1].start and 0 <= x and x + w <= 1080 and w > 100 and h > 60
        assert x % 2 == y % 2 == w % 2 == h % 2 == 0
    assert glass_pills(to_ass(WORDS, "highlight", 6.0), "highlight") == []


def test_glass_render(tmp_path):
    import json, shutil, subprocess
    import pytest
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg")
    from scripts.shared.crop.smart_crop_adapter import fit_track, render_tracked
    from scripts.shared.subtitle_styles import glass_pills
    v = tmp_path / "in.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=25:duration=2",
                    "-pix_fmt", "yuv420p", str(v)], check=True)
    ass = tmp_path / "clip.ass"
    txt = to_ass(WORDS[:4], "glass", 2.0)
    ass.write_text(txt, encoding="utf-8")
    (tmp_path / "clip.glass.json").write_text(json.dumps(glass_pills(txt, "glass")))
    out = tmp_path / "o.mp4"
    try:
        render_tracked(str(v), str(out), 0.0, 2.0, fit_track(1920, 1080, 2.0), subtitle_path=str(ass))
    except RuntimeError as e:
        if "No such filter: 'ass'" in str(e):
            pytest.skip("ffmpeg without libass")
        raise
    assert out.stat().st_size > 10_000


def test_hook_title_on_top_and_keywords_stay_emphasised():
    ass = to_ass(WORDS, "champagne", 6.0, keywords=["энергии"], hook="Откуда у него энергия")
    ev = events(ass)
    hook = [e for e in ev if ",Hook," in e]
    assert len(hook) == 1 and "Откуда у него энергия" in hook[0] and hook[0].split(",")[2].startswith("0:00:03")
    assert "Style: Hook," in ass
    kw_ev = [e for e in ev if ",Hook," not in e and "энергии" in e]
    assert kw_ev and all(PRESETS["champagne"].active in e.split("энергии")[0][-40:] for e in kw_ev)
    assert len(events(to_ass(WORDS, "champagne", 6.0))) == len(ev) - 1       # no hook → no title event


def test_glass_pills_skip_the_hook_and_stay_inside_the_frame():
    from scripts.shared.subtitle_styles import glass_pills
    long = [Word(t, i * 0.4, i * 0.4 + 0.35) for i, t in enumerate("Сверхдлинноеслововообщенепомещающеесявстроку да".split())]
    ass = to_ass(long, "glass", 3.0, hook="Почему подросток стал зависимым от всего этого")
    pills = glass_pills(ass, "glass")
    assert pills and all(x >= 0 and x + w <= 1080 and y >= 0 and y + h <= 1920 for _, _, x, y, w, h in pills)
    assert len(pills) == len(glass_pills(to_ass(long, "glass", 3.0), "glass"))


def test_long_hook_wraps_into_two_balanced_lines_inside_the_frame():
    from scripts.shared.subtitle_styles import _text_width, _wrap_hook
    t = _wrap_hook("Почему подросток стал зависимым от всего этого")
    scale = int(re.match(r"\{\\fscx(\d+)", t).group(1)) / 100 if t.startswith("{") else 1.0
    rows = re.sub(r"\{[^}]*\}", "", t).split("\\N")
    assert len(rows) == 2 and all(_text_width(r, "Montserrat Black", 62) * scale <= 901 for r in rows)
    assert _wrap_hook("Коротко") == "Коротко"


def test_font_choice_uses_full_name_and_keeps_line_width(tmp_path, monkeypatch):
    import shutil
    from scripts.shared import subtitle_styles as ss
    src = __import__("pathlib").Path("/Users/agent/Business/SONYA/fixtures/fonts_candidates/Unbounded-ExtraBold.ttf")
    if not src.is_file():
        import pytest
        pytest.skip("candidate fonts are local fixtures")
    for f in ss.FONTS_DIR.glob("*.ttf"):
        shutil.copy(f, tmp_path)
    shutil.copy(src, tmp_path)
    monkeypatch.setattr(ss, "FONTS_DIR", tmp_path)
    assert "unbounded" in ss.available_fonts() and "tektur" not in ss.available_fonts()
    ass = to_ass(WORDS, "highlight", 6.0, font="unbounded")
    style = [l for l in ass.splitlines() if l.startswith("Style: Sonya,")][0].split(",")
    assert style[1] == "Unbounded ExtraBold" and int(style[2]) < PRESETS["highlight"].size   # wide face → smaller
    assert to_ass(WORDS, "highlight", 6.0, font="tektur") == to_ass(WORDS, "highlight", 6.0)   # no file → preset font


def test_owner_font_dela_is_bundled_with_cyrillic_and_used_for_hook():
    from scripts.shared.subtitle_styles import _text_width, available_fonts
    assert "dela" in available_fonts()
    assert (FONTS_DIR.parent / "fonts_license" / "OFL-DelaGothicOne.txt").is_file()
    assert _text_width("ЁЖЩЪЮЯ", "Dela Gothic One", 40) != _text_width("ЁЖЩЪЮЯ", "Montserrat ExtraBold", 40)
    ass = to_ass(WORDS, "highlight", 6.0, font="dela", hook="Где он берёт энергию")
    assert "Style: Sonya,Dela Gothic One," in ass and "Style: Hook,Dela Gothic One," in ass
