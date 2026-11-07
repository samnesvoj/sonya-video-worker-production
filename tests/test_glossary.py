from scripts.shared.glossary import apply


def W(*ts):
    return [{"word": " " + t, "start": i, "end": i + 0.5} for i, t in enumerate(ts)]


def test_name_fixed_with_inflection_and_case_only_in_context():
    out = apply(W("я", "смотрел", "вписку", "с", "Вавилоном,", "и", "он"))
    assert out[4]["word"] == " Эвелоном," and out[4]["glossary"] == "Вавилоном→Эвелоном" and out[4]["start"] == 4
    assert apply(W("древний", "Вавилон", "был", "огромным", "городом"))[1]["word"] == " Вавилон"   # no context
    assert apply(W("стример", "ВАВИЛОН"))[1]["word"] == " ЭВЕЛОН"
    assert apply(W("стример", "вавилонская"))[1]["word"] == " вавилонская"     # different word, long suffix


def test_per_run_entries_and_no_context_rule():
    out = apply(W("привет", "Моргинштерн"), extra=[{"wrong": "Моргинштерн", "right": "Моргенштерн"}])
    assert out[1]["word"] == " Моргенштерн"
