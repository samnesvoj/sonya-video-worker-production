import json
from modes.sonya_gen import translate as tr
from scripts.shared import llm_provider as lp

P = lp.ProviderConfig("timeweb", "https://x/v1", "k", "m")


def tx(answer, calls=None):
    def t(url, headers, body, timeout):
        if calls is not None:
            calls.append(timeout)
        return 200, json.dumps({"choices": [{"message": {"content": answer}}], "usage": {}}).encode()
    return t


def test_latin_untouched():
    assert tr.to_english("a dog in a field") == ("a dog in a field", None)


def test_russian_translated():
    calls = []
    out, w = tr.to_english("собака лабрадор бежит по пшеничному полю", providers=[P],
                           transport=tx(json.dumps({"prompt": "A Labrador dog running through a wheat field."}), calls))
    assert out == "A Labrador dog running through a wheat field, no text, no logos" and w is None and calls == [10.0]


def test_failures_fall_back():
    ru = "собака"
    assert tr.to_english(ru, providers=[]) == (ru, "prompt_translation_failed")
    assert tr.to_english(ru, providers=[P], transport=tx("not json"))[1] == "prompt_translation_failed"
    assert tr.to_english(ru, providers=[P], transport=tx('{"prompt": "собака"}')) == (ru, "prompt_translation_failed")
    def boom(*a): raise TimeoutError()
    assert tr.to_english(ru, providers=[P], transport=boom) == (ru, "prompt_translation_failed")
