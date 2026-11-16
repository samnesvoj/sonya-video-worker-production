"""Site language chip -> ASR language; auto -> Whisper language check (production 2026-11-15: English video got GigaAM)."""
from scripts.shared import asr


def test_job_language_from_site_chip():
    assert asr.job_language({"settings": {"translation": "en"}}) == "en"
    assert asr.job_language({"settings": {"translation": "ru"}}) == "ru"
    assert asr.job_language({"settings": {"translation": "none"}}) == "auto"
    assert asr.job_language({}) == "auto"
    assert asr.job_language(None) == "auto"
    assert asr.job_language({"language": "de", "settings": {"translation": "en"}}) == "de"
    assert asr.job_language({"settings": {"translation": "<script>"}}) == "auto"


def test_auto_uses_detection(monkeypatch):
    monkeypatch.setattr(asr, "detect_language", lambda p: "en")
    assert asr.resolve_language("x.mp4", None) == "en"
    assert asr.resolve_language("x.mp4", "auto") == "en"
    assert asr.resolve_language("x.mp4", "ru") == "ru"          # an explicit choice is never second-guessed


def test_unsure_detection_falls_back_to_default(monkeypatch):
    monkeypatch.setattr(asr, "detect_language", lambda p: None)
    monkeypatch.delenv("SONYA_ASR_LANGUAGE", raising=False)
    assert asr.resolve_language("x.mp4", None) == "ru"


def test_detect_off(monkeypatch):
    monkeypatch.setenv("SONYA_ASR_DETECT", "off")
    assert asr.detect_language("/nonexistent.mp4") is None


def test_english_goes_to_whisper(monkeypatch):
    monkeypatch.setattr(asr, "detect_language", lambda p: "en")
    seen = {}
    monkeypatch.setattr(asr, "_whisper", lambda path, lang, name: seen.update(lang=lang, name=name) or {"language": lang})
    monkeypatch.setattr(asr, "_cuda", lambda: False)
    asr.transcribe("x.mp4", language=None, backend="gigaam")
    assert seen == {"lang": "en", "name": "faster_whisper_cpu"}
