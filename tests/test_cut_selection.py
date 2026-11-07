"""ASR-grounded Cut selection + LLM provider adapter (no network)."""
from __future__ import annotations

import json
import random
from types import SimpleNamespace

import pytest

from scripts.shared import cut_selection as cs
from scripts.shared import llm_provider as lp
from scripts.shared.boundaries import normalize_words


def speech(n_sentences=30, words_per=8, gap=0.5):
    """Synthetic interview: sentences of 8 words, 0.4 s each, 0.1 s apart, pauses between."""
    out, t = [], 0.3
    for s in range(n_sentences):
        for k in range(words_per):
            txt = f"s{s}w{k}" + ("." if k == words_per - 1 else "")
            out.append({"word": " " + txt, "start": round(t, 3), "end": round(t + 0.4, 3)})
            t += 0.5
        t += gap
    return out


CFG = cs.CandidateConfig(min_duration=8, max_duration=20, target_duration=12)


def test_candidates_are_complete_phrases_with_stable_ids():
    words, _ = normalize_words(speech())
    c1 = cs.build_candidates(words, CFG)
    assert c1 and all(c.complete for c in c1)
    assert all(CFG.min_duration - 1e-6 <= c.duration <= CFG.max_duration + 1e-6 for c in c1)
    # starts/ends sit exactly on ASR words
    starts, ends = {w.start for w in words}, {w.end for w in words}
    assert all(c.start in starts and c.end in ends for c in c1)
    # same words (even shuffled input) -> same ids
    shuffled = speech()
    random.Random(3).shuffle(shuffled)
    w2, _ = normalize_words(shuffled)
    assert [c.id for c in cs.build_candidates(w2, CFG)] == [c.id for c in c1]
    assert len({c.id for c in c1}) == len(c1)


def test_local_ranking_is_deterministic_and_non_overlapping():
    a, ma = cs.select_clips(speech(), 3, cfg=CFG)
    b, mb = cs.select_clips(speech(), 3, cfg=CFG)
    assert a == b and ma["selection_method"] == "local" and ma["llm"]["status"] == "not_opted_in"
    assert len(a) == 3
    spans = sorted((s["start"], s["start"] + s["duration"]) for s in a)
    assert all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1))
    assert all(s["source"] == "asr_candidate:local" for s in a)


def fake_chat(text, **extra):
    calls = []

    def chat(messages, *, prompt_version, opt_in):
        calls.append({"messages": messages, "prompt_version": prompt_version, "opt_in": opt_in})
        return SimpleNamespace(text=text(messages) if callable(text) else text,
                               telemetry=lambda: {"provider": "fake", "model": "m", "prompt_version": prompt_version,
                                                  "input_tokens": 100, "output_tokens": 20, "cost_usd": 0.001,
                                                  "latency_ms": 5, **extra})
    return chat, calls


def ids_in_prompt(messages):
    return [c["id"] for c in json.loads(messages[1]["content"].split("\n", 1)[1])]


def test_llm_picks_ids_and_timestamps_come_from_asr():
    def answer(messages):
        ids = ids_in_prompt(messages)
        return json.dumps({"selected": [{"id": ids[-1], "score": 9, "reason": "punchline"},
                                        {"id": ids[0], "score": 7, "reason": "hook"}]})
    chat, calls = fake_chat(answer)
    segs, meta = cs.select_clips(speech(), 2, llm_opt_in=True, cfg=CFG, chat=chat)
    words, _ = normalize_words(speech())
    by_id = {c.id: c for c in cs.build_candidates(words, CFG)}
    assert meta["selection_method"] == "llm" and calls[0]["prompt_version"] == cs.PROMPT_VERSION
    for s in segs:
        c = by_id[s["candidate_id"]]
        assert (s["start"], s["duration"]) == (c.start, c.duration)  # from ASR, not from the model
    assert segs[0]["selection_reason"] == "punchline"
    assert meta["llm"]["input_tokens"] == 100 and meta["llm"]["cost_usd"] == 0.001


@pytest.mark.parametrize("reply", [
    "not json at all",
    json.dumps({"selected": [{"id": "c_made_up", "start": 1.0, "end": 9.0}]}),
    json.dumps({"wrong": []}),
])
def test_malformed_or_hallucinated_reply_falls_back_to_local(reply):
    chat, _ = fake_chat(reply)
    segs, meta = cs.select_clips(speech(), 2, llm_opt_in=True, cfg=CFG, chat=chat)
    assert meta["selection_method"] == "local" and segs
    assert meta["llm"]["problems"] or meta["llm"].get("fallback_reason")
    local, _ = cs.select_clips(speech(), 2, cfg=CFG)
    assert segs == local


def test_unknown_ids_are_dropped_known_kept():
    def answer(messages):
        ids = ids_in_prompt(messages)
        return json.dumps({"selected": [{"id": "ghost", "score": 10}, {"id": ids[2], "score": 5}]})
    chat, _ = fake_chat(answer)
    segs, meta = cs.select_clips(speech(), 2, llm_opt_in=True, cfg=CFG, chat=chat)
    assert meta["selection_method"] == "llm" and len(segs) == 1
    assert "unknown_id:ghost" in meta["llm"]["problems"]


def test_no_paid_call_without_opt_in(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network must not be touched")
    monkeypatch.setattr(lp, "_urllib_transport", boom)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    segs, meta = cs.select_clips(speech(), 1, llm_opt_in=False, cfg=CFG)
    assert segs and meta["llm"] == {"status": "not_opted_in"}
    with pytest.raises(lp.LLMDisabled):
        lp.chat_json([{"role": "user", "content": "x"}], prompt_version="v", opt_in=False)


def test_opt_in_without_provider_is_disabled_not_error(monkeypatch):
    for k in ("OPENROUTER_API_KEY", "TIMEWEB_GEMINI_BASE_URL", "TIMEWEB_GEMINI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    segs, meta = cs.select_clips(speech(), 1, llm_opt_in=True, cfg=CFG)
    assert meta["llm"]["status"] == "disabled" and meta["selection_method"] == "local" and segs


def test_no_words_and_no_candidates():
    assert cs.select_clips([], 2)[1]["selection_status"] == "no_words"
    short = speech(n_sentences=1)  # ~4 s of speech < min 8 s
    assert cs.select_clips(short, 2, cfg=CFG)[1]["selection_status"] == "no_candidates"


# ── provider adapter ────────────────────────────────────────────────────

def providers():
    return [lp.ProviderConfig("timeweb", "https://tw.example/v1", "TW-KEY", "gemini-2.5-flash"),
            lp.ProviderConfig("openrouter", "https://openrouter.ai/api/v1", "OR-KEY", "google/gemini-3.8-flash",
                              {"usage": {"include": True}})]


def ok_payload(model="google/gemini-3.8-flash"):
    return json.dumps({"id": "gen-1", "model": model, "choices": [{"message": {"content": '{"selected": []}'},
                                                                   "finish_reason": "stop"}],
                       "usage": {"prompt_tokens": 1200, "completion_tokens": 80, "cost": 0.0021}}).encode()


def test_timeweb_primary_openrouter_fallback_with_telemetry():
    seen = []

    def transport(url, headers, body, timeout):
        seen.append((url, headers["Authorization"], json.loads(body)))
        if "tw.example" in url:
            return 503, b'{"error": {"message": "overloaded"}}'
        return 200, ok_payload()
    res = lp.chat_json([{"role": "user", "content": "hi"}], prompt_version="cut-select-v1", opt_in=True,
                       providers=providers(), transport=transport)
    assert [u for u, _, _ in seen] == ["https://tw.example/v1/chat/completions",
                                       "https://openrouter.ai/api/v1/chat/completions"]
    assert res.provider == "openrouter" and res.prompt_version == "cut-select-v1"
    assert (res.input_tokens, res.output_tokens, res.cost_usd) == (1200, 80, 0.0021)
    assert res.attempts[0] == {**res.attempts[0], "provider": "timeweb", "ok": False, "status": 503}
    tel = json.dumps(res.telemetry())
    assert "TW-KEY" not in tel and "OR-KEY" not in tel
    assert seen[1][2]["usage"] == {"include": True} and seen[1][2]["temperature"] == 0.0


def test_all_providers_fail_reports_attempts():
    def transport(url, headers, body, timeout):
        raise TimeoutError("read timed out")
    with pytest.raises(lp.LLMError) as e:
        lp.chat_json([{"role": "user", "content": "x"}], prompt_version="v", opt_in=True,
                     providers=providers(), transport=transport)
    assert [a["provider"] for a in e.value.attempts] == ["timeweb", "openrouter"]
    assert all(not a["ok"] for a in e.value.attempts)


def test_configured_providers_from_env():
    assert lp.configured_providers({}) == []
    names = [p.name for p in lp.configured_providers({"TIMEWEB_GEMINI_BASE_URL": "https://x", "TIMEWEB_GEMINI_API_KEY": "k",
                                                       "OPENROUTER_API_KEY": "o"})]
    assert names == ["timeweb", "openrouter"]
    assert "k" not in repr(lp.configured_providers({"OPENROUTER_API_KEY": "secret-k"})[0])


@pytest.mark.parametrize("payload", [b"[]", b'{"choices": "x"}', b'{"choices": [{"message": {"content": 5}}]}',
                                     b'{"choices": [null]}', b"<html>"])
def test_structurally_malformed_200_is_a_provider_failure(payload):
    with pytest.raises(lp.LLMError):
        lp.chat_json([{"role": "user", "content": "x"}], prompt_version="v", opt_in=True,
                     providers=providers()[1:], transport=lambda *a: (200, payload))


def test_selection_survives_unexpected_chat_exception():
    def chat(*a, **k):
        raise KeyError("weird")
    segs, meta = cs.select_clips(speech(), 1, llm_opt_in=True, cfg=CFG, chat=chat)
    assert segs and meta["llm"]["status"] == "error" and meta["selection_method"] == "local"


@pytest.mark.parametrize("kw", [{"target_duration": 0}, {"min_duration": 30, "max_duration": 10},
                                {"pause_sec": 0}, {"max_per_start": 0}])
def test_candidate_config_validation(kw):
    with pytest.raises(ValueError):
        cs.CandidateConfig(**kw)


def test_nan_and_out_of_range_scores_are_clamped():
    def answer(messages):
        ids = ids_in_prompt(messages)
        return '{"selected": [{"id": "%s", "score": NaN}, {"id": "%s", "score": 1e9}]}' % (ids[0], ids[-1])
    chat, _ = fake_chat(answer)
    segs, meta = cs.select_clips(speech(), 2, llm_opt_in=True, cfg=CFG, chat=chat)
    assert all(0.0 <= s["score"] <= 10.0 for s in segs)
    assert any(p.startswith("bad_score:") for p in meta["llm"]["problems"])
    json.dumps(meta, allow_nan=False)
