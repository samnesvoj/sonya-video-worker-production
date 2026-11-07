"""Timeweb (Gemini) integration: llm_provider additions + trailer budget/limits/kill switch. Offline: fake transports only,
no network, no Keychain. The key is a sentinel string; every test that can leak it checks for it."""
import json
import subprocess
from types import SimpleNamespace

import pytest

from modes.trailer_cinematic import budget, recap_runner, visual
from scripts.shared import llm_provider as lp

KEY = "tw-SECRET-KEY-0123456789"
MSG = [{"role": "user", "content": "hi"}]


def ok_body(text='{"a": 1}', usage=None):
    return json.dumps({"id": "x", "model": "gemini/gemini-3.1-flash-lite",
                       "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                       "usage": usage or {"prompt_tokens": 100, "completion_tokens": 20,
                                          "prompt_tokens_details": {"text_tokens": 40, "image_tokens": 50, "video_tokens": 10}}}).encode()


def prov():
    return lp.ProviderConfig("timeweb", "https://api.timeweb.ai/v1", KEY, "gemini/gemini-3.1-flash-lite")


def seq(*responses):
    """transport returning/raising the given items in order; records calls."""
    calls = []

    def t(url, headers, body, timeout):
        calls.append((url, headers, body))
        r = responses[min(len(calls) - 1, len(responses) - 1)]
        if isinstance(r, BaseException):
            raise r
        return r
    t.calls = calls
    return t


def chat(transport, **kw):
    return lp.chat_json(MSG, prompt_version="t", opt_in=True, providers=[prov()], transport=transport,
                        sleep=lambda s: None, **kw)


# ── regression: nothing changes without the new variables ───────────────────

def test_configured_providers_unchanged_without_new_vars():
    env = {"TIMEWEB_GEMINI_BASE_URL": "https://x/v1", "TIMEWEB_GEMINI_API_KEY": KEY, "OPENROUTER_API_KEY": "or"}
    ps = lp.configured_providers(env)
    assert [p.name for p in ps] == ["timeweb", "openrouter"] and ps[0].model == "gemini-2.5-flash"
    assert lp.configured_providers({"OPENROUTER_API_KEY": "or"})[0].name == "openrouter"
    assert lp.configured_providers({}) == []


def test_chat_json_default_has_no_retry_and_falls_through_providers():
    t = seq((500, b"{}"), (200, ok_body()))
    p2 = lp.ProviderConfig("openrouter", "https://o/api", "k2", "m2")
    r = lp.chat_json(MSG, prompt_version="t", opt_in=True, providers=[prov(), p2], transport=t)
    assert r.provider == "openrouter" and len(t.calls) == 2          # old behaviour: next provider, one try each


def test_not_opted_in_and_no_provider():
    with pytest.raises(lp.LLMDisabled):
        lp.chat_json(MSG, prompt_version="t", opt_in=False, providers=[prov()])
    with pytest.raises(lp.LLMDisabled):
        lp.chat_json(MSG, prompt_version="t", opt_in=True, providers=[])


# ── kill switch / key sources ───────────────────────────────────────────────

@pytest.mark.parametrize("v", ["false", "FALSE", "0", "no", "off", " False "])
def test_kill_switch_removes_timeweb_everywhere(v):
    env = {"TIMEWEB_GEMINI_BASE_URL": "https://x/v1", "TIMEWEB_GEMINI_API_KEY": KEY, "OPENROUTER_API_KEY": "or",
           "SONYA_LLM_TIMEWEB_ENABLED": v}
    assert [p.name for p in lp.configured_providers(env)] == ["openrouter"]
    assert lp.timeweb_provider(env, keychain=False) is None
    assert lp.timeweb_enabled(env) is False


def test_timeweb_provider_env_and_defaults():
    p = lp.timeweb_provider({"TIMEWEB_GEMINI_API_KEY": KEY}, keychain=False)
    assert (p.name, p.base_url, p.model) == ("timeweb", "https://api.timeweb.ai/v1", "gemini/gemini-3.1-flash-lite")
    assert KEY not in repr(p)                                         # api_key is repr=False
    assert lp.timeweb_provider({}, keychain=False) is None


def test_keychain_read_uses_stdout_only_and_never_argv(monkeypatch):
    seen = {}

    def fake_run(argv, **kw):
        seen["argv"] = argv
        return SimpleNamespace(returncode=0, stdout=KEY + "\n", stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)
    p = lp.timeweb_provider({})
    assert p.api_key == KEY
    assert seen["argv"][:6] == ["security", "find-generic-password", "-s", "group.sonya.timeweb-ai", "-a", "api_key"]
    assert KEY not in " ".join(seen["argv"])


def test_keychain_failure_is_none_not_exception(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=44, stdout="", stderr="could not be found"))
    assert lp.timeweb_provider({}) is None

    def boom(*a, **k):
        raise FileNotFoundError("security")
    monkeypatch.setattr(subprocess, "run", boom)
    assert lp.keychain_secret("a", "b") is None


# ── errors: classification, retries, no provider switch ─────────────────────

@pytest.mark.parametrize("status,kind", [(401, "auth"), (403, "auth"), (429, "rate_limit"), (500, "server"),
                                         (503, "server"), (400, "bad_request"), (413, "too_large")])
def test_status_classes(status, kind):
    with pytest.raises(lp.LLMError) as e:
        chat(seq((status, json.dumps({"error": {"message": "nope"}}).encode())))
    assert e.value.kind == kind and e.value.attempts[-1]["kind"] == kind


@pytest.mark.parametrize("status", [401, 403, 400, 413])
def test_non_retryable_statuses_are_sent_once(status):
    t = seq((status, b"{}"))
    with pytest.raises(lp.LLMError):
        chat(t, retries=3)
    assert len(t.calls) == 1


def test_retry_429_then_success():
    sleeps = []
    t = seq((429, b"{}"), (503, b"{}"), (200, ok_body()))
    r = lp.chat_json(MSG, prompt_version="t", opt_in=True, providers=[prov()], transport=t, retries=2, backoff=1.0,
                     sleep=sleeps.append)
    assert r.text == '{"a": 1}' and len(t.calls) == 3 and sleeps == [1.0, 2.0]
    assert [a["ok"] for a in r.attempts] == [False, False, True]


def test_retries_exhausted_reports_last_kind_and_stays_on_provider():
    t = seq((500, b"{}"))
    with pytest.raises(lp.LLMError) as e:
        chat(t, retries=2)
    assert e.value.kind == "server" and len(t.calls) == 3
    assert {c[0] for c in t.calls} == {"https://api.timeweb.ai/v1/chat/completions"}   # never another provider


def test_timeout_and_network_are_retryable_and_classified():
    import socket
    import urllib.error
    t = seq(socket.timeout("timed out"), urllib.error.URLError(socket.timeout()), (200, ok_body()))
    assert chat(t, retries=2).text
    with pytest.raises(lp.LLMError) as e:
        chat(seq(socket.timeout("x")), retries=0)
    assert e.value.kind == "timeout"
    with pytest.raises(lp.LLMError) as e:
        chat(seq(ConnectionResetError("x")), retries=0)
    assert e.value.kind == "network"


def test_empty_and_broken_body_are_errors():
    with pytest.raises(lp.LLMError) as e:
        chat(seq((200, b"not json at all")))
    assert e.value.kind == "empty"
    with pytest.raises(lp.LLMError):
        chat(seq((200, json.dumps({"choices": [{"message": {"content": ""}}]}).encode())))


def test_oversized_request_is_not_sent():
    t = seq((200, ok_body()))
    big = [{"role": "user", "content": "x" * 5000}]
    with pytest.raises(lp.LLMError) as e:
        lp.chat_json(big, prompt_version="t", opt_in=True, providers=[prov()], transport=t, max_request_bytes=1000)
    assert e.value.kind == "too_large" and t.calls == []


# ── the key never leaks ─────────────────────────────────────────────────────

def test_key_not_in_errors_attempts_results_or_repr():
    echo = json.dumps({"error": {"message": f"Invalid key {KEY} for request"}}).encode()
    with pytest.raises(lp.LLMError) as e:
        chat(seq((401, echo)))
    blob = repr(e.value) + str(e.value) + json.dumps(e.value.attempts)
    assert KEY not in blob and "***" in blob
    r = chat(seq((200, ok_body())))
    assert KEY not in json.dumps(r.telemetry(), default=str) and KEY not in repr(r)
    # the key travels only in the Authorization header
    t = seq((200, ok_body()))
    chat(t)
    url, headers, body = t.calls[0]
    assert headers["Authorization"] == f"Bearer {KEY}" and KEY not in url and KEY.encode() not in body


def test_key_not_in_llm_calls_summary_or_exceptions_through_budget_llm(tmp_path, capsys):
    echo = json.dumps({"error": {"message": f"bad {KEY}"}}).encode()
    llm = budget.LLM(lp.chat_json, 1.0, "gemini/gemini-3.1-flash-lite", reasoning=False,
                     chat_kwargs=dict(providers=[prov()], transport=seq((401, echo))))
    with pytest.raises(budget.FatalLLMError) as e:
        llm.ask("s", MSG, "v", lambda d: d)
    out = capsys.readouterr()
    assert KEY not in str(e.value) + json.dumps(llm.calls, default=str) + json.dumps(llm.summary()) + out.out + out.err


# ── budget: price, tokens, limits ───────────────────────────────────────────

def test_price_known_and_flagged_estimate():
    assert budget.price("gemini/gemini-3.1-flash-lite") == (0.5e-6, 3.0e-6)
    assert "gemini/gemini-3.1-flash-lite" in budget.ESTIMATED_PRICES


def test_token_split_by_modality_and_booking():
    llm = budget.LLM(lp.chat_json, 1.0, "gemini/gemini-3.1-flash-lite", reasoning=False,
                     chat_kwargs=dict(providers=[prov()], transport=seq((200, ok_body()))))
    assert llm.ask("s", MSG, "v", lambda d: d) == {"a": 1}
    c = llm.calls[-1]
    assert c["tokens"] == {"text": 40, "image": 50, "video": 10}
    assert c["cost_usd"] == pytest.approx(100 * 0.5e-6 + 20 * 3.0e-6) and c["cost_source"] == "estimate"
    s = llm.summary()
    assert s["tokens_by_modality"]["image"] == 50 and s["price_is_estimate"] is True


def test_reasoning_field_not_sent_to_timeweb():
    t = seq((200, ok_body()))
    llm = budget.LLM(lp.chat_json, 1.0, "gemini/gemini-3.1-flash-lite", reasoning=False,
                     chat_kwargs=dict(providers=[prov()], transport=t))
    llm.ask("s", MSG, "v", lambda d: d)
    body = json.loads(t.calls[0][2])
    assert "reasoning" not in body and body["max_tokens"] == 8000 and body["response_format"] == {"type": "json_object"}
    t2 = seq((200, ok_body()))
    budget.LLM(lp.chat_json, 1.0, "m", chat_kwargs=dict(providers=[prov()], transport=t2)).ask("s", MSG, "v", lambda d: d)
    assert json.loads(t2.calls[0][2])["reasoning"] == {"effort": "low"}     # old (OpenRouter) path unchanged


def test_run_limit_stops_before_the_request():
    t = seq((200, ok_body()))
    llm = budget.LLM(lp.chat_json, 0.001, "gemini/gemini-3.1-flash-lite", reasoning=False,
                     chat_kwargs=dict(providers=[prov()], transport=t))
    with pytest.raises(budget.BudgetExceeded):
        llm.ask("s", MSG, "v", lambda d: d)                  # worst case 8000 out tokens = $0.024 > $0.001
    assert t.calls == [] and llm.calls[-1]["stopped"] == "budget"


def test_daily_limit_stops_before_request_and_is_shared_between_runs(tmp_path):
    led = budget.DailyLedger(str(tmp_path / "led.json"), 0.05)
    led.add(0.04)
    t = seq((200, ok_body()))
    llm = budget.LLM(lp.chat_json, 5.0, "gemini/gemini-3.1-flash-lite", reasoning=False, ledger=led,
                     chat_kwargs=dict(providers=[prov()], transport=t))
    with pytest.raises(budget.BudgetExceeded) as e:
        llm.ask("s", MSG, "v", lambda d: d, max_out=5000)    # forecast $0.015 → 0.04 + 0.015 > 0.05
    assert "daily" in str(e.value) and t.calls == [] and llm.calls[-1]["stopped"] == "daily_budget"
    llm.ask("s2", MSG, "v", lambda d: d, max_out=100)        # small call fits
    assert led.spent() == pytest.approx(0.04 + 100 * 0.5e-6 + 20 * 3e-6)
    # another process / run sees the same file
    assert budget.DailyLedger(str(tmp_path / "led.json"), 0.05).spent() == led.spent()


def test_daily_ledger_resets_on_new_day_and_survives_garbage(tmp_path):
    p = tmp_path / "l.json"
    budget.DailyLedger(str(p), 2, today="2026-11-04").add(1.5)
    assert budget.DailyLedger(str(p), 2, today="2026-11-05").spent() == 0.0
    assert budget.DailyLedger(str(p), 2, today="2026-11-04").spent() == 1.5
    p.write_text("{garbage")
    assert budget.DailyLedger(str(p), 2).spent() == 0.0


def test_auth_error_stops_the_run_without_retries():
    t = seq((401, b"{}"))
    llm = budget.LLM(lp.chat_json, 1.0, "gemini/gemini-3.1-flash-lite", reasoning=False,
                     chat_kwargs=dict(providers=[prov()], transport=t, retries=2))
    with pytest.raises(budget.FatalLLMError) as e:
        llm.ask("s", MSG, "v", lambda d: d)
    assert len(t.calls) == 1 and "401" in str(e.value) and isinstance(e.value, budget.LLMStop)


def _analyze_with(monkeypatch, tmp_path, exc):
    def fake_prepare(src, cands, frames_dir, sha, workers=4):
        for k, c in enumerate(cands):
            c.update(frames=[], dhash=1 << k, motion=1.0, sharpness=1.0, brightness=1.0)
        return cands
    monkeypatch.setattr(visual, "prepare_candidates", fake_prepare)
    calls = []

    class L:
        def ask(self, tag, *a, **k):
            calls.append(tag)
            raise exc
    shots = [{"id": 1, "start": 0.0, "end": 30.0}]
    try:
        visual.analyze(L(), "src.mp4", shots, [], [], [], 30.0, "sha", tmp_path, max_candidates=3, batch=3, log=lambda m: None)
    except Exception as e:                                   # noqa: BLE001
        return calls, e
    return calls, None


def test_visual_analyst_stops_at_once_on_fatal_error(monkeypatch, tmp_path):
    calls, err = _analyze_with(monkeypatch, tmp_path, budget.FatalLLMError("key rejected"))
    assert len(calls) == 1 and isinstance(err, budget.FatalLLMError)      # no per-window repeats of a rejected key


def test_visual_analyst_still_falls_back_per_window_on_ordinary_failure(monkeypatch, tmp_path):
    calls, err = _analyze_with(monkeypatch, tmp_path, RuntimeError("provider hiccup"))
    assert len(calls) > 1 and err is None                                 # old behaviour kept


# ── structured answers: array instead of object ─────────────────────────────

def test_array_wrapping_one_object_is_normalised():
    assert budget.parse_json('[{"windows": []}]') == {"windows": []}
    assert budget.parse_json('```json\n[ {"a": 1} ]\n```') == {"a": 1}
    assert budget.parse_json('Вот ответ: [{"a": 1}]') == {"a": 1}


def test_other_arrays_stay_arrays_for_the_strict_validator():
    assert budget.parse_json('[]') == []
    assert budget.parse_json('[{"a": 1}, {"b": 2}]') == [{"a": 1}, {"b": 2}]
    assert budget.parse_json('[1, 2]') == [1, 2]
    assert budget.parse_json("not json") is None
    with pytest.raises(budget.SchemaError):
        visual.validate_vision(budget.parse_json('[{"a": 1}, {"b": 2}]'), ["V1"])


def test_array_answer_validates_without_a_paid_repair_call():
    good = {"windows": [{"id": "V1", "shot_type": "closeup"}]}
    t = seq((200, ok_body(json.dumps([good]))))
    llm = budget.LLM(lp.chat_json, 1.0, "gemini/gemini-3.1-flash-lite", reasoning=False,
                     chat_kwargs=dict(providers=[prov()], transport=t))
    out = llm.ask("v", MSG, "v", lambda d: visual.validate_vision(d, ["V1"]))
    assert out["V1"]["shot_type"] == "closeup" and len(t.calls) == 1


# ── video input ─────────────────────────────────────────────────────────────

def test_video_part_counts_as_video_tokens_not_base64_text():
    big = "A" * 3_000_000
    msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"},
                                         {"type": "image_url", "image_url": {"url": "data:video/mp4;base64," + big}}]}]
    n = budget.estimate_input_tokens(msgs)
    assert budget.VIDEO_TOKENS <= n < budget.VIDEO_TOKENS + 200 and budget.count_videos(msgs) == 1


def test_clips_mode_message_has_one_video_per_window(tmp_path):
    clip = tmp_path / "c.mp4"
    clip.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    c = {"id": "V1", "start": 1.0, "end": 4.0, "frames": [], "clip": str(clip)}
    m = visual.vision_messages([c], [], {"V1": "текст"}, vision="clips")
    parts = m[1]["content"]
    vids = [p for p in parts if p.get("type") == "image_url"]
    assert len(vids) == 1 and vids[0]["image_url"]["url"].startswith("data:video/mp4;base64,")
    assert "видеофрагмент" in m[0]["content"]


# ── Trailer provider selection / kill switch ────────────────────────────────

def test_select_provider_auto_is_previous_behaviour():
    assert recap_runner.select_provider("auto") == ("auto", None)


def test_select_provider_timeweb_ok_and_off_and_missing_key():
    kind, cfg = recap_runner.select_provider("timeweb", env={"TIMEWEB_GEMINI_API_KEY": KEY}, keychain=False)
    assert kind == "timeweb" and cfg.model == "gemini/gemini-3.1-flash-lite"
    with pytest.raises(recap_runner.LLMUnavailable, match="выключен"):
        recap_runner.select_provider("timeweb", env={"TIMEWEB_GEMINI_API_KEY": KEY, "SONYA_LLM_TIMEWEB_ENABLED": "false"})
    with pytest.raises(recap_runner.LLMUnavailable, match="выключен"):
        recap_runner.select_provider("timeweb", timeweb_off=True, env={"TIMEWEB_GEMINI_API_KEY": KEY})
    with pytest.raises(recap_runner.LLMUnavailable, match="Ключ Timeweb не найден"):
        recap_runner.select_provider("timeweb", env={}, keychain=False)


def test_cli_no_timeweb_explicit_provider_exits_2_with_plain_message(capsys, monkeypatch):
    monkeypatch.setenv("SONYA_LLM_TIMEWEB_ENABLED", "true")
    rc = recap_runner.main(["--input", "/nonexistent.mp4", "--llm-provider", "timeweb", "--no-timeweb"])
    assert rc == 2 and "LLM недоступен" in capsys.readouterr().out


def test_cli_defaults_unchanged():
    # budget default stays 1.0 on the old path; Timeweb path defaults 0.50 / day 2.00
    assert budget.DEFAULT_RUN_USD == 0.50 and budget.DEFAULT_DAY_USD == 2.00
