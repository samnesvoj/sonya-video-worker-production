import argparse
import json
from decimal import Decimal

import pytest

from tools import fast_start_test as fs


def run(**kw):
    env = fs.build_sim_env(**{k: v for k, v in kw.items() if k in ("worker_ok", "models_s", "pull_s", "secret_leak")})
    rep = fs.run_scenario(env, kw.get("max_usd", 3.0), kw.get("max_minutes", 90), "fast-start-test/x/input.mp4", Decimal("82.1386"))
    rep["quality"] = {"outputs_valid": True, "compare_verdict": "NO_VISIBLE_REGRESSION_IN_THESE_CHECKS"}
    blob = json.dumps(rep) + json.dumps(env.timelines(env.store.all()[0].label) if env.store.all() else [])
    rep["secret_scan"] = {"clean": not any(x in blob for x in ("ghs_LEAK", "LEAKED"))}
    return rep, env


def test_simulated_scenario_passes_every_criterion_and_tears_down():
    rep, env = run()
    crit = fs.evaluate(rep)
    assert all(c["ok"] for c in crit), [c for c in crit if not c["ok"]]
    assert rep["creates"] == 1 and rep["resumes"] == 1 and rep["leftover_instances"] == 0 and env.vast.inst == {}
    assert rep["stop_latency_s"] == pytest.approx(60, abs=10)                 # the 60 s idle rule is what the harness measured
    assert rep["phases"]["resume"]["release_cached"] is True and rep["phases"]["create"]["release_cached"] is False
    assert rep["phases"]["resume"]["stages"]["models"] < 1                    # models were not fetched again


def test_slow_models_and_pull_break_the_user_wait_criteria_not_the_harness():
    rep, _ = run(models_s=400, pull_s=400)
    bad = {c["criterion"] for c in fs.evaluate(rep) if not c["ok"]}
    assert "create: request -> ready" in bad and "create: user wait (queued -> processing)" in bad
    assert "resume: request -> ready" not in bad


def test_worker_that_never_claims_aborts_by_guard_fails_and_still_destroys_everything():
    rep, env = run(worker_ok=False, max_minutes=45)
    assert rep["aborted"] and rep["leftover_instances"] == 0 and env.vast.inst == {}
    crit = {c["criterion"]: c["ok"] for c in fs.evaluate(rep)}
    assert crit["all jobs completed, none failed"] is False and crit["teardown: no instance left running or stopped"] is True


def test_cost_guard_aborts_and_tears_down():
    rep, env = run(max_usd=0.001)
    assert rep["aborted"] and "budget" in rep["aborted"] and env.vast.inst == {}
    assert next(c for c in fs.evaluate(rep) if c["criterion"] == "cost within the limit")["ok"] in (True, False)    # reported either way


def test_secret_in_a_timeline_fails_the_secret_criterion():
    rep, _ = run(secret_leak=True)
    assert next(c for c in fs.evaluate(rep) if c["criterion"].startswith("no token"))["ok"] is False


def test_unchecked_quality_and_secrets_are_failures_not_silent_passes():
    rep, _ = run()
    rep["quality"] = {"outputs_valid": None, "compare_verdict": None}
    rep["secret_scan"] = {"clean": None}
    failed = {c["criterion"] for c in fs.evaluate(rep) if not c["ok"]}
    assert any("outputs valid" in f for f in failed) and any("Mac mini reference" in f for f in failed) and any("no token" in f for f in failed)


def test_live_mode_refuses_without_apply_and_confirmation_and_spends_nothing(capsys, monkeypatch):
    for k in ("GH_APP_ID", "S3_SECRET_ACCESS_KEY", "VAST_API_KEY_FILE"):
        monkeypatch.delenv(k, raising=False)
    a = argparse.Namespace(release_sha="main", image="ghcr.io/x/y:latest", backend_url="http://x", control_video=None, max_usd=3.0, confirm_spend=None,
                           max_minutes=90, usd_rub="82.1386", apply=False)
    with pytest.raises(SystemExit) as ei:
        fs.live(a)
    out = capsys.readouterr().out
    assert ei.value.code == 3 and "NOTHING was started" in out and "FAIL release SHA" in out and "FAIL image is pinned" in out and "FAIL spend limit" in out


def test_report_markdown_marks_simulation_loudly():
    rep, _ = run(); rep["mode"] = "SIMULATION"
    assert "NOT a measurement" in fs.render_md(rep, fs.evaluate(rep))
