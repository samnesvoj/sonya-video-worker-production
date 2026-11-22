"""«Мидролл» API side (scripts/midroll_routes.py + presets_routes.clean_settings): own keys only, claim-time params."""
from scripts import midroll_routes as mr
from scripts.presets_routes import clean_settings

U = "55e4d9fd-1111-2222-3333-444455556666"
KEY = f"users/{U}/midroll/0123456789abcdef.mp4"
OTHER = "users/aaaaaaaa-1111-2222-3333-444455556666/midroll/0123456789abcdef.mp4"


def test_own_key_and_clean():
    assert mr.own_key(KEY, U) and not mr.own_key(OTHER, U)
    assert not mr.own_key(f"users/{U}/jobs/x/0123456789abcdef.mp4", U)
    assert not mr.own_key(f"users/{U}/midroll/../0123456789abcdef.mp4", U)
    c = mr.clean({"key": KEY, "name": "<b>промо</b>.mp4", "place": "pre", "every": 2, "max_total": 75, "after_s": 1}, U)
    assert c["place"] == "pre" and c["every"] == 2 and c["max_total"] == 0 and c["after_s"] == 3 and "<" not in c["name"]
    assert mr.clean({"key": OTHER}, U) is None and mr.clean("x", U) is None
    assert mr.job_value({"key": KEY, "on": False}, U) is None
    assert set(mr.job_value({"key": KEY}, U)) == {"key", "place", "at", "after_s", "every", "fit", "max_total", "dur"}


def test_presets_keep_only_own_midroll():
    assert clean_settings({"midroll": {"key": KEY, "fit": "blur"}}, U)["midroll"]["fit"] == "blur"
    assert clean_settings({"midroll": {"key": OTHER}}, U)["midroll"] is None
    assert "midroll" not in clean_settings({"quality": "high"}, U)


def _job(params):
    return {"id": "j1", "user_id": U, "params": params}


def test_claim_explicit_value_checked():
    assert mr.with_midroll(_job({"midroll": {"key": KEY}}))["params"]["midroll"]["key"] == KEY
    assert mr.with_midroll(_job({"midroll": {"key": OTHER}}))["params"]["midroll"] is None
    assert mr.with_midroll(_job({"midroll": None}))["params"]["midroll"] is None


def test_claim_default_preset_for_bot_jobs(monkeypatch):
    doc = {"default_id": "p1", "presets": [{"id": "p1", "settings": {"midroll": {"key": KEY, "every": 2}}},
                                           {"id": "p2", "settings": {}}]}
    j = mr.with_midroll(_job({"settings": {}}), default_preset_loader=lambda uid: doc)
    assert j["params"]["midroll"]["every"] == 2
    off = {"default_id": "p2", "presets": doc["presets"]}
    assert "midroll" not in mr.with_midroll(_job({}), default_preset_loader=lambda uid: off)["params"]

    def boom(uid):
        raise RuntimeError("db down")
    assert mr.with_midroll(_job({"a": 1}), default_preset_loader=boom)["params"] == {"a": 1}
    monkeypatch.setenv("SONYA_MIDROLL", "off")
    assert mr.with_midroll(_job({"midroll": {"key": KEY}}))["params"]["midroll"] is None
