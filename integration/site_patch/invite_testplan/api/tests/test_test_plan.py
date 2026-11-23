"""«Тест-драйв» (test_all): pure rules -- no DB. Run: python3 -m pytest -q tests/test_test_plan.py"""
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from scripts import entitlements as E, feature_flags, pricing, quota_view

NOW = datetime(2026, 11, 23, 12, tzinfo=timezone.utc)
USER = {"id": "u1", "plan_type": "free", "plan_status": None, "plan_active_until": None,
        "free_video_used": 1, "free_video_limit": 1}


def sub(mode, used=0, limit=10, plan="test_all", days=14, ended=False):
    start = NOW - timedelta(days=30 if ended else 1)
    return {"id": uuid.uuid4(), "plan_id": plan, "plan_mode": mode, "ops_limit": limit, "ops_used": used,
            "max_source_sec": 3600, "period_start": start, "period_end": start + timedelta(days=days),
            "payment_id": "p1"}


def make_pack(used=None):
    used = used or {}
    return [sub("test:" + s, used.get(s, 0)) for s in pricing.TEST_SLOTS]


@pytest.mark.parametrize("mode,clip,slot", [
    ("virality", "viral", "viral"), ("virality", "", "viral"), ("virality", "hooks", "hooks"),
    ("virality", "storytelling", "story"), ("stories", None, "story"), ("educational", None, "edu"),
    ("streamer", None, "streamer"), ("beep", None, "beep"), ("virality", "trailer", None),
    ("trailer_film_breaker", None, None), ("sonya_gen", None, None)])
def test_slot_map(mode, clip, slot):
    assert E.test_slot(mode, clip) == slot


def test_each_mode_spends_own_counter():
    subs = make_pack()
    for mode, clip, slot in [("virality", "hooks", "hooks"), ("educational", None, "edu"), ("beep", None, "beep")]:
        ent = E.resolve_entitlement(USER, mode, subs, NOW, clip_type=clip)
        assert ent.kind == "subscription" and ent.plan_mode == "test:" + slot and ent.max_source_sec == 3600
        assert ent.subscription_id == str(next(s["id"] for s in subs if s["plan_mode"] == "test:" + slot))


def test_spent_counter_refuses_plainly_other_modes_still_work():
    subs = make_pack({"viral": 10})
    with pytest.raises(E.EntitlementDenied) as e:
        E.resolve_entitlement(USER, "virality", subs, NOW, clip_type="viral")
    assert e.value.code == E.PLAN_LIMIT_REACHED and "Вирал" in e.value.message and "10 из 10" in e.value.message
    assert E.resolve_entitlement(USER, "virality", subs, NOW, clip_type="hooks").plan_mode == "test:hooks"


def test_spent_beep_falls_through_to_pack_flow():
    with pytest.raises(E.EntitlementDenied) as e:
        E.resolve_entitlement(USER, "beep", make_pack({"beep": 10}), NOW)
    assert e.value.code == E.FREE_PLAN_USED          # prod_generation_api then checks the bought BEEP pack


def test_regular_plan_first_then_test():
    subs = make_pack() + [sub("cut", 3, 30, plan="cut_start", days=30)]
    assert E.resolve_entitlement(USER, "virality", subs, NOW, clip_type="viral").plan_id == "cut_start"
    subs[-1]["ops_used"] = 30
    assert E.resolve_entitlement(USER, "virality", subs, NOW, clip_type="viral").plan_mode == "test:viral"


def test_test_before_free_and_trailer_not_included():
    user = dict(USER, free_video_used=0)
    assert E.resolve_entitlement(user, "virality", make_pack(), NOW, clip_type="viral").kind == "subscription"
    assert E.resolve_entitlement(user, "virality", make_pack(), NOW, clip_type="trailer").kind == "free"
    with pytest.raises(E.EntitlementDenied) as e:            # test rows are not «a plan without this mode»
        E.resolve_entitlement(USER, "trailer_film_breaker", make_pack(), NOW)
    assert e.value.code == E.FREE_PLAN_USED


def test_expired_test_is_ignored():
    subs = [sub("test:" + s, ended=True) for s in pricing.TEST_SLOTS]
    with pytest.raises(E.EntitlementDenied) as e:
        E.resolve_entitlement(USER, "virality", subs, NOW, clip_type="viral")
    assert e.value.code == E.FREE_PLAN_USED


def test_no_test_rows_behaviour_unchanged():
    subs = [sub("cut", 30, 30, plan="cut_start", days=30)]
    with pytest.raises(E.EntitlementDenied) as e:
        E.resolve_entitlement(USER, "virality", subs, NOW, clip_type="viral")
    assert e.value.code == E.PLAN_LIMIT_REACHED and "Тест" not in e.value.message


def test_plan_hidden_until_flag(monkeypatch):
    monkeypatch.delenv("SONYA_TEST_PLAN", raising=False)
    plan = pricing.get_plan("test_all")
    assert plan is not None and plan.plan_mode == "test" and plan.ops_limit == 10 and plan.duration_days == 14
    assert "test_all" not in pricing.PLAN_CATALOG                 # bot plan list does not show it
    assert feature_flags.plan_disabled(plan) is True
    assert feature_flags.disabled_detail("t", plan)["code"] == "TEST_PLAN_DISABLED"
    assert feature_flags.public_flags()["test_plan"] is False
    monkeypatch.setenv("SONYA_TEST_PLAN", "on")
    assert feature_flags.plan_disabled(plan) is False and feature_flags.public_flags()["test_plan"] is True


def test_existing_prices_untouched():
    want = {"cut_start": "1090.00", "cut_pro": "2690.00", "cut_studio": "4990.00", "trailer_start": "1190.00",
            "trailer_pro": "2190.00", "trailer_studio": "4990.00", "streamer_start": "1990.00",
            "gen_100": "349.00", "beep_10": "249.00"}
    assert {k: str(v.amount) for k, v in pricing.PLAN_CATALOG.items()} == want


def test_quota_view_shows_test_counter():
    q = quota_view.build(USER, "hooks", subscriptions=make_pack({"hooks": 4}), now=NOW)
    assert q["kind"] == "test" and q["left"] == 6 and q["limit"] == 10 and "Хуки" in q["plan_label"]
    assert all(not o["mode"].startswith("test:") for o in q["others"])
    q = quota_view.build(USER, "virality", subscriptions=make_pack({"viral": 10}), now=NOW, clip_type="viral")
    assert q["kind"] == "test" and q["left"] == 0 and q["action"] == "choose"
    assert any("закончились" in l for l in quota_view.text_lines(q))
    q = quota_view.build(USER, "trailer", subscriptions=make_pack(), now=NOW)
    assert q["kind"] == "free"
    assert quota_view.plan_label("test_all") == "Тест-драйв"
