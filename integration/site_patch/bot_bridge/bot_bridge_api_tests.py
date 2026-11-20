"""@sonya_group_ai bridge (scripts/bot_bridge_routes.py): secret, linked user, params as the site builds them,
the site's own job / checkout functions are called with the linked user."""
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from scripts import bot_bridge_routes as B
from scripts import prod_generation_api as api
from scripts import payment_routes

SECRET = "bridge-test-secret"
H = {"X-Sonya-Bot-Secret": SECRET}
USER = {"id": "u-1", "email": "anna@example.com", "plan_type": "free", "plan_status": None,
        "plan_active_until": None, "free_video_limit": 1, "free_video_used": 0, "telegram_linked": True}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("SONYA_BOT_BRIDGE_SECRET", SECRET)
    monkeypatch.setattr(B, "find_linked_user", lambda tg: dict(USER) if tg == 777 else None)
    monkeypatch.setattr(B, "preset_of", lambda uid: ("Мой стиль", dict(B.PRESET_DEFAULTS, subtitleStyle="gradient",
                                                                     max_clips=8, hook=True)))
    return TestClient(api.app)


def test_secret_required(client):
    assert client.post("/api/telegram/bot/account", json={"telegram_user_id": 777}).status_code == 401
    assert client.post("/api/telegram/bot/account", json={"telegram_user_id": 777},
                       headers={"X-Sonya-Bot-Secret": "nope"}).status_code == 401


def test_off_switch(client, monkeypatch):
    monkeypatch.setenv("SONYA_BOT_BRIDGE", "off")
    assert client.post("/api/telegram/bot/account", json={"telegram_user_id": 777}, headers=H).status_code == 404


def test_not_linked(client):
    r = client.post("/api/telegram/bot/account", json={"telegram_user_id": 5}, headers=H)
    assert r.status_code == 200 and r.json()["linked"] is False and r.json()["connect_url"].endswith("/?tg=connect")
    r = client.post("/api/telegram/bot/jobs/url", json={"telegram_user_id": 5, "url": "https://vk.com/video1_2",
                                                        "clip": "viral"}, headers=H)
    assert r.status_code == 401 and r.json()["detail"]["error"] == "not_linked"


def test_bad_tg_id(client):
    assert client.post("/api/telegram/bot/account", json={"telegram_user_id": "x"}, headers=H).status_code == 400
    assert client.post("/api/telegram/bot/account", json={"telegram_user_id": True}, headers=H).status_code == 400


def test_account_access(client, monkeypatch):
    monkeypatch.setattr(api.entitlements, "get_user_subscriptions", lambda uid: [])
    r = client.post("/api/telegram/bot/account", json={"telegram_user_id": 777}, headers=H).json()
    assert r["linked"] and not r["is_admin"] and r["email_hint"] == "an•••@example.com"
    assert "trailer" not in r["access"]                      # admins only
    assert r["access"]["viral"]["allowed"] and r["access"]["viral"]["left"] == 1
    assert r["preset"]["name"] == "Мой стиль"


def test_account_denied_carries_code(client, monkeypatch):
    monkeypatch.setattr(B, "find_linked_user", lambda tg: dict(USER, free_video_used=1))
    monkeypatch.setattr(api.entitlements, "get_user_subscriptions", lambda uid: [])
    monkeypatch.setattr(api, "pack_status", lambda uid, mode: {"left": 0})
    r = client.post("/api/telegram/bot/account", json={"telegram_user_id": 777}, headers=H).json()
    a = r["access"]["viral"]
    assert a["allowed"] is False and a["code"] == "FREE_PLAN_USED" and a["max_source_minutes"] > 0
    assert r["access"]["beep"]["allowed"] is False


def test_admin_sees_trailer(client, monkeypatch):
    monkeypatch.setattr(B, "find_linked_user", lambda tg: dict(USER, plan_type="admin"))
    monkeypatch.setattr(api.entitlements, "get_user_subscriptions", lambda uid: [])
    r = client.post("/api/telegram/bot/account", json={"telegram_user_id": 777}, headers=H).json()
    assert "trailer" in r["access"] and r["quality"]["units"] == 1


def test_params_like_site():
    preset = ("Мой стиль", dict(B.PRESET_DEFAULTS, subtitleStyle="gradient", max_clips=8, hook=True, edu_template=3))
    p = B.build_params("story", "https://youtu.be/x", "youtube", preset, "high", None, None)
    assert p["clipType"] == "storytelling" and p["settings"]["quality"] == "high"
    assert p["settings"]["subtitleStyle"] == "gradient" and p["max_clips"] == 8 and p["hook"] == 1
    assert p["settings"]["preset"] == "Мой стиль" and p["frontend"] == "telegram_bot"
    assert p["source"] == {"mode": "url", "url": "https://youtu.be/x", "platform": "youtube"}
    e = B.build_params("educational", "u", "vk", preset, "high", None, None)
    assert e["settings"]["quality"] == "standard" and e["settings"]["subtitleStyle"] == "viral"
    assert e["settings"]["edu_template"] == 3 and "hook" not in e
    s = B.build_params("streamer", "u", "twitch", ("", B.PRESET_DEFAULTS), "standard", None, None)
    assert s["settings"]["streamer_layout"] == 1 and s["settings"]["moment_types"] == [] and "preset" not in s["settings"]
    b = B.build_params("beep", "u", "vk", preset, "high", 0, 600)
    assert b["beep"] == {"effect": "beep", "categories": ["mat"]} and "clipType" not in b
    assert b["trim_end_sec"] == 600 and b["trim_first_sec"] == 600 and b["trim_start_sec"] == 0
    t = B.build_params("viral", None, "upload", preset, "standard", 300, 900, file_name="a.mp4")
    assert t["trim_start_sec"] == 300 and "trim_first_sec" not in t and t["source"]["mode"] == "upload"


def test_job_from_url_calls_site_path(client, monkeypatch):
    seen = {}

    async def fake(**kw):
        seen.update(kw)
        return {"ingest_id": "ing-1", "status": "checking", "platform": "vk"}
    monkeypatch.setattr(api, "create_generation_job_from_url", fake)
    r = client.post("/api/telegram/bot/jobs/url", headers=H, json={
        "telegram_user_id": 777, "url": "https://vk.com/video1_2", "clip": "hooks", "quality": "high", "key": "abc"})
    assert r.status_code == 202 and r.json()["ingest_id"] == "ing-1"
    assert seen["user"]["id"] == "u-1" and seen["body"].mode == "virality"
    assert seen["body"].params["clipType"] == "hooks" and seen["body"].params["settings"]["quality"] == "high"
    assert seen["idempotency_key_header"] == "tg-abc"


def test_job_402_passes_through(client, monkeypatch):
    async def fake(**kw):
        raise HTTPException(status_code=402, detail={"error": "payment_required", "code": "FREE_PLAN_USED"})
    monkeypatch.setattr(api, "create_generation_job_from_url", fake)
    r = client.post("/api/telegram/bot/jobs/url", headers=H,
                    json={"telegram_user_id": 777, "url": "https://vk.com/video1_2", "clip": "viral"})
    assert r.status_code == 402 and r.json()["detail"]["code"] == "FREE_PLAN_USED"


def test_trailer_refused_for_users(client):
    r = client.post("/api/telegram/bot/jobs/url", headers=H,
                    json={"telegram_user_id": 777, "url": "https://vk.com/video1_2", "clip": "trailer"})
    assert r.status_code == 400


def test_trim_range(client):
    r = client.post("/api/telegram/bot/jobs/url", headers=H, json={
        "telegram_user_id": 777, "url": "https://vk.com/video1_2", "clip": "viral", "trim_start_sec": 600, "trim_end_sec": 60})
    assert r.status_code == 422


def test_job_file(client, monkeypatch):
    seen = {}

    async def fake(**kw):
        seen.update(kw)
        return api.JobResponse(job_id="j-9", status="queued", mode=kw["mode"])
    monkeypatch.setattr(api, "create_generation_job", fake)
    r = client.post("/api/telegram/bot/jobs/file", headers=H, data={"telegram_user_id": "777", "clip": "beep"},
                    files={"file": ("v.mp4", b"\x00\x00\x00\x18ftypmp42", "video/mp4")})
    assert r.status_code == 202 and r.json()["job_id"] == "j-9" and seen["mode"] == "beep"
    assert '"frontend": "telegram_bot"' in seen["params"]


def test_job_status_owner_only(client, monkeypatch):
    monkeypatch.setattr(api, "get_job", lambda jid: {"id": jid, "user_id": "someone-else", "status": "queued"})
    assert client.post("/api/telegram/bot/job", headers=H, json={"telegram_user_id": 777, "job_id": "j"}).status_code == 404
    monkeypatch.setattr(api, "get_job", lambda jid: {"id": jid, "user_id": "u-1", "status": "assembling",
                                                     "params": {"clipType": "viral"}})
    r = client.post("/api/telegram/bot/job", headers=H, json={"telegram_user_id": 777, "job_id": "j"}).json()
    assert r["stage"] == "edit" and r["error"] is None and r["clip_type"] == "viral"


def test_job_status_done_and_failed(client, monkeypatch):
    monkeypatch.setattr(api, "get_job", lambda jid: {"id": jid, "user_id": "u-1", "status": "completed", "params": {}})
    monkeypatch.setattr(api, "list_job_files", lambda jid: [
        {"file_type": "output", "s3_key": "a/1.mp4"}, {"file_type": "output", "s3_key": "a/2.mp4"},
        {"file_type": "preview", "s3_key": "a/1_preview.mp4"}])
    r = client.post("/api/telegram/bot/job", headers=H, json={"telegram_user_id": 777, "job_id": "j"}).json()
    assert r["stage"] == "done" and r["clips"] == 2
    monkeypatch.setattr(api, "get_job", lambda jid: {"id": jid, "user_id": "u-1", "status": "failed", "error": "x",
                                                     "last_error": "SOURCE_UNSAFE_URL", "params": {"quota_refunded": True}})
    r = client.post("/api/telegram/bot/job", headers=H, json={"telegram_user_id": 777, "job_id": "j"}).json()
    assert r["stage"] == "failed" and r["error_code"] == "SOURCE_UNSAFE_URL" and r["refunded"] is True


def test_high_load_notice(client, monkeypatch):
    monkeypatch.setattr(api, "get_job", lambda jid: {"id": jid, "user_id": "u-1", "status": "queued", "params": {}})
    monkeypatch.setattr(api, "_fleet_no_funds", lambda: True)
    r = client.post("/api/telegram/bot/job", headers=H, json={"telegram_user_id": 777, "job_id": "j"}).json()
    assert r["notice"] == api.NOTICE_HIGH_LOAD


def test_pricing_prices_as_sold(client):
    from scripts import pricing
    plans = {p["plan_id"]: p for p in client.post("/api/telegram/bot/pricing", headers=H, json={}).json()["plans"]}
    for pid in ("cut_start", "cut_pro", "cut_studio", "beep_10", "gen_100", "streamer_start"):
        assert plans[pid]["price"] == int(pricing.PLAN_CATALOG[pid].amount)
    assert plans["cut_pro"]["ops"] == 50 and plans["cut_pro"]["max_source_minutes"] == 120


def test_checkout_is_site_checkout(client, monkeypatch):
    seen = {}

    async def fake(**kw):
        seen.update(kw)
        return {"invoice_id": 42, "status": "pending", "redirect_url": "https://yoomoney.ru/checkout/x"}
    monkeypatch.setattr(payment_routes, "checkout", fake)
    r = client.post("/api/telegram/bot/checkout", headers=H, json={"telegram_user_id": 777, "plan_id": "cut_pro", "key": "k1"})
    assert r.json()["redirect_url"].startswith("https://yoomoney.ru/")
    assert seen["body"].plan_id == "cut_pro" and seen["user"]["id"] == "u-1" and seen["idempotency_key_header"] == "tg-k1"


def test_payment_status(client, monkeypatch):
    async def fake(**kw):
        assert kw["invoice_id"] == 42 and kw["user"]["id"] == "u-1"
        return {"status": "paid", "plan_id": "cut_pro"}
    monkeypatch.setattr(payment_routes, "payment_status", fake)
    r = client.post("/api/telegram/bot/payment", headers=H, json={"telegram_user_id": 777, "invoice_id": "42"})
    assert r.json()["status"] == "paid"


def test_delivery_count(client, monkeypatch, tmp_path):
    import sqlite3
    db = tmp_path / "state.db"
    c = sqlite3.connect(db)
    c.execute("CREATE TABLE sent(job_id TEXT, s3_key TEXT, status TEXT, attempts INT DEFAULT 0, ts TEXT)")
    c.executemany("INSERT INTO sent VALUES(?,?,?,?,?)", [("j", "a", "ok", 1, ""), ("j", "b", "fail", 5, ""),
                                                         ("j", "c", "fail", 1, ""), ("x", "d", "ok", 1, "")])
    c.commit(); c.close()
    monkeypatch.setenv("SONYA_TG_STATE", str(db))
    monkeypatch.setattr(api, "get_job", lambda jid: {"id": jid, "user_id": "u-1", "status": "completed", "params": {}})
    monkeypatch.setattr(api, "list_job_files", lambda jid: [{"file_type": "output", "s3_key": f"{i}.mp4"} for i in range(3)])
    r = client.post("/api/telegram/bot/job", headers=H, json={"telegram_user_id": 777, "job_id": "j"}).json()
    assert r["delivered"] == 1 and r["delivery_failed"] == 1 and r["clips"] == 3
    monkeypatch.setenv("SONYA_TG_STATE", str(tmp_path / "none.db"))
    r = client.post("/api/telegram/bot/job", headers=H, json={"telegram_user_id": 777, "job_id": "j"}).json()
    assert r["delivered"] is None
