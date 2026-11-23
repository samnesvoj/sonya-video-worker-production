"""
pricing.py
==========
Server-side source of truth for SONYA's public plans: price, period, and
the entitlement each plan grants. Payment-provider independent -- a
provider (scripts/payment_providers.py) only collects and confirms the
money for a payment whose plan and amount were fixed here at checkout:

    plan_id -> PLAN_CATALOG -> payments row (snapshot) -> provider confirms
            -> payment_store.process_successful_payment() -> user_subscriptions

Ids must match SONYA_PRICING in auth.js. Streamer Pro / Max are
intentionally absent (not public).
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import os
from typing import Optional


@dataclass(frozen=True)
class Plan:
    plan_id: str
    # Snapshot label stored on payments.plan_type (NOT NULL, migration 010).
    # Public plans use "subscription" -- they never set users.plan_type; their
    # rights live in user_subscriptions (migration 016, entitlements.py).
    plan_type: str
    amount: Decimal
    duration_days: int
    description: str
    # Entitlement terms -- snapshotted onto the payment row at checkout and
    # copied into user_subscriptions on activation.
    plan_mode: str        # "cut" | "trailer" | "streamer"
    ops_limit: int        # operations per period
    max_source_sec: int   # max source video length


def _monthly(plan_id: str, mode: str, amount: str, ops: int, max_source_min: int,
             description: str) -> Plan:
    return Plan(plan_id=plan_id, plan_type="subscription", amount=Decimal(amount),
                duration_days=30, description=description, plan_mode=mode,
                ops_limit=ops, max_source_sec=max_source_min * 60)


PLAN_CATALOG: dict[str, Plan] = {p.plan_id: p for p in (
    _monthly("cut_start",      "cut",      "1090.00", 30,  60, "SONYA Нарезка Start — подписка на 1 месяц"),
    _monthly("cut_pro",        "cut",      "2690.00", 50, 120, "SONYA Нарезка Pro — подписка на 1 месяц"),
    _monthly("cut_studio",     "cut",      "4990.00", 80, 180, "SONYA Нарезка Studio — подписка на 1 месяц"),
    _monthly("trailer_start",  "trailer",  "1190.00",  8,  90, "SONYA Трейлер Start — подписка на 1 месяц"),
    _monthly("trailer_pro",    "trailer",  "2190.00", 12, 120, "SONYA Трейлер Pro — подписка на 1 месяц"),
    _monthly("trailer_studio", "trailer",  "4990.00", 24, 180, "SONYA Трейлер Studio — подписка на 1 месяц"),
    _monthly("streamer_start", "streamer", "1990.00", 10, 240, "SONYA Стример Start — подписка на 1 месяц"),
    # SONYA Gen image pack: one-off purchase (not a subscription), never
    # expires. Opens no user_subscriptions row (payment_store); the balance is
    # counted from paid payments (plan_mode "gen") and generation_jobs
    # (prod_generation_api._gen_quota). description = receipt item name.
    Plan(plan_id="gen_100", plan_type="pack", amount=Decimal("349.00"), duration_days=0,
         description="Генерация изображений SONYA Gen, 100 шт.", plan_mode="gen",
         ops_limit=100, max_source_sec=0),
    # SONYA BEEP pack: 10 video processings, one-off, never expires. Spent only after the free quota
    # (a legacy Pro / admin never spends it); balance = paid plan_mode "beep" payments - beep jobs billed
    # "pack:beep" that did not fail (prod_job_store.create_job_with_quota).
    Plan(plan_id="beep_10", plan_type="pack", amount=Decimal("249.00"), duration_days=0,
         description="Обработка видео SONYA BEEP, 10 шт.", plan_mode="beep",
         ops_limit=10, max_source_sec=0),
)}

# Plan modes sold as one-off packs (no subscription period).
PACK_PLAN_MODES = frozenset({"gen", "beep"})

# ── «Тест-драйв» (owner 2026-11-22: «в каждом режиме 10 разных обработок видео») ──────────────────────────
# One purchase per account, 14 days, ops_limit processings in EACH video mode below (6 x 10 = 60), source up
# to 60 min. On payment payment_store opens one user_subscriptions row per slot (plan_mode "test:<slot>",
# migration 026); entitlements.test_slot() maps a job to its slot, the debit/refund path is the normal
# subscription one (charged_subscription_id, trigger 025). 2K takes quality.HIGH_UNITS like everywhere.
# NOT ON SALE until the owner approves the price: new checkouts are refused unless SONYA_TEST_PLAN=on
# (feature_flags.plan_disabled, like Trailer). get_plan() always knows it, so a payment made while it was on
# is still priced, receipted and activated. Not in PLAN_CATALOG -> the bot's plan list does not show it. The amount below is the PROPOSAL sent to the owner (cost calc ~/SONYA/test-tariff).
TEST_PLAN_ID = "test_all"
TEST_PLAN_MODE = "test"
TEST_SLOTS = ("viral", "hooks", "story", "edu", "streamer", "beep")
TEST_SLOT_LABELS = {"viral": "Вирал", "hooks": "Хуки", "story": "Сторителлинг", "edu": "Образовательный",
                    "streamer": "Стример", "beep": "BEEP"}
TEST_PLAN = Plan(plan_id=TEST_PLAN_ID, plan_type="subscription",
                 amount=Decimal(os.environ.get("SONYA_TEST_PLAN_PRICE", "1490.00")), duration_days=14,
                 description="SONYA Тест-драйв — 10 обработок в каждом режиме, 14 дней",
                 plan_mode=TEST_PLAN_MODE, ops_limit=10, max_source_sec=60 * 60)


def test_plan_enabled() -> bool:
    return os.environ.get("SONYA_TEST_PLAN", "off").strip().lower() in ("on", "1", "true", "yes")


def get_plan(plan_id: str) -> Optional[Plan]:
    if plan_id == TEST_PLAN_ID:
        return TEST_PLAN
    return PLAN_CATALOG.get(plan_id)
