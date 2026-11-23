"""
quota_view.py
=============
One answer to «сколько видео у меня осталось» (owner 2026-11-22), for GET /api/me/quota, the account menu,
the Mini App and @sonya_group_ai («Мой лимит»). Read-only: nothing is debited here.

Same rules as the real debit (entitlements.resolve_entitlement + prod_job_store.create_job_with_quota):
  admin / legacy Pro      -> unlimited
  active subscription     -> limit = ops_limit of the period, used = ops_used, resets_at = period_end
  otherwise               -> free quota (users.free_video_used / free_video_limit; never resets)
  a subscription for the mode that ended, no free left -> kind "expired" (left 0, «Продлить»)
2K («Высокое качество») costs quality.HIGH_UNITS processings (admins: 1).

Prices and plans are NOT defined here -- labels only, from scripts/pricing.py.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from scripts import entitlements, pricing, quality

# Next plan up (same mode) for «Повысить». Only ids that exist in pricing.PLAN_CATALOG.
_UPGRADE = {"cut_start": "cut_pro", "cut_pro": "cut_studio", "trailer_start": "trailer_pro",
            "trailer_pro": "trailer_studio"}

# Accepted ?mode=: a job mode (virality, streamer, ...) or a plan mode (cut, trailer, streamer).
_PLAN_TO_JOB_MODE = {"cut": "virality", "trailer": "trailer_film_breaker", "streamer": "streamer"}


def _iso(v: Any) -> Optional[str]:
    if v is None:
        return None
    return v if isinstance(v, str) else v.isoformat()


def plan_label(plan_id: str) -> str:
    """cut_pro -> «Нарезка Pro»."""
    if plan_id == pricing.TEST_PLAN_ID:
        return "Тест-драйв"
    plan = pricing.get_plan(plan_id)
    mode = plan.plan_mode if plan else plan_id.split("_", 1)[0]
    tier = plan_id.split("_", 1)[1] if "_" in plan_id else plan_id
    return f"{entitlements.PLAN_MODE_LABELS.get(mode, mode)} {tier.capitalize()}".strip()


def _sub_view(sub: Dict[str, Any]) -> Dict[str, Any]:
    limit, used = int(sub["ops_limit"]), int(sub["ops_used"])
    return {"plan": sub["plan_id"], "plan_label": plan_label(sub["plan_id"]), "mode": sub["plan_mode"],
            "used": min(used, limit), "limit": limit, "left": max(0, limit - used),
            "resets_at": _iso(sub["period_end"]), "unlimited": False}


# ?mode=hooks / storytelling (bot, site chips) -> job mode virality + that clip type («Тест-драйв» counter).
_CLIP_MODES = {"viral": "viral", "hooks": "hooks", "storytelling": "storytelling", "story": "storytelling"}


def job_mode_of(mode: Optional[str]) -> str:
    m = str(mode or "virality").strip().lower()
    if m in _CLIP_MODES:
        return "virality"
    return _PLAN_TO_JOB_MODE.get(m, m)


def clip_of(mode: Optional[str]) -> str:
    return _CLIP_MODES.get(str(mode or "").strip().lower(), "")


def _test_view(subs: List[dict], active: List[dict], job_mode: str, clip_type: str) -> Optional[Dict[str, Any]]:
    """«Тест-драйв» counter for this mode (running, or the latest one that ended), else None."""
    slot = entitlements.test_slot(job_mode, clip_type)
    if slot is None:
        return None
    key = entitlements.TEST_MODE_PREFIX + slot
    run = max((s for s in active if s["plan_mode"] == key), key=lambda s: s["period_end"], default=None)
    if run is None:
        return None
    v = _sub_view(run)
    v.update(plan_label=f"Тест-драйв · {pricing.TEST_SLOT_LABELS.get(slot, slot)}", mode=key)
    return v


def build(user: Dict[str, Any], mode: Optional[str] = "virality", *, subscriptions: Optional[List[dict]] = None,
          is_admin: bool = False, pack: Optional[Dict[str, int]] = None,
          now: Optional[datetime] = None, clip_type: str = "") -> Dict[str, Any]:
    """{plan, plan_label, kind, mode, used, limit, left, resets_at, unlimited, cost, action, upgrade_plan, others}.

    subscriptions: every user_subscriptions row of the user (None = look them up).
    pack: BEEP pack_status {bought, used, left}, only for mode=beep (None = not counted)."""
    now = now or datetime.now(timezone.utc)
    job_mode = job_mode_of(mode)
    clip_type = clip_type or clip_of(mode)
    plan_mode = entitlements.PLAN_MODE_BY_JOB_MODE.get(job_mode)
    subs = entitlements.get_user_subscriptions(str(user["id"])) if subscriptions is None else subscriptions
    active = entitlements.active_subscriptions(subs, now)
    high = 1 if is_admin else max(1, int(quality.HIGH_UNITS))
    out: Dict[str, Any] = {"mode": plan_mode or job_mode, "cost": {"standard": 1, "high": high},
                           "action": None, "upgrade_plan": None, "resets_at": None}
    others = [_sub_view(s) for s in active if s["plan_mode"] != plan_mode
              and not str(s["plan_mode"]).startswith(entitlements.TEST_MODE_PREFIX)]

    if is_admin or entitlements.legacy_pro_active(user, now):
        out.update(kind="admin" if is_admin else "legacy_pro", plan="admin" if is_admin else "pro",
                   plan_label="Безлимит" if is_admin else "Pro", used=None, limit=None, left=None,
                   unlimited=True, resets_at=None if is_admin else _iso(user.get("plan_active_until")))
        out["others"] = others
        return out

    sub = max((s for s in active if s["plan_mode"] == plan_mode), key=lambda s: s["period_end"], default=None) \
        if plan_mode else None
    free_limit = max(0, int(user.get("free_video_limit") or 0))
    free_used = min(free_limit, max(0, int(user.get("free_video_used") or 0)))

    if sub is not None:
        out.update(kind="subscription", **_sub_view(sub))
        if out["left"] <= 0:
            out["action"] = "renew"
            out["upgrade_plan"] = _UPGRADE.get(sub["plan_id"])
    else:
        ended = max((s for s in subs if plan_mode and s["plan_mode"] == plan_mode and s["period_end"] <= now),
                    key=lambda s: s["period_end"], default=None)
        if ended is not None and free_used >= free_limit:
            v = _sub_view(ended)
            out.update(kind="expired", plan=v["plan"], plan_label=v["plan_label"], used=v["limit"],
                       limit=v["limit"], left=0, unlimited=False, resets_at=None, ended_at=v["resets_at"],
                       action="renew", upgrade_plan=_UPGRADE.get(v["plan"]))
        else:
            out.update(kind="free", plan="free", plan_label="Бесплатно", used=free_used, limit=free_limit,
                       left=free_limit - free_used, unlimited=False, action="choose")
            if pack and int(pack.get("bought") or 0) > 0:
                bought, pused = int(pack["bought"]), min(int(pack.get("used") or 0), int(pack["bought"]))
                out.update(kind="pack", plan="beep_10", plan_label="BEEP · пакет",
                           limit=free_limit + bought, used=free_used + pused,
                           left=(free_limit - free_used) + (bought - pused))
                out["action"] = "choose" if out["left"] <= 0 else None
    test = _test_view(subs, active, job_mode, clip_type)
    if test is not None and (out.get("kind") != "subscription" or (out.get("left") or 0) <= 0) \
            and (test["left"] > 0 or out.get("kind") in ("free", "expired") and (out.get("left") or 0) <= 0):
        # Spent the same way as entitlements.resolve_entitlement: regular plan first, then the test counter.
        out.update(kind="test", **test)
        out["action"] = "choose" if test["left"] <= 0 else None
        out["upgrade_plan"] = None
    out["others"] = others
    return out


def _plural(n: int, one: str, few: str, many: str) -> str:
    return entitlements._plural(abs(int(n)), one, few, many)


_MONTHS = ("янв.", "февр.", "марта", "апр.", "мая", "июня", "июля", "авг.", "сент.", "окт.", "нояб.", "дек.")


def short_date(iso: Optional[str]) -> str:
    """'2026-12-12T…' -> «12 дек.» (Moscow date)."""
    if not iso:
        return ""
    try:
        from zoneinfo import ZoneInfo
        d = datetime.fromisoformat(str(iso).replace("Z", "+00:00")).astimezone(ZoneInfo("Europe/Moscow"))
    except Exception:
        return ""
    return f"{d.day} {_MONTHS[d.month - 1]}"


def text_lines(q: Dict[str, Any]) -> List[str]:
    """The same words as the site's glass card, as plain lines (bot «Мой лимит»)."""
    if q.get("unlimited"):
        lines = [f"{q['plan_label']}", "Видео без ограничений."]
        if q.get("resets_at"):
            lines.append(f"Подписка до {short_date(q['resets_at'])}")
        return lines
    left, limit = int(q.get("left") or 0), int(q.get("limit") or 0)
    lines = [q.get("plan_label") or "Бесплатно",
             f"Осталось {left} из {limit} {_plural(limit, 'видео', 'видео', 'видео')}"]
    if q.get("kind") == "test" and q.get("resets_at"):
        lines.append(f"Тест-драйв до {short_date(q['resets_at'])}")
    elif q.get("kind") == "subscription" and q.get("resets_at"):
        lines.append(f"Обновится {short_date(q['resets_at'])}")
    elif q.get("kind") == "expired":
        lines.append("Подписка закончилась — продлите тариф.")
    elif q.get("kind") == "free" and left <= 0:
        lines.append("Бесплатное видео использовано — выберите тариф.")
    if q.get("kind") == "test" and left <= 0:
        lines.append("Обработки этого режима в Тест-драйве закончились — выберите тариф.")
    if q.get("kind") == "subscription" and left <= 0:
        lines.append("Видео в этом периоде закончились — продлите или повысьте тариф.")
    high = (q.get("cost") or {}).get("high") or 3
    lines.append(f"Одна нарезка — 1 видео, в качестве 2K — {high}.")
    return lines
