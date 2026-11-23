"""
entitlements.py
================
Server-side source of truth for "may this user start this operation, and
what is it billed against" (migration 016).

Model:
  * Each purchase of a public plan (scripts/pricing.py::PLAN_CATALOG)
    becomes one user_subscriptions row: plan_id, plan_mode (cut / trailer /
    streamer), ops_limit, ops_used, max_source_sec, period_start/end.
  * users.plan_type is NOT a source of rights for new plans. It only keeps
    meaning for legacy 500 ₽ "pro" buyers (unlimited, every mode, until
    plan_active_until) -- honored as sold, never mapped onto a new plan.
  * The free plan (users.free_video_used / free_video_limit) is unchanged.

Decision order for one operation (resolve_entitlement):
  1. legacy Pro active                     -> "legacy_pro"   (no debit, no plan cap)
  2. active subscription for the job's mode -> "subscription" (debit 1 op; plan cap)
       ... with ops exhausted              -> PLAN_LIMIT_REACHED
  3. free quota left                       -> "free"         (debit free quota)
  4. otherwise                              -> MODE_NOT_IN_PLAN | SUBSCRIPTION_EXPIRED | FREE_PLAN_USED

resolve_entitlement() is only the fast, pre-I/O decision. The actual debit
happens atomically in prod_job_store.create_job_with_quota(), which
re-checks the subscription row under the user's row lock -- so a stale
decision can never overspend (it becomes "plan_limit_reached" there).

Source length is checked with ffprobe, on the file already on this server
(after upload / after URL download), before S3 upload and before any debit.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

PLAN_MODES = ("cut", "trailer", "streamer")

# Job mode (ALLOWED_MODES in prod_generation_api.py) -> billable plan mode.
# sonya_gen is a placeholder mode with no public plan: free quota or
# legacy Pro only.
PLAN_MODE_BY_JOB_MODE: Dict[str, str] = {
    "virality": "cut",
    "stories": "cut",
    "educational": "cut",
    "trailer_film_breaker": "trailer",
    "streamer": "streamer",
}

PLAN_MODE_LABELS = {"cut": "Нарезка", "trailer": "Трейлер", "streamer": "Стример"}

# «Тест-драйв» (pricing.TEST_PLAN): one user_subscriptions row per slot, plan_mode "test:<slot>" (migration 026).
TEST_MODE_PREFIX = "test:"
_TEST_SLOT_BY_CLIP = {"": "viral", "viral": "viral", "virality": "viral", "hooks": "hooks",
                      "storytelling": "story", "stories": "story", "story": "story"}


def test_slot(job_mode: str, clip_type: Optional[str] = None) -> Optional[str]:
    """Which «Тест-драйв» counter a job spends: viral / hooks / story / edu / streamer / beep (None = not in it,
    e.g. Trailer, Gen)."""
    if job_mode == "virality":
        return _TEST_SLOT_BY_CLIP.get(str(clip_type or "").strip().lower())
    return {"stories": "story", "educational": "edu", "streamer": "streamer", "beep": "beep"}.get(job_mode)


def clip_type_of(params: Optional[dict]) -> str:
    return str((params or {}).get("clipType") or "").strip().lower()

# Denial codes -- the frontend keys its UI off these (auth.js).
FREE_PLAN_USED = "FREE_PLAN_USED"
MODE_NOT_IN_PLAN = "MODE_NOT_IN_PLAN"
SUBSCRIPTION_EXPIRED = "SUBSCRIPTION_EXPIRED"
PLAN_LIMIT_REACHED = "PLAN_LIMIT_REACHED"
SOURCE_TOO_LONG = "SOURCE_TOO_LONG"
SOURCE_DURATION_UNKNOWN = "SOURCE_DURATION_UNKNOWN"
DURATION_CHECK_UNAVAILABLE = "DURATION_CHECK_UNAVAILABLE"

_FFPROBE_TIMEOUT_SEC = 60


@dataclass(frozen=True)
class Entitlement:
    kind: str                              # "legacy_pro" | "subscription" | "free"
    plan_mode: Optional[str]
    subscription_id: Optional[str] = None
    plan_id: Optional[str] = None
    max_source_sec: Optional[int] = None   # None = no plan cap (existing infra caps only)

    @property
    def is_paid(self) -> bool:
        return self.kind in ("legacy_pro", "subscription")


class EntitlementDenied(Exception):
    """A user-facing refusal. Nothing has been debited when this is raised."""

    def __init__(self, code: str, message: str, status_code: int = 402, **extra: Any):
        super().__init__(code)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.extra = extra

    def detail(self, trace_id: str) -> Dict[str, Any]:
        error = "payment_required" if self.status_code == 402 else self.code.lower()
        return {"error": error, "code": self.code, "message": self.message,
                "trace_id": trace_id, **self.extra}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _plural(n: int, one: str, few: str, many: str) -> str:
    n10, n100 = n % 10, n % 100
    if n10 == 1 and n100 != 11:
        return one
    if 2 <= n10 <= 4 and not 12 <= n100 <= 14:
        return few
    return many


def legacy_pro_active(user: dict, now: Optional[datetime] = None) -> bool:
    """Legacy 500 ₽ Pro (pre-016): plan_type='pro' with an unexpired period."""
    if user.get("plan_type") != "pro" or user.get("plan_status") != "active":
        return False
    until = user.get("plan_active_until")
    return bool(until) and until > (now or _now())


def _test_entitlement(active: List[dict], job_mode: str, clip_type: Optional[str]) -> Optional[Entitlement]:
    """A running «Тест-драйв» counter for this job with something left, else None."""
    slot = test_slot(job_mode, clip_type)
    if slot is None:
        return None
    sub = max((s for s in active if s["plan_mode"] == TEST_MODE_PREFIX + slot and s["ops_used"] < s["ops_limit"]),
              key=lambda s: s["period_end"], default=None)
    if sub is None:
        return None
    return Entitlement(kind="subscription", plan_mode=sub["plan_mode"], subscription_id=str(sub["id"]),
                       plan_id=sub["plan_id"], max_source_sec=int(sub["max_source_sec"]))


def resolve_entitlement(user: dict, job_mode: str, subscriptions: List[dict],
                        now: Optional[datetime] = None, clip_type: Optional[str] = None) -> Entitlement:
    """Pure decision; raises EntitlementDenied. `subscriptions` = every
    user_subscriptions row of this user (active or not). clip_type (params.clipType) only picks the
    «Тест-драйв» counter (Вирал / Хуки / Сторителлинг are all job mode virality); a regular plan for the
    mode with operations left is always spent first, then the test counter, then the free quota."""
    now = now or _now()
    plan_mode = PLAN_MODE_BY_JOB_MODE.get(job_mode)

    if legacy_pro_active(user, now):
        return Entitlement(kind="legacy_pro", plan_mode=plan_mode)

    active = [s for s in subscriptions if s["period_start"] <= now < s["period_end"]]
    test_ent = _test_entitlement(active, job_mode, clip_type)
    if plan_mode is not None:
        sub = max((s for s in active if s["plan_mode"] == plan_mode),
                  key=lambda s: s["period_end"], default=None)
        if sub is not None:
            if sub["ops_used"] >= sub["ops_limit"] and test_ent is not None:
                return test_ent
            if sub["ops_used"] >= sub["ops_limit"]:
                raise EntitlementDenied(
                    PLAN_LIMIT_REACHED,
                    f"Лимит тарифа исчерпан: использовано {sub['ops_used']} из {sub['ops_limit']}. "
                    "Новый период начнётся после окончания текущего.",
                    plan_mode=plan_mode, plan_id=sub["plan_id"], ops_limit=sub["ops_limit"],
                    period_end=sub["period_end"].isoformat(),
                )
            return Entitlement(kind="subscription", plan_mode=plan_mode,
                               subscription_id=str(sub["id"]), plan_id=sub["plan_id"],
                               max_source_sec=int(sub["max_source_sec"]))

    if test_ent is not None:
        return test_ent

    if user["free_video_used"] < user["free_video_limit"]:
        return Entitlement(kind="free", plan_mode=plan_mode)

    # «Тест-драйв» rows never count as «a plan» for the refusals below; a spent test counter says so plainly
    # (BEEP falls through: its bought pack is checked after FREE_PLAN_USED in prod_generation_api).
    slot = test_slot(job_mode, clip_type)
    spent = next((s for s in active if slot and s["plan_mode"] == TEST_MODE_PREFIX + slot), None)
    if spent is not None and job_mode != "beep":
        from scripts.pricing import TEST_SLOT_LABELS
        raise EntitlementDenied(
            PLAN_LIMIT_REACHED,
            f"В Тест-драйве закончились обработки режима «{TEST_SLOT_LABELS.get(slot, slot)}»: "
            f"использовано {spent['ops_used']} из {spent['ops_limit']}. Выберите тариф, чтобы продолжить.",
            plan_mode=spent["plan_mode"], plan_id=spent["plan_id"], ops_limit=spent["ops_limit"],
            period_end=spent["period_end"].isoformat(),
        )
    active = [s for s in active if not str(s.get("plan_mode") or "").startswith(TEST_MODE_PREFIX)]

    label = PLAN_MODE_LABELS.get(plan_mode or "", "этого режима")
    if active:
        raise EntitlementDenied(
            MODE_NOT_IN_PLAN,
            f"Ваш тариф не включает режим «{label}». Выберите тариф для этого режима.",
            plan_mode=plan_mode,
        )
    if plan_mode is not None and any(s["plan_mode"] == plan_mode for s in subscriptions):
        raise EntitlementDenied(
            SUBSCRIPTION_EXPIRED,
            f"Подписка на режим «{label}» закончилась. Продлите тариф, чтобы продолжить.",
            plan_mode=plan_mode,
        )
    raise EntitlementDenied(
        FREE_PLAN_USED,
        "Бесплатная генерация уже использована. Выберите тариф SONYA.",
        plan_mode=plan_mode,
    )


def enforce_source_duration(ent: Entitlement, duration_sec: Optional[float]) -> None:
    """Plan cap only; free/legacy keep the existing infra caps. Raises
    EntitlementDenied (nothing debited yet at every call site)."""
    if ent.max_source_sec is None:
        return
    if duration_sec is None:
        raise EntitlementDenied(
            SOURCE_DURATION_UNKNOWN,
            "Не удалось определить длительность видео. Проверьте файл и попробуйте снова.",
            status_code=422,
        )
    if duration_sec > ent.max_source_sec:
        max_min = ent.max_source_sec // 60
        raise EntitlementDenied(
            SOURCE_TOO_LONG,
            f"Видео длиннее, чем позволяет тариф: {int(duration_sec // 60)} мин, "
            f"максимум — {max_min} {_plural(max_min, 'минута', 'минуты', 'минут')}.",
            status_code=413,
            plan_id=ent.plan_id, max_source_minutes=max_min,
            source_minutes=round(duration_sec / 60, 1),
        )


# ── Source duration (ffprobe) ────────────────────────────────────────────

class DurationCheckUnavailable(RuntimeError):
    """ffprobe binary missing -- a deploy problem, never the user's fault."""


def ffprobe_available() -> bool:
    return shutil.which(os.environ.get("FFPROBE_BIN", "ffprobe")) is not None


def probe_duration_sec(path: str) -> Optional[float]:
    """Container duration (falls back to the longest stream) of a local
    file, via ffprobe in a subprocess with a timeout -- untrusted media is
    never parsed inside the API process. None = file has no readable
    duration. Raises DurationCheckUnavailable if ffprobe isn't installed."""
    binary = shutil.which(os.environ.get("FFPROBE_BIN", "ffprobe"))
    if binary is None:
        raise DurationCheckUnavailable("ffprobe not found")
    try:
        proc = subprocess.run(
            [binary, "-v", "error", "-show_entries", "format=duration:stream=duration",
             "-of", "json", path],
            capture_output=True, timeout=_FFPROBE_TIMEOUT_SEC, check=False,
        )
    except subprocess.TimeoutExpired:
        logger.warning("[entitlements] ffprobe_timeout path=%s", path)
        return None
    if proc.returncode != 0:
        return None
    try:
        info = json.loads(proc.stdout or b"{}")
    except ValueError:
        return None
    candidates = [(info.get("format") or {}).get("duration")]
    candidates += [s.get("duration") for s in info.get("streams") or []]
    values = []
    for c in candidates:
        try:
            v = float(c)
        except (TypeError, ValueError):
            continue
        if v > 0:
            values.append(v)
    if not values:
        return None
    fmt = (info.get("format") or {}).get("duration")
    try:
        return float(fmt) if fmt and float(fmt) > 0 else max(values)
    except (TypeError, ValueError):
        return max(values)


def measure_and_enforce(ent: Entitlement, path: str) -> Optional[float]:
    """Measure only when a plan cap applies (free/legacy: unchanged flow).
    ffprobe missing -> fail closed with 503, no debit."""
    if ent.max_source_sec is None:
        return None
    try:
        duration = probe_duration_sec(path)
    except DurationCheckUnavailable:
        logger.error("[entitlements] ffprobe_missing -- paid processing blocked until installed")
        raise EntitlementDenied(
            DURATION_CHECK_UNAVAILABLE,
            "Проверка видео временно недоступна. Попробуйте позже.",
            status_code=503,
        )
    enforce_source_duration(ent, duration)
    return duration


# ── DB access ────────────────────────────────────────────────────────────

def _get_conn():
    from scripts.payment_store import _get_conn as conn  # same DATABASE_URL / driver
    return conn()


def get_user_subscriptions(user_id: str) -> List[dict]:
    """Every subscription row for the user, newest period first."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM user_subscriptions WHERE user_id = %s ORDER BY period_end DESC",
                (user_id,),
            )
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def active_subscriptions(subscriptions: List[dict], now: Optional[datetime] = None) -> List[dict]:
    now = now or _now()
    return [s for s in subscriptions if s["period_start"] <= now < s["period_end"]]


def serialize_subscription(sub: dict) -> Dict[str, Any]:
    """User-facing view -- no payment/amount/internal fields."""
    return {
        "plan_id": sub["plan_id"],
        "mode": sub["plan_mode"],
        "ops_limit": sub["ops_limit"],
        "ops_used": sub["ops_used"],
        "ops_remaining": max(0, sub["ops_limit"] - sub["ops_used"]),
        "max_source_minutes": sub["max_source_sec"] // 60,
        "period_start": sub["period_start"].isoformat(),
        "period_end": sub["period_end"].isoformat(),
    }
