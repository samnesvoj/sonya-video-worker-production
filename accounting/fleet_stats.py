"""Server creates / destroys of the Vast pool for the accounting bot (owner 2026-11-18: «серверов создано за сутки» and
«из них по причинам») + the fleet budget alerts. Reads the fleet journal only (dispatcher/store.py), never Vast."""
from __future__ import annotations

from collections import Counter
from decimal import Decimal
from typing import List, Optional

from accounting import tgformat as tf

CREATE_WHY = {"demand": "под задания", "prime": "прогрев warm", "reserve": "резерв POWER"}
DESTROY_WHY = {
    "burst_idle": "простой после нагрузки", "nvenc_missing": "нет NVENC", "resume_failures": "не проснулся (GPU занят)",
    "youtube_bot_check": "YouTube «не бот»", "stale_release": "новая версия Сони", "retention": "долго стоял",
    "spare_retention": "запасной стоял > срока", "host_error": "сбой хоста", "create_timeout": "не запустился",
    "worker_stalled": "воркер завис",
}
BLOCK_WHY = {
    "day_limit": "исчерпан лимит новых серверов на сутки", "budget": "достигнут дневной лимит трат",
    "futile_wakes": "серверы будились, но задание никто не взял",
}


def _why(detail: dict) -> str:
    if detail.get("why"):
        return detail["why"]
    if detail.get("reserve"):
        return "reserve"
    return "prime" if detail.get("prime") else "demand"


def counts(store, lo: float, hi: float) -> dict:
    created: Counter = Counter()
    destroyed: Counter = Counter()
    resumed = 0
    for e in store.events_since(lo, ("created", "destroy_requested", "resume_requested")):
        if e["ts"] > hi:
            continue
        if e["event"] == "created":
            created[_why(e["detail"])] += 1
        elif e["event"] == "destroy_requested":
            destroyed[e["detail"].get("reason") or "other"] += 1
        else:
            resumed += 1
    return {"created": created, "destroyed": destroyed, "resumed": resumed}


def _parts(c: Counter, names: dict) -> str:
    return ", ".join(f"{names.get(k, k)} {n}" for k, n in c.most_common())


def lines(store, now: float, hours: float = 24.0, cfg=None, spend_usd: Optional[Decimal] = None) -> List[str]:
    """HTML lines (tgformat) for the daily report and /vast."""
    c = counts(store, now - hours * 3600, now)
    n_c, n_d = sum(c["created"].values()), sum(c["destroyed"].values())
    out = [tf.esc(f"Серверов создано за сутки: {n_c}" + (f" (из них {_parts(c['created'], CREATE_WHY)})" if n_c else "")
                  + (f"; запусков уже готовых (resume): {c['resumed']}" if c["resumed"] else ""))]
    if n_d:
        out.append(tf.esc(f"Удалено за сутки: {n_d} — {_parts(c['destroyed'], DESTROY_WHY)}"))
    if cfg is not None and (cfg.max_creates_per_day or cfg.day_budget_usd > 0):
        bits = []
        if cfg.max_creates_per_day:
            bits.append(f"новых серверов {store.creates_since(now - 86400)} из {cfg.max_creates_per_day} за 24 ч")
        if cfg.day_budget_usd > 0 and spend_usd is not None:
            bits.append(f"траты сегодня ≈ ${spend_usd:.2f} из ${cfg.day_budget_usd}")
        out.append(tf.esc("Лимиты: " + "; ".join(bits)))
    return out


def check_budget_alerts(ledger, store, now: float, cfg=None) -> List[str]:
    """One alert per MSK day per reason when the fleet refused a create (event create_blocked)."""
    from accounting.balance import ops_alert
    day = (now + 3 * 3600) // 86400
    keys = []
    for e in store.events_since(now - 900, ("create_blocked",)):
        why = e["detail"].get("reason")
        if why not in BLOCK_WHY:
            continue
        d = e["detail"]
        text = [tf.title("⚠️ Vast: новые серверы не создаются"),
                tf.esc(f"Причина: {BLOCK_WHY[why]}."),
                tf.esc(f"Создано за 24 ч: {d.get('creates_24h')}" + (f"; траты сегодня ≈ ${d['spend_today']}" if d.get("spend_today") else "")
                       + "."),
                tf.esc("Задания не теряются: ждут ближайшего свободного или остановленного сервера. "
                       "Если очередь растёт — подними лимит (SONYA_MAX_CREATES_PER_DAY / SONYA_FLEET_DAY_BUDGET_USD) или проверь /vast.")]
        if why == "futile_wakes":
            text[-1] = tf.esc("Сервер будили под задание, но воркер его не взял — пауза 20 мин, чтобы не жечь деньги. "
                              "Проверь задание в очереди.")
        k = ops_alert(ledger, f"fleet-budget:{why}:{int(day)}", text, now)
        if k:
            keys.append(k)
    return keys
