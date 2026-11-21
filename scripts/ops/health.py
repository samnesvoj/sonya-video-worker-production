"""SONYA self-check (night task 2026-11-21): one place that says «всё ок / что сломано» and alerts the owner once per
problem (+ «починилось»), through the accounting bot's outbox like the balance alerts.

Pure part (unit-tested, no network): the checks' decision rules, the incident state machine and the texts.
The collector that feeds it runs on the VPS every 15 min: scripts/ops/health_run.py (systemd sonya-health.timer).

Incident rules (anti-spam):
  - a failing check opens an incident; it is announced after `confirm` failing runs in a row (1 for money/SSL/disk,
    2 = 15+ min for flappy network things);
  - while it stays open nothing more is sent - except ONE escalation if its level gets worse (warn -> crit);
  - it closes after `clear` OK runs in a row; «✅ Починилось» is sent only if the opening was announced;
  - a check whose cause is already open is not announced separately (queue stuck <- no money on Vast,
    relay down <- its host down, ...); it is announced later if it outlives the cause;
  - a check with no data (collector could not ask) keeps its previous state and only shows «нет данных» in /status.
"""
from __future__ import annotations

import html
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

MSK = timezone(timedelta(hours=3))
OK, WARN, CRIT, UNKNOWN = "ok", "warn", "crit", "unknown"
CAUSE_GRACE_S = 5400                 # e.g. Vast create errors stay in the 1-h window after the top-up
_RANK = {OK: 0, UNKNOWN: 0, WARN: 1, CRIT: 2}


@dataclass
class Check:
    name: str
    level: str                       # ok / warn / crit / unknown
    title: str                       # short, for /status and the alert head
    detail: str = ""                 # one or two sentences: what happens + what to do
    value: str = ""                  # short state for the «всё ок» line


# name -> (label for /status, confirm, clear, causes)
SPEC: Dict[str, Tuple[str, int, int, Tuple[str, ...]]] = {      # ORDER matters: causes before their consequences
    "api":          ("сайт/API", 2, 1, ()),
    "site_public":  ("сайт снаружи", 2, 2, ("api",)),
    "postgres":     ("база", 2, 1, ()),
    "fleet":        ("флот Vast", 2, 1, ()),
    "vast_balance": ("баланс Vast", 1, 1, ("fleet",)),
    "vast_create":  ("создание серверов", 1, 2, ("vast_balance", "fleet")),
    "queue":        ("очередь", 1, 1, ("vast_balance", "fleet", "postgres", "api")),
    "bot_host":     ("бот-сервер", 2, 1, ()),
    "mac_host":     ("Mac mini", 3, 1, ()),
    "yt_all":       ("YouTube", 2, 2, ()),
    "yt_bot":       ("YouTube через бот-сервер", 2, 2, ("bot_host", "yt_all")),
    "yt_mac":       ("YouTube через Mac", 2, 2, ("mac_host", "yt_all")),
    "yt_cookies":   ("cookies YouTube", 1, 1, ()),
    "tg_deliver":   ("доставка в Telegram", 2, 2, ()),
    "accountant":   ("бот-бухгалтер", 2, 1, ("bot_host",)),
    "disk_vps":     ("диск сервера сайта", 1, 1, ()),
    "disk_bot":     ("диск бот-сервера", 1, 1, ("bot_host",)),
    "disk_mac":     ("диск Mac mini", 1, 1, ("mac_host",)),
    "ssl":          ("SSL sonya.group", 1, 1, ()),
}
ORDER = list(SPEC)


def esc(s: Any) -> str:
    return html.escape(str(s), quote=False)


def ago(sec: float) -> str:
    sec = max(0, int(sec))
    if sec < 90:
        return "только что"
    if sec < 5400:
        return f"{sec // 60} мин"
    if sec < 172800:
        return f"{sec // 3600} ч {sec % 3600 // 60} мин" if sec < 36000 else f"{sec // 3600} ч"
    return f"{sec // 86400} дн"


def msk(ts: float) -> str:
    return datetime.fromtimestamp(ts, MSK).strftime("%d.%m %H:%M")


# ─────────────────────────────────────────── decision rules (inputs are plain data gathered by the collector)
def check_api(local: Optional[dict], err: str = "") -> Check:
    if local is None:
        return Check("api", CRIT, "Сайт/API не отвечает", f"Локальный /health не ответил ({err or 'нет ответа'}). Сайт и "
                     "нарезка не работают. sonya-api перезапускается сам; если не поднялся — смотреть journalctl -u sonya-api.")
    if local.get("status") != "ok":
        return Check("api", CRIT, "Сайт/API отвечает с ошибкой", f"/health: {str(local)[:120]}")
    return Check("api", OK, "API", value="ok")


def check_public(code: Optional[int], err: str = "") -> Check:
    if code == 200:
        return Check("site_public", OK, "сайт снаружи", value="200")
    return Check("site_public", CRIT, "sonya.group не открывается снаружи",
                 f"https://sonya.group/api/health → {code or err or 'нет ответа'}. API внутри живо — проблема в nginx/сертификате/сети.")


def check_postgres(latency_ms: Optional[float], err: str = "") -> Check:
    if latency_ms is None:
        return Check("postgres", CRIT, "База (Postgres) недоступна", f"Подключение не удалось ({err or 'ошибка'}). "
                     "Сайт не может создавать и показывать задания.")
    if latency_ms > 3000:
        return Check("postgres", WARN, "База отвечает медленно", f"Простой запрос {latency_ms:.0f} мс.", value=f"{latency_ms:.0f} мс")
    return Check("postgres", OK, "база", value=f"{latency_ms:.0f} мс")


def check_queue(q: Optional[dict], queued_min: int = 45, stale_min: int = 40, wait_min: int = 150) -> Check:
    """q = {queued, oldest_queued_s, active, stale_active, oldest_stale_s, source_wait, oldest_wait_s}"""
    if q is None:
        return Check("queue", UNKNOWN, "очередь", "нет данных (база недоступна)")
    bad = []
    if q.get("stale_active"):
        bad.append(f"{q['stale_active']} в работе без признаков жизни {ago(q.get('oldest_stale_s') or 0)}")
    if q.get("queued") and (q.get("oldest_queued_s") or 0) > queued_min * 60:
        bad.append(f"{q['queued']} ждут в очереди, самое старое {ago(q['oldest_queued_s'])}")
    if q.get("source_wait") and (q.get("oldest_wait_s") or 0) > wait_min * 60:
        bad.append(f"{q['source_wait']} ждут YouTube дольше {wait_min // 60} ч")
    if bad:
        return Check("queue", CRIT if q.get("stale_active") or (q.get("oldest_queued_s") or 0) > 3 * 3600 else WARN,
                     "Очередь стоит", "; ".join(bad) + ". Пользователи ждут клипы.")
    return Check("queue", OK, "очередь", value=f"ждут {q.get('queued', 0)}, в работе {q.get('active', 0)}")


def check_fleet(active: Optional[bool], tick_age_s: Optional[float]) -> Check:
    if active is False:
        return Check("fleet", CRIT, "Диспетчер Vast (sonya-fleet) не запущен",
                     "Без него серверы не создаются и очередь не разбирается. systemctl restart sonya-fleet.")
    if tick_age_s is None or tick_age_s > 600:
        return Check("fleet", CRIT, "Диспетчер Vast завис",
                     f"Служба запущена, но не отчитывалась {ago(tick_age_s or 0) if tick_age_s else 'давно'}. systemctl restart sonya-fleet.")
    return Check("fleet", OK, "флот", value="жив")


def check_balance(balance: Optional[float], burn_h: Optional[float], pending: int = 0, low: float = 3.0, crit: float = 1.0) -> Check:
    if balance is None:
        return Check("vast_balance", UNKNOWN, "баланс Vast", "нет данных")
    v = f"${balance:.2f}"
    if balance < crit:
        what = ("Новые серверы не создаются; задания НЕ падают — ждут, пользователи видят «Высокая нагрузка, клипы будут чуть позже»"
                + (f" (сейчас ждут {pending})" if pending else "") + ".")
        return Check("vast_balance", CRIT, "Vast: деньги кончились" if balance <= 0.3 else "Vast: денег почти нет",
                     f"На балансе {v}. {what} Пополни vast.ai → Billing (или смени карту) — серверы поднимутся сами, "
                     "напоминать больше не буду, напишу, когда деньги появятся.", value=v)
    if balance < low:
        runway = f" — хватит примерно на {int(balance / burn_h)} ч" if burn_h and burn_h > 0 else ""
        return Check("vast_balance", WARN, f"Vast: баланс ниже ${low:.0f}", f"На балансе {v}{runway}. Пополни заранее.", value=v)
    return Check("vast_balance", OK, "баланс", value=v)


def check_create(failures: Optional[int], reasons: Iterable[str] = (), threshold: int = 3) -> Check:
    if failures is None:
        return Check("vast_create", UNKNOWN, "создание серверов", "нет данных")
    if failures >= threshold:
        rs = ", ".join(sorted({r for r in reasons if r})[:3])
        return Check("vast_create", WARN, "Vast: ошибки создания серверов",
                     f"{failures} неудачных попыток за час" + (f" ({rs})" if rs else "") + ". Диспетчер пробует другие предложения сам.")
    return Check("vast_create", OK, "создание", value=f"ошибок за час {failures}")


RELAY_NAMES = {"bot": "бот-сервер", "mac": "Mac mini"}


def check_relays(hbs: Optional[Dict[str, dict]], now: float, fresh_s: int = 900) -> List[Check]:
    """hbs = S3 yt-relay/hb/<name>.json (ts, canary {ok, bot_check})."""
    if hbs is None:
        return [Check(n, UNKNOWN, SPEC[n][0], "нет данных (S3)") for n in ("yt_all", "yt_bot", "yt_mac")]
    out, healthy = [], []
    for name in ("bot", "mac"):
        h, cn = hbs.get(name), f"yt_{name}"
        label = RELAY_NAMES[name]
        if not h:
            out.append(Check(cn, WARN, f"YouTube: путь через {label} не найден", "Нет heartbeat в S3."))
            continue
        age = now - float(h.get("ts") or 0)
        c = h.get("canary") or {}
        if age > fresh_s:
            out.append(Check(cn, WARN, f"YouTube: путь через {label} не отвечает", f"Нет связи {ago(age)}. Остальные пути работают сами."))
        elif c and not c.get("ok"):
            out.append(Check(cn, WARN, f"YouTube: канарейка через {label} не прошла",
                             ("YouTube просит «не бот». " if c.get("bot_check") else "") + "Задания идут другим путём, если он жив."))
        else:
            healthy.append(name)
            out.append(Check(cn, OK, label, value="ok" if c else "канарейки ещё не было"))
    if healthy:
        out.insert(0, Check("yt_all", OK, "YouTube", value="пути: " + ", ".join(RELAY_NAMES[n] for n in healthy)))
    else:
        out.insert(0, Check("yt_all", CRIT, "YouTube: не работает ни один путь скачивания",
                            "Задания по ссылкам YouTube ждут до 2 ч, потом просьба загрузить файл и возврат попытки. "
                            "Проверь Mac mini и бот-сервер; если cookies протухли — положи новые."))
    return out


COOKIES_HOWTO = ("Зайди в отдельный (НЕ личный) аккаунт YouTube, выгрузи cookies.txt расширением «Get cookies.txt LOCALLY» "
                 "и положи файл в _incoming/yt-cookies на Mac mini.")


def check_cookies(pool: Optional[dict], now: float) -> Check:
    if not pool or not pool.get("total"):
        return Check("yt_cookies", UNKNOWN, "cookies YouTube", "нет данных о пуле")
    live, total = int(pool.get("live") or 0), int(pool.get("total") or 0)
    stale = now - float(pool.get("ts") or now) > 6 * 3600
    if live == 0:
        return Check("yt_cookies", WARN, "YouTube: живых cookies нет",
                     f"Файлов {total}, рабочих 0. Пока YouTube отдаёт видео и так — всё работает; запасной уровень защиты выключен. "
                     + COOKIES_HOWTO, value="0 живых")
    return Check("yt_cookies", OK, "cookies", value=f"{live} из {total}" + (" (данные старые)" if stale else ""))


def check_tg(active: Optional[bool], errors: Optional[int], threshold: int = 5) -> Check:
    if active is False:
        return Check("tg_deliver", CRIT, "Доставка клипов в Telegram остановлена",
                     "sonya-tg-deliver не запущен: готовые клипы не приходят в Telegram (на сайте они есть). systemctl restart sonya-tg-deliver.")
    if errors is not None and errors >= threshold:
        return Check("tg_deliver", WARN, "Доставка в Telegram: ошибки", f"{errors} ошибок за 15 мин в журнале sonya-tg-deliver.")
    if active is None:
        return Check("tg_deliver", UNKNOWN, "доставка в Telegram", "нет данных")
    return Check("tg_deliver", OK, "доставка", value="ok" + (f", ошибок {errors}" if errors else ""))


def check_accountant(last_claim_age_s: Optional[float], stuck_alerts: int, beacon_active: Optional[bool]) -> Check:
    if beacon_active is False:
        return Check("accountant", CRIT, "Бот-бухгалтер не запущен",
                     "sonya-accounting-bot на бот-сервере остановлен: чеки и алерты не приходят.")
    if last_claim_age_s is None:
        return Check("accountant", UNKNOWN, "бот-бухгалтер", "ещё нет данных о его опросах")
    if last_claim_age_s > 900:
        return Check("accountant", CRIT, "Бот-бухгалтер не забирает сообщения",
                     f"Последний опрос {ago(last_claim_age_s)} назад. Алерты копятся и придут, когда он оживёт.")
    if stuck_alerts:
        return Check("accountant", WARN, "Бот-бухгалтер: сообщения не уходят",
                     f"{stuck_alerts} сообщений не доставлены за 30 мин (Telegram отказывает?).")
    return Check("accountant", OK, "бухгалтер", value="ok")


def check_host(name: str, beacon: Optional[dict], now: float, max_age_s: int, services: Iterable[str] = ()) -> Check:
    label = SPEC[name][0]
    if not beacon:
        return Check(name, UNKNOWN, label, "маячок ещё не присылал данных")
    age = now - float(beacon.get("ts") or 0)
    if age > max_age_s:
        return Check(name, CRIT if name == "bot_host" else WARN, f"{label[:1].upper() + label[1:]} не на связи",
                     f"Маячок молчит {ago(age)}." + (" Mac мог уснуть или пропал интернет дома — YouTube идёт через бот-сервер."
                                                    if name == "mac_host" else " Бухгалтер, бот @sonya_group_ai и YouTube-путь там."))
    down = [s for s in services if (beacon.get("services") or {}).get(s) not in (None, "active")]
    if down:
        return Check(name, CRIT, f"{label}: остановлены службы", ", ".join(down) + ". systemctl restart …")
    return Check(name, OK, label, value="на связи")


def check_disk(name: str, d: Optional[dict], min_free_gb: float = 5.0, max_used_pct: float = 95.0) -> Check:
    label = SPEC[name][0]
    if not d or not d.get("total_gb"):
        return Check(name, UNKNOWN, label, "нет данных")
    free, total = float(d["free_gb"]), float(d["total_gb"])
    used = 100.0 * (1 - free / total)
    v = f"свободно {free:.0f} ГБ ({100 - used:.0f}%)"
    if free < min_free_gb or used >= max_used_pct:
        return Check(name, CRIT if free < 2 else WARN, f"Мало места: {label}", v + ". Когда кончится — служба встанет.", value=v)
    return Check(name, OK, label, value=v)


def check_ssl(days_left: Optional[float], err: str = "") -> Check:
    if days_left is None:
        return Check("ssl", UNKNOWN, "SSL", f"не смог прочитать сертификат ({err})")
    v = f"ещё {int(days_left)} дн"
    if days_left < 3:
        return Check("ssl", CRIT, "SSL sonya.group истекает", v + ". certbot не продлил — сайт скоро будет «небезопасным». certbot renew.", value=v)
    if days_left < 14:
        return Check("ssl", WARN, "SSL sonya.group скоро истечёт", v + ". certbot должен продлить сам (обычно за 30 дн) — он не справился.", value=v)
    return Check("ssl", OK, "SSL", value=v)


# ─────────────────────────────────────────── incidents
@dataclass
class Alert:
    key: str
    lines: List[str]


def _new_alert_lines(c: Check, label: str) -> List[str]:
    icon = "🚨" if c.level == CRIT else "⚠️"
    return [f"<b>{icon} {esc(c.title)}</b>", esc(c.detail), f"<i>Самопроверка SONYA · {esc(label)} · /status — сводка</i>"]


def update(state: Dict[str, Any], checks: List[Check], now: float) -> Tuple[Dict[str, Any], List[Alert]]:
    """Advance the incident state with this run's checks; returns (new state, alerts to enqueue)."""
    inc: Dict[str, dict] = dict(state.get("incidents") or {})
    closed: Dict[str, float] = {k: v for k, v in (state.get("closed") or {}).items() if now - v < CAUSE_GRACE_S}
    by = {c.name: c for c in checks}
    alerts: List[Alert] = []

    def open_and_announced(n: str) -> bool:
        i = inc.get(n)
        return bool(i and i.get("announced"))

    def cause_open(n: str) -> Optional[str]:
        for p in SPEC.get(n, ("", 1, 1, ()))[3]:
            i = inc.get(p)
            if (i and i.get("fails", 0) >= 1) or p in closed:     # a cause that just healed still explains the tail
                return p
        return None

    for n in sorted(by, key=lambda x: ORDER.index(x) if x in ORDER else 99):     # causes come first in ORDER
        c = by[n]
        label, confirm, clear, _ = SPEC.get(n, (n, 2, 1, ()))
        i = inc.get(n)
        if c.level == UNKNOWN:
            continue                                     # no data: keep whatever we knew
        if c.level == OK:
            if not i:
                continue
            i["oks"] = i.get("oks", 0) + 1
            if i["oks"] >= clear:
                if i.get("announced"):
                    alerts.append(Alert(f"health-ok:{n}:{int(i['since'])}", [
                        f"<b>✅ Починилось: {esc(i.get('title') or label)}</b>",
                        esc(f"Проблема длилась {ago(now - i['since'])} (с {msk(i['since'])} МСК)."
                            + (f" Сейчас: {c.value}." if c.value else ""))]))
                inc.pop(n)
                closed[n] = now
            continue
        if not i:
            i = inc[n] = {"since": now, "fails": 0, "announced": False, "level": c.level, "title": c.title}
        i.update(fails=i.get("fails", 0) + 1, oks=0, last=now, detail=c.detail, cur_level=c.level)
        if i.get("announced"):
            if _RANK[c.level] > _RANK.get(i.get("level", WARN), 1):       # one escalation, never a repeat
                i.update(level=c.level, title=c.title)
                alerts.append(Alert(f"health:{n}:{int(i['since'])}:{c.level}", _new_alert_lines(c, label)))
            continue
        if i["fails"] < confirm:
            continue
        cause = cause_open(n)
        if cause:
            i["suppressed_by"] = cause
            continue
        i.update(announced=True, level=c.level, title=c.title, announced_at=now)
        i.pop("suppressed_by", None)
        alerts.append(Alert(f"health:{n}:{int(i['since'])}", _new_alert_lines(c, label)))
    return {"incidents": inc, "closed": closed, "ts": now}, alerts


# ─────────────────────────────────────────── /status
def status_text(snap: Optional[dict], now: Optional[float] = None) -> str:
    """snap = health.json written by the collector: {ts, checks: [{name, level, title, detail, value}], incidents}."""
    now = now or time.time()
    if not snap or not snap.get("checks"):
        return "<b>Самопроверка ещё не запускалась.</b>"
    age = now - float(snap.get("ts") or 0)
    checks = [Check(**{k: c.get(k, "") for k in ("name", "level", "title", "detail", "value")}) for c in snap["checks"]]
    inc = snap.get("incidents") or {}
    bad = [c for c in checks if c.level in (WARN, CRIT)]
    unk = [c for c in checks if c.level == UNKNOWN]
    lines = []
    if not bad:
        lines.append("<b>✅ Всё работает</b>")
    else:
        lines.append(f"<b>{'🚨' if any(c.level == CRIT for c in bad) else '⚠️'} Сломано: {len(bad)}</b>")
        for c in sorted(bad, key=lambda c: -_RANK[c.level]):
            i = inc.get(c.name) or {}
            since = f" — с {msk(i['since'])}" if i.get("since") else ""
            lines.append(f"{'🚨' if c.level == CRIT else '⚠️'} <b>{esc(c.title)}</b>{esc(since)}")
            if c.detail:
                lines.append("   " + esc(_short(c.detail)))
    oks = {c.name: c for c in checks if c.level == OK}
    ok = []
    for c in oks.values():
        if c.name in ("site_public", "yt_bot", "yt_mac", "vast_create", "disk_vps", "disk_bot", "disk_mac"):
            continue                                   # covered by their parent / grouped below
        lab = SPEC.get(c.name, (c.name,))[0]
        ok.append(lab + (f" {c.value}" if c.value and c.name in ("vast_balance", "queue", "yt_cookies", "ssl") else "")
                  + (f" ({c.value.replace('пути: ', '')})" if c.name == "yt_all" and c.value else ""))
    disks = [n for n in ("disk_vps", "disk_bot", "disk_mac") if n in oks]
    if disks:
        ok.append("диски" + ("" if len(disks) == 3 else " (" + ", ".join(SPEC[n][0].replace("диск ", "") for n in disks) + ")"))
    if ok:
        lines.append("")
        lines.append("<b>Работает:</b> " + esc("; ".join(ok)))
    if unk:
        lines.append("<i>" + esc("Нет данных: " + ", ".join(SPEC.get(c.name, (c.name,))[0] for c in unk)) + "</i>")
    stale = f" ⚠️ самопроверка давно не запускалась" if age > 2700 else ""
    lines.append(f"<i>Проверено {esc(ago(age))}{' назад' if age >= 90 else ''} ({esc(msk(float(snap['ts'])))} МСК){stale}</i>")
    return "\n".join(lines)


def _short(text: str, limit: int = 170) -> str:
    """Whole sentences up to `limit` characters (the alert has the full text)."""
    out = ""
    for part in text.replace("! ", "!|").replace(". ", ".|").split("|"):
        if out and len(out) + len(part) + 1 > limit:
            break
        out = (out + " " + part).strip()
    return out if len(out) <= limit + 60 else out[:limit].rsplit(" ", 1)[0] + "…"


def snapshot(checks: List[Check], state: Dict[str, Any], now: float, errors: Optional[Dict[str, str]] = None) -> dict:
    return {"ts": now, "checks": [c.__dict__ for c in checks], "incidents": state.get("incidents") or {},
            "collector_errors": errors or {}}
