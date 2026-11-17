"""YouTube download health -> owner alerts (via the accounting bot's outbox).

Inputs: the relay heartbeats (scripts/yt_relay.py hb/<relay>.json: ts, priority, canary {ok, ts, bot_check}, cookies).
Alerts (each at most once per period):
  all paths down        no relay alive with a working canary            -> every 2 h, URGENT
  one relay down        its heartbeat older than 15 min                   -> every 12 h
  canary failed         its last hourly test download failed              -> every 6 h
  cookies expired       the separate YouTube account's cookies are signed out -> once a day
The automatic switch needs nothing from here: workers ask every relay, the first working one answers.
"""
from __future__ import annotations

from typing import Dict, List

from accounting import tgformat as tf
from accounting.balance import ops_alert

NAMES = {"bot": "бот-сервер (Москва)", "mac": "Mac mini (дом)"}


def _n(name: str) -> str:
    return NAMES.get(name, name)


def check(ledger, hbs: Dict[str, dict], now: float) -> List[str]:
    out: List[str] = []
    healthy = [n for n, h in hbs.items() if now - float(h.get("ts") or 0) <= 900 and (h.get("canary") or {}).get("ok")]
    if hbs and not healthy:
        k = ops_alert(ledger, f"yt-all-down:{int(now // 7200)}", [
            tf.title("🚨 YouTube: не работает ни один путь скачивания"),
            tf.esc("Задания по ссылкам YouTube сейчас не скачаются (повторятся автоматически, потом — ошибка пользователю). "
                   "Проверь, включён ли Mac mini и жив ли бот-сервер; если cookies протухли — положи новые."),
            tf.esc("Пути: " + "; ".join(f"{_n(n)} — " + _state(h, now) for n, h in sorted(hbs.items())))], now)
        if k:
            out.append(k)
    for n, h in hbs.items():
        age = now - float(h.get("ts") or 0)
        if age > 900:
            k = ops_alert(ledger, f"yt-relay-down:{n}:{int(now // 43200)}", [
                tf.title(f"⚠️ YouTube: запасной путь «{_n(n)}» не отвечает"),
                tf.esc(f"Нет связи {int(age // 60)} мин. Остальные пути работают сами; проверь, включён ли он.")], now)
            if k:
                out.append(k)
            continue
        c = h.get("canary") or {}
        if c and not c.get("ok"):
            k = ops_alert(ledger, f"yt-canary:{n}:{int(now // 21600)}", [
                tf.title(f"⚠️ YouTube: проверочное скачивание через «{_n(n)}» не прошло"),
                tf.esc(("YouTube просит подтвердить «не бот». " if c.get("bot_check") else "") +
                       "Задания идут через другой путь, если он жив.")], now)
            if k:
                out.append(k)
        if h.get("cookies") == "expired":
            k = ops_alert(ledger, f"yt-cookies:{n}:{int(now // 86400)}", [
                tf.title("⚠️ YouTube: cookies отдельного аккаунта протухли"),
                tf.esc("Зайди в отдельный (НЕ личный) аккаунт YouTube в браузере, выгрузи cookies.txt расширением "
                       "«Get cookies.txt LOCALLY» и положи файл в папку _incoming/yt-cookies на Mac mini. "
                       "Без cookies скачивание работает, но запасной уровень защиты выключен.")], now)
            if k:
                out.append(k)
    return out


def _state(h: dict, now: float) -> str:
    if now - float(h.get("ts") or 0) > 900:
        return "не на связи"
    c = h.get("canary") or {}
    if not c:
        return "проверки ещё не было"
    return "ок" if c.get("ok") else ("«не бот»" if c.get("bot_check") else "ошибка")
