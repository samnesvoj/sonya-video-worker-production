"""SONYA self-check collector — VPS 217.199.252.163, systemd sonya-health.timer (every 15 min).

  python -m scripts.ops.health_run run          gather -> checks -> incidents -> alerts into the accounting outbox
  python -m scripts.ops.health_run dry          gather + print /status text and would-be alerts; writes nothing
  python -m scripts.ops.health_run test-alert   one test message through the accounting bot (proves delivery)

Read-only everywhere except its own files (/var/lib/sonya-health/{state,health,fallback}.json) and new rows in the
outbox table of /var/lib/sonya-fleet/ledger.db (the same way the balance alerts go). Never calls the Vast API:
the balance comes from the dispatcher's capacity.json (refreshed every minute). Switch: SONYA_HEALTH=off.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import ssl
import sqlite3
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from scripts.ops import health as H

HOME = Path(os.environ.get("SONYA_HEALTH_DIR", "/var/lib/sonya-health"))
FLEET_DIR = Path(os.environ.get("SONYA_FLEET_STATE_DIR", "/var/lib/sonya-fleet"))
ACTIVE = ("claimed", "worker_started", "preflight_running", "downloading", "model_downloading", "mode_running",
          "analyzing", "yolo", "scripting", "tts", "subtitles", "assembling", "uploading_result")


def _atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, default=str))
    os.chmod(tmp, 0o600)
    tmp.replace(path)


def _read(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return None


def _systemctl_active(unit: str) -> Optional[bool]:
    try:
        r = subprocess.run(["systemctl", "is-active", unit], capture_output=True, text=True, timeout=10)
        return r.stdout.strip() == "active"
    except Exception:  # noqa: BLE001
        return None


def _http_json(url: str, timeout: float = 8.0) -> Tuple[Optional[int], Optional[dict], str]:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "sonya-health"}), timeout=timeout) as r:
            body = r.read(4096)
            try:
                return r.status, json.loads(body), ""
            except ValueError:
                return r.status, None, "не JSON"
    except urllib.error.HTTPError as e:
        return e.code, None, f"HTTP {e.code}"
    except Exception as e:  # noqa: BLE001
        return None, None, type(e).__name__


def gather_db(env) -> Tuple[Optional[float], Optional[dict], str]:
    import psycopg2
    t0 = time.time()
    try:
        conn = psycopg2.connect(env["DATABASE_URL"], connect_timeout=8)
    except Exception as e:  # noqa: BLE001
        return None, None, type(e).__name__
    try:
        conn.set_session(readonly=True, autocommit=True)
        cur = conn.cursor()
        cur.execute("SET statement_timeout = 10000")
        cur.execute("SELECT 1")
        lat = (time.time() - t0) * 1000
        q: Dict[str, Any] = {}
        cur.execute("SELECT COUNT(*), EXTRACT(EPOCH FROM now() - MIN(created_at)) FROM generation_jobs WHERE status='queued'")
        q["queued"], q["oldest_queued_s"] = (lambda r: (int(r[0]), float(r[1] or 0)))(cur.fetchone())
        cur.execute("SELECT COUNT(*) FROM generation_jobs WHERE status = ANY(%s)", (list(ACTIVE),))
        q["active"] = int(cur.fetchone()[0])
        stale_min = int(env.get("SONYA_HEALTH_STALE_MIN", "60"))
        cur.execute("SELECT COUNT(*), EXTRACT(EPOCH FROM now() - MIN(GREATEST(COALESCE(heartbeat_at, claimed_at, updated_at), "
                    "updated_at))) FROM generation_jobs WHERE status = ANY(%s) AND GREATEST(COALESCE(heartbeat_at, claimed_at, "
                    "updated_at), updated_at) < now() - make_interval(mins => %s)", (list(ACTIVE), stale_min))
        q["stale_active"], q["oldest_stale_s"] = (lambda r: (int(r[0]), float(r[1] or 0)))(cur.fetchone())
        cur.execute("SELECT COUNT(*), EXTRACT(EPOCH FROM now() - MIN(updated_at)) FROM generation_jobs WHERE status='source_wait'")
        q["source_wait"], q["oldest_wait_s"] = (lambda r: (int(r[0]), float(r[1] or 0)))(cur.fetchone())
        return lat, q, ""
    except Exception as e:  # noqa: BLE001
        return (time.time() - t0) * 1000, None, type(e).__name__
    finally:
        conn.close()


def gather_create_failures(now: float) -> Tuple[Optional[int], List[str]]:
    db = FLEET_DIR / "fleet.db"
    if not db.exists():
        return None, []
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=10)
    try:
        rows = c.execute("SELECT event, detail FROM fleet_events WHERE ts>=? AND event IN ('create_failed','create_failed_next',"
                         "'resume_failed')", (now - 3600,)).fetchall()
    finally:
        c.close()
    reasons = []
    for _ev, d in rows:
        try:
            dd = json.loads(d or "{}")
            reasons.append(str(dd.get("reason") or dd.get("status") or "")[:40])
        except ValueError:
            pass
    return len(rows), reasons


def _s3_env(env) -> bool:
    if not env.get("S3_ACCESS_KEY_ID"):
        return False
    os.environ.setdefault("S3_BUCKET_NAME", env.get("MODELS_S3_BUCKET", ""))
    return True


def gather_s3(env, now: float) -> Tuple[Optional[dict], Optional[dict], Optional[dict], str]:
    """relay heartbeats, cookies pool, bot-server beacon."""
    if not _s3_env(env):
        return None, None, None, "нет S3 ключей"
    from scripts import yt_relay
    try:
        hbs = yt_relay.heartbeats(now)
        pool = yt_relay.get_json(yt_relay.prefix() + "pool.json")
        bot = yt_relay.get_json("worker-v2/health/beacon/bot.json")
        return hbs, pool, bot, ""
    except Exception as e:  # noqa: BLE001
        return None, None, None, type(e).__name__


def tg_errors(minutes: int = 15) -> Optional[int]:
    try:
        r = subprocess.run(["journalctl", "-u", "sonya-tg-deliver", "--since", f"-{minutes} min", "-o", "cat", "--no-pager"],
                           capture_output=True, text=True, timeout=20)
        return sum(1 for ln in r.stdout.splitlines()
                   if any(w in ln for w in ("send failed", "send error", "presign error", "pass failed", "Traceback")))
    except Exception:  # noqa: BLE001
        return None


def stuck_outbox(now: float) -> Tuple[int, List[dict]]:
    """Alerts the accounting bot has not delivered for 30+ min."""
    db = FLEET_DIR / "ledger.db"
    if not db.exists():
        return 0, []
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=10)
    c.row_factory = sqlite3.Row
    try:
        rows = [dict(r) for r in c.execute("SELECT report_key, text, created_at FROM report_outbox WHERE status='pending' "
                                           "AND kind='alert' AND created_at < ? AND created_at > ?", (now - 1800, now - 2 * 86400))]
    finally:
        c.close()
    return len(rows), rows


def ssl_days(host: str = "sonya.group") -> Tuple[Optional[float], str]:
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((host, 443), timeout=8) as s, ctx.wrap_socket(s, server_hostname=host) as t:
            na = t.getpeercert()["notAfter"]
        exp = datetime.strptime(na, "%b %d %H:%M:%S %Y %Z").replace(tzinfo=timezone.utc).timestamp()
        return (exp - time.time()) / 86400, ""
    except Exception as e:  # noqa: BLE001
        return None, type(e).__name__


def disk(path: str = "/") -> dict:
    u = shutil.disk_usage(path)
    return {"total_gb": u.total / 1e9, "free_gb": u.free / 1e9}


def gather(env=os.environ, now: Optional[float] = None) -> Tuple[List[H.Check], Dict[str, str]]:
    now = now or time.time()
    errs: Dict[str, str] = {}
    out: List[H.Check] = []

    def safe(name, fn):
        try:
            r = fn()
            out.extend(r if isinstance(r, list) else [r])
        except Exception as e:  # noqa: BLE001 - one broken probe must not hide the others
            errs[name] = type(e).__name__
            out.append(H.Check(name, H.UNKNOWN, H.SPEC.get(name, (name,))[0], f"проверка упала: {type(e).__name__}"))

    def api():
        st, data, err = _http_json("http://127.0.0.1:8000/health")
        return H.check_api(data if st == 200 else None, err or (f"HTTP {st}" if st else ""))
    safe("api", api)
    safe("site_public", lambda: H.check_public(*(lambda r: (r[0], r[2]))(_http_json("https://sonya.group/api/health"))))

    db = {}

    def pg():
        lat, q, err = gather_db(env)
        db["q"] = q
        return H.check_postgres(lat if q is not None or err == "" else None, err)
    safe("postgres", pg)

    cap = _read(FLEET_DIR / "capacity.json") or {}
    cap_age = now - float(cap.get("ts") or 0) if cap else None
    safe("fleet", lambda: H.check_fleet(_systemctl_active("sonya-fleet"), cap_age))

    def bal():
        if cap_age is None or cap_age > 900 or cap.get("balance") in (None, "None"):
            return H.check_balance(None, None)
        return H.check_balance(float(cap["balance"]), None, int(cap.get("pending") or 0),
                               float(env.get("SONYA_VAST_BALANCE_LOW_USD", "3")), float(env.get("SONYA_VAST_BALANCE_CRITICAL_USD", "1")))
    safe("vast_balance", bal)
    safe("vast_create", lambda: H.check_create(*gather_create_failures(now)))
    safe("queue", lambda: H.check_queue(db.get("q"), int(env.get("SONYA_HEALTH_QUEUED_MIN", "45")),
                                        int(env.get("SONYA_HEALTH_STALE_MIN", "60"))))

    hbs, pool, bot, s3err = gather_s3(env, now)
    if s3err:
        errs["s3"] = s3err
    mac = _read(HOME / "beacon" / "mac.json")
    safe("bot_host", lambda: H.check_host("bot_host", bot, now, 1200, ("sonya-accounting-bot", "sonya-group-bot", "sonya-ytrelay")))
    safe("mac_host", lambda: H.check_host("mac_host", mac, now, 2700))
    safe("yt_all", lambda: H.check_relays(hbs, now))
    safe("yt_cookies", lambda: H.check_cookies(pool, now) if not s3err else H.Check("yt_cookies", H.UNKNOWN, "cookies", "нет данных"))
    safe("tg_deliver", lambda: H.check_tg(_systemctl_active("sonya-tg-deliver"), tg_errors()))

    def acct():
        poll = _read(FLEET_DIR / "acct_poll.json")
        age = now - float(poll["ts"]) if poll and poll.get("ts") else None
        svc = ((bot or {}).get("services") or {}).get("sonya-accounting-bot") if bot and now - float(bot.get("ts") or 0) < 1200 else None
        return H.check_accountant(age, stuck_outbox(now)[0], None if svc is None else svc == "active")
    safe("accountant", acct)
    safe("disk_vps", lambda: H.check_disk("disk_vps", disk("/")))
    safe("disk_bot", lambda: H.check_disk("disk_bot", (bot or {}).get("disk")))
    safe("disk_mac", lambda: H.check_disk("disk_mac", (mac or {}).get("disk"), min_free_gb=8.0))
    safe("ssl", lambda: H.check_ssl(*ssl_days()))
    return out, errs


def enqueue(alerts: List[H.Alert], now: float) -> List[str]:
    from accounting.balance import ops_alert
    from accounting.ledger import Ledger
    led = Ledger(str(FLEET_DIR / "ledger.db"))
    led.c.execute("PRAGMA busy_timeout = 30000")
    out = []
    for a in alerts:
        k = ops_alert(led, a.key, a.lines, now)
        if k:
            out.append(k)
    led.c.close()
    return out


def write_fallback(snap_incidents: Dict[str, dict], stuck: List[dict], now: float) -> None:
    """For the Mac mini beacon: if the accounting bot itself is down, the Mac sends these via its own channel once."""
    import re
    items = [{"id": r["report_key"], "text": re.sub(r"<[^>]+>", "", r["text"])[:900]} for r in stuck]
    a = snap_incidents.get("accountant")
    if a and a.get("announced"):
        items.insert(0, {"id": f"health:accountant:{int(a['since'])}", "text": "SONYA самопроверка: бот-бухгалтер не забирает "
                         "сообщения — алерты не доходят. " + str(a.get("detail") or "")})
    _atomic(HOME / "fallback.json", {"ts": now, "items": items})


def run(mode: str = "run", env=os.environ) -> int:
    now = time.time()
    if env.get("SONYA_HEALTH", "on") == "off":
        print("SONYA_HEALTH=off")
        return 0
    if mode == "test-alert":
        k = enqueue([H.Alert(f"health-test:{int(now)}", [
            "<b>🧪 Тест самопроверки SONYA</b>",
            H.esc("Это проверочное сообщение: алерты самопроверки доходят. Отвечать не нужно. /status — короткая сводка.")])], now)
        print("enqueued", k)
        return 0
    checks, errs = gather(env, now)
    state = _read(HOME / "state.json") or {}
    new_state, alerts = H.update(state, checks, now)
    snap = H.snapshot(checks, new_state, now, errs)
    if mode == "dry":
        print(H.status_text(snap, now))
        print("\n--- would alert:")
        for a in alerts:
            print(a.key, "|", " / ".join(a.lines)[:300])
        print("collector errors:", errs)
        return 0
    keys = enqueue(alerts, now) if alerts else []
    _atomic(HOME / "state.json", new_state)
    _atomic(HOME / "health.json", snap)
    os.chmod(HOME / "health.json", 0o644)
    try:
        write_fallback(new_state["incidents"], stuck_outbox(now)[1], now)
    except Exception as e:  # noqa: BLE001
        errs["fallback"] = type(e).__name__
    bad = [f"{c.name}={c.level}" for c in checks if c.level not in (H.OK,)]
    print(f"health run: {len(checks)} checks, not ok: {', '.join(bad) or '-'}; alerts: {', '.join(keys) or '-'}; errors: {errs or '-'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(run(sys.argv[1] if len(sys.argv) > 1 else "run"))
