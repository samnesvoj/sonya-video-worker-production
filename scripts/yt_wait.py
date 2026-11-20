"""YouTube patience: a link YouTube would not give us does NOT fail the job (owner 2026-11-19: «чтоб мы не обосрались
перед клиентами из-за YouTube»).

Flow
  1. The GPU worker tried every path (Vast direct -> relay `bot` -> relay `mac` with the cookies pool) and all refused:
     it reports error_code SOURCE_YT_WAIT; the API parks the job in status 'source_wait' (no GPU, no server is held,
     the fleet sees no demand). The user sees «Скачиваем видео… это может занять дольше обычного».
  2. This waiter (runs in the dispatcher loop on the VPS, `tick()` every fleet tick) asks the relays again with growing
     pauses (2, 4, 8, 15, 20, 30, 30 … min) for up to SONYA_YT_WAIT_MAX_SEC (2 h). Same S3 protocol as the worker
     (scripts/yt_relay.py: req/<job>.json -> st/<job>/<relay>.json); the relay uploads the file to the job's
     s3_input_key.
  3. Got it -> the job goes back to 'queued' with params.source_fetch.prefetched=true: the worker reads s3_input_key like
     an uploaded file. A real answer about the video (private, deleted, too long) -> the job fails with that text.
     2 h without success -> the job fails with a polite text and the attempt is given back (the API refunds SOURCE_*
     failures: scripts/prod_job_store.fail_job on the site server).
  4. Admins (accounting bot outbox): right away «YouTube перестал отдавать видео, нужен новый файл cookies» + how many
     jobs wait, a reminder every hour, «снова работает» when it recovers, one line per given-up job.

State per job lives in params.yt_wait (JSONB): {since, attempt, next_at, req_at, last, done_at}. Nothing in memory, so a
dispatcher restart continues where it stopped. SONYA_YT_WAIT=off: the waiter does nothing (jobs in 'source_wait' stay).
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger("sonya.yt_wait")

WAIT_STATUS = "source_wait"
WAIT_CODE = "SOURCE_YT_WAIT"                  # worker -> API: park the job
GIVEUP_CODE = "SOURCE_YT_UNAVAILABLE"         # waiter -> API: final, attempt refunded
WAIT_TEXT = "Скачиваем видео… это может занять дольше обычного"
GIVEUP_TEXT = ("Не получилось получить видео с YouTube. Загрузите файл или ссылку VK/Rutube — эта попытка не списана.")
BACKOFF_MIN = (2, 4, 8, 15, 20, 30)           # pause before attempt n (n = 1, 2, ...); the last value repeats


def max_wait(env=os.environ) -> float:
    return float(env.get("SONYA_YT_WAIT_MAX_SEC", "7200"))


def pause_before(attempt: int) -> float:
    """Seconds of pause before attempt `attempt` (1 = the first try of the waiter, right after the worker gave up)."""
    return 60.0 * BACKOFF_MIN[min(max(attempt, 1), len(BACKOFF_MIN)) - 1]


def decide(w: Dict[str, Any], rnd: Optional[Dict[str, Any]], now: float, limit: float) -> Dict[str, Any]:
    """Pure: what to do with one waiting job.
    w   = params.yt_wait ({} on first sight); rnd = yt_relay.outcome() of the round in flight (None = no round in flight).
    -> {"do": "init"|"submit"|"requeue"|"fail_user"|"give_up"|"schedule"|"none", "w": new state, ...}"""
    w = dict(w or {})
    if not w.get("since"):
        w.update(since=now, attempt=0, next_at=now + pause_before(1), req_at=None)
        return {"do": "init", "w": w}
    age = now - float(w["since"])
    if w.get("req_at"):
        kind = (rnd or {}).get("kind", "wait")
        if kind == "done":
            return {"do": "requeue", "w": dict(w, req_at=None, done_at=now), "relay": rnd.get("relay"), "duration": rnd.get("duration")}
        if kind == "user":
            return {"do": "fail_user", "w": dict(w, req_at=None), "code": rnd["code"], "message": rnd["message"]}
        if kind == "wait":
            # a relay is downloading: let it finish, but never past limit + 30 min
            if age > limit + 1800:
                return {"do": "give_up", "w": dict(w, req_at=None, last="hard timeout")}
            return {"do": "none", "w": w}
        deadline = float(w["since"]) + limit
        nxt = min(now + pause_before(int(w.get("attempt") or 0) + 1), deadline - 60)   # one last try before the deadline
        w.update(req_at=None, last=str(rnd.get("reason") or "refused")[:200], next_at=nxt)
        if age >= limit or nxt <= now + 30:
            return {"do": "give_up", "w": w}
        return {"do": "schedule", "w": w}
    if age >= limit:
        return {"do": "give_up", "w": w}
    if now >= float(w.get("next_at") or 0):
        return {"do": "submit", "w": dict(w, attempt=int(w.get("attempt") or 0) + 1, req_at=now)}
    return {"do": "none", "w": w}


# ── IO ──────────────────────────────────────────────────────────────────────────
class Db:  # pragma: no cover - production Postgres
    def __init__(self, url: str):
        self.url = url

    def _conn(self):
        import psycopg2
        import psycopg2.extras
        return psycopg2.connect(self.url, cursor_factory=psycopg2.extras.RealDictCursor)

    def waiting(self) -> List[Dict[str, Any]]:
        c = self._conn()
        try:
            with c, c.cursor() as cur:
                cur.execute("SELECT id::text, user_id::text, mode, params, s3_input_key, updated_at FROM generation_jobs "
                            "WHERE status=%s ORDER BY created_at", (WAIT_STATUS,))
                return [dict(r) for r in cur.fetchall()]
        finally:
            c.close()

    def save_wait(self, job_id: str, w: Dict[str, Any]) -> bool:
        c = self._conn()
        try:
            with c, c.cursor() as cur:
                cur.execute("UPDATE generation_jobs SET params = jsonb_set(COALESCE(params,'{}'::jsonb), '{yt_wait}', %s::jsonb) "
                            "WHERE id=%s AND status=%s", (json.dumps(w), job_id, WAIT_STATUS))
                return cur.rowcount == 1
        finally:
            c.close()

    def requeue(self, job_id: str, sf: Dict[str, Any], w: Dict[str, Any]) -> bool:
        c = self._conn()
        try:
            with c, c.cursor() as cur:
                cur.execute("UPDATE generation_jobs SET status='queued', worker_id=NULL, claimed_at=NULL, heartbeat_at=NULL, "
                            "last_error=NULL, error=NULL, updated_at=NOW(), "
                            "params = COALESCE(params,'{}'::jsonb) || jsonb_build_object('source_fetch', %s::jsonb, 'yt_wait', %s::jsonb) "
                            "WHERE id=%s AND status=%s", (json.dumps(sf), json.dumps(w), job_id, WAIT_STATUS))
                return cur.rowcount == 1
        finally:
            c.close()


def api_fail(job_id: str, code: str, message: str, env=os.environ) -> bool:  # pragma: no cover - network
    """The worker API's own fail call (retry=False): the API refunds a SOURCE_* failure, reconciles streamer batches."""
    base = (env.get("BACKEND_API_URL") or "http://127.0.0.1:8000").rstrip("/")
    req = urllib.request.Request(f"{base}/api/worker/jobs/{job_id}/fail", method="POST",
                                 data=json.dumps({"error_code": code, "error_message": message, "retry": False}).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + env.get("WORKER_SECRET", "")})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return 200 <= r.status < 300
    except Exception as exc:  # noqa: BLE001 - next tick tries again
        log.warning("yt_wait_api_fail_error job=%s %s", job_id, type(exc).__name__)
        return False


def _params(job: Dict[str, Any]) -> Dict[str, Any]:
    p = job.get("params") or {}
    if isinstance(p, str):
        try:
            p = json.loads(p)
        except ValueError:
            p = {}
    return p if isinstance(p, dict) else {}


def tick(db, relay, now: Optional[float] = None, ledger=None, env=os.environ,
         fail: Callable[[str, str, str], bool] = None, pool: Optional[Dict[str, Any]] = None) -> List[str]:
    """One pass over the waiting jobs. relay = scripts.yt_relay (or a test double with submit/withdraw/statuses/
    heartbeats/outcome). -> actions (for the log)."""
    if env.get("SONYA_YT_WAIT", "on") == "off":
        return []
    now = now or time.time()
    fail = fail or (lambda j, c, m: api_fail(j, c, m, env))
    limit = max_wait(env)
    jobs = db.waiting()
    acts: List[str] = []
    hbs = None
    events: Dict[str, List[Dict[str, Any]]] = {"requeued": [], "gave_up": []}
    for job in jobs:
        jid = job["id"]
        p = _params(job)
        sf = p.get("source_fetch") if isinstance(p.get("source_fetch"), dict) else None
        w = p.get("yt_wait") if isinstance(p.get("yt_wait"), dict) else {}
        if not sf or not sf.get("url"):
            if fail(jid, GIVEUP_CODE, GIVEUP_TEXT):          # nothing to fetch (should not happen): do not hang forever
                acts.append(f"yt_wait_no_source {jid[:8]}")
            continue
        rnd = None
        if w.get("req_at"):
            try:
                if hbs is None:
                    hbs = relay.heartbeats(now)
                rnd = relay.outcome(jid, relay.statuses(jid), hbs, float(w["req_at"]), now)
            except Exception as exc:  # noqa: BLE001 - S3 blip: look again next tick
                log.warning("yt_wait_poll_error job=%s %s", jid, type(exc).__name__)
                continue
        d = decide(w, rnd, now, limit)
        do, nw = d["do"], d["w"]
        try:
            if do == "submit":
                relay.submit(jid, sf, job.get("mode") or "virality", job["s3_input_key"], now)
                db.save_wait(jid, nw)
                acts.append(f"yt_wait_submit {jid[:8]} attempt={nw['attempt']}")
            elif do in ("init", "schedule"):
                db.save_wait(jid, nw)
                acts.append(f"yt_wait_{do} {jid[:8]} next_in={int(float(nw['next_at']) - now)}s")
            elif do == "requeue":
                relay.withdraw(jid)
                sf2 = dict(sf, prefetched=True, prefetched_duration=d.get("duration"), prefetched_relay=d.get("relay"))
                if db.requeue(jid, sf2, nw):
                    events["requeued"].append({"id": jid, "waited": now - float(nw["since"])})
                    acts.append(f"yt_wait_requeue {jid[:8]} relay={d.get('relay')} waited={int(now - float(nw['since']))}s")
            elif do == "fail_user":
                relay.withdraw(jid)
                if fail(jid, d["code"], d["message"]):
                    acts.append(f"yt_wait_user_fail {jid[:8]} {d['code']}")
            elif do == "give_up":
                relay.withdraw(jid)
                db.save_wait(jid, dict(nw, gave_up=now))
                if fail(jid, GIVEUP_CODE, GIVEUP_TEXT):
                    events["gave_up"].append({"id": jid, "user_id": job.get("user_id"), "mode": job.get("mode"),
                                              "attempts": nw.get("attempt"), "last": nw.get("last")})
                    acts.append(f"yt_wait_give_up {jid[:8]} attempts={nw.get('attempt')}")
        except Exception as exc:  # noqa: BLE001 - one job never stops the others
            log.warning("yt_wait_job_error job=%s %s: %s", jid, type(exc).__name__, str(exc)[:200])
    if ledger is not None:
        try:
            still = [j for j in db.waiting()] if (events["requeued"] or events["gave_up"]) else jobs
            acts += alerts(ledger, still, events, now, pool=pool)
        except Exception as exc:  # noqa: BLE001
            log.warning("yt_wait_alert_error %s", type(exc).__name__)
    return acts


# ── admin alerts ────────────────────────────────────────────────────────────────
COOKIES_HOWTO = ("Зайди в отдельный (НЕ личный) аккаунт YouTube в браузере, выгрузи cookies.txt расширением "
                 "«Get cookies.txt LOCALLY» и положи файл в папку _incoming/yt-cookies на Mac mini "
                 "(можно несколько файлов от разных аккаунтов — система переключится сама).")


def _since(j: Dict[str, Any], now: float) -> float:
    w = _params(j).get("yt_wait") or {}
    try:
        return float(w.get("since") or now)
    except (TypeError, ValueError):
        return now


def alerts(ledger, waiting: List[Dict[str, Any]], events: Dict[str, List[Dict[str, Any]]], now: float,
           pool: Optional[Dict[str, Any]] = None) -> List[str]:
    from accounting import tgformat as tf
    from accounting.balance import ops_alert
    out: List[str] = []
    pool_line = ""
    if pool and pool.get("total") is not None:
        pool_line = f"Пул cookies: живых {pool.get('live', 0)} из {pool.get('total', 0)}."
    if waiting:
        start = min(_since(j, now) for j in waiting)
        oldest = int((now - start) // 60)
        n = len(waiting)
        k = ops_alert(ledger, f"yt-wait:{int(start)}:{int((now - start) // 3600)}", [
            tf.title("🚨 YouTube перестал отдавать видео, нужен новый файл cookies"),
            tf.esc(f"Ждут скачивания: {n} {_jobs_word(n)} (самое долгое — {oldest} мин). Пользователи видят «Скачиваем видео… "
                   "это может занять дольше обычного», попытка не списана. Система сама повторяет попытки до 2 ч, потом вернёт "
                   "попытку и попросит загрузить файл."),
            tf.esc(COOKIES_HOWTO + (" " + pool_line if pool_line else ""))], now)
        if k:
            out.append(k)
    for e in events.get("requeued") or []:
        if not waiting:
            k = ops_alert(ledger, f"yt-wait-ok:{e['id']}", [
                tf.title("✅ YouTube снова отдаёт видео"),
                tf.esc(f"Задание {e['id'][:8]} скачано после {int(e['waited'] // 60)} мин ожидания и пошло в обработку. "
                       "Больше никто не ждёт.")], now)
            if k:
                out.append(k)
    for e in events.get("gave_up") or []:
        k = ops_alert(ledger, f"yt-wait-gaveup:{e['id']}", [
            tf.title("⚠️ YouTube: видео не скачалось за 2 ч"),
            tf.esc(f"Задание {e['id'][:8]} ({e.get('mode')}): {e.get('attempts')} попыток, последняя — {e.get('last') or '?'}. "
                   "Пользователю показано «Не получилось получить видео с YouTube. Загрузите файл или ссылку VK/Rutube — "
                   "эта попытка не списана», попытка возвращена.")], now)
        if k:
            out.append(k)
    return out


def _jobs_word(n: int) -> str:
    if n % 10 == 1 and n % 100 != 11:
        return "задание"
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return "задания"
    return "заданий"
