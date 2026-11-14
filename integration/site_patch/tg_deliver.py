#!/usr/bin/env python3
"""SONYA: send finished clips to the user's linked Telegram (@sonya_group_bot).

Runs on the site VPS next to sonya-api (cwd /opt/sonya, env from .env.local), as
its own service so a Telegram failure can never touch a job. No DB schema change:
idempotency state is a local sqlite file (job_id + s3_key primary key).

Rules: only completed video jobs of users with telegram_linked + chat_id; clip
<= 49 MB -> sendVideo, bigger -> a download link; one message per clip, never
twice; a send error is logged (type only, never the token) and retried later,
at most MAX_ATTEMPTS. Nothing is sent before SONYA_TG_DELIVER_SINCE (set on first
start = now) so enabling it does not flood users with old jobs.

  python -m scripts.tg_deliver            # loop
  python -m scripts.tg_deliver --once     # one pass
  python -m scripts.tg_deliver --job ID --force   # test one job (ignores 'since')
"""
from __future__ import annotations

import argparse, logging, os, sqlite3, sys, tempfile, time
from datetime import datetime, timezone

import requests

log = logging.getLogger("tg_deliver")
STATE = os.environ.get("SONYA_TG_STATE", "/var/lib/sonya-tg/state.db")
API = os.environ.get("TELEGRAM_API_BASE", "https://api.telegram.org")
MAX_VIDEO = int(os.environ.get("SONYA_TG_MAX_VIDEO_BYTES", str(49 * 1024 * 1024)))
MAX_ATTEMPTS = 5
VIDEO_MODES = ("virality", "hooks", "educational", "streamer", "beep")
LINK_TTL = 7 * 86400
MODE_RU = {"virality": "вирусные клипы", "hooks": "хуки", "educational": "обучающие клипы",
           "streamer": "клипы стримера", "beep": "бип-цензура"}


def _db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    c = sqlite3.connect(STATE)
    c.execute("CREATE TABLE IF NOT EXISTS sent(job_id TEXT, s3_key TEXT, status TEXT, attempts INT DEFAULT 0, ts TEXT, PRIMARY KEY(job_id, s3_key))")
    c.execute("CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT)")
    return c


def _since(c) -> datetime:
    v = os.environ.get("SONYA_TG_DELIVER_SINCE")
    if not v:
        r = c.execute("SELECT v FROM meta WHERE k='since'").fetchone()
        if r:
            v = r[0]
        else:
            v = datetime.now(timezone.utc).isoformat()
            c.execute("INSERT INTO meta VALUES('since',?)", (v,)); c.commit()
    return datetime.fromisoformat(v)


def _tg(method: str, **kw):
    tok = os.environ["TELEGRAM_BOT_TOKEN"]
    return requests.post("%s/bot%s/%s" % (API, tok, method), timeout=kw.pop("timeout", 30), **kw)


def _send_clip(chat_id: int, url: str, size: int, caption: str, link_url: str) -> bool:
    try:
        if size and size <= MAX_VIDEO:
            with tempfile.NamedTemporaryFile(suffix=".mp4") as tf:
                with requests.get(url, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    for ch in r.iter_content(1 << 20):
                        tf.write(ch)
                tf.flush()
                with open(tf.name, "rb") as fh:
                    resp = _tg("sendVideo", timeout=300, data={"chat_id": chat_id, "caption": caption,
                               "supports_streaming": "true"}, files={"video": ("sonya_clip.mp4", fh, "video/mp4")})
        else:
            resp = _tg("sendMessage", json={"chat_id": chat_id,
                       "text": caption + "\nФайл большой — скачать (ссылка 7 дней): " + link_url})
        if not resp.ok:
            log.warning("send failed chat=%s status=%s", chat_id, resp.status_code)
        return resp.ok
    except Exception as exc:  # never include str(exc): it may hold the URL / token
        log.warning("send error chat=%s type=%s", chat_id, type(exc).__name__)
        return False


def run_once(job_only: str | None = None, force: bool = False) -> int:
    from scripts.telegram_store import _get_conn
    from scripts.prod_s3_storage import generate_presigned_get_url
    st = _db(); since = datetime.min.replace(tzinfo=timezone.utc) if force else _since(st)
    sent = 0
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            q = ("SELECT j.id, j.mode, u.telegram_chat_id FROM generation_jobs j JOIN users u ON u.id::text=j.user_id::text "
                 "WHERE j.status::text='completed' AND j.mode::text = ANY(%s) AND u.telegram_linked AND u.telegram_chat_id IS NOT NULL "
                 "AND j.completed_at >= %s ")
            args = [list(VIDEO_MODES), since]
            if job_only:
                q += "AND j.id::text=%s "; args.append(job_only)
            else:
                q += "AND j.completed_at > now() - interval '3 days' "
            cur.execute(q + "ORDER BY j.completed_at", args)
            jobs = [dict(r) for r in cur.fetchall()]
            for j in jobs:
                cur.execute("SELECT s3_key,size_bytes FROM generation_files WHERE job_id=%s AND file_type='output' "
                            "AND lower(s3_key) LIKE '%%.mp4' ORDER BY s3_key", (j["id"],))
                files = [dict(r) for r in cur.fetchall()]
                for n, f in enumerate(files, 1):
                    row = st.execute("SELECT status,attempts FROM sent WHERE job_id=? AND s3_key=?", (j["id"], f["s3_key"])).fetchone()
                    if row and (row[0] == "ok" or row[1] >= MAX_ATTEMPTS):
                        continue
                    try:
                        url = generate_presigned_get_url(f["s3_key"], expires_in=LINK_TTL)
                        link = generate_presigned_get_url(f["s3_key"], expires_in=LINK_TTL, download_name="sonya_clip_%d.mp4" % n)
                    except Exception as exc:
                        log.warning("presign error job=%s type=%s", j["id"][:8], type(exc).__name__); continue
                    cap = "SONYA · %s · клип %d из %d" % (MODE_RU.get(j["mode"], j["mode"]), n, len(files))
                    ok = _send_clip(int(j["telegram_chat_id"]), url, int(f.get("size_bytes") or 0), cap, link)
                    st.execute("INSERT INTO sent(job_id,s3_key,status,attempts,ts) VALUES(?,?,?,1,?) ON CONFLICT(job_id,s3_key) "
                               "DO UPDATE SET status=excluded.status, attempts=attempts+1, ts=excluded.ts",
                               (j["id"], f["s3_key"], "ok" if ok else "fail", datetime.now(timezone.utc).isoformat()))
                    st.commit(); sent += ok
                    time.sleep(0.5)
    finally:
        conn.close()
    return sent


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(); ap.add_argument("--once", action="store_true")
    ap.add_argument("--job"); ap.add_argument("--force", action="store_true"); a = ap.parse_args()
    if not os.environ.get("TELEGRAM_BOT_TOKEN"):
        log.error("TELEGRAM_BOT_TOKEN is not set - delivery disabled"); sys.exit(2)
    if a.once or a.job:
        print("sent", run_once(a.job, a.force)); return
    while True:
        try:
            run_once()
        except Exception as exc:
            log.warning("pass failed type=%s", type(exc).__name__)
        time.sleep(30)


if __name__ == "__main__":
    main()
