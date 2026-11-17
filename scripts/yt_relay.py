"""YouTube relay: download a link on a machine YouTube trusts, hand the file to the GPU worker through S3.

Why (2026-11-17): YouTube answers "Sign in to confirm you're not a bot" to Vast data-centre IPs. A probe on a fresh
Vast host: every player client (tv, web_safari, mweb, ios, android_vr, web, tv_simply, web_embedded), with the newest
yt-dlp AND a bgutil PO token, failed on 3 live links. The same links download from the bot server (Timeweb Moscow,
~12 MB/s) and from the owner's Mac mini (home IP, ~2 MB/s with 16 parallel fragments). docs/YOUTUBE_FETCH.md.

Protocol (all objects under SONYA_YT_RELAY_PREFIX, default worker-v2/yt-relay/):
  req/<job>.json            worker -> relays: {job_id, sf, mode, s3_key, max_height, created}
  st/<job>/<relay>.json     relay  -> worker: {state: claimed|downloading|done|failed, ts, pct, code, message, duration}
  hb/<relay>.json           relay heartbeat every 30 s: {ts, priority, version, ytdlp, canary, cookies, active}
The relay downloads with the SAME code as the worker (source_fetch.fetch_source: length check first, user's segment,
<=1080p) and uploads to the job's s3_input_key; the worker then reads that key like an uploaded file.

Who takes a request: relays are ordered by priority (bot server 0, Mac mini 1). A relay takes a job when no relay with
a smaller priority is handling it: that relay failed it, did not claim it within CLAIM_GRACE s, went silent, or has no
fresh heartbeat. So the fast relay goes first and the home relay is the automatic backup.

Run a relay:   python -m scripts.yt_relay serve --name bot --priority 0
One check:     python -m scripts.yt_relay canary
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("sonya.yt_relay")

HB_FRESH = 90            # a relay with an older heartbeat is considered down
CLAIM_GRACE = 25         # a higher-priority relay must claim within this many seconds of the request
STATUS_STALE = 240       # a claimed job whose status did not move for this long is taken over / given up
CANARY_URL = os.environ.get("SONYA_YT_CANARY_URL", "https://www.youtube.com/watch?v=jNQXAC9IVRw")   # 19 s, public since 2005
TERMINAL = ("done", "failed")


def install_s3_tunnel() -> None:
    """Mac mini: the home ISP sometimes drops every connection to the S3 host (TCP opens, TLS dies; 2026-11-17).
    SONYA_S3_TUNNEL=127.0.0.1:19443 (an `ssh -L 19443:s3.twcstorage.ru:443` to the VPS, run.sh keeps it up) makes THIS
    process connect to the tunnel for the S3 host only; TLS still verifies s3.twcstorage.ru end to end."""
    spec = os.environ.get("SONYA_S3_TUNNEL", "").strip()
    host = (os.environ.get("S3_ENDPOINT_URL", "").split("//")[-1].split("/")[0].split(":")[0]).lower()
    if not spec or not host:
        return
    import socket
    th, tp = spec.rsplit(":", 1)
    orig = socket.getaddrinfo
    if getattr(orig, "_sonya_tunnel", False):
        return

    def gai(h, port, *a, **kw):
        if isinstance(h, str) and h.lower() == host and int(port or 443) == 443:
            return orig(th, int(tp), *a, **kw)
        return orig(h, port, *a, **kw)
    gai._sonya_tunnel = True  # type: ignore[attr-defined]
    socket.getaddrinfo = gai


install_s3_tunnel()


def prefix() -> str:
    p = os.environ.get("SONYA_YT_RELAY_PREFIX", "worker-v2/yt-relay/")
    return p if p.endswith("/") else p + "/"


# ── S3 (the worker's own client/bucket) ───────────────────────────────────────
def _s3():
    from scripts import prod_s3_storage as st
    return st._client(), st._bucket()


def put_json(key: str, data: Dict[str, Any]) -> None:
    c, b = _s3()
    c.put_object(Bucket=b, Key=key, Body=json.dumps(data, ensure_ascii=False, default=str).encode(),
                 ContentType="application/json")


def get_json(key: str) -> Optional[Dict[str, Any]]:
    c, b = _s3()
    try:
        return json.loads(c.get_object(Bucket=b, Key=key)["Body"].read())
    except Exception:  # noqa: BLE001 - missing / half-written -> None
        return None


def list_keys(pfx: str) -> List[str]:
    c, b = _s3()
    out, tok = [], None
    while True:
        kw = {"Bucket": b, "Prefix": pfx, "MaxKeys": 1000}
        if tok:
            kw["ContinuationToken"] = tok
        r = c.list_objects_v2(**kw)
        out += [o["Key"] for o in r.get("Contents", [])]
        if not r.get("IsTruncated"):
            return out
        tok = r.get("NextContinuationToken")


def delete_keys(keys: List[str]) -> None:
    c, b = _s3()
    for i in range(0, len(keys), 500):
        c.delete_objects(Bucket=b, Delete={"Objects": [{"Key": k} for k in keys[i:i + 500]], "Quiet": True})


def heartbeats(now: Optional[float] = None) -> Dict[str, Dict[str, Any]]:
    """name -> heartbeat of every relay (fresh or not; 'fresh' key added)."""
    now = now or time.time()
    out = {}
    for k in list_keys(prefix() + "hb/"):
        hb = get_json(k)
        if hb:
            name = k.rsplit("/", 1)[-1][:-5]
            hb["fresh"] = now - float(hb.get("ts") or 0) <= HB_FRESH
            out[name] = hb
    return out


def statuses(job_id: str) -> Dict[str, Dict[str, Any]]:
    out = {}
    for k in list_keys(f"{prefix()}st/{job_id}/"):
        st = get_json(k)
        if st:
            out[k.rsplit("/", 1)[-1][:-5]] = st
    return out


# ── the decision rule (pure, unit-tested) ─────────────────────────────────────
def should_take(me: str, my_priority: int, req: Dict[str, Any], sts: Dict[str, Dict[str, Any]],
                hbs: Dict[str, Dict[str, Any]], now: float) -> bool:
    if any(s.get("state") == "done" for s in sts.values()):
        return False
    mine = sts.get(me)
    if mine is not None:
        # my own status that stopped moving = this relay restarted mid-download: take it again
        return mine.get("state") not in TERMINAL and now - float(mine.get("ts") or 0) > STATUS_STALE
    age = now - float(req.get("created") or now)
    for name, hb in hbs.items():
        if name == me or int(hb.get("priority", 99)) >= my_priority or not hb.get("fresh"):
            continue
        st = sts.get(name)
        if st is None:
            if age <= CLAIM_GRACE:
                return False                        # its turn: wait
            continue                                # did not claim in time (busy / broken): take over
        if st.get("state") == "failed":
            continue
        if now - float(st.get("ts") or 0) > STATUS_STALE:
            continue                                # claimed and went silent
        return False                                # still working on it
    return True


# ── worker side ───────────────────────────────────────────────────────────────
class RelayUnavailable(Exception):
    """No relay is alive (or all of them gave up for a non-user reason). The caller decides what is next."""


_USER_CODES_ORDER = ("SOURCE_TOO_LONG", "SOURCE_TRIM_RANGE", "SOURCE_PRIVATE", "SOURCE_AGE_RESTRICTED", "SOURCE_UNAVAILABLE",
                     "SOURCE_GEO", "SOURCE_LIVE", "SOURCE_UNSUPPORTED", "SOURCE_UNSAFE_URL", "SOURCE_DURATION_UNKNOWN")


def fetch_via_relay(job_id: str, sf: Dict[str, Any], mode: str, s3_key: str, work_dir: Path,
                    tick: Optional[Callable[[str], None]] = None, wait_alive: float = 120.0,
                    max_total: float = 2.5 * 3600, poll: float = 3.0) -> Dict[str, Any]:
    """Ask the relays for sf['url'], wait, then read the file from s3_key. -> {"path", "duration", "relay"}.
    Raises source_fetch.SourceFetchError for a user-facing reason (private, too long, ...), RelayUnavailable otherwise."""
    from scripts.prod_s3_storage import download_file
    from scripts.source_fetch import SourceFetchError
    t0 = time.time()
    put_json(f"{prefix()}req/{job_id}.json", {"job_id": job_id, "sf": sf, "mode": mode, "s3_key": s3_key, "created": t0,
                                              "max_height": os.environ.get("SONYA_YTDLP_MAX_HEIGHT", "1080")})
    logger.info("[yt_relay] request job_id=%s", job_id)
    last_move, last_sig, last_tick = t0, None, 0.0
    net_err_since: Optional[float] = None
    try:
        while True:
            now = time.time()
            try:
                sts = statuses(job_id)
                hbs = heartbeats(now)
                net_err_since = None
            except Exception as exc:  # noqa: BLE001 - a network blip to S3 must not drop the job
                net_err_since = net_err_since or now
                logger.warning("[yt_relay] poll error %s", type(exc).__name__)
                if now - net_err_since > 300:
                    raise RelayUnavailable(f"S3 unreachable: {type(exc).__name__}") from exc
                time.sleep(max(poll, 5))
                continue
            done = [n for n, s in sts.items() if s.get("state") == "done"]
            if done:
                dest = work_dir / "input.mp4"
                for i in range(5):
                    try:
                        download_file(s3_key, dest)
                        break
                    except Exception:  # noqa: BLE001
                        if i == 4:
                            raise
                        time.sleep(5 * (i + 1))
                st = sts[done[0]]
                logger.info("[yt_relay] done job_id=%s relay=%s in %.0f s", job_id, done[0], now - t0)
                return {"path": dest, "duration": st.get("duration"), "relay": done[0], "wait_s": round(now - t0, 1)}
            sig = json.dumps({n: (s.get("state"), s.get("ts")) for n, s in sts.items()}, sort_keys=True)
            if sig != last_sig:
                last_sig, last_move = sig, now
            alive = {n for n, h in hbs.items() if h.get("fresh")}
            failed = {n: s for n, s in sts.items() if s.get("state") == "failed"}
            running = [n for n, s in sts.items() if s.get("state") not in TERMINAL and now - float(s.get("ts") or 0) <= STATUS_STALE]
            if any(h.get("fresh") and job_id in (h.get("active") or []) for h in hbs.values()):
                last_move = now                           # a live relay holds the job (its heartbeat says so)
            if failed:
                for code in _USER_CODES_ORDER:            # a real answer about the video: no other relay changes it
                    for s in failed.values():
                        if s.get("code") == code:
                            raise SourceFetchError(code, s.get("message") or code)
            if not running and alive and alive <= set(failed):
                raise RelayUnavailable("all relays failed: " + ", ".join(f"{n}={s.get('code')}" for n, s in failed.items()))
            if not alive and not running and now - t0 > wait_alive:
                raise RelayUnavailable("no relay alive")
            if now - last_move > STATUS_STALE + 60:
                raise RelayUnavailable("relays stalled")
            if now - t0 > max_total:
                raise RelayUnavailable("relay timeout")
            if tick and now - last_tick > 30:
                last_tick = now
                try:
                    tick(",".join(f"{n}:{s.get('state')}:{s.get('pct', '')}" for n, s in sts.items()) or "waiting")
                except Exception:  # noqa: BLE001
                    pass
            time.sleep(poll)
    finally:
        try:
            c, b = _s3()
            c.delete_object(Bucket=b, Key=f"{prefix()}req/{job_id}.json")
        except Exception:  # noqa: BLE001 - the relay's janitor removes it a day later
            pass


# ── relay side ────────────────────────────────────────────────────────────────
def ytdlp_version() -> str:
    try:
        import yt_dlp
        return yt_dlp.version.__version__
    except Exception:  # noqa: BLE001
        return "?"


def run_canary(url: str = CANARY_URL) -> Dict[str, Any]:
    """The users' path in miniature: probe + download (+cookies retry) of a tiny public video, no S3."""
    from scripts import url_ingest
    t = time.time()
    path = None
    try:
        url_ingest.probe(url, "youtube", max_duration_sec=3600)
        path, _ = url_ingest.download_video(url, "youtube", max_height=360)
        size = os.path.getsize(path)
        return {"ok": size > 10000, "ts": time.time(), "seconds": round(time.time() - t, 1), "bytes": size}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "ts": time.time(), "seconds": round(time.time() - t, 1),
                "bot_check": url_ingest._is_bot_check(exc), "error": str(exc)[:200]}
    finally:
        if path:
            url_ingest.cleanup(path)


def check_cookies() -> str:
    """'none' | 'ok' | 'expired' | 'error'. Loads the cookie jar and asks youtube.com whether we are signed in.
    Never logs a cookie value."""
    from scripts import url_ingest
    if not url_ingest.cookies_configured():
        return "none"
    import http.cookiejar
    import requests
    try:
        with url_ingest._CookieCopy() as ck:
            jar = http.cookiejar.MozillaCookieJar(ck)
            jar.load(ignore_discard=True, ignore_expires=True)
        r = requests.get("https://www.youtube.com/account", cookies=jar, timeout=30, allow_redirects=True,
                         headers={"User-Agent": url_ingest._UA, "Accept-Language": "en-US,en;q=0.8"})
        if '"LOGGED_IN":true' in r.text or '"logged_in":"1"' in r.text.lower():
            return "ok"
        return "expired"
    except Exception as exc:  # noqa: BLE001
        logger.warning("[yt_relay] cookie check error %s", type(exc).__name__)
        return "error"


class Relay:
    def __init__(self, name: str, priority: int, concurrency: int, work_root: Path):
        self.name, self.priority, self.concurrency, self.work_root = name, priority, concurrency, work_root
        self.active: Dict[str, threading.Thread] = {}
        self.canary: Dict[str, Any] = {}
        self.cookies, self.cookies_checked = "none", 0.0
        self.started = time.time()
        self.lock = threading.Lock()

    # status of one job, written by this relay
    def _st(self, job_id: str, **kw) -> None:
        put_json(f"{prefix()}st/{job_id}/{self.name}.json", {"ts": time.time(), "relay": self.name, **kw})

    def heartbeat(self) -> None:
        put_json(f"{prefix()}hb/{self.name}.json", {
            "ts": time.time(), "priority": self.priority, "ytdlp": ytdlp_version(), "active": sorted(self.active),
            "canary": self.canary, "cookies": self.cookies, "cookies_checked": self.cookies_checked,
            "started": self.started, "pid": os.getpid()})

    def _job(self, req: Dict[str, Any]) -> None:
        from scripts.source_fetch import SourceFetchError, fetch_source
        job_id = req["job_id"]
        wd = Path(tempfile.mkdtemp(prefix=f"ytr_{job_id[:8]}_", dir=str(self.work_root)))
        last = {"t": 0.0, "pct": None}
        stop = threading.Event()

        def prog(pct: float) -> None:
            last["pct"] = round(float(pct), 1)
            if time.time() - last["t"] > 10:
                last["t"] = time.time()
                try:
                    self._st(job_id, state="downloading", pct=last["pct"])
                except Exception:  # noqa: BLE001
                    pass

        def keepalive() -> None:
            # a segment download (ffmpeg) reports no progress: the status still moves every 30 s while we work
            while not stop.wait(30):
                if time.time() - last["t"] >= 25:
                    last["t"] = time.time()
                    try:
                        self._st(job_id, state="downloading", pct=last["pct"])
                    except Exception:  # noqa: BLE001
                        pass
        t0 = time.time()
        ka = threading.Thread(target=keepalive, name=f"relay-ka-{job_id[:8]}", daemon=True)
        try:
            self._st(job_id, state="claimed")
            ka.start()
            os.environ["SONYA_YTDLP_MAX_HEIGHT"] = str(req.get("max_height") or "1080")
            got = fetch_source(req["sf"], req.get("mode") or "virality", req["s3_key"], wd, progress=prog)
            stop.set()
            ka.join(5)
            self._st(job_id, state="done", duration=got.get("duration"), bytes=got.get("bytes"),
                     seconds=round(time.time() - t0, 1))
            logger.info("[yt_relay] done job=%s %.0f s %d bytes", job_id, time.time() - t0, got.get("bytes") or 0)
        except SourceFetchError as exc:
            stop.set()
            ka.join(5)
            logger.warning("[yt_relay] failed job=%s code=%s", job_id, exc.code)
            self._st(job_id, state="failed", code=exc.code, message=exc.message, seconds=round(time.time() - t0, 1))
        except Exception as exc:  # noqa: BLE001
            stop.set()
            ka.join(5)
            logger.exception("[yt_relay] crashed job=%s", job_id)
            self._st(job_id, state="failed", code="RELAY_ERROR", message=str(exc)[:200])
        finally:
            stop.set()
            shutil.rmtree(wd, ignore_errors=True)
            with self.lock:
                self.active.pop(job_id, None)

    def scan(self) -> None:
        now = time.time()
        reqs = [k for k in list_keys(prefix() + "req/") if k.endswith(".json")]
        if not reqs:
            return
        hbs = heartbeats(now)
        for k in reqs:
            job_id = k.rsplit("/", 1)[-1][:-5]
            with self.lock:
                if job_id in self.active or len(self.active) >= self.concurrency:
                    continue
            req = get_json(k)
            if not req:
                continue
            if now - float(req.get("created") or now) > 3 * 3600:
                continue                                       # orphan (worker died): the janitor removes it
            if not should_take(self.name, self.priority, req, statuses(job_id), hbs, now):
                continue
            th = threading.Thread(target=self._job, args=(req,), name=f"relay-{job_id[:8]}", daemon=True)
            with self.lock:
                self.active[job_id] = th
            th.start()

    def janitor(self) -> None:
        """Requests/statuses older than a day (crashed workers) are removed; only the priority-0 relay does it."""
        if self.priority != 0:
            return
        c, b = _s3()
        old = time.time() - 86400
        stale = []
        for pfx in ("req/", "st/"):
            kw = {"Bucket": b, "Prefix": prefix() + pfx}
            for o in c.list_objects_v2(**kw).get("Contents", []):
                if o["LastModified"].timestamp() < old:
                    stale.append(o["Key"])
        if stale:
            delete_keys(stale)

    def self_update(self) -> bool:
        """Newest yt-dlp (nightly: YouTube changes often) into this venv; True if the version changed (caller re-execs)."""
        if os.environ.get("SONYA_YT_AUTOUPDATE", "1") in ("0", "off", "false"):
            return False
        before = ytdlp_version()
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "--pre", "yt-dlp[default]"],
                           capture_output=True, text=True, timeout=600)
        if r.returncode:
            logger.warning("[yt_relay] self-update failed: %s", r.stderr[-300:])
            return False
        after = subprocess.run([sys.executable, "-c", "import yt_dlp;print(yt_dlp.version.__version__)"],
                               capture_output=True, text=True).stdout.strip()
        logger.info("[yt_relay] yt-dlp %s -> %s", before, after)
        return bool(after) and after != before

    def serve(self) -> None:
        self.work_root.mkdir(parents=True, exist_ok=True)
        # yt-dlp/url_ingest temp files go to the work dir: /tmp is a ~1 GB RAM tmpfs on the bot server (Ubuntu 26.04),
        # a 1-hour video filled it (ENOSPC, 2026-11-17)
        os.environ["TMPDIR"] = str(self.work_root)
        tempfile.tempdir = str(self.work_root)
        last_hb = last_canary = last_jan = 0.0
        last_update = time.time()
        restart = False
        while True:
            now = time.time()
            try:
                ck = os.environ.get("SONYA_YT_COOKIES", "")
                mt = os.path.getmtime(ck) if ck and os.path.exists(ck) else 0.0
                if mt != getattr(self, "_ck_mtime", None):      # new / rotated / removed cookies: check now
                    self._ck_mtime, self.cookies_checked = mt, 0.0
                if now - self.cookies_checked > 6 * 3600:
                    self.cookies, self.cookies_checked = check_cookies(), now
                if now - last_canary > float(os.environ.get("SONYA_YT_CANARY_EVERY", "3600")):
                    last_canary = now
                    self.canary = run_canary()
                    logger.info("[yt_relay] canary ok=%s %.1fs", self.canary.get("ok"), self.canary.get("seconds", 0))
                    if not self.canary.get("ok") and self.canary.get("bot_check") and self.cookies == "ok":
                        self.cookies, self.cookies_checked = check_cookies(), now
                if now - last_hb > 30:
                    last_hb = now
                    self.heartbeat()
                if not restart:
                    self.scan()
                if now - last_jan > 3600:
                    last_jan = now
                    self.janitor()
                if now - last_update > 86400:
                    last_update = now
                    restart = self.self_update()
            except Exception as exc:  # noqa: BLE001 - network blips (the Mac's line drops): keep going
                logger.warning("[yt_relay] loop error %s: %s", type(exc).__name__, str(exc)[:200])
                time.sleep(10)
            if restart and not self.active:
                logger.info("[yt_relay] restarting on the new yt-dlp")
                os.execv(sys.executable, [sys.executable, "-m", "scripts.yt_relay", *sys.argv[1:]])
            time.sleep(3)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--name", required=True)
    s.add_argument("--priority", type=int, default=0)
    s.add_argument("--concurrency", type=int, default=2)
    s.add_argument("--work", default=os.environ.get("SONYA_YT_RELAY_WORK", tempfile.gettempdir() + "/sonya-ytrelay"))
    s.add_argument("--update-first", action="store_true", help="pip -U yt-dlp before serving")
    sub.add_parser("canary")
    sub.add_parser("cookies")
    sub.add_parser("status")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    if a.cmd == "canary":
        print(json.dumps(run_canary()))
        return 0
    if a.cmd == "cookies":
        print(check_cookies())
        return 0
    if a.cmd == "status":
        print(json.dumps(heartbeats(), indent=1, default=str))
        return 0
    r = Relay(a.name, a.priority, a.concurrency, Path(a.work))
    if a.update_first and r.self_update():
        os.execv(sys.executable, [sys.executable, "-m", "scripts.yt_relay", *[x for x in sys.argv[1:] if x != "--update-first"]])
    r.serve()
    return 0


if __name__ == "__main__":
    sys.exit(main())
