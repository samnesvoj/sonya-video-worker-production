"""
Entry points on the SONYA VPS (central side only).
  build_fleet()   -> FleetManager wired to Vast, the production queue (read-only) and the GitHub App
  python -m dispatcher.main run      loop: fleet tick every SONYA_FLEET_TICK_SECONDS + due accounting reports
  python -m dispatcher.main once     single tick (cron / manual check)
  python -m dispatcher.main status   servers + the ranked spare offers per class (reads fleet.db only)
Secrets come from files/env on THIS host: VAST_API_KEY_FILE (0600), GH_APP_* , WORKER_SECRET, S3_*.
SONYA_FLEET_ENABLED=false (default) makes everything a no-op; SONYA_FLEET_DRY_RUN=true (default) sends no write call to Vast.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import date
from pathlib import Path
from typing import Optional

from dispatcher.config import FleetConfig
from dispatcher.fleet import FleetManager
from dispatcher.queue import PostgresQueueView
from dispatcher.store import FleetStore
from dispatcher.token_provider import github_app_token_provider, release_token_provider
from dispatcher.vast_api import VastClient

log = logging.getLogger("sonya.fleet.main")
STATE_DIR = Path(os.environ.get("SONYA_FLEET_STATE_DIR", "/var/lib/sonya-fleet"))


def _vast_key(env=os.environ) -> str:
    f = env.get("VAST_API_KEY_FILE")
    if f:
        mode = Path(f).stat().st_mode & 0o077
        if mode:
            raise RuntimeError("VAST_API_KEY_FILE must not be group/world accessible (chmod 600)")
        return Path(f).read_text().strip()
    return env.get("VAST_API_KEY", "")


def build_fleet(env=os.environ) -> FleetManager:
    cfg = FleetConfig.from_env(dict(env))
    STATE_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    return FleetManager(cfg, FleetStore(str(STATE_DIR / "fleet.db")), VastClient(_vast_key(env), dry_run=cfg.dry_run),
                        PostgresQueueView(env["DATABASE_URL"], dedicated=cfg.dedicated), release_token_provider(cfg.release_repo, env), env,
                        nvenc_missing=_nvenc_missing_checker(env))


def _nvenc_missing_checker(env=os.environ):  # pragma: no cover - S3 on the VPS
    """label -> True when the worker published fleet-metrics/<label>/nvenc_missing.json (scripts/ops/readiness.py).
    Off when SONYA_NVENC_MISSING is not "replace" or there are no S3 credentials."""
    bucket = env.get("S3_BUCKET_NAME") or env.get("S3_BUCKET")
    if env.get("SONYA_NVENC_MISSING", "").strip().lower() != "replace" or not bucket or not env.get("S3_ACCESS_KEY_ID"):
        return None
    import boto3
    from botocore.config import Config
    client = boto3.client("s3", endpoint_url=env.get("S3_ENDPOINT_URL") or None, region_name=env.get("S3_REGION") or None,
                          aws_access_key_id=env["S3_ACCESS_KEY_ID"], aws_secret_access_key=env.get("S3_SECRET_ACCESS_KEY"),
                          config=Config(connect_timeout=3, read_timeout=5, retries={"max_attempts": 1}))

    def check(label: str) -> bool:
        try:
            client.head_object(Bucket=bucket, Key=f"fleet-metrics/{label}/nvenc_missing.json")
            return True
        except Exception:  # noqa: BLE001 - 404 = fine
            return False
    return check


def build_accounting(fleet: FleetManager, env=os.environ):
    from accounting.cbr import CbrRates
    from accounting.ledger import Ledger
    from accounting.money import UTC, msk_day_bounds
    from accounting.outbox import ReportService
    from accounting.usage import SqlJobsSource, cost_between
    from datetime import datetime, timedelta

    ledger = Ledger(str(STATE_DIR / "ledger.db"))

    def counter(lo: float, hi: float) -> int:  # completed pool jobs in [lo, hi): read-only SELECT
        import psycopg2
        conn = psycopg2.connect(env["DATABASE_URL"])
        try:
            conn.set_session(readonly=True, autocommit=True)
            cur = conn.cursor()
            # test accounts (…@sonya.invalid: load tests, cutover tests) are not videos of the business
            cur.execute("SELECT COUNT(*) FROM generation_jobs WHERE status='completed' AND mode = ANY(%s) AND worker_id LIKE %s "
                        "AND completed_at >= to_timestamp(%s) AND completed_at < to_timestamp(%s) "
                        "AND user_id::text NOT IN (SELECT id::text FROM users WHERE email LIKE %s)",
                        (list(fleet.cfg.pool_modes), fleet.cfg.label_prefix + "%", lo, hi, "%@sonya.invalid"))
            return int(cur.fetchone()[0])
        finally:
            conn.close()

    def utc_cost(day: date, now: float) -> dict:
        lo = datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp()
        r = cost_between(fleet.store, lo, lo + 86400, now)
        return {k: r[k] for k in ("gpu", "disk", "other")}

    first = date.fromisoformat(env.get("SONYA_ACCOUNTING_FIRST_DAY", date.today().isoformat()))
    svc = ReportService(ledger, fleet.store, CbrRates(ledger), SqlJobsSource(counter), utc_cost, first)
    return svc, ledger


def _accounting_secret(env=os.environ) -> str:
    f = env.get("SONYA_ACCOUNTING_SECRET_FILE", "/etc/sonya-fleet/accounting_api_secret")
    if Path(f).stat().st_mode & 0o077:
        raise RuntimeError("accounting secret file must be chmod 600")
    return Path(f).read_text().strip()


HEALTH_JSON = Path(os.environ.get("SONYA_HEALTH_DIR", "/var/lib/sonya-health")) / "health.json"
_POLL = {"t": 0.0}


def health_owns_alerts(env=os.environ, now: Optional[float] = None) -> bool:
    """SONYA_HEALTH_ALERTS=on and the self-check ran in the last 45 min: the balance / YouTube / no-funds alerts are sent
    by it (once per problem + «починилось») instead of the repeating ones here. If it stops, the old ones come back."""
    if env.get("SONYA_HEALTH_ALERTS", "off") != "on":
        return False
    try:
        return (now or time.time()) - HEALTH_JSON.stat().st_mtime < 2700
    except OSError:
        return False


def start_accounting_api(fleet: FleetManager, svc, env=os.environ):  # pragma: no cover - thread on the VPS
    """Signed internal API for the accounting bot on 127.0.0.1 (nginx proxies /api/internal/accounting/vast/ here)."""
    import threading
    from decimal import Decimal
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from accounting.internal_api import AccountingApi
    from accounting.summary import build_summary

    def summary(now: float) -> str:
        try:
            bal = fleet.vast.balance_usd()
            bal = Decimal(bal) if bal is not None else None
        except Exception:  # noqa: BLE001
            bal = None
        try:
            import_gemini(svc.ledger, env)     # /vast shows the calls of the last minutes too
        except Exception as exc:  # noqa: BLE001
            log.warning("import_gemini_error %s", type(exc).__name__)
        text = build_summary(fleet.store, svc.rates, now, bal, svc.ledger, timeweb_balance(env))
        try:
            text += "\n\n" + "\n".join(_fleet_lines(fleet, now))
        except Exception as exc:  # noqa: BLE001
            log.warning("fleet_lines_error %s", type(exc).__name__)
        return text

    def status(now: float) -> str:
        from scripts.ops import health
        try:
            snap = json.loads(HEALTH_JSON.read_text())
        except Exception:  # noqa: BLE001
            snap = None
        return health.status_text(snap, now)

    def on_claim(now: float) -> None:
        if now - _POLL["t"] >= 60:
            _POLL["t"] = now
            tmp = STATE_DIR / "acct_poll.tmp"
            tmp.write_text(json.dumps({"ts": now}))
            tmp.replace(STATE_DIR / "acct_poll.json")

    api = AccountingApi(svc, _accounting_secret(env), summary=summary, status=status, on_claim=on_claim)

    class H(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            n = min(int(self.headers.get("Content-Length") or 0), 65536)
            status, data = api.handle("POST", self.path.split("?")[0], dict(self.headers), self.rfile.read(n) if n else b"")
            out = json.dumps(data).encode()
            self.send_response(status); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)

        def log_message(self, *a):  # no request logging (headers carry signatures)
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", int(env.get("SONYA_ACCOUNTING_API_PORT", "8765"))), H)
    threading.Thread(target=srv.serve_forever, name="accounting-api", daemon=True).start()
    log.info("accounting_api_listening 127.0.0.1:%s", srv.server_address[1])
    return srv


def run(once: bool = False) -> int:  # pragma: no cover - exercised on the VPS
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    fleet = build_fleet()
    svc = ledger = None
    if os.environ.get("SONYA_ACCOUNTING_ENABLED", "false").lower() == "true":
        svc, ledger = build_accounting(fleet)
        svc.extra_lines = lambda now: [_balance_line(fleet, now), *_fleet_lines(fleet, now)]
        if os.environ.get("VAST_REPORTS_ENABLED", "false").lower() == "true":
            start_accounting_api(fleet, svc)
    while True:
        try:
            publish_release_pointer(fleet)
        except Exception as exc:  # noqa: BLE001 - workers then simply stay on the release they have
            log.warning("release_pointer_error %s", type(exc).__name__)
        try:
            for a in fleet.tick():
                log.info("fleet_action %s", a)
            for a in watch_yt_wait(ledger) + watch_capacity(fleet, ledger):
                log.info("fleet_action %s", a)
            if svc:
                try:
                    import_confirmed(fleet, ledger)   # a charges-API hiccup must not stop the reports
                except Exception as exc:  # noqa: BLE001
                    log.warning("import_confirmed_error %s", type(exc).__name__)
                try:
                    import_gemini(ledger)             # Gemini token counts from the workers (S3) -> ledger
                except Exception as exc:  # noqa: BLE001
                    log.warning("import_gemini_error %s", type(exc).__name__)
                for k in svc.run_due(time.time()):
                    log.info("report_built %s", k)
                from accounting.summary import check_alert
                from decimal import Decimal
                k = check_alert(ledger, fleet.store, svc.rates, time.time(), Decimal(os.environ.get("SONYA_ACCOUNTING_ALERT_RUB_DAY", "500")))
                if k:
                    log.info("report_built %s", k)
                if not health_owns_alerts():
                    for k in watch_balance(fleet, ledger) + watch_youtube(ledger):
                        log.info("report_built %s", k)
                from accounting.fleet_stats import check_budget_alerts
                for k in check_budget_alerts(ledger, fleet.store, time.time()):
                    log.info("report_built %s", k)
        except Exception as exc:  # noqa: BLE001
            log.error("tick_error %s", type(exc).__name__)
        if once:
            return 0
        time.sleep(fleet.cfg.tick_seconds)


_PTR = {"t": 0.0, "sha": ""}


def publish_release_pointer(fleet: FleetManager, env=os.environ, every: float = 600.0) -> bool:  # pragma: no cover - S3
    """SONYA_RELEASE_POINTER=on: worker-v2/release/current.json = this dispatcher's release + image (checked every 10 min;
    written only when it differs). Deploy of a new Sonya version = new SONYA_RELEASE_SHA here + restart: no server is
    re-created, each worker switches on its next start / resume (deploy/vast/run_worker.sh)."""
    cfg = fleet.cfg
    now = time.time()
    if not cfg.release_pointer or fleet.dry or (now - _PTR["t"] < every and _PTR["sha"] == cfg.release_sha):
        return False
    _PTR["t"] = now
    from scripts.ops import release_pointer
    wrote = release_pointer.publish(cfg.release_sha, cfg.image, env=env)
    _PTR["sha"] = cfg.release_sha
    if wrote:
        fleet.store.event(now, None, "release_pointer", sha=cfg.release_sha)
        log.info("release_pointer_published %s", cfg.release_sha[:12])
    return wrote


def _fleet_lines(fleet: FleetManager, now: float) -> list:
    from accounting import fleet_stats
    spend = fleet.day_spend_usd(now) if fleet.cfg.day_budget_usd > 0 else None
    return fleet_stats.lines(fleet.store, now, cfg=fleet.cfg, spend_usd=spend)


_BAL = {"t": 0.0, "v": None}


def _balance(fleet: FleetManager, now: float, every: float = 300.0):  # pragma: no cover - network
    """Vast credit (Decimal) read at most every 5 min; None if unknown."""
    from decimal import Decimal
    if now - _BAL["t"] >= every:
        _BAL["t"] = now
        try:
            b = fleet.vast.balance_usd()
            _BAL["v"] = Decimal(str(b)) if b is not None else None
        except Exception as exc:  # noqa: BLE001
            log.warning("balance_error %s", type(exc).__name__)
    return _BAL["v"]


def _balance_line(fleet: FleetManager, now: float):  # pragma: no cover
    from accounting import balance as bal
    _BAL["t"] = 0.0                       # fresh value for the report
    b = _balance(fleet, now)
    return bal.balance_line(b, bal.burn_usd_per_hour(fleet.store, now) if b is not None else None)


def watch_balance(fleet: FleetManager, ledger, env=os.environ) -> list:  # pragma: no cover - VPS loop
    """Alert when the Vast balance is low (< $3) or critical (< $1). 2026-11-17: it ran out unnoticed."""
    from decimal import Decimal
    from accounting import balance as bal
    now = time.time()
    b = _balance(fleet, now)
    if b is None:
        return []
    k = bal.check_balance_alert(ledger, b, bal.burn_usd_per_hour(fleet.store, now), now,
                                low=Decimal(env.get("SONYA_VAST_BALANCE_LOW_USD", "3")),
                                critical=Decimal(env.get("SONYA_VAST_BALANCE_CRITICAL_USD", "1")))
    return [k] if k else []


_YT = {"t": 0.0}


def watch_youtube(ledger, env=os.environ, every: float = 300.0) -> list:  # pragma: no cover - S3 on the VPS
    """YouTube relays (scripts/yt_relay.py): heartbeat + hourly canary + account cookies -> alerts via the outbox."""
    now = time.time()
    if now - _YT["t"] < every or env.get("SONYA_YT_WATCH", "on") == "off":
        return []
    _YT["t"] = now
    if not env.get("S3_ACCESS_KEY_ID"):
        return []
    os.environ.setdefault("S3_BUCKET_NAME", env.get("MODELS_S3_BUCKET", ""))
    from accounting import yt_watch
    from scripts import yt_relay
    try:
        hbs = yt_relay.heartbeats(now)
    except Exception as exc:  # noqa: BLE001
        log.warning("yt_watch_error %s", type(exc).__name__)
        return []
    return yt_watch.check(ledger, hbs, now) + yt_watch.check_pool(ledger, _yt_pool(now), now)


_POOL = {"t": 0.0, "v": None}


def _yt_pool(now: float, every: float = 300.0):  # pragma: no cover - S3
    """Cookies pool summary of the Mac mini (tools/yt_cookies.py -> yt-relay/pool.json, no secrets); None if unknown."""
    if now - _POOL["t"] >= every:
        _POOL["t"] = now
        try:
            from scripts import yt_relay
            _POOL["v"] = yt_relay.get_json(yt_relay.prefix() + "pool.json")
        except Exception as exc:  # noqa: BLE001
            log.warning("yt_pool_error %s", type(exc).__name__)
    return _POOL["v"]


_YTW = {"t": 0.0}


def watch_yt_wait(ledger, env=os.environ, every: float = 30.0) -> list:  # pragma: no cover - Postgres + S3 on the VPS
    """Jobs parked in 'source_wait' (YouTube gave the video to no path): retried through the relays for up to 2 h,
    then failed politely with the attempt given back (scripts/yt_wait.py). Runs without the accountant too."""
    now = time.time()
    if now - _YTW["t"] < every or env.get("SONYA_YT_WAIT", "on") == "off":
        return []
    _YTW["t"] = now
    if not env.get("S3_ACCESS_KEY_ID") or not env.get("DATABASE_URL"):
        return []
    os.environ.setdefault("S3_BUCKET_NAME", env.get("MODELS_S3_BUCKET", ""))
    from scripts import yt_relay, yt_wait
    try:
        return yt_wait.tick(yt_wait.Db(env["DATABASE_URL"]), yt_relay, now, ledger=ledger, env=env, pool=_yt_pool(now))
    except Exception as exc:  # noqa: BLE001
        log.warning("yt_wait_error %s: %s", type(exc).__name__, str(exc)[:200])
        return []


_CAP = {"t": 0.0}


def watch_capacity(fleet: FleetManager, ledger, env=os.environ, every: float = 60.0) -> list:  # pragma: no cover - VPS
    """«Ждёт денег Vast»: capacity.json for the site API (notice to users) + admin alert (dispatcher/capacity.py)."""
    now = time.time()
    if now - _CAP["t"] < every or env.get("SONYA_CAPACITY_WATCH", "on") == "off":
        return []
    _CAP["t"] = now
    from decimal import Decimal
    from dispatcher import capacity
    try:
        snap = fleet.queue.snapshot(fleet.cfg.pool_modes, fleet.cfg.label_prefix)
        up = len(fleet.store.all(capacity.UP_STATES))
        ev = fleet.store.events_since(now - 1200, ("create_failed", "create_failed_next", "resume_failed"))
        cur = capacity.tick(STATE_DIR / "capacity.json", snap.pending, up, _balance(fleet, now), ev, now,
                            ledger=None if health_owns_alerts(env, now) else ledger,
                            threshold=Decimal(env.get("SONYA_VAST_NOFUNDS_USD", "0.30")))
    except Exception as exc:  # noqa: BLE001
        log.warning("capacity_error %s", type(exc).__name__)
        return []
    return [f"capacity {cur['state']}"] * (cur["state"] != "ok") + list(cur.get("alerts") or [])


def import_confirmed(fleet: FleetManager, ledger) -> None:  # pragma: no cover
    from accounting import vast_billing
    now = time.time()
    rows = fleet.vast.charges(int(now - 3 * 86400), int(now))
    vast_billing.import_charges(ledger, rows, now)


_GEMINI_LAST = {"t": 0.0}


def import_gemini(ledger, env=os.environ, every: float = 300.0) -> int:  # pragma: no cover - S3 on the VPS
    """Worker JSONs (scripts/shared/llm_usage.py) -> ledger.llm_usage, at most every 5 min (idempotent)."""
    now = time.time()
    if now - _GEMINI_LAST["t"] < every:
        return 0
    _GEMINI_LAST["t"] = now
    bucket = env.get("S3_BUCKET_NAME") or env.get("S3_BUCKET")
    if not bucket or not env.get("S3_ACCESS_KEY_ID"):
        return 0
    import boto3
    from accounting import gemini
    client = boto3.client("s3", endpoint_url=env.get("S3_ENDPOINT_URL") or None, region_name=env.get("S3_REGION") or None,
                          aws_access_key_id=env["S3_ACCESS_KEY_ID"], aws_secret_access_key=env.get("S3_SECRET_ACCESS_KEY"))
    n = gemini.import_from_s3(ledger, client, bucket, env.get("SONYA_LLM_USAGE_PREFIX") or gemini.DEFAULT_PREFIX, now)
    if n:
        log.info("gemini_usage_imported calls=%d", n)
    return n


def timeweb_balance(env=os.environ):  # pragma: no cover - network
    """Timeweb Cloud account balance (RUB) if a Cloud API token file is configured; the Gemini key cannot read it."""
    from accounting import gemini
    f = env.get("TIMEWEB_CLOUD_TOKEN_FILE", "/etc/sonya-fleet/timeweb_cloud_token")
    try:
        if not Path(f).exists() or Path(f).stat().st_mode & 0o077:
            return None
        return gemini.timeweb_balance_text(Path(f).read_text().strip())
    except Exception:  # noqa: BLE001
        return None


def status(state_dir: Path = STATE_DIR) -> int:
    from dispatcher.status import status_text
    db = state_dir / "fleet.db"
    if not db.exists():
        print(f"no fleet state yet ({db})")
        return 1
    print(status_text(FleetStore(str(db))))
    return 0


if __name__ == "__main__":  # pragma: no cover
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        sys.exit(status())
    sys.exit(run(once=(len(sys.argv) > 1 and sys.argv[1] == "once")))
