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

import logging
import os
import sys
import time
from datetime import date
from pathlib import Path

from dispatcher.config import FleetConfig
from dispatcher.fleet import FleetManager
from dispatcher.queue import PostgresQueueView
from dispatcher.store import FleetStore
from dispatcher.token_provider import github_app_token_provider
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
                        PostgresQueueView(env["DATABASE_URL"]), github_app_token_provider(cfg.release_repo.split("/")[-1], env), env)


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
            cur.execute("SELECT COUNT(*) FROM generation_jobs WHERE status='completed' AND mode = ANY(%s) AND worker_id LIKE %s "
                        "AND completed_at >= to_timestamp(%s) AND completed_at < to_timestamp(%s)",
                        (list(fleet.cfg.pool_modes), fleet.cfg.label_prefix + "%", lo, hi))
            return int(cur.fetchone()[0])
        finally:
            conn.close()

    def utc_cost(day: date, now: float) -> dict:
        lo = datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp()
        r = cost_between(fleet.store, lo, lo + 86400, now)
        return {k: r[k] for k in ("gpu", "disk", "other")}

    first = date.fromisoformat(env.get("SONYA_ACCOUNTING_FIRST_DAY", date.today().isoformat()))
    return ReportService(ledger, fleet.store, CbrRates(ledger), SqlJobsSource(counter), utc_cost, first), ledger


def run(once: bool = False) -> int:  # pragma: no cover - exercised on the VPS
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    fleet = build_fleet()
    svc = ledger = None
    if os.environ.get("SONYA_ACCOUNTING_ENABLED", "false").lower() == "true":
        svc, ledger = build_accounting(fleet)
    while True:
        try:
            for a in fleet.tick():
                log.info("fleet_action %s", a)
            if svc:
                import_confirmed(fleet, ledger)
                for k in svc.run_due(time.time()):
                    log.info("report_built %s", k)
        except Exception as exc:  # noqa: BLE001
            log.error("tick_error %s", type(exc).__name__)
        if once:
            return 0
        time.sleep(fleet.cfg.tick_seconds)


def import_confirmed(fleet: FleetManager, ledger) -> None:  # pragma: no cover
    from accounting import vast_billing
    now = time.time()
    rows = fleet.vast.charges(int(now - 3 * 86400), int(now))
    vast_billing.import_charges(ledger, rows, now)


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
