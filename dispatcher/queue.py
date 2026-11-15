"""
Read-only view of the production job queue. The fleet only ever READS jobs; it never claims, updates or
deletes them (claiming is the worker's job through /api/worker/claim). Two implementations:
  PostgresQueueView - read-only SELECTs against the existing generation_jobs table
  SqliteQueueView   - same SQL shape for tests / simulations
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Protocol, Set

ACTIVE_STATUSES = ("claimed", "worker_started", "preflight_running", "downloading", "model_downloading", "mode_running",
                   "analyzing", "yolo", "scripting", "tts", "subtitles", "assembling", "uploading_result")
TERMINAL_STATUSES = ("completed", "failed", "cancelled")


@dataclass
class QueueSnapshot:
    pending: int = 0                                   # queued jobs of the pool modes (claimable now)
    active_by_worker: Dict[str, int] = field(default_factory=dict)   # non-terminal jobs held by each worker_id
    last_terminal_by_worker: Dict[str, float] = field(default_factory=dict)  # epoch of last completed/failed/cancelled
    active_unassigned: int = 0                         # active jobs with no worker_id (never stop under them)
    bot_check_workers: Set[str] = field(default_factory=set)  # workers whose IP YouTube refused ("not a bot") just now
    # Dedicated servers (CEO 2026-11-16, SONYA_DEDICATED=on; same rule as the API claim, scripts/prod_job_store.py):
    dedicated_users: int = 0                           # paying users / admins with queued or running jobs: 1 whole server each
    dedicated_jobs: int = 0                            # their jobs (queued + running)
    dedicated_workers: Set[str] = field(default_factory=set)   # servers that hold a dedicated job right now

    @property
    def active_total(self) -> int:
        return sum(self.active_by_worker.values()) + self.active_unassigned


BOT_CHECK_CODE = "SOURCE_BOT_CHECK"


# Users whose jobs get a server of their own: identical to DEDICATED_USERS_SQL in the API's scripts/prod_job_store.py
DEDICATED_USERS_SQL = """
    SELECT u.id::text AS uid FROM users u
     WHERE (%(ded_admin)s AND lower(coalesce(u.plan_type, '')) = 'admin')
        OR (%(ded_legacy)s AND u.plan_type = 'pro' AND u.plan_status = 'active' AND u.plan_active_until > now())
    UNION
    SELECT s.user_id::text FROM user_subscriptions s
     WHERE s.plan_id = ANY(%(ded_plans)s) AND now() >= s.period_start AND now() < s.period_end
"""
_Q_DEDICATED = ("WITH ded AS (" + DEDICATED_USERS_SQL + ") "
                "SELECT j.user_id::text, j.status, j.worker_id FROM generation_jobs j "
                "WHERE j.mode = ANY(%(modes)s) AND j.mode = ANY(%(ded_modes)s) AND (j.status = 'queued' OR j.status = ANY(%(active)s)) "
                "AND j.user_id::text IN (SELECT uid FROM ded)")


def dedicated_params(env=None) -> dict:
    import os
    e = os.environ if env is None else env
    on = lambda k, d: e.get(k, d).strip().lower() in ("1", "on", "true", "yes")
    csv = lambda k, d: [x.strip() for x in e.get(k, d).split(",") if x.strip()]
    return {"ded_admin": on("SONYA_DEDICATED_ADMINS", "on"), "ded_legacy": on("SONYA_DEDICATED_LEGACY_PRO", "on"),
            "ded_plans": csv("SONYA_DEDICATED_PLANS", "cut_pro,cut_studio,streamer_start"),
            "ded_modes": csv("SONYA_DEDICATED_MODES", "virality,beep,educational,streamer")}


def fold_dedicated(snap: "QueueSnapshot", rows) -> None:
    """rows: (user_id, status, worker_id) of dedicated jobs -> snap.dedicated_*."""
    users, n = set(), 0
    for uid, st, wid in rows:
        n += 1
        users.add(uid)
        if st != "queued" and wid:
            snap.dedicated_workers.add(wid)
    snap.dedicated_users, snap.dedicated_jobs = len(users), n


class QueueView(Protocol):
    def snapshot(self, modes: Iterable[str], label_prefix: str) -> QueueSnapshot: ...


_Q_PENDING = "SELECT COUNT(*) FROM generation_jobs WHERE status='queued' AND mode = ANY(%s)"
_Q_ACTIVE = ("SELECT worker_id, COUNT(*) FROM generation_jobs WHERE status = ANY(%s) AND mode = ANY(%s) GROUP BY worker_id")
_Q_TERMINAL = ("SELECT worker_id, EXTRACT(EPOCH FROM MAX(updated_at)) FROM generation_jobs "
               "WHERE status = ANY(%s) AND mode = ANY(%s) AND worker_id LIKE %s GROUP BY worker_id")
# Requeued after a YouTube bot check: worker_id still names the refused host until another worker claims the job.
_Q_BOTCHECK = ("SELECT DISTINCT worker_id FROM generation_jobs WHERE status='queued' AND last_error=%s "
               "AND worker_id LIKE %s AND updated_at > now() - interval '1 hour'")


class PostgresQueueView:
    """Uses a READ ONLY transaction; point it at a read-only DB role if one exists."""

    def __init__(self, dsn: str, dedicated: bool = False):
        self._dsn = dsn
        self._dedicated = dedicated

    def snapshot(self, modes, label_prefix):
        import psycopg2  # lazy: only on the VPS

        modes = list(modes)
        conn = psycopg2.connect(self._dsn)
        try:
            conn.set_session(readonly=True, autocommit=True)
            cur = conn.cursor()
            snap = QueueSnapshot()
            cur.execute(_Q_PENDING, (modes,))
            snap.pending = int(cur.fetchone()[0])
            cur.execute(_Q_ACTIVE, (list(ACTIVE_STATUSES), modes))
            for wid, n in cur.fetchall():
                if wid:
                    snap.active_by_worker[wid] = int(n)
                else:
                    snap.active_unassigned += int(n)
            cur.execute(_Q_TERMINAL, (list(TERMINAL_STATUSES), modes, label_prefix + "%"))
            snap.last_terminal_by_worker = {w: float(t) for w, t in cur.fetchall() if w}
            cur.execute(_Q_BOTCHECK, (BOT_CHECK_CODE, label_prefix + "%"))
            snap.bot_check_workers = {w for (w,) in cur.fetchall() if w}
            if self._dedicated:
                cur.execute(_Q_DEDICATED, dict(dedicated_params(), modes=modes, active=list(ACTIVE_STATUSES)))
                fold_dedicated(snap, cur.fetchall())
            return snap
        finally:
            conn.close()


class SqliteQueueView:
    """Test double with the same semantics over a table shaped like generation_jobs."""

    def __init__(self, conn):
        self.c = conn
        self.dedicated_user_ids: Set[str] = set()     # tests: users with a server of their own
        self._users: Dict[str, str] = {}
        conn.execute("CREATE TABLE IF NOT EXISTS generation_jobs (id TEXT PRIMARY KEY, mode TEXT, status TEXT, worker_id TEXT, updated_at REAL, completed_at REAL, last_error TEXT)")

    def add(self, job_id, mode="virality", status="queued", worker_id=None, ts=0.0, last_error=None, user_id=None):
        self.c.execute("INSERT OR REPLACE INTO generation_jobs VALUES (?,?,?,?,?,?,?)",
                       (job_id, mode, status, worker_id, ts, ts if status in TERMINAL_STATUSES else None, last_error))
        if user_id is not None:
            self._users[job_id] = user_id

    def set(self, job_id, status, worker_id=None, ts=0.0):
        self.c.execute("UPDATE generation_jobs SET status=?, worker_id=COALESCE(?, worker_id), updated_at=?, completed_at=? WHERE id=?",
                       (status, worker_id, ts, ts if status in TERMINAL_STATUSES else None, job_id))

    def snapshot(self, modes, label_prefix):
        modes = list(modes)
        mq = ",".join("?" * len(modes))
        s = QueueSnapshot()
        s.pending = self.c.execute(f"SELECT COUNT(*) FROM generation_jobs WHERE status='queued' AND mode IN ({mq})", modes).fetchone()[0]
        aq = ",".join("?" * len(ACTIVE_STATUSES))
        for wid, n in self.c.execute(f"SELECT worker_id, COUNT(*) FROM generation_jobs WHERE status IN ({aq}) AND mode IN ({mq}) GROUP BY worker_id", [*ACTIVE_STATUSES, *modes]):
            if wid:
                s.active_by_worker[wid] = n
            else:
                s.active_unassigned += n
        tq = ",".join("?" * len(TERMINAL_STATUSES))
        s.last_terminal_by_worker = {w: t for w, t in self.c.execute(
            f"SELECT worker_id, MAX(updated_at) FROM generation_jobs WHERE status IN ({tq}) AND mode IN ({mq}) AND worker_id LIKE ? GROUP BY worker_id",
            [*TERMINAL_STATUSES, *modes, label_prefix + "%"]) if w}
        s.bot_check_workers = {w for (w,) in self.c.execute(
            "SELECT DISTINCT worker_id FROM generation_jobs WHERE status='queued' AND last_error=? AND worker_id LIKE ?",
            (BOT_CHECK_CODE, label_prefix + "%")) if w}
        if self.dedicated_user_ids:
            rows = [(self._users.get(jid), st, wid) for jid, st, wid in self.c.execute(
                f"SELECT id, status, worker_id FROM generation_jobs WHERE mode IN ({mq}) AND (status='queued' OR status IN ({aq}))",
                [*modes, *ACTIVE_STATUSES])]
            fold_dedicated(s, [r for r in rows if r[0] in self.dedicated_user_ids])
        return s
