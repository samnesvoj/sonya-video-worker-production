"""
Read-only view of the production job queue. The fleet only ever READS jobs; it never claims, updates or
deletes them (claiming is the worker's job through /api/worker/claim). Two implementations:
  PostgresQueueView - read-only SELECTs against the existing generation_jobs table
  SqliteQueueView   - same SQL shape for tests / simulations
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Protocol

ACTIVE_STATUSES = ("claimed", "worker_started", "preflight_running", "downloading", "model_downloading", "mode_running",
                   "analyzing", "yolo", "scripting", "tts", "subtitles", "assembling", "uploading_result")
TERMINAL_STATUSES = ("completed", "failed", "cancelled")


@dataclass
class QueueSnapshot:
    pending: int = 0                                   # queued jobs of the pool modes (claimable now)
    active_by_worker: Dict[str, int] = field(default_factory=dict)   # non-terminal jobs held by each worker_id
    last_terminal_by_worker: Dict[str, float] = field(default_factory=dict)  # epoch of last completed/failed/cancelled
    active_unassigned: int = 0                         # active jobs with no worker_id (never stop under them)

    @property
    def active_total(self) -> int:
        return sum(self.active_by_worker.values()) + self.active_unassigned


class QueueView(Protocol):
    def snapshot(self, modes: Iterable[str], label_prefix: str) -> QueueSnapshot: ...


_Q_PENDING = "SELECT COUNT(*) FROM generation_jobs WHERE status='queued' AND mode = ANY(%s)"
_Q_ACTIVE = ("SELECT worker_id, COUNT(*) FROM generation_jobs WHERE status = ANY(%s) AND mode = ANY(%s) GROUP BY worker_id")
_Q_TERMINAL = ("SELECT worker_id, EXTRACT(EPOCH FROM MAX(updated_at)) FROM generation_jobs "
               "WHERE status = ANY(%s) AND mode = ANY(%s) AND worker_id LIKE %s GROUP BY worker_id")


class PostgresQueueView:
    """Uses a READ ONLY transaction; point it at a read-only DB role if one exists."""

    def __init__(self, dsn: str):
        self._dsn = dsn

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
            return snap
        finally:
            conn.close()


class SqliteQueueView:
    """Test double with the same semantics over a table shaped like generation_jobs."""

    def __init__(self, conn):
        self.c = conn
        conn.execute("CREATE TABLE IF NOT EXISTS generation_jobs (id TEXT PRIMARY KEY, mode TEXT, status TEXT, worker_id TEXT, updated_at REAL, completed_at REAL)")

    def add(self, job_id, mode="virality", status="queued", worker_id=None, ts=0.0):
        self.c.execute("INSERT OR REPLACE INTO generation_jobs VALUES (?,?,?,?,?,?)", (job_id, mode, status, worker_id, ts, ts if status in TERMINAL_STATUSES else None))

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
        return s
