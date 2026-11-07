"""
PHASE A heartbeat (E): a long-running job that keeps calling
touch_job_heartbeat() must never be mistaken for stuck by
requeue_stale_jobs(), while a job that never heartbeats (every mode before
this change, and any job whose worker genuinely died) keeps the original
claimed_at-only behavior exactly as before.

No real Postgres — same approach as tests/test_job_store_claim.py: a
minimal fake cursor that evaluates the actual predicate
(COALESCE(heartbeat_at, claimed_at) < now - stale_minutes, status IN (...),
retry_count < max_retries) against in-memory rows, so this fails if the
liveness check regresses to claimed_at-only (or, in the other direction,
if it stops honoring claimed_at at all for a job that never heartbeats).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from scripts import prod_job_store


class _FakeRequeueCursor:
    def __init__(self, rows: dict) -> None:
        self._rows = rows
        self._rowcount = 0

    def execute(self, sql: str, params: tuple) -> None:
        sql_norm = " ".join(sql.split())
        assert sql_norm.startswith("UPDATE generation_jobs")
        assert "COALESCE(heartbeat_at, claimed_at)" in sql_norm, (
            "requeue_stale_jobs() must key staleness off "
            "COALESCE(heartbeat_at, claimed_at), not claimed_at alone"
        )

        # params = (new_status, updated_at, *active_statuses, stale_minutes)
        new_status = params[0]
        updated_at = params[1]
        stale_minutes = params[-1]
        active_statuses = set(params[2:-1])

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(minutes=stale_minutes)

        self._rowcount = 0
        for row in self._rows.values():
            if row["status"] not in active_statuses:
                continue
            if row["retry_count"] >= row["max_retries"]:
                continue
            effective = row.get("heartbeat_at") or row.get("claimed_at")
            if effective is not None and effective < cutoff:
                row["status"] = new_status
                row["worker_id"] = None
                row["claimed_at"] = None
                row["heartbeat_at"] = None
                row["updated_at"] = updated_at
                self._rowcount += 1

    @property
    def rowcount(self) -> int:
        return self._rowcount

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeHeartbeatCursor:
    def __init__(self, rows: dict) -> None:
        self._rows = rows
        self.executed = []

    def execute(self, sql: str, params: tuple) -> None:
        sql_norm = " ".join(sql.split())
        self.executed.append((sql_norm, params))
        assert sql_norm.startswith("UPDATE generation_jobs")
        assert "SET heartbeat_at=%s" in sql_norm
        assert "status" not in sql_norm.split("WHERE")[0].split("SET")[1], (
            "touch_job_heartbeat() must only ever write heartbeat_at, "
            "never status"
        )
        heartbeat_at, job_id = params
        if job_id in self._rows:
            self._rows[job_id]["heartbeat_at"] = heartbeat_at

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeConn:
    def __init__(self, rows: dict, cursor_cls) -> None:
        self._rows = rows
        self._cursor_cls = cursor_cls
        self.last_cursor = None

    def cursor(self):
        self.last_cursor = self._cursor_cls(self._rows)
        return self.last_cursor

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def close(self) -> None:
        pass


def _job_row(status: str, *, claimed_at, heartbeat_at=None, retry_count=0, max_retries=3) -> dict:
    return {
        "id": "job-1",
        "status": status,
        "worker_id": "worker-1",
        "claimed_at": claimed_at,
        "heartbeat_at": heartbeat_at,
        "retry_count": retry_count,
        "max_retries": max_retries,
    }


def test_job_with_recent_heartbeat_is_not_requeued(monkeypatch):
    """E (protected case): claimed 40 minutes ago (well past the 30-minute
    default), but heartbeating every couple of minutes -- must survive."""
    now = datetime.now(timezone.utc)
    rows = {
        "job-1": _job_row(
            "mode_running",
            claimed_at=now - timedelta(minutes=40),
            heartbeat_at=now - timedelta(minutes=2),
        ),
    }
    monkeypatch.setattr(prod_job_store, "_get_conn", lambda: _FakeConn(rows, _FakeRequeueCursor))

    requeued = prod_job_store.requeue_stale_jobs(stale_minutes=30)

    assert requeued == 0
    assert rows["job-1"]["status"] == "mode_running"  # untouched


def test_job_without_heartbeat_is_still_requeued_when_stale(monkeypatch):
    """E (regression guard): a job that never heartbeats -- every mode
    today except streamer's long phase, and any job whose worker actually
    died -- must still be requeued via the claimed_at fallback, exactly as
    before this change."""
    now = datetime.now(timezone.utc)
    rows = {
        "job-1": _job_row(
            "mode_running",
            claimed_at=now - timedelta(minutes=40),
            heartbeat_at=None,
        ),
    }
    monkeypatch.setattr(prod_job_store, "_get_conn", lambda: _FakeConn(rows, _FakeRequeueCursor))

    requeued = prod_job_store.requeue_stale_jobs(stale_minutes=30)

    assert requeued == 1
    assert rows["job-1"]["status"] == "queued"
    assert rows["job-1"]["claimed_at"] is None
    assert rows["job-1"]["heartbeat_at"] is None


def test_job_with_stale_heartbeat_is_requeued(monkeypatch):
    """A heartbeat that itself stopped updating (worker actually died mid-
    run) must not protect the job forever -- once IT is older than the
    stale window, the job is still fair game for requeue."""
    now = datetime.now(timezone.utc)
    rows = {
        "job-1": _job_row(
            "mode_running",
            claimed_at=now - timedelta(hours=5),        # claimed long ago
            heartbeat_at=now - timedelta(minutes=45),    # but heartbeat also died 45m ago
        ),
    }
    monkeypatch.setattr(prod_job_store, "_get_conn", lambda: _FakeConn(rows, _FakeRequeueCursor))

    requeued = prod_job_store.requeue_stale_jobs(stale_minutes=30)

    assert requeued == 1
    assert rows["job-1"]["status"] == "queued"


def test_job_at_max_retries_is_never_requeued(monkeypatch):
    """Unrelated existing guard must still hold after the query change."""
    now = datetime.now(timezone.utc)
    rows = {
        "job-1": _job_row(
            "mode_running",
            claimed_at=now - timedelta(minutes=40),
            retry_count=3, max_retries=3,
        ),
    }
    monkeypatch.setattr(prod_job_store, "_get_conn", lambda: _FakeConn(rows, _FakeRequeueCursor))

    requeued = prod_job_store.requeue_stale_jobs(stale_minutes=30)

    assert requeued == 0


def test_touch_job_heartbeat_only_writes_heartbeat_column(monkeypatch):
    """Heartbeat must never change status or imply a status transition."""
    rows = {"job-1": {"id": "job-1", "status": "mode_running", "heartbeat_at": None}}
    conn = _FakeConn(rows, _FakeHeartbeatCursor)
    monkeypatch.setattr(prod_job_store, "_get_conn", lambda: conn)

    prod_job_store.touch_job_heartbeat("job-1")

    assert rows["job-1"]["heartbeat_at"] is not None
    assert rows["job-1"]["status"] == "mode_running"  # unchanged
