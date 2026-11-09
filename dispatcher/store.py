"""
Fleet state + lifecycle journal in the dispatcher's OWN SQLite file (never the production PostgreSQL).
Single dispatcher process; every transition is a compare-and-swap on (state, generation) inside
BEGIN IMMEDIATE, so a repeated or concurrent tick cannot double-stop / double-start an instance.
The journal (fleet_events) is the source of truth for the cost accounting of the lifecycle.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS fleet_instances (
  id INTEGER PRIMARY KEY,                -- Vast instance id (contract id)
  label TEXT NOT NULL UNIQUE,            -- sonya-pool-xxxx == WORKER_ID of the worker inside
  state TEXT NOT NULL,                   -- creating|running|stop_pending|stopping|stopped|starting|destroying|destroyed|lost
  generation INTEGER NOT NULL DEFAULT 0,
  gpu_name TEXT, dph_base TEXT, storage_gb_month TEXT, inet_up_tb TEXT, inet_down_tb TEXT,
  disk_gb TEXT, release_sha TEXT, machine_id INTEGER,
  created_at REAL, running_since REAL, last_work_at REAL,
  stop_requested_at REAL, stopped_at REAL, resume_requested_at REAL, destroyed_at REAL,
  resume_failures INTEGER NOT NULL DEFAULT 0, note TEXT,
  role TEXT NOT NULL DEFAULT 'burst',    -- warm: kept STOPPED between jobs | burst: created for the load, destroyed when idle
  slots INTEGER,                         -- power servers: parallel videos counted (NULL = jobs_per_instance, default class)
  num_gpus INTEGER NOT NULL DEFAULT 1,
  gpu_class TEXT NOT NULL DEFAULT 'standard'   -- standard: RTX 3060 class | power: multi-GPU rush server (never the warm)
);
CREATE TABLE IF NOT EXISTS fleet_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL NOT NULL, instance_id INTEGER, event TEXT NOT NULL, detail TEXT
);
CREATE INDEX IF NOT EXISTS ix_events_inst ON fleet_events(instance_id, ts);
CREATE TABLE IF NOT EXISTS fleet_creates (ts REAL NOT NULL);
CREATE TABLE IF NOT EXISTS bad_machines (machine_id INTEGER PRIMARY KEY, until REAL NOT NULL, reason TEXT);
CREATE TABLE IF NOT EXISTS good_machines (machine_id INTEGER PRIMARY KEY, last_ok REAL NOT NULL);  -- booted our image there
CREATE TABLE IF NOT EXISTS fleet_spares (   -- the latest ranked spare offers per class (warm|burst|power|reserve), for status
  cls TEXT NOT NULL, rank INTEGER NOT NULL, ts REAL NOT NULL, detail TEXT NOT NULL, PRIMARY KEY (cls, rank)
);
"""

STATES_BILLING_GPU = ("running", "stop_pending")          # GPU is attached and billed per second
STATES_EXISTING = ("creating", "running", "stop_pending", "stopping", "stopped", "starting", "destroying")


@dataclass
class Inst:
    id: int
    label: str
    state: str
    generation: int
    gpu_name: str
    dph_base: Decimal
    storage_gb_month: Decimal
    disk_gb: Decimal
    release_sha: str
    machine_id: Optional[int]
    created_at: Optional[float]
    running_since: Optional[float]
    last_work_at: Optional[float]
    stop_requested_at: Optional[float]
    stopped_at: Optional[float]
    resume_requested_at: Optional[float]
    resume_failures: int
    role: str = "burst"
    note: str = ""                       # "warm_eligible": default-class host with cheap storage (may become the warm)
    slots: int = 0                       # parallel videos counted for it; 0 = cfg.jobs_per_instance (rows from before multi-GPU)
    num_gpus: int = 1
    gpu_class: str = "standard"          # "power": multi-GPU rush server, always burst


def _dec(v: Any) -> Decimal:
    return Decimal(str(v)) if v not in (None, "") else Decimal("0")


class FleetStore:
    def __init__(self, path: str = ":memory:"):
        self._c = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        self._c.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._c.executescript(SCHEMA)
        cols = {r[1] for r in self._c.execute("PRAGMA table_info(fleet_instances)")}
        if "role" not in cols:              # a fleet.db from before the warm/burst split: every old row is burst
            self._c.execute("ALTER TABLE fleet_instances ADD COLUMN role TEXT NOT NULL DEFAULT 'burst'")
        for col, ddl in (("slots", "INTEGER"), ("num_gpus", "INTEGER NOT NULL DEFAULT 1"),   # before multi-GPU: 1 card, standard
                         ("gpu_class", "TEXT NOT NULL DEFAULT 'standard'")):
            if col not in cols:
                self._c.execute(f"ALTER TABLE fleet_instances ADD COLUMN {col} {ddl}")

    # -- helpers
    def _row_to_inst(self, r: sqlite3.Row) -> Inst:
        return Inst(id=r["id"], label=r["label"], state=r["state"], generation=r["generation"], gpu_name=r["gpu_name"] or "",
                    dph_base=_dec(r["dph_base"]), storage_gb_month=_dec(r["storage_gb_month"]), disk_gb=_dec(r["disk_gb"]),
                    release_sha=r["release_sha"] or "", machine_id=r["machine_id"], created_at=r["created_at"],
                    running_since=r["running_since"], last_work_at=r["last_work_at"], stop_requested_at=r["stop_requested_at"],
                    stopped_at=r["stopped_at"], resume_requested_at=r["resume_requested_at"], resume_failures=r["resume_failures"],
                    role=r["role"] or "burst", note=r["note"] or "", slots=r["slots"] or 0, num_gpus=r["num_gpus"] or 1,
                    gpu_class=r["gpu_class"] or "standard")

    def event(self, ts: float, instance_id: Optional[int], event: str, **detail: Any) -> None:
        with self._lock:
            self._c.execute("INSERT INTO fleet_events(ts, instance_id, event, detail) VALUES (?,?,?,?)",
                            (ts, instance_id, event, json.dumps(detail, default=str, sort_keys=True)))

    def events(self, instance_id: Optional[int] = None) -> List[dict]:
        q, a = "SELECT * FROM fleet_events", ()
        if instance_id is not None:
            q, a = q + " WHERE instance_id=?", (instance_id,)
        with self._lock:
            return [dict(r) | {"detail": json.loads(r["detail"] or "{}")} for r in self._c.execute(q + " ORDER BY ts, id", a)]

    # -- reads
    def all(self, states: Optional[tuple] = None) -> List[Inst]:
        with self._lock:
            rows = self._c.execute("SELECT * FROM fleet_instances").fetchall()
        out = [self._row_to_inst(r) for r in rows]
        return [i for i in out if states is None or i.state in states]

    def get(self, instance_id: int) -> Optional[Inst]:
        with self._lock:
            r = self._c.execute("SELECT * FROM fleet_instances WHERE id=?", (instance_id,)).fetchone()
        return self._row_to_inst(r) if r else None

    # -- writes
    def add(self, ts: float, *, id: int, label: str, state: str, **cols: Any) -> None:
        keys = ["id", "label", "state", "created_at"] + list(cols)
        vals = [id, label, state, ts] + [str(v) if isinstance(v, Decimal) else v for v in cols.values()]
        with self._lock:
            self._c.execute(f"INSERT OR IGNORE INTO fleet_instances({','.join(keys)}) VALUES ({','.join('?' * len(keys))})", vals)

    def cas(self, instance_id: int, expect_state: str, expect_gen: int, new_state: str, **cols: Any) -> bool:
        """Atomic transition; returns False if somebody else moved the instance first."""
        sets = ["state=?", "generation=generation+1"] + [f"{k}=?" for k in cols]
        vals = [new_state] + list(cols.values()) + [instance_id, expect_state, expect_gen]
        with self._lock:
            cur = self._c.execute(f"UPDATE fleet_instances SET {','.join(sets)} WHERE id=? AND state=? AND generation=?", vals)
            return cur.rowcount == 1

    def touch(self, instance_id: int, **cols: Any) -> None:
        if not cols:
            return
        with self._lock:
            self._c.execute(f"UPDATE fleet_instances SET {','.join(f'{k}=?' for k in cols)} WHERE id=?", [*cols.values(), instance_id])

    def record_create(self, ts: float) -> None:
        with self._lock:
            self._c.execute("INSERT INTO fleet_creates(ts) VALUES (?)", (ts,))

    def creates_since(self, ts: float) -> int:
        with self._lock:
            return self._c.execute("SELECT COUNT(*) FROM fleet_creates WHERE ts>=?", (ts,)).fetchone()[0]

    def ban_machine(self, machine_id: int, until: float, reason: str) -> None:
        with self._lock:
            self._c.execute("INSERT OR REPLACE INTO bad_machines VALUES (?,?,?)", (machine_id, until, reason))

    def mark_good_machine(self, machine_id: int, ts: float) -> None:
        with self._lock:
            self._c.execute("INSERT OR REPLACE INTO good_machines VALUES (?,?)", (machine_id, ts))

    def good_machines(self, since: float = 0.0) -> set:
        with self._lock:
            return {r[0] for r in self._c.execute("SELECT machine_id FROM good_machines WHERE last_ok>=?", (since,))}

    def save_spares(self, cls: str, offers: List[dict], ts: float) -> None:
        """Replace the spares list of one class (offers = launcher.offer_brief dicts, best first). An empty list is kept
        as one {"offer": None} row, so the status shows "no acceptable offer" with its time."""
        offers = offers or [{"offer": None}]
        with self._lock:
            self._c.execute("BEGIN IMMEDIATE")
            try:
                self._c.execute("DELETE FROM fleet_spares WHERE cls=?", (cls,))
                self._c.executemany("INSERT INTO fleet_spares VALUES (?,?,?,?)",
                                    [(cls, k, ts, json.dumps(o, default=str, sort_keys=True)) for k, o in enumerate(offers)])
                self._c.execute("COMMIT")
            except Exception:
                self._c.execute("ROLLBACK")
                raise

    def spares(self) -> Dict[str, List[dict]]:
        """{class: [offer brief + 'ts', best first]}."""
        out: Dict[str, List[dict]] = {}
        with self._lock:
            for r in self._c.execute("SELECT cls, ts, detail FROM fleet_spares ORDER BY cls, rank"):
                out.setdefault(r["cls"], []).append(json.loads(r["detail"]) | {"ts": r["ts"]})
        return out

    def banned_machines(self, now: float) -> set:
        with self._lock:
            return {r[0] for r in self._c.execute("SELECT machine_id FROM bad_machines WHERE until>?", (now,))}
