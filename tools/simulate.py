#!/usr/bin/env python3
"""
SIMULATION with SYNTHETIC data (fake Vast, scripted queue) - never touches Vast, GitHub, Telegram or the production DB.
Shows the 60 s idle-stop behaviour end to end and prints example daily/monthly bot reports built from the simulated journal.
Usage: python tools/simulate.py
"""
import sys
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from accounting import reports, usage
from accounting.ledger import Ledger
from accounting.money import MSK
from accounting.outbox import ReportService
from dispatcher.config import FleetConfig
from dispatcher.fleet import FleetManager
from dispatcher.queue import SqliteQueueView
from dispatcher.store import FleetStore
from tests.fleet.fakes import ENV, FakeVast, IMAGE, SHA
import sqlite3


def msk(y, m, d, hh=0, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=MSK).timestamp()


class Rates:
    def get(self, day):
        from accounting.cbr import Rate
        return Rate(day, Decimal("82.1386"), day.strftime("%d.%m.%Y"), False)


class Jobs:
    def __init__(self, q): self.q = q
    def completed_count(self, lo, hi):
        return self.q.c.execute("SELECT COUNT(*) FROM generation_jobs WHERE status='completed' AND completed_at>=? AND completed_at<?", (lo, hi)).fetchone()[0]


def main():
    clock = [msk(2026, 11, 1)]
    now = lambda: clock[0]
    vast, conn = FakeVast(), sqlite3.connect(":memory:", check_same_thread=False)
    vast.offers = [{"id": 1, "gpu_name": "RTX 3090", "dph_total": 0.2014, "dph_base": 0.2, "machine_id": 7, "storage_cost": 0.2,
                    "internet_up_cost_per_tb": 4, "internet_down_cost_per_tb": 2.67}]
    q = SqliteQueueView(conn)
    store = FleetStore()
    cfg = FleetConfig(enabled=True, dry_run=False, image=IMAGE, release_sha=SHA, stopped_retention_seconds=1800)
    mgr = FleetManager(cfg, store, vast, q, lambda: "ghs_SIMULATED", ENV, now)
    # (day, hour, minute, minutes of GPU work) - synthetic arrivals
    arrivals = []
    for day, spec in {1: [(9, 0, 8), (9, 5, 8), (9, 10, 8), (15, 30, 8)], 2: [(11, 0, 12), (11, 2, 12)], 3: [(10, 0, 8), (13, 0, 8), (16, 0, 8), (19, 0, 8), (22, 0, 8)],
                      4: [], 5: [(12, 0, 10), (12, 1, 10), (12, 2, 10), (12, 3, 10), (12, 4, 10), (12, 5, 10)]}.items():
        arrivals += [(msk(2026, 11, day, h, m), dur) for h, m, dur in spec]
    arrivals.sort()
    jobs, pending, n, log = {}, list(arrivals), 0, []
    end = msk(2026, 11, 6)
    last_actions = 0
    while clock[0] < end:
        t = clock[0]
        while pending and pending[0][0] <= t:
            n += 1
            q.add(f"j{n}", ts=t); jobs[f"j{n}"] = {"dur": pending[0][1] * 60, "state": "queued", "t": t}; pending.pop(0)
        for iid, d in list(vast.inst.items()):                 # worker emulation
            if d["status"] != "running":
                continue
            booted = store.get(iid) and store.get(iid).running_since and t - store.get(iid).running_since >= 90
            run = [j for j, v in jobs.items() if v["state"] == "running" and v["w"] == d["label"]]
            if run:
                j = run[0]
                if t - jobs[j]["t0"] >= jobs[j]["dur"]:
                    q.set(j, "uploading_result", d["label"], t); jobs[j]["state"] = "uploading"; jobs[j]["t0"] = t
            else:
                up = [j for j, v in jobs.items() if v["state"] == "uploading" and v["w"] == d["label"]]
                if up and t - jobs[up[0]]["t0"] >= 20:
                    q.set(up[0], "completed", d["label"], t); jobs[up[0]]["state"] = "done"
                elif booted:
                    nxt = next((j for j, v in jobs.items() if v["state"] == "queued"), None)
                    if nxt:
                        q.set(nxt, "mode_running", d["label"], t); jobs[nxt].update(state="running", w=d["label"], t0=t)
        acts = mgr.tick()
        for a in acts:
            log.append(f"{datetime.fromtimestamp(t, MSK):%d.%m %H:%M:%S} {a}")
        clock[0] += 10
    print("=== SIMULATION (synthetic data) - fleet actions ===")
    print("\n".join(log))
    led = Ledger()
    svc = ReportService(led, store, Rates(), Jobs(conn_q := q), lambda d, now: {"gpu": Decimal(0), "disk": Decimal(0), "other": Decimal(0)}, date(2026, 11, 1))
    # make jobs.completed_count use the sim DB
    svc.run_due(msk(2026, 11, 6, 0, 6))
    print("\n=== Example DAILY report (day 5) ===")
    print(led.c.execute("SELECT text FROM report_outbox WHERE report_key='vast-daily:2026-11-05'").fetchone()[0])
    reports_m = reports.build_monthly(led, 2026, 11, Rates())
    print("\n=== Example MONTHLY report (synthetic: only days 1-5 of the month have activity) ===")
    print(reports_m["text"])


if __name__ == "__main__":
    main()
