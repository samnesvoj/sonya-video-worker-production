"""
Calculated costs from the fleet lifecycle journal (dispatcher/store.py) x the price snapshot taken when the instance
was created. These are CALCULATED numbers, not Vast charges. Billing model (Vast docs, docs.vast.ai billing + storage):
  GPU      per second while the instance is running (not while 'Loading', not while stopped)
  disk     per GB-hour for every second the instance EXISTS (running or stopped): price_gb_month / 720
  traffic  per TB, any state (not measurable here -> estimate only)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Callable, Dict, List, Optional, Tuple

from accounting.money import HOURS_PER_MONTH, D, msk_day_bounds

GPU_START = ("running", "adopted_running", "external_start")
GPU_END = ("stopped", "unexpected_stop", "destroy_requested", "destroyed", "lost", "create_timeout")
DAY = Decimal(24)
EST_USD_PER_TB = Decimal("2.7")   # ASSUMPTION: typical cheap-host rate seen in offers (2.7 USD/TB)


@dataclass
class EstimateParams:
    model_gb_per_cold_start: Decimal = Decimal("5")      # ASSUMPTION - real size unknown until models.lock.json is filled
    video_in_gb: Decimal = Decimal("0.5")                # ASSUMPTION
    video_out_gb: Decimal = Decimal("0.1")               # ASSUMPTION
    stopped_storage_multiplier: Decimal = Decimal("1")   # Vast docs: stopped storage "may be higher" - factor unknown


def _hours(a: float, b: float, lo: float, hi: float) -> Decimal:
    s, e = max(a, lo), min(b, hi)
    return D(max(0.0, e - s)) / Decimal(3600)


def instance_intervals(store, inst, now: float) -> Tuple[List[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """-> (gpu intervals, existence interval) in epoch seconds; open intervals end at `now`."""
    gpu: List[Tuple[float, float]] = []
    open_start: Optional[float] = None
    exist_start = inst.created_at
    exist_end: Optional[float] = None
    for e in store.events(inst.id):
        ev, ts = e["event"], e["ts"]
        if ev == "adopted":
            exist_start = exist_start or ts
            if e["detail"].get("state") == "running":
                open_start = ts
        elif ev in ("running", "external_start") and open_start is None:
            open_start = ts
        elif ev in ("stopped", "unexpected_stop", "destroy_requested", "destroyed", "lost") and open_start is not None:
            gpu.append((open_start, ts))
            open_start = None
        if ev in ("destroy_requested", "destroyed", "lost") and exist_end is None:
            exist_end = ts
    if open_start is not None:
        gpu.append((open_start, now))
    return gpu, ((exist_start, exist_end if exist_end is not None else now) if exist_start else None)


def cost_between(store, lo: float, hi: float, now: float, est: EstimateParams = EstimateParams()) -> dict:
    """Pure calculation for [lo, hi): no writes. Used for MSK days (reports) and UTC days (reconciliation)."""
    gpu_cost = disk_cost = hours_total = Decimal(0)
    cold_starts = 0
    per_inst: Dict[int, dict] = {}
    for inst in store.all():
        gpu, exist = instance_intervals(store, inst, now)
        g_hours = sum((_hours(a, b, lo, hi) for a, b in gpu), Decimal(0))
        e_hours = _hours(exist[0], exist[1], lo, hi) if exist else Decimal(0)
        stopped_hours = max(Decimal(0), e_hours - g_hours)
        per_gb_h = inst.storage_gb_month / HOURS_PER_MONTH
        g = g_hours * inst.dph_base
        d = inst.disk_gb * per_gb_h * (g_hours + stopped_hours * est.stopped_storage_multiplier)
        if g_hours or e_hours:
            per_inst[inst.id] = {"gpu_hours": str(g_hours), "exist_hours": str(e_hours), "gpu": str(g), "disk": str(d), "dph_base": str(inst.dph_base)}
        gpu_cost, disk_cost, hours_total = gpu_cost + g, disk_cost + d, hours_total + g_hours
        cold_starts += sum(1 for e in store.events(inst.id) if lo <= e["ts"] < hi and e["event"] in ("running", "external_start"))
    return {"gpu": gpu_cost, "disk": disk_cost, "other": Decimal(0), "gpu_hours": hours_total, "cold_starts": cold_starts, "per_instance": per_inst}


def compute_day(ledger, store, day: date, jobs_source, now: float, est: EstimateParams = EstimateParams()) -> dict:
    lo, hi = msk_day_bounds(day)
    r = cost_between(store, lo, hi, now, est)
    for iid, d in r["per_instance"].items():
        ledger.upsert_cost(f"calc:{day}:{iid}:gpu", "calculated", day, "gpu", D(d["gpu"]), iid, d, now)
        ledger.upsert_cost(f"calc:{day}:{iid}:disk", "calculated", day, "disk", D(d["disk"]), iid, d, now)
    videos = jobs_source.completed_count(lo, hi)
    # traffic: ESTIMATE only (needs byte counters from Vast; unavailable) - never part of totals
    traffic_gb = r["cold_starts"] * est.model_gb_per_cold_start + videos * (est.video_in_gb + est.video_out_gb)
    ledger.upsert_cost(f"est:{day}:traffic", "estimate", day, "other", traffic_gb / Decimal(1000) * EST_USD_PER_TB, None,
                       {"gb": str(traffic_gb), "assumed_usd_per_tb": str(EST_USD_PER_TB), "cold_starts": r["cold_starts"]}, now)
    ledger.put_facts(day, videos, r["gpu_hours"], {"instances": r["per_instance"]})
    if now > hi + 3600:
        ledger.freeze_facts(day)
    return {"gpu": r["gpu"], "disk": r["disk"], "gpu_hours": r["gpu_hours"], "videos": videos}


class SqlJobsSource:
    """Counts completed pool jobs via a callable(sql_params) so it works with psycopg2 on the VPS or sqlite in tests."""

    def __init__(self, counter: Callable[[float, float], int]):
        self._counter = counter

    def completed_count(self, lo: float, hi: float) -> int:
        return int(self._counter(lo, hi))
