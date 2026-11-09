"""`python -m dispatcher.main status` - what the fleet has now and which spare offers it would take next.
Reads ONLY the dispatcher's own fleet.db (no Vast call, no production DB, no secrets)."""
from __future__ import annotations

import time
from typing import List, Optional

from dispatcher.store import FleetStore

LIVE = ("creating", "running", "stop_pending", "stopping", "stopped", "starting", "destroying")
CLASS_TITLES = {"warm": "warm (European RTX 3060 class, cheap pause)", "burst": "burst (any default-class host)",
                "power": "power (multi-GPU rush server)", "reserve": "power reserve (paused multi-GPU, pause <= cap)"}


def _age(now: float, ts: Optional[float]) -> str:
    if not ts:
        return "-"
    s = int(max(0, now - ts))
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def status_text(store: FleetStore, now: Optional[float] = None) -> str:
    now = time.time() if now is None else now
    out: List[str] = ["Servers:"]
    insts = sorted(store.all(LIVE), key=lambda i: (i.role, i.id))
    for i in insts:
        out.append(f"  {i.id:>10}  {i.state:<12} {i.role:<8} {i.num_gpus}x{i.gpu_name or '?'} machine={i.machine_id} "
                   f"slots={i.slots or 'default'} since={_age(now, i.running_since if i.state == 'running' else i.stopped_at or i.created_at)}")
    if not insts:
        out.append("  (none)")
    spares = store.spares()
    out.append("Spare offers (best first; 'cached' = our image booted on that host before):")
    if not spares:
        out.append("  (no list yet: SONYA_SPARES_ENABLED / SONYA_POWER_RESERVE off, or the first refresh has not run)")
    for cls in ("warm", "burst", "power", "reserve"):
        if cls not in spares:
            continue
        rows = spares[cls]
        out.append(f"  {CLASS_TITLES[cls]} - refreshed {_age(now, rows[0]['ts'])} ago:")
        if rows[0].get("offer") is None:
            out.append("    (no acceptable offer right now)")
            continue
        for k, o in enumerate(rows, 1):
            out.append(f"    {k}. offer {o['offer']} machine {o['machine']}: {o['num_gpus']}x {o['gpu']}, {o['geo']}, "
                       f"${o['dph']}/h, disk ${o['storage']}/GB-month, {o['cpu']}{'  cached' if o['cached'] else ''}")
    banned = sorted(store.banned_machines(now))
    out.append(f"Banned hosts (1 h): {', '.join(map(str, banned)) if banned else 'none'}")
    return "\n".join(out)
