"""
FleetManager - the CENTRAL decision maker for GPU instances (the worker never stops itself).

Lifecycle:  (create) -> creating -> running -> stop_pending -> stopping -> stopped -> starting -> running ...
                                                   \\-> destroying -> destroyed   (retention / stale release / failures)
Rules
  * demand (queued jobs or active jobs needing a worker) -> RESUME a stopped instance of the SAME release,
    else CREATE; never more than max_instances; never more than max_creates_per_hour.
  * idle (no queued, no active jobs on this worker, no unassigned active jobs) for IDLE_TIMEOUT_SECONDS
    after the last terminal job -> STOP. A never-used fresh instance gets boot_grace first.
  * stop is a 3-step fenced transition (CAS running->stop_pending, fresh re-check of the queue, Vast call);
    any new job aborts it. If a job slips in anyway, SIGTERM makes the worker requeue it (WORKER_SHUTDOWN, retry).
  * stopped longer than stopped_retention_seconds, or built from an older release -> DESTROY (stopped disks bill storage).
  * resume stuck in 'scheduling' longer than resume_timeout_seconds -> cancel (stop), count a failure; after
    max_resume_failures -> DESTROY and CREATE a new one.
All Vast calls are no-ops in dry-run. Repeated ticks are idempotent.
"""
from __future__ import annotations

import logging
import math
import secrets
import time
from typing import Callable, Dict, List, Mapping, Optional

from dispatcher import launcher
from dispatcher.config import FleetConfig
from dispatcher.queue import QueueSnapshot, QueueView
from dispatcher.store import FleetStore, Inst
from dispatcher.vast_api import VastClient, VastError, VastInstance

log = logging.getLogger("sonya.fleet")
RUNNING_LIKE = ("running",)
VAST_STOPPED = ("exited", "stopped", "created", "inactive")


class FleetManager:
    def __init__(self, cfg: FleetConfig, store: FleetStore, vast: VastClient, queue: QueueView,
                 token_provider: Callable[[], str], environ: Mapping[str, str], clock: Callable[[], float] = time.time):
        self.cfg, self.store, self.vast, self.queue = cfg, store, vast, queue
        self._token, self._env, self.now = token_provider, environ, clock
        self._dry_seen: set = set()
        self._create_backoff_until = 0.0

    @property
    def dry(self) -> bool:
        return self.cfg.dry_run or self.vast.dry_run

    # ------------------------------------------------------------------ tick
    def tick(self) -> List[str]:
        actions: List[str] = []
        if not self.cfg.enabled:
            return actions
        now = self.now()
        live = {i.id: i for i in self.vast.list_instances() if i.label.startswith(self.cfg.label_prefix)}
        snap = self.queue.snapshot(self.cfg.pool_modes, self.cfg.label_prefix)
        self._reconcile(live, now, actions)
        for inst in self.store.all():
            self._advance(inst, live.get(inst.id), snap, now, actions)
        self._demand(snap, now, actions)
        self._retention(snap, now, actions)
        return actions

    # ------------------------------------------------------------ reconcile
    def _reconcile(self, live: Dict[int, VastInstance], now: float, actions: List[str]) -> None:
        known = {i.id: i for i in self.store.all()}
        for iid, v in live.items():
            if iid not in known:  # adopt: never leak an instance we did not record
                state = "running" if v.actual_status == "running" else "stopped"
                self.store.add(now, id=iid, label=v.label, state=state, gpu_name=v.gpu_name, dph_base=v.dph_total,
                               storage_gb_month=v.storage_cost_gb_month, disk_gb=v.disk_gb, running_since=now if state == "running" else None,
                               stopped_at=now if state == "stopped" else None, release_sha="")
                self.store.event(now, iid, "adopted", state=state)
                actions.append(f"adopt {iid}")
        for inst in known.values():
            v = live.get(inst.id)
            if v is None and inst.state in ("destroying",):
                self.store.cas(inst.id, inst.state, inst.generation, "destroyed", destroyed_at=now)
                self.store.event(now, inst.id, "destroyed")
            elif v is None and inst.state in ("creating", "running", "stop_pending", "stopping", "stopped", "starting") \
                    and not (inst.state == "creating" and now - (inst.created_at or now) < 120):   # Vast lists new contracts with a delay
                self.store.cas(inst.id, inst.state, inst.generation, "lost", destroyed_at=now)
                self.store.event(now, inst.id, "lost", previous=inst.state)
                actions.append(f"lost {inst.id}")

    # -------------------------------------------------------------- advance
    def _advance(self, inst: Inst, v: Optional[VastInstance], snap: QueueSnapshot, now: float, actions: List[str]) -> None:
        inst = self.store.get(inst.id) or inst
        s, vs = inst.state, (v.actual_status if v else "")
        if v is None:
            return
        if s in ("creating", "starting") and vs == "running":
            self.store.cas(inst.id, s, inst.generation, "running", running_since=now)
            self.store.event(now, inst.id, "running", via=s)
            return
        if s == "creating" and now - (inst.created_at or now) > self.cfg.create_timeout_seconds:
            self._destroy(inst, now, actions, reason="create_timeout")
            if inst.machine_id:
                self.store.ban_machine(inst.machine_id, now + 3600, "create_timeout")
            return
        if s == "starting" and now - (inst.resume_requested_at or now) > self.cfg.resume_timeout_seconds:
            self._resume_failed(inst, now, actions)
            return
        if s == "stopping" and vs in VAST_STOPPED:
            self.store.cas(inst.id, s, inst.generation, "stopped", stopped_at=now)
            self.store.event(now, inst.id, "stopped")
            actions.append(f"stopped {inst.id}")
            return
        if s == "stopping" and vs == "running" and now - (inst.stop_requested_at or now) > 120:
            self._call(lambda: self.vast.stop(inst.id), inst, now, "stop_retry")  # idempotent
            self.store.touch(inst.id, stop_requested_at=now)
            return
        if s == "running" and vs in VAST_STOPPED:  # host/balance/manual stop behind our back
            self.store.cas(inst.id, s, inst.generation, "stopped", stopped_at=now)
            self.store.event(now, inst.id, "unexpected_stop", vast_status=vs)
            return
        if s == "stopped" and vs == "running":    # started manually
            self.store.cas(inst.id, s, inst.generation, "running", running_since=now)
            self.store.event(now, inst.id, "external_start")
            return
        if s == "stop_pending":                   # crash between CAS and Vast call: re-validate and finish
            self._finish_stop(inst, now, actions)
            return
        if s == "destroying" and now - (inst.stop_requested_at or now) > 120:
            self._call(lambda: self.vast.destroy(inst.id), inst, now, "destroy_retry")
            self.store.touch(inst.id, stop_requested_at=now)
            return
        if s == "running":
            self._maybe_stop(inst, snap, now, actions)

    # ----------------------------------------------------------------- stop
    def _maybe_stop(self, inst: Inst, snap: QueueSnapshot, now: float, actions: List[str]) -> None:
        busy = snap.active_by_worker.get(inst.label, 0) > 0
        terminal = snap.last_terminal_by_worker.get(inst.label)
        if busy:
            self.store.touch(inst.id, last_work_at=now)
            return
        if snap.pending > 0 and snap.active_unassigned == 0:
            # Jobs are waiting and this worker holds none. Normally it claims one within a poll interval. If it has been up
            # and idle for stall_seconds it is hung / unreachable / misconfigured: replace it (it has NO active job, so no
            # work is lost). Active jobs are never interrupted by this rule.
            since = max(terminal or 0.0, inst.last_work_at or 0.0, (inst.running_since or now) + self.cfg.boot_grace_seconds)
            if now - since > self.cfg.stall_seconds:
                self.store.event(now, inst.id, "worker_stalled", pending=snap.pending, idle_s=int(now - since))
                if inst.machine_id:
                    self.store.ban_machine(inst.machine_id, now + 3600, "worker_stalled")
                self._destroy(inst, now, actions, reason="worker_stalled")
            return
        if snap.pending > 0 or snap.active_unassigned > 0:
            return                                  # work is coming / cannot prove it is safe
        worked = bool(terminal or inst.last_work_at)
        last = max(terminal or 0.0, inst.last_work_at or 0.0)
        due = (last + self.cfg.idle_timeout_seconds) if worked else ((inst.running_since or now) + self.cfg.boot_grace_seconds)
        too_old = inst.running_since and now - inst.running_since > self.cfg.max_running_hours * 3600
        if now >= due or too_old:
            self._begin_stop(inst, now, actions)

    def _begin_stop(self, inst: Inst, now: float, actions: List[str]) -> None:
        if self._dry_log(inst, "stop", now):
            return
        if not self.store.cas(inst.id, "running", inst.generation, "stop_pending", stop_requested_at=now):
            return                                  # somebody else moved it: do nothing
        self.store.event(now, inst.id, "stop_pending")
        self._finish_stop(self.store.get(inst.id), now, actions)

    def _finish_stop(self, inst: Inst, now: float, actions: List[str]) -> None:
        fresh = self.queue.snapshot(self.cfg.pool_modes, self.cfg.label_prefix)   # fence: re-read right before the call
        if fresh.pending > 0 or fresh.active_unassigned > 0 or fresh.active_by_worker.get(inst.label, 0) > 0:
            self.store.cas(inst.id, "stop_pending", inst.generation, "running", last_work_at=now)
            self.store.event(now, inst.id, "stop_cancelled", pending=fresh.pending)
            actions.append(f"stop_cancelled {inst.id}")
            return
        try:
            self.vast.stop(inst.id)
        except VastError as exc:
            self.store.cas(inst.id, "stop_pending", inst.generation, "running")
            self.store.event(now, inst.id, "stop_failed", status=exc.status)
            return
        self.store.cas(inst.id, "stop_pending", inst.generation, "stopping")
        self.store.event(now, inst.id, "stop_requested")
        actions.append(f"stop {inst.id}")

    # --------------------------------------------------------------- demand
    def _demand(self, snap: QueueSnapshot, now: float, actions: List[str]) -> None:
        demand = snap.pending + snap.active_total
        if demand <= 0:
            return
        need = min(self.cfg.max_instances, math.ceil(demand / self.cfg.jobs_per_instance))
        have = len(self.store.all(("creating", "running", "starting")))
        if have >= need:
            return
        if self.store.all(("stop_pending", "stopping")):
            return                                  # let the stop finish, then resume the same disk
        stopped = sorted((i for i in self.store.all(("stopped",)) if i.resume_failures < self.cfg.max_resume_failures
                          and i.release_sha == self.cfg.release_sha), key=lambda i: i.stopped_at or 0, reverse=True)
        if stopped:
            self._resume(stopped[0], now, actions)
        else:
            self._create(now, actions)

    def _resume(self, inst: Inst, now: float, actions: List[str]) -> None:
        if self._dry_log(inst, "resume", now):
            return
        if not self.store.cas(inst.id, "stopped", inst.generation, "starting", resume_requested_at=now):
            return
        try:
            self.vast.resume(inst.id)
        except VastError as exc:
            self.store.cas(inst.id, "starting", inst.generation + 1, "stopped", resume_failures=inst.resume_failures + 1)
            self.store.event(now, inst.id, "resume_failed", status=exc.status)
            return
        self.store.event(now, inst.id, "resume_requested")
        actions.append(f"resume {inst.id}")

    def _resume_failed(self, inst: Inst, now: float, actions: List[str]) -> None:
        self._call(lambda: self.vast.stop(inst.id), inst, now, "resume_cancel")   # per Vast docs: cancel scheduling by stopping
        fails = inst.resume_failures + 1
        self.store.event(now, inst.id, "resume_timeout", failures=fails)
        if fails >= self.cfg.max_resume_failures:
            self.store.cas(inst.id, "starting", inst.generation, "stopped", resume_failures=fails)
            self._destroy(self.store.get(inst.id), now, actions, reason="resume_failures")
        else:
            self.store.cas(inst.id, "starting", inst.generation, "stopped", resume_failures=fails, stopped_at=now)

    def _create(self, now: float, actions: List[str]) -> None:
        if now < self._create_backoff_until:
            return
        if self.store.creates_since(now - 3600) >= self.cfg.max_creates_per_hour:
            self.store.event(now, None, "create_rate_limited")
            self._create_backoff_until = now + 300
            return
        try:
            launcher.validate_release(self.cfg)
            offer = launcher.pick_offer(self.vast.search_offers(launcher.search_query(self.cfg)), self.cfg, self.store.banned_machines(now))
            if offer is None:
                raise launcher.LaunchError("no suitable offer")
            label = self.cfg.label_prefix + secrets.token_hex(4)
            if self.dry:
                if "create" not in self._dry_seen:
                    self._dry_seen.add("create")
                    self.store.event(now, None, "dry_run_create", offer=offer.get("id"), dph=str(offer.get("dph_total")))
                return
            payload = launcher.create_payload(self.cfg, label, self._token(), self._env)
            iid = self.vast.create(int(offer["id"]), payload)
        except (launcher.LaunchError, VastError) as exc:
            self._create_backoff_until = now + 60
            self.store.event(now, None, "create_failed", reason=str(exc)[:160])
            return
        self.store.record_create(now)
        self.store.add(now, id=int(iid), label=label, state="creating", gpu_name=offer.get("gpu_name"), dph_base=offer.get("dph_base", offer.get("dph_total")),
                       storage_gb_month=offer.get("storage_cost"), disk_gb=self.cfg.disk_gb, release_sha=self.cfg.release_sha,
                       machine_id=offer.get("machine_id"), inet_up_tb=offer.get("internet_up_cost_per_tb"), inet_down_tb=offer.get("internet_down_cost_per_tb"))
        self.store.event(now, int(iid), "created", offer=offer.get("id"), gpu=offer.get("gpu_name"), dph_base=str(offer.get("dph_base")))
        actions.append(f"create {iid}")

    # ------------------------------------------------------------ retention
    def _retention(self, snap: QueueSnapshot, now: float, actions: List[str]) -> None:
        for inst in self.store.all(("stopped",)):
            stale = inst.release_sha != self.cfg.release_sha
            expired = now - (inst.stopped_at or now) > self.cfg.stopped_retention_seconds
            if (expired or stale) and snap.pending == 0:
                self._destroy(inst, now, actions, reason="stale_release" if stale else "retention")

    def _destroy(self, inst: Optional[Inst], now: float, actions: List[str], reason: str) -> None:
        if inst is None or self._dry_log(inst, "destroy", now):
            return
        if not self.store.cas(inst.id, inst.state, inst.generation, "destroying", stop_requested_at=now):
            return
        self._call(lambda: self.vast.destroy(inst.id), inst, now, "destroy_requested", reason=reason)
        actions.append(f"destroy {inst.id} ({reason})")

    # -------------------------------------------------------------- helpers
    def _call(self, fn, inst: Inst, now: float, event: str, **detail) -> None:
        try:
            fn()
            self.store.event(now, inst.id, event, **detail)
        except VastError as exc:
            self.store.event(now, inst.id, event + "_error", status=exc.status)

    def _dry_log(self, inst: Inst, action: str, now: float) -> bool:
        if not self.dry:
            return False
        key = (inst.id, action, inst.generation)
        if key not in self._dry_seen:
            self._dry_seen.add(key)
            self.store.event(now, inst.id, f"dry_run_{action}")
        return True
