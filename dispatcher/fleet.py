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
  * WARM / BURST (owner decision 2026-11-08): exactly cfg.warm_instances (1) instance has role "warm" - on idle it is
    STOPPED (disk keeps image + release + models: resume ~10-40 s instead of a ~4-6 min cold start). Every other
    instance is "burst": created for the extra load and DESTROYED (not stopped) when idle, so its disk never bills.
    The warm is never removed by retention (only by a stale release / host error / failed resume). A warm that cannot
    resume (GPU taken by another renter) is destroyed + its machine banned, and the next instance becomes the warm.
    No warm at all (first start, lost, stale) and no load -> one is PRIMED: created, booted, stopped.
    Only a default-class host with cheap storage may be the warm (launcher.warm_eligible).
  * CAPACITY in video slots: demand is covered by the slots of running / booting servers (default class: jobs_per_instance
    each, so the count is the same as before). POWER class (cfg.power_enabled, off by default): when the jobs not yet
    covered are >= cfg.power_gap(), the next create is ONE multi-GPU server (e.g. 2x A40) with slots from its cards, CPU and
    RAM (launcher.offer_slots); it is always a burst (destroyed when idle), never the warm. No power offer -> default class.
  * SPARES (cfg.spares_enabled, off by default; owner 2026-11-09): every spares_refresh_seconds the ranked best
    spares_count offers per class (warm / burst / power / reserve) are saved (fleet_spares, event "spares" when the list
    changes). Banned hosts are left out; within a class a host that already booted our image (good_machines) comes first.
    A create Vast refuses goes straight to the next offer of the ranking in the same tick (at most spares_count tries); a
    warm whose resume call fails is destroyed + banned and replaced by a create in the same tick.
  * POWER RESERVE (cfg.power_reserve, off by default; owner 2026-11-09): ONE power server (role "reserve", e.g. 2x A40) is
    primed like the warm and kept STOPPED. Demand >= cfg.reserve_wake_at() wakes it (outside max_instances); idle after the
    rush -> stopped again. Not woken in resume_timeout_seconds (90 s) or the resume call fails -> destroyed + host banned
    and the next reserve-eligible power offer (pause <= power_reserve_max_storage_rub / month) is created at once.
All Vast calls are no-ops in dry-run. Repeated ticks are idempotent.
"""
from __future__ import annotations

import logging
import secrets
import time
from typing import Callable, Dict, List, Mapping, Optional

from dispatcher import launcher
from dispatcher.config import FleetConfig
from dispatcher.queue import QueueSnapshot, QueueView
from dispatcher.store import FleetStore, Inst
from dispatcher.vast_api import VastClient, VastError, VastInstance, is_host_error

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
        self._reserve_backoff_until = 0.0
        self._spares_at = float("-inf")

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
        self._refresh_spares(now)
        self._demand(snap, now, actions)
        self._retention(snap, now, actions)
        self._prime_warm(snap, now, actions)
        self._prime_reserve(snap, now, actions)
        return actions

    # ------------------------------------------------------------------ roles
    def _warm(self, exclude: Optional[int] = None) -> List[Inst]:
        """Instances that ARE the warm server (any live state)."""
        return [i for i in self.store.all(("creating", "running", "stop_pending", "stopping", "stopped", "starting"))
                if i.role == "warm" and i.id != exclude]

    def _reserves(self, exclude: Optional[int] = None) -> List[Inst]:
        """Instances that ARE the power reserve (any live state)."""
        return [i for i in self.store.all(("creating", "running", "stop_pending", "stopping", "stopped", "starting"))
                if i.role == "reserve" and i.id != exclude]

    def _idle_action(self, inst: Inst) -> str:
        """What an idle instance becomes: 'stop' (it is / becomes the warm, or it is the power reserve) or 'destroy'."""
        if inst.role == "reserve":                           # reserve switched off / stale release -> it goes away
            return "stop" if self.cfg.power_reserve and inst.release_sha == self.cfg.release_sha else "destroy"
        if inst.role == "warm" and inst.gpu_class != "power":
            return "stop"
        if inst.gpu_class == "power":                        # a rush server never becomes the warm (pause cost, owner scheme)
            return "destroy" if self.cfg.warm_instances > 0 else "stop"
        if self.cfg.warm_instances > 0 and not self._warm(exclude=inst.id) and inst.release_sha == self.cfg.release_sha \
                and (inst.note or "") == "warm_eligible":
            return "stop"                                    # promoted below, in _finish_stop, after the fence
        return "destroy" if self.cfg.warm_instances > 0 else "stop"

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
            if inst.machine_id:
                self.store.mark_good_machine(inst.machine_id, now)   # our image layers are now in this host's cache
            return
        if s in ("creating", "starting", "running") and vs != "running" and is_host_error(v.status_msg):
            # run 5 (2026-11-09): Vast showed "running" for seconds, then "created" + "failed to inject CDI devices";
            # treating that as an ordinary stop meant 2 resumes, a destroy WITHOUT a ban, and a re-create on the
            # same broken machine. A host that cannot start the container is dropped and banned instead.
            since = {"creating": inst.created_at, "starting": inst.resume_requested_at}.get(s)
            if s == "running" or now - (since or now) > self.cfg.host_error_grace_seconds:
                self.store.event(now, inst.id, "host_error", state=s, msg=v.status_msg[:200])
                if inst.machine_id:
                    self.store.ban_machine(inst.machine_id, now + 3600, "host_error")
                self._destroy(inst, now, actions, reason="host_error")
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
        grace = self.cfg.boot_grace_seconds
        if not worked and self._idle_action(inst) == "destroy" and snap.active_total == 0:
            grace = self.cfg.idle_timeout_seconds   # a burst that came too late: no job anywhere -> nothing to boot for
        due = (last + self.cfg.idle_timeout_seconds) if worked else ((inst.running_since or now) + grace)
        too_old = inst.running_since and now - inst.running_since > self.cfg.max_running_hours * 3600
        if now >= due or too_old:
            self._begin_stop(inst, now, actions)

    def _begin_stop(self, inst: Inst, now: float, actions: List[str]) -> None:
        if self._dry_log(inst, "stop", now):
            return
        if not self.store.cas(inst.id, "running", inst.generation, "stop_pending", stop_requested_at=now):
            return                                  # somebody else moved it: do nothing
        self.store.event(now, inst.id, "stop_pending", role=inst.role)
        self._finish_stop(self.store.get(inst.id), now, actions)

    def _finish_stop(self, inst: Inst, now: float, actions: List[str]) -> None:
        fresh = self.queue.snapshot(self.cfg.pool_modes, self.cfg.label_prefix)   # fence: re-read right before the call
        if fresh.pending > 0 or fresh.active_unassigned > 0 or fresh.active_by_worker.get(inst.label, 0) > 0:
            self.store.cas(inst.id, "stop_pending", inst.generation, "running", last_work_at=now)
            self.store.event(now, inst.id, "stop_cancelled", pending=fresh.pending)
            actions.append(f"stop_cancelled {inst.id}")
            return
        if self._idle_action(inst) == "destroy":    # burst: never paused, its disk would keep billing
            if self.store.cas(inst.id, "stop_pending", inst.generation, "destroying", stop_requested_at=now):
                self._call(lambda: self.vast.destroy(inst.id), inst, now, "destroy_requested", reason="burst_idle")
                actions.append(f"destroy {inst.id} (burst_idle)")
            return
        if inst.role not in ("warm", "reserve"):
            self.store.touch(inst.id, role="warm")
            self.store.event(now, inst.id, "warm_assigned")
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
    def slots(self, inst: Inst) -> int:
        """Parallel videos counted for this server (a multi-GPU power server carries more than one RTX 3060)."""
        return inst.slots or self.cfg.jobs_per_instance

    def _demand(self, snap: QueueSnapshot, now: float, actions: List[str]) -> None:
        """Cover demand with SLOTS, not servers. With every server at jobs_per_instance slots this is exactly the old rule
        need = min(max_instances, ceil(demand / jobs_per_instance)); a power server counts with its own slots."""
        demand = snap.pending + snap.active_total
        if demand <= 0:
            return
        up = self.store.all(("creating", "running", "starting", "stop_pending", "stopping"))   # waking: let a stop finish,
        cap = sum(self.slots(i) for i in up)                                                   # then resume the same disk
        count = sum(1 for i in up if i.role != "reserve")       # the power reserve is outside max_instances
        used: set = set()
        if self.cfg.power_reserve and demand >= self.cfg.reserve_wake_at() and cap < demand:
            cap += self._wake_reserve(now, actions, used)
        if cap >= demand or count >= self.cfg.max_instances:
            return
        stopped = sorted((i for i in self.store.all(("stopped",)) if i.resume_failures < self.cfg.max_resume_failures
                          and i.release_sha == self.cfg.release_sha and i.role != "reserve"),
                         key=lambda i: (i.role != "warm", -(i.stopped_at or 0)))
        while cap < demand and count < self.cfg.max_instances:   # the whole gap in ONE tick: 6 jobs -> warm resume + 2 creates
            if stopped:
                inst = stopped.pop(0)
                if self._resume(inst, now, actions) or not self.cfg.spares_enabled:
                    got = self.slots(inst)
                else:                                            # spares: the warm is gone -> the next offer, now
                    got = self._create(now, actions, exclude=used, gap=demand - cap)
                    if not got:
                        break
            else:
                got = self._create(now, actions, exclude=used, gap=demand - cap)
                if not got:
                    break
            cap, count = cap + got, count + 1

    def _wake_reserve(self, now: float, actions: List[str], used: set) -> int:
        """Rush: wake the paused power reserve -> its slots. No reserve at all (lost / replaced) -> create the next power
        offer of the ranking right now. A reserve already up is counted in `up` by the caller (0 here)."""
        res = self._reserves()
        if not res:
            return self._create(now, actions, exclude=used, reserve=True)
        r = next((i for i in res if i.state == "stopped" and i.release_sha == self.cfg.release_sha), None)
        if r is None:
            return 0
        self.store.event(now, r.id, "reserve_wake", wake_at=self.cfg.reserve_wake_at())
        if self._resume(r, now, actions):
            return self.slots(r)
        return self._create(now, actions, exclude=used, reserve=True)     # resume refused: replaced at once

    def _resume(self, inst: Inst, now: float, actions: List[str]) -> bool:
        """False only when Vast refused the resume call."""
        if self._dry_log(inst, "resume", now):
            return True
        if not self.store.cas(inst.id, "stopped", inst.generation, "starting", resume_requested_at=now):
            return True
        try:
            self.vast.resume(inst.id)
        except VastError as exc:
            fails = inst.resume_failures + 1
            self.store.cas(inst.id, "starting", inst.generation + 1, "stopped", resume_failures=fails)
            self.store.event(now, inst.id, "resume_failed", status=exc.status)
            if (self.cfg.spares_enabled or inst.role == "reserve") and fails >= self.cfg.max_resume_failures:
                if inst.machine_id:
                    self.store.ban_machine(inst.machine_id, now + 3600, "resume_failures")
                self.store.event(now, inst.id, f"{inst.role}_replaced", reason="resume_failed")
                self._destroy(self.store.get(inst.id), now, actions, reason="resume_failures")
            return False
        self.store.event(now, inst.id, "resume_requested")
        actions.append(f"resume {inst.id}")
        return True

    def _resume_failed(self, inst: Inst, now: float, actions: List[str]) -> None:
        self._call(lambda: self.vast.stop(inst.id), inst, now, "resume_cancel")   # per Vast docs: cancel scheduling by stopping
        fails = inst.resume_failures + 1
        self.store.event(now, inst.id, "resume_timeout", failures=fails)
        if fails >= self.cfg.max_resume_failures:
            if inst.machine_id:            # never re-create on the machine that could not resume
                self.store.ban_machine(inst.machine_id, now + 3600, "resume_failures")
            if inst.role in ("warm", "reserve"):   # host busy: the next instance becomes the warm (owner decision 2026-11-08);
                self.store.event(now, inst.id, f"{inst.role}_replaced", reason="resume_failed")   # reserve: next power offer
            self.store.cas(inst.id, "starting", inst.generation, "stopped", resume_failures=fails)
            self._destroy(self.store.get(inst.id), now, actions, reason="resume_failures")
        else:
            self.store.cas(inst.id, "starting", inst.generation, "stopped", resume_failures=fails, stopped_at=now)

    def _candidates(self, offers: List[dict], banned: set, known: set, exclude: set, *, want_warm: bool, prime: bool,
                    gap: int, reserve: bool):
        """(offer, is_power) in the order they are tried; lazy, so the power search runs only when it is reached.
        The first one is exactly the old single choice (warm-eligible -> power on a big gap -> any default-class host)."""
        if reserve:
            for o in launcher.rank_power_offers(self.vast.search_offers(launcher.power_search_query(self.cfg)), self.cfg,
                                                banned, known, exclude=exclude, environ=self._env, reserve=True):
                yield o, True
            return
        if want_warm or prime:
            for o in launcher.rank_offers(offers, self.cfg, banned, known, warm=True, exclude=exclude):
                yield o, False
        if prime:
            return
        if self.cfg.power_enabled and gap >= self.cfg.power_gap():
            for o in launcher.rank_power_offers(self.vast.search_offers(launcher.power_search_query(self.cfg)), self.cfg,
                                                banned, known, exclude=exclude, environ=self._env):
                yield o, True
        for o in launcher.rank_offers(offers, self.cfg, banned, known, exclude=exclude):
            yield o, False

    def _create(self, now: float, actions: List[str], *, exclude: Optional[set] = None, prime: bool = False,
                gap: int = 0, reserve: bool = False) -> int:
        """One new instance -> its slots (0 = nothing created). The FIRST instance while there is no warm goes to a
        warm-eligible host if one is offered (it becomes the warm when idle); otherwise, when the power class is on and
        gap (jobs not covered yet) >= power_gap, a multi-GPU POWER server; otherwise any acceptable host, as a burst.
        prime=True: warm-eligible only. reserve=True: the paused POWER reserve (reserve-eligible power hosts only).
        Spares on: an offer Vast refuses -> the next one of the ranking, up to spares_count tries (off: one try)."""
        if now < (self._reserve_backoff_until if reserve else self._create_backoff_until):
            return 0
        if self.store.creates_since(now - 3600) >= self.cfg.max_creates_per_hour:
            self.store.event(now, None, "create_rate_limited")
            if reserve:
                self._reserve_backoff_until = now + 300
            else:
                self._create_backoff_until = now + 300
            return 0
        exclude = set() if exclude is None else exclude
        want_warm = not reserve and self.cfg.warm_instances > 0 and not self._warm() and not any(
            (i.note or "") == "warm_eligible" for i in self.store.all(("creating", "running", "starting")))
        tries = self.cfg.spares_count if self.cfg.spares_enabled else 1
        offer, power, iid, tried = None, False, None, 0
        try:
            launcher.validate_release(self.cfg)
            offers = [] if reserve else self.vast.search_offers(launcher.search_query(self.cfg))
            banned, known = self.store.banned_machines(now), self.store.good_machines(now - 30 * 86400)
            for offer, power in self._candidates(offers, banned, known, exclude, want_warm=want_warm, prime=prime, gap=gap,
                                                 reserve=reserve):
                if offer.get("id") in exclude or offer.get("machine_id") in exclude:
                    continue                             # refused a moment ago (it may be in two lists)
                slots = launcher.offer_slots(offer, self.cfg, power=power, environ=self._env)
                if self.dry:
                    key = "prime" if prime else "create_reserve" if reserve else "create_power" if power else "create"
                    if key not in self._dry_seen:
                        self._dry_seen.add(key)
                        self.store.event(now, None, f"dry_run_{key}", offer=offer.get("id"), dph=str(offer.get("dph_total")),
                                         gpu=offer.get("gpu_name"), cpu=offer.get("cpu_name"), geo=offer.get("geolocation"),
                                         num_gpus=launcher.num_gpus(offer), slots=slots)
                    return 0
                label = self.cfg.label_prefix + secrets.token_hex(4)
                payload = launcher.create_payload(self.cfg, label, self._token(), self._env, power_slots=slots if power else 0)
                tried += 1
                try:
                    iid = self.vast.create(int(offer["id"]), payload)
                    break
                except VastError as exc:
                    if tried >= tries:
                        raise
                    exclude.update({offer.get("id"), offer.get("machine_id")})   # e.g. rented a second ago: next spare
                    self.store.event(now, None, "create_failed_next", offer=offer.get("id"), machine=offer.get("machine_id"),
                                     status=exc.status, reason=str(exc)[:120], attempt=tried)
            if iid is None:
                raise launcher.LaunchError("no suitable reserve-eligible power offer" if reserve else
                                           "no suitable warm-eligible offer" if prime else "no suitable offer")
        except (launcher.LaunchError, VastError) as exc:
            if reserve:
                self._reserve_backoff_until = now + 600
            else:
                self._create_backoff_until = now + (600 if prime else 60)
            self.store.event(now, None, "reserve_failed" if reserve else "prime_failed" if prime else "create_failed",
                             reason=str(exc)[:160])
            return 0
        eligible = not power and launcher.warm_eligible(offer, self.cfg)
        exclude.update({offer.get("id"), offer.get("machine_id")})
        self.store.record_create(now)
        self.store.add(now, id=int(iid), label=label, state="creating", gpu_name=offer.get("gpu_name"), dph_base=offer.get("dph_base", offer.get("dph_total")),
                       storage_gb_month=offer.get("storage_cost"), disk_gb=self.cfg.disk_gb, release_sha=self.cfg.release_sha,
                       machine_id=offer.get("machine_id"), inet_up_tb=offer.get("internet_up_cost_per_tb"), inet_down_tb=offer.get("internet_down_cost_per_tb"),
                       role="reserve" if reserve else "warm" if prime else "burst", note="warm_eligible" if eligible else "",
                       # default class: NULL = cfg.jobs_per_instance at every tick, exactly the old count
                       slots=slots if power else None, num_gpus=launcher.num_gpus(offer), gpu_class="power" if power else "standard")
        extra = {"reserve": True} if reserve else {}
        if tried > 1:
            extra["attempt"] = tried
        self.store.event(now, int(iid), "created", offer=offer.get("id"), gpu=offer.get("gpu_name"), dph_base=str(offer.get("dph_base")),
                         cpu=offer.get("cpu_name"), geo=offer.get("geolocation"), inet_down=offer.get("inet_down"),
                         machine=offer.get("machine_id"), cached=offer.get("machine_id") in known, prime=prime, warm_eligible=eligible,
                         num_gpus=launcher.num_gpus(offer), slots=slots, gpu_class="power" if power else "standard", **extra)
        actions.append(f"{'prime' if prime else 'create_reserve' if reserve else 'create_power' if power else 'create'} {iid}")
        return slots

    def _prime_warm(self, snap: QueueSnapshot, now: float, actions: List[str]) -> None:
        """No warm server at all and nothing happening -> create one; it boots (image, release, models) and is stopped
        by the normal idle rule after boot_grace. Bounded by max_creates_per_hour and a 10 min back-off on failure."""
        if not (self.cfg.warm_instances > 0 and self.cfg.warm_prime) or snap.pending + snap.active_total > 0:
            return
        if self._warm() or self.store.all(("creating", "running", "starting", "stop_pending", "stopping", "destroying")):
            return
        self._create(now, actions, prime=True)

    def _prime_reserve(self, snap: QueueSnapshot, now: float, actions: List[str]) -> None:
        """Power reserve on, none exists and nothing happening -> create one; it boots and is stopped by the idle rule
        after boot_grace (then only its disk bills, <= power_reserve_max_storage_rub / month). The warm is primed first."""
        if not self.cfg.power_reserve or snap.pending + snap.active_total > 0 or self._reserves():
            return
        if self.cfg.warm_instances > 0 and self.cfg.warm_prime and not self._warm():
            return
        if self.store.all(("creating", "starting")):
            return
        self._create(now, actions, reserve=True)

    # --------------------------------------------------------------- spares
    def _refresh_spares(self, now: float) -> None:
        """Every spares_refresh_seconds: the ranked best spares_count offers per class -> fleet_spares (+ event when the
        list changes). Read-only Vast searches (also in dry-run)."""
        cfg = self.cfg
        if not (cfg.spares_enabled or cfg.power_reserve) or now - self._spares_at < cfg.spares_refresh_seconds:
            return
        self._spares_at = now
        banned, known, n = self.store.banned_machines(now), self.store.good_machines(now - 30 * 86400), cfg.spares_count
        lists: Dict[str, List[dict]] = {}
        try:
            if cfg.spares_enabled:
                offers = self.vast.search_offers(launcher.search_query(cfg))
                if cfg.warm_instances > 0:
                    lists["warm"] = launcher.rank_offers(offers, cfg, banned, known, warm=True)[:n]
                lists["burst"] = launcher.rank_offers(offers, cfg, banned, known)[:n]
            if (cfg.spares_enabled and cfg.power_enabled) or cfg.power_reserve:
                power = self.vast.search_offers(launcher.power_search_query(cfg))
                if cfg.spares_enabled and cfg.power_enabled:
                    lists["power"] = launcher.rank_power_offers(power, cfg, banned, known, environ=self._env)[:n]
                if cfg.power_reserve:
                    lists["reserve"] = launcher.rank_power_offers(power, cfg, banned, known, environ=self._env, reserve=True)[:n]
        except VastError as exc:
            self.store.event(now, None, "spares_failed", status=exc.status)
            return
        before = self.store.spares()
        for cls, offs in lists.items():
            brief = [launcher.offer_brief(o, known) for o in offs]
            self.store.save_spares(cls, brief, now)
            if [b["offer"] for b in brief] != [b["offer"] for b in before.get(cls, []) if b.get("offer") is not None]:
                self.store.event(now, None, "spares", cls=cls, offers=brief)
            log.info("spares %s: %s", cls, ", ".join(f"{b['offer']}/{b['machine']} {b['num_gpus']}x{b['gpu']} {b['geo']} "
                                                     f"${b['dph']}{' cached' if b['cached'] else ''}" for b in brief) or "none")

    # ------------------------------------------------------------ retention
    def _retention(self, snap: QueueSnapshot, now: float, actions: List[str]) -> None:
        for inst in self.store.all(("stopped",)):
            stale = inst.release_sha != self.cfg.release_sha
            warm = inst.role == "warm" and self.cfg.warm_instances > 0 and not any(
                w.stopped_at and (w.stopped_at > (inst.stopped_at or 0)) for w in self._warm(exclude=inst.id) if w.state == "stopped")
            reserve = inst.role == "reserve" and self.cfg.power_reserve and not any(
                r.stopped_at and (r.stopped_at > (inst.stopped_at or 0)) for r in self._reserves(exclude=inst.id) if r.state == "stopped")
            expired = not (warm or reserve) and now - (inst.stopped_at or now) > self.cfg.stopped_retention_seconds
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
