"""
Vast.ai REST client (central dispatcher only). Endpoints mirror the official `vastai` CLI:
  GET    /api/v1/instances/            list (paginated)
  PUT    /api/v0/instances/{id}/ {"state": "running"|"stopped"}   resume / stop
  DELETE /api/v0/instances/{id}/              destroy
  POST   /api/v0/bundles/                     search offers
  PUT    /api/v0/asks/{offer_id}/             create instance from an offer
  GET    /api/v0/charges/              confirmed charges
The HTTP transport is injectable (tests use a fake Vast). The API key never appears in logs, errors or reprs.
In dry-run mode every state-changing call is only logged.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Tuple

log = logging.getLogger("sonya.fleet.vast")
BASE = "https://console.vast.ai"

Transport = Callable[[str, str, Dict[str, str], Optional[dict], int], Tuple[int, Any]]


class VastError(RuntimeError):
    def __init__(self, msg: str, status: Optional[int] = None, retryable: bool = False):
        super().__init__(msg)
        self.status = status
        self.retryable = retryable


def urllib_transport(method: str, url: str, headers: Dict[str, str], body: Optional[dict], timeout: int):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            return exc.code, json.loads(raw) if raw else {}
        except ValueError:
            return exc.code, {}
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise VastError(f"network error: {type(exc).__name__}", retryable=True) from None


@dataclass
class VastInstance:
    id: int
    label: str
    actual_status: str          # running | loading | exited | created | scheduling | ...
    intended_status: str        # running | stopped
    gpu_name: str = ""
    dph_total: Decimal = Decimal("0")
    disk_gb: Decimal = Decimal("0")
    storage_cost_gb_month: Decimal = Decimal("0")
    machine_id: Optional[int] = None
    start_date: Optional[float] = None
    status_msg: str = ""        # Vast's last container message (docker / GPU errors land here)

    @classmethod
    def parse(cls, d: Dict[str, Any]) -> "VastInstance":
        def dec(k):
            try:
                return Decimal(str(d.get(k) or 0))
            except Exception:  # noqa: BLE001
                return Decimal("0")
        return cls(id=int(d["id"]), label=str(d.get("label") or ""), actual_status=str(d.get("actual_status") or d.get("cur_state") or ""),
                   intended_status=str(d.get("intended_status") or d.get("next_state") or ""), gpu_name=str(d.get("gpu_name") or ""),
                   dph_total=dec("dph_total"), disk_gb=dec("disk_space"), storage_cost_gb_month=dec("storage_cost"),
                   machine_id=d.get("machine_id"), start_date=d.get("start_date"),
                   status_msg=str(d.get("status_msg") or ""))


# A container that the host could not start at all (run 5, 2026-11-09: "failed to inject CDI devices:
# unresolvable CDI devices ...gpu=4"). Vast keeps such a contract in "created" forever.
HOST_ERROR_MARKERS = ("Error response from daemon", "OCI runtime create failed", "failed to inject CDI devices",
                      "nvidia-container-cli", "could not select device driver")


def is_host_error(status_msg: str) -> bool:
    return any(m in (status_msg or "") for m in HOST_ERROR_MARKERS)


class VastClient:
    def __init__(self, api_key: str, *, dry_run: bool = True, transport: Transport = urllib_transport,
                 base_url: str = BASE, timeout: int = 30, retries: int = 3, sleep: Callable[[float], None] = time.sleep):
        self._key = api_key
        self.dry_run = dry_run
        self._t = transport
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._retries = retries
        self._sleep = sleep

    def __repr__(self) -> str:  # never leaks the key
        return f"VastClient(dry_run={self.dry_run})"

    def _call(self, method: str, path: str, body: Optional[dict] = None, *, write: bool = False) -> Any:
        if write and self.dry_run:
            log.info("DRY-RUN vast %s %s", method, path)
            return {"success": True, "dry_run": True}
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json", "Accept": "application/json"}
        last: Optional[VastError] = None
        for attempt in range(self._retries):
            try:
                status, data = self._t(method, self._base + path, headers, body, self._timeout)
            except VastError as exc:
                last = exc
            else:
                if status == 429 or status >= 500:
                    last = VastError(f"vast HTTP {status}", status, retryable=True)
                elif status >= 400:
                    raise VastError(f"vast HTTP {status} on {method} {path}", status)  # body deliberately not echoed
                else:
                    return data
            self._sleep(min(2 ** attempt, 8))
        raise last or VastError("vast call failed", retryable=True)

    # reads
    def list_instances(self) -> List[VastInstance]:
        data = self._call("GET", "/api/v1/instances/")
        rows = data.get("instances", data) if isinstance(data, dict) else data
        return [VastInstance.parse(r) for r in (rows or [])]

    def search_offers(self, query: dict) -> List[dict]:
        data = self._call("POST", "/api/v0/bundles/", query)
        return list(data.get("offers", [])) if isinstance(data, dict) else []

    def charges(self, start_ts: int, end_ts: int, charge_types: Tuple[str, ...] = ("instance", "volume")) -> List[dict]:
        out: List[dict] = []
        token = None
        while True:
            q = {"select_filters": {"day": {"gte": start_ts, "lte": end_ts}, "type": {"in": list(charge_types)}},
                 "limit": 100, "format": "table"}
            if token:
                q["after_token"] = token
            data = self._call("GET", "/api/v0/charges/?q=" + urllib.request.quote(json.dumps(q)))
            out += list(data.get("results", []))
            token = data.get("next_token")
            if not token:
                return out

    # writes (idempotent per Vast semantics: same state twice is a no-op)
    def stop(self, instance_id: int) -> None:
        self._call("PUT", f"/api/v0/instances/{int(instance_id)}/", {"state": "stopped"}, write=True)

    def resume(self, instance_id: int) -> None:
        self._call("PUT", f"/api/v0/instances/{int(instance_id)}/", {"state": "running"}, write=True)

    def destroy(self, instance_id: int) -> None:
        self._call("DELETE", f"/api/v0/instances/{int(instance_id)}/", {}, write=True)

    def create(self, offer_id: int, payload: dict) -> Optional[int]:
        data = self._call("PUT", f"/api/v0/asks/{int(offer_id)}/", payload, write=True)
        if isinstance(data, dict) and data.get("dry_run"):
            return None
        if not isinstance(data, dict) or not data.get("success") or "new_contract" not in data:
            raise VastError("vast create did not return new_contract")
        return int(data["new_contract"])
