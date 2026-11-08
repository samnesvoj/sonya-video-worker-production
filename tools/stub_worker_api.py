#!/usr/bin/env python3
"""
stub_worker_api.py - a TINY stand-in for the production worker API, used ONLY by the Fast Start test (tools/fast_start_test.py).
It implements the exact endpoints gpu_worker.py calls (claim / status / heartbeat / complete / fail / files) for an in-memory job list,
so the paid GPU test never touches the production queue, database or WORKER_SECRET. It is also a QueueView for the fleet manager.
Security: bearer secret compared in constant time (a throwaway secret generated per test run), request bodies capped at 64 KB, no
request logging, binds to 127.0.0.1 unless told otherwise (expose it with a tunnel for the duration of the test, then stop it).
"""
from __future__ import annotations

import hmac
import json
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, Iterable, List, Optional

from dispatcher.queue import ACTIVE_STATUSES, TERMINAL_STATUSES, QueueSnapshot

MAX_BODY = 64 * 1024


class StubState:
    def __init__(self, clock: Callable[[], float] = time.time):
        self.clock, self.lock = clock, threading.RLock()
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.failures: List[Dict[str, Any]] = []

    def enqueue(self, mode: str, s3_input_key: str, params: Optional[dict] = None, user_id: str = "fast-start-test") -> str:
        jid = str(uuid.uuid4())
        with self.lock:
            self.jobs[jid] = {"id": jid, "user_id": user_id, "mode": mode, "params": params or {}, "s3_input_key": s3_input_key, "status": "queued",
                              "worker_id": None, "queued_at": self.clock(), "history": [("queued", self.clock())], "files": [], "error": None}
        return jid

    def _set(self, j: dict, status: str) -> None:
        j["status"] = status
        j["history"].append((status, self.clock()))

    def claim(self, worker_id: str, modes: Optional[List[str]] = None, job_id: Optional[str] = None) -> Optional[dict]:
        with self.lock:
            for j in sorted(self.jobs.values(), key=lambda j: j["queued_at"]):
                if j["status"] == "queued" and (job_id is None or j["id"] == job_id) and (not modes or j["mode"] in modes):
                    j["worker_id"] = worker_id
                    self._set(j, "claimed")
                    return {k: j[k] for k in ("id", "user_id", "mode", "params", "s3_input_key", "status")}
        return None

    def status(self, jid: str, status: str) -> bool:
        with self.lock:
            j = self.jobs.get(jid)
            if not j or j["status"] in TERMINAL_STATUSES:
                return False
            self._set(j, status)
            return True

    def complete(self, jid: str, body: dict) -> bool:
        with self.lock:
            j = self.jobs.get(jid)
            if not j:
                return False
            j["result"] = {k: body.get(k) for k in ("s3_output_key", "clip_count", "processing_ms")}
            self._set(j, "completed")
            return True

    def fail(self, jid: str, body: dict) -> bool:
        with self.lock:
            j = self.jobs.get(jid)
            if not j:
                return False
            self.failures.append({"job": jid, "code": body.get("error_code"), "retry": bool(body.get("retry")), "t": self.clock()})
            j["error"] = body.get("error_code")
            if body.get("retry"):
                self._set(j, "queued")                      # the real server requeues retryable failures
            else:
                self._set(j, "failed")
            return True

    def add_file(self, jid: str, body: dict) -> bool:
        with self.lock:
            j = self.jobs.get(jid)
            if j is not None:
                j["files"].append({k: body.get(k) for k in ("file_type", "s3_key", "filename", "size_bytes")})
            return j is not None

    # QueueView
    def snapshot(self, modes: Iterable[str], label_prefix: str) -> QueueSnapshot:
        modes = set(modes)
        s = QueueSnapshot()
        with self.lock:
            for j in self.jobs.values():
                if j["mode"] not in modes:
                    continue
                if j["status"] == "queued":
                    s.pending += 1
                elif j["status"] in ACTIVE_STATUSES:
                    if j["worker_id"]:
                        s.active_by_worker[j["worker_id"]] = s.active_by_worker.get(j["worker_id"], 0) + 1
                    else:
                        s.active_unassigned += 1
                elif j["status"] in TERMINAL_STATUSES and j["worker_id"] and j["worker_id"].startswith(label_prefix):
                    s.last_terminal_by_worker[j["worker_id"]] = max(s.last_terminal_by_worker.get(j["worker_id"], 0.0), j["history"][-1][1])
        return s


ROUTES = [(re.compile(r"^/api/worker/claim$"), "claim"), (re.compile(r"^/api/worker/jobs/([0-9a-f-]{36})/(status|heartbeat|complete|fail|files)$"), "job")]


def make_server(state: StubState, secret: str, host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):          # no request logging at all (no paths, no headers)
            pass

        def _send(self, code: int, obj: dict) -> None:
            b = json.dumps(obj).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

        def do_GET(self):  # noqa: N802 - liveness only (tunnel check); no job data, no auth needed
            return self._send(200, {"ok": True}) if self.path == "/healthz" else self._send(404, {"error": "not_found"})

        def do_POST(self):  # noqa: N802
            tok = self.headers.get("Authorization", "")
            if not hmac.compare_digest(tok.encode(), f"Bearer {secret}".encode()):
                return self._send(401, {"error": "unauthorized"})
            n = int(self.headers.get("Content-Length", 0))
            if n > MAX_BODY:
                return self._send(413, {"error": "too_large"})
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return self._send(400, {"error": "bad_json"})
            for rx, kind in ROUTES:
                m = rx.match(self.path)
                if not m:
                    continue
                if kind == "claim":
                    return self._send(200, {"job": state.claim(str(body.get("worker_id", "")), body.get("modes"), body.get("job_id"))})
                jid, action = m.group(1), m.group(2)
                ok = {"status": lambda: state.status(jid, str(body.get("status", ""))), "heartbeat": lambda: jid in state.jobs,
                      "complete": lambda: state.complete(jid, body), "fail": lambda: state.fail(jid, body), "files": lambda: state.add_file(jid, body)}[action]()
                return self._send(200 if ok else 404, {"ok": bool(ok)})
            self._send(404, {"error": "not_found"})

    return ThreadingHTTPServer((host, port), H)


def serve_in_thread(state: StubState, secret: str, host: str = "127.0.0.1", port: int = 0):
    srv = make_server(state, secret, host, port)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://{host}:{srv.server_address[1]}"
