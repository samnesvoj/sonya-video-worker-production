#!/usr/bin/env python3
"""healthd.py - loopback-only health endpoint (stdlib). /healthz: worker alive. /readyz: alive AND ready file."""
from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

STATE_DIR = Path(os.environ.get("SONYA_STATE_DIR", "/run/sonya"))
READY_FILE = Path(os.environ.get("SONYA_READY_FILE", str(STATE_DIR / "ready")))
PID_FILE = Path(os.environ.get("SONYA_WORKER_PID_FILE", str(STATE_DIR / "worker.pid")))


def worker_alive() -> bool:
    try:
        os.kill(int(PID_FILE.read_text().strip()), 0)
        return True
    except (OSError, ValueError):
        return False


def status_for(path: str):
    alive = worker_alive()
    if path == "/healthz":
        return (200, "alive") if alive else (503, "worker_not_running")
    if path == "/readyz":
        if not alive:
            return 503, "worker_not_running"
        return (200, "ready") if READY_FILE.is_file() else (503, "not_ready")
    return 404, "not_found"


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        code, state = status_for(self.path.split("?", 1)[0])
        body = json.dumps({"status": state}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_a):
        pass


def main() -> int:
    HTTPServer(("127.0.0.1", int(os.environ.get("SONYA_HEALTH_PORT", "8099"))), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
