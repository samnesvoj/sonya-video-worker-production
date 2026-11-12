#!/usr/bin/env python3
"""
CPU-only simulation of the Vast start chain on a dev machine:
entrypoint.sh -> bootstrap (pinned SHA from this checkout as a local stand-in remote) -> run_worker.sh
-> readiness -> gpu_worker --poll against a stub backend -> SIGTERM.
NOT a GPU test: proves wiring, token hygiene, readiness gating and shutdown only. Needs a committed HEAD.
"""
import hashlib, http.server, json, os, shutil, signal, subprocess, sys, tempfile, threading, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "ghs_SMOKE_TOKEN_MUST_NEVER_APPEAR_0123456789"
SECRET = "SMOKE-WORKER-SECRET-MUST-NOT-APPEAR"
claims = []


class Stub(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        claims.append((self.path, json.loads(body or b"{}"), self.headers.get("Authorization")))
        out = b'{"job": null}'
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)

    def log_message(self, *a):
        pass


def main():
    srv = http.server.HTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    subprocess.check_call(["git", "config", "uploadpack.allowAnySHA1InWant", "true"], cwd=ROOT)
    tmp = Path(tempfile.mkdtemp(prefix="smoke-"))
    lock = hashlib.sha256((ROOT / "requirements/worker.lock.txt").read_bytes()).hexdigest()
    for line in (ROOT / "modes/virality/mode.yaml").read_text().splitlines():  # dummy model files: no S3 here
        if line.strip().startswith("local_path:"):
            f = tmp / "models" / line.split(":", 1)[1].strip()
            f.parent.mkdir(parents=True, exist_ok=True); f.write_bytes(b"dummy")
    env = {**os.environ, "SONYA_ENV_TEST": "1", "SONYA_TEST_REMOTE_URL": f"file://{ROOT}", "SONYA_RELEASE_SHA": sha,
           "SONYA_RELEASE_REPO": "o/r", "GITHUB_RELEASE_TOKEN": TOKEN, "SONYA_BASE_LOCK_SHA256": lock,
           "SONYA_RELEASES_DIR": str(tmp / "releases"), "SONYA_CURRENT_LINK": str(tmp / "current"),
           "SONYA_STATE_DIR": str(tmp / "run"), "SONYA_BOOTSTRAP_DIR": str(ROOT / "bootstrap"),
           "SONYA_SKIP_GPU_CHECK": "1", "SONYA_REQUIRE_ASR": "0", "SONYA_HEALTH_PORT": "18099", "TMPDIR": str(tmp / "work"),
           "MODELS_LOCAL_DIR": str(tmp / "models"), "WORKER_POLL_INTERVAL": "1",
           "BACKEND_API_URL": f"http://127.0.0.1:{srv.server_address[1]}", "WORKER_SECRET": SECRET,
           "S3_ENDPOINT_URL": "http://127.0.0.1:9", "S3_ACCESS_KEY_ID": "x", "S3_SECRET_ACCESS_KEY": "y",
           "S3_REGION": "r", "S3_BUCKET_NAME": "b", "MODELS_S3_BUCKET": "b", "ENABLED_MODES": "virality,trailer_film_breaker"}
    log = open(tmp / "out.log", "w")
    p = subprocess.Popen(["bash", str(ROOT / "bootstrap/entrypoint.sh")], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    ready = None
    for _ in range(90):
        time.sleep(1)
        try:
            ready = urllib.request.urlopen("http://127.0.0.1:18099/readyz", timeout=2).status
            break
        except Exception:
            if p.poll() is not None:
                break
    time.sleep(3)
    try:
        os.killpg(p.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        rc = p.wait(timeout=30)
    except subprocess.TimeoutExpired:
        rc = "TIMEOUT"
    text = (tmp / "out.log").read_text()
    r = {"ready_200": ready == 200, "claims_sent": bool(claims),
         "claim_modes": claims[0][1].get("modes") if claims else None,
         "bearer_used": bool(claims) and claims[0][2] == f"Bearer {SECRET}", "exit_code": rc,
         "ready_file_removed": not (tmp / "run/ready").exists(),
         "token_not_in_log": TOKEN not in text, "secret_not_in_log": SECRET not in text,
         "no_git_dir_in_release": not (tmp / "releases" / sha / ".git").exists()}
    print(json.dumps(r, indent=2))
    ok = r["ready_200"] and r["claim_modes"] == ["virality"] and r["bearer_used"] and r["token_not_in_log"] \
        and r["secret_not_in_log"] and r["ready_file_removed"] and rc in (143, 0)
    if not ok:
        print(text[-3000:])
    shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
