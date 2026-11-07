#!/usr/bin/env bash
# Runs INSIDE the freshly built base image (CI), as the image's non-root user, against a real pinned release fetched from the
# private GitHub repo with the short-lived Actions token. CPU only: it proves dependency resolution, entrypoint plumbing,
# bootstrap identity checks, Trailer lock, readiness gate and loopback health - NOT GPU behaviour.
set -Eeuo pipefail
fail() { echo "CHECK FAILED: $*" >&2; exit 1; }
ok()   { echo "ok  - $*"; }

# 1. dependency set resolves and is import-compatible (torch / CUDA build / NumPy 2.x / OpenCV 4 / MediaPipe / ultralytics / ctranslate2)
python3 - <<'PY'
import importlib, sys
import numpy as np, torch, cv2
print("python", sys.version.split()[0], "| numpy", np.__version__, "| torch", torch.__version__, "cuda-build", torch.version.cuda, "| cv2", cv2.__version__)
assert cv2.__version__.startswith("4."), "OpenCV must stay 4.x (Haar cascades)"
assert hasattr(cv2, "CascadeClassifier") and cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml").empty() is False
t = torch.from_numpy(np.ones((2, 3), dtype="float32")); assert t.sum().item() == 6 and t.numpy().shape == (2, 3)   # torch<->numpy interop
for m in ("mediapipe", "ultralytics", "faster_whisper", "ctranslate2", "scenedetect", "boto3", "fastapi", "yaml", "openai", "google.genai", "elevenlabs"):
    importlib.import_module(m)
print("imports ok")
PY
ok "dependencies import together (torch/numpy/opencv/mediapipe/ultralytics/faster-whisper)"
pip check >/dev/null 2>&1 && ok "pip check clean" || { pip check || true; fail "pip check reported conflicts"; }
command -v ffmpeg >/dev/null && ffmpeg -hide_banner -version | head -1

# 2. image hygiene: non-root, no SONYA source, no secrets in the environment
[ "$(id -u)" != "0" ] || fail "running as root"
ok "non-root user ($(id -un))"
if find / -xdev \( -name gpu_worker.py -o -name mode_policy.py -o -path "*/modes/virality" \) -not -path "/proc/*" 2>/dev/null | grep -q .; then fail "SONYA source found in the image"; fi
[ -z "$(ls -A /opt/sonya/releases 2>/dev/null)" ] || fail "releases dir not empty in a fresh image"
ok "no SONYA application code in the image"
if env | grep -iE '(token|secret|password|api[_-]?key)=' | grep -vE '^(SONYA_|GITHUB_RELEASE_TOKEN=)' ; then fail "secret-looking env present in image"; fi
ok "no secret-looking environment variables baked in"

# 3. bootstrap: fetch EXACTLY the pinned SHA from the private repo, verify, drop credentials
: "${SONYA_RELEASE_SHA:?}"; : "${GITHUB_RELEASE_TOKEN:?}"
out="$(python3 /opt/sonya-bootstrap/sonya_bootstrap.py 2>boot.err)" || { sed -E "s/${GITHUB_RELEASE_TOKEN}/***/g" boot.err; fail "bootstrap failed"; }
grep -q "$GITHUB_RELEASE_TOKEN" boot.err && fail "token leaked into bootstrap log"
REL="$(echo "$out" | tail -1)"
[ "$(basename "$REL")" = "$SONYA_RELEASE_SHA" ] || fail "release dir is not the pinned SHA"
[ ! -e "$REL/.git" ] || fail ".git present in release"
[ -f "$REL/deploy/vast/run_worker.sh" ] && [ -f "$REL/release/manifest.json" ] || fail "release incomplete"
ok "bootstrap fetched pinned SHA ${SONYA_RELEASE_SHA:0:12}, verified manifest + dependency lock, no .git, token not logged"
# a wrong SHA, a branch name and a missing token must all be refused
SONYA_RELEASE_SHA=main python3 /opt/sonya-bootstrap/sonya_bootstrap.py >/dev/null 2>&1 && fail "branch name accepted"
SONYA_RELEASE_SHA=0000000000000000000000000000000000000000 SONYA_RELEASES_DIR=/tmp/r2 SONYA_CURRENT_LINK=/tmp/c2 python3 /opt/sonya-bootstrap/sonya_bootstrap.py >/dev/null 2>&1 && fail "unknown SHA accepted"
env -u GITHUB_RELEASE_TOKEN SONYA_RELEASES_DIR=/tmp/r3 SONYA_CURRENT_LINK=/tmp/c3 python3 /opt/sonya-bootstrap/sonya_bootstrap.py >/dev/null 2>&1 && fail "download without token accepted"
ok "branch name / unknown SHA / missing token refused"
unset GITHUB_RELEASE_TOKEN

# 4. Trailer is locked inside the released code
cd "$REL"; export PYTHONPATH="$REL"
ENABLED_MODES=trailer_film_breaker,trailer_cinematic,virality python3 - <<'PY'
from scripts import mode_policy, mode_registry
assert mode_policy.enabled_modes() == ["virality"], mode_policy.enabled_modes()
for m in ("trailer_film_breaker", "trailer_cinematic"):
    try:
        mode_registry.get_runner(m); raise SystemExit("Trailer runner was returned!")
    except mode_policy.ModeDisabledError:
        pass
PY
ok "Trailer modes refused even when requested through ENABLED_MODES"

# 5. readiness gate refuses to open without env, then health reflects state (loopback only)
export SONYA_STATE_DIR=/tmp/state SONYA_READY_FILE=/tmp/state/ready SONYA_WORKER_PID_FILE=/tmp/state/worker.pid SONYA_SKIP_GPU_CHECK=1
mkdir -p /tmp/state
if python3 -m scripts.ops.readiness >ready.out 2>&1; then fail "readiness passed without worker env"; fi
grep -q "missing env" ready.out && [ ! -e "$SONYA_READY_FILE" ] || fail "readiness did not block"
ok "readiness blocks (no ready file) when the worker environment is incomplete"
python3 /opt/sonya-bootstrap/healthd.py & HP=$!; sleep 1
code() { curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:8099$1"; }
[ "$(code /readyz)" = 503 ] || fail "readyz should be 503"
echo $$ > "$SONYA_WORKER_PID_FILE"; [ "$(code /healthz)" = 200 ] && [ "$(code /readyz)" = 503 ] || fail "healthz/readyz state wrong"
echo ready > "$SONYA_READY_FILE"; [ "$(code /readyz)" = 200 ] || fail "readyz should be 200"
kill $HP
ok "healthz/readyz behave (503 -> alive -> ready)"
echo "ALL DOCKER CHECKS PASSED"
