#!/usr/bin/env bash
# macmini_setup.sh - one-time, idempotent setup of the Mac mini as a SONYA mode-development station.
# Run it ON THE MAC MINI. It never touches existing projects, global git config, or other GitHub logins.
#   bash macmini_setup.sh --dry-run     show what would happen
#   bash macmini_setup.sh               do it (asks for the token once, hidden input)
# Creates:  ~/SONYA-dev/sonya-video-worker-production   (partial clone: history only, blobs on demand -> small transfers)
#           ~/SONYA-models                               (models stay OUTSIDE git; use tools/link_models.py to link existing ones)
#           ~/SONYA-controls                             (control videos stay OUTSIDE git)
# Keychain item: service "sonya-video-worker-github" (fine-grained token, ONE repository). Used only through a repo-scoped credential helper.
set -Eeuo pipefail
REPO="samnesvoj/sonya-video-worker-production"
DEV="$HOME/SONYA-dev"; DIR="$DEV/sonya-video-worker-production"; SERVICE="sonya-video-worker-github"; ACCT="${GITHUB_USER:-samnesvoj}"
DRY=0; [[ "${1:-}" == "--dry-run" ]] && DRY=1
run() { if (( DRY )); then echo "[dry-run] $*"; else eval "$@"; fi; }
say() { printf '\n== %s\n' "$*"; }

say "1. prerequisites"
for t in git python3 ffmpeg ffprobe security; do command -v "$t" >/dev/null || { echo "missing: $t"; (( DRY )) || exit 1; }; done
python3 - <<'PY' || { echo "python >= 3.11 required"; exit 1; }
import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)
PY

say "2. token in the macOS Keychain (never printed, never in argv, never in a file)"
if security find-generic-password -s "$SERVICE" -a "$ACCT" >/dev/null 2>&1; then
  echo "Keychain item '$SERVICE' already exists - keeping it."
else
  cat <<TXT
Create a FINE-GRAINED token on github.com (Settings > Developer settings > Personal access tokens > Fine-grained tokens):
  Resource owner: $ACCT | Repository access: ONLY $REPO | Expiration: 30-90 days
  Permissions: Contents = Read and write, Pull requests = Read and write, Actions = Read, Metadata = Read. Nothing else.
TXT
  if (( DRY )); then echo "[dry-run] security add-generic-password -s $SERVICE -a $ACCT -w   (prompts, hidden)"; else
    security add-generic-password -U -s "$SERVICE" -a "$ACCT" -w   # -w with no value => prompts without echo
  fi
fi

say "3. partial clone (small transfers) into a NEW folder"
if [[ -e "$DIR" && -n "$(ls -A "$DIR" 2>/dev/null)" ]]; then echo "$DIR exists and is not empty - leaving it alone."; else
  run "mkdir -p '$DEV'"
  # the credential helper is injected for this one command only (-c), so global git config is untouched
  run "git -c credential.helper= -c 'credential.https://github.com.helper=!f(){ echo username=$ACCT; echo password=\$(security find-generic-password -s $SERVICE -a $ACCT -w); }; f' clone --filter=blob:none --no-tags --single-branch --branch main https://github.com/$REPO.git '$DIR'"
fi

say "4. repo-local git settings (this clone only)"
cd "$DIR" 2>/dev/null || { (( DRY )) || exit 1; }
cfg() { run "git -C '$DIR' config --local $*"; }
cfg "credential.helper ''"
cfg "credential.https://github.com/$REPO.helper '!f(){ echo username=$ACCT; echo password=\$(security find-generic-password -s $SERVICE -a $ACCT -w); }; f'"
cfg "user.name 'SONYA Mac mini'" ; cfg "user.email 'macmini@sonya.local'"
cfg "fetch.prune true" ; cfg "fetch.negotiationAlgorithm skipping" ; cfg "pack.compression 9" ; cfg "http.postBuffer 524288000" ; cfg "core.compression 9"
cfg "push.default current" ; cfg "pull.ff only" ; cfg "gc.auto 0"
cfg "core.hooksPath tools/git-hooks"
cfg "remote.origin.tagOpt --no-tags"

say "5. folders outside git"
run "mkdir -p '$HOME/SONYA-models' '$HOME/SONYA-controls'"
echo "Add to your shell profile for this project:  export MODELS_LOCAL_DIR=\$HOME/SONYA-models"

say "6. checks"
run "python3 -m venv '$DIR/.venv' && '$DIR/.venv/bin/pip' install -q pytest pyyaml requests pydantic fastapi boto3 numpy pillow python-dotenv httpx scenedetect cryptography && '$DIR/.venv/bin/pip' install -q --no-deps opencv-python==4.14.0.94 opencv-contrib-python==4.14.0.94 opencv-python-headless==4.14.0.94"
run "'$DIR/.venv/bin/python' -m pytest -q -p no:cacheprovider '$DIR/tests/test_mode_policy.py' '$DIR/tests/test_api_contract.py'"
echo; echo "Done. Next: docs/dev/MAC_MINI_SETUP.md (link existing models, register control videos), docs/dev/DEV_WORKFLOW.md"
