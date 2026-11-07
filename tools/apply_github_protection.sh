#!/usr/bin/env bash
# Applies .github/rulesets/*.json to the repo. NOT run automatically. Requires: GitHub Pro (or a public repo) - on a free private
# repo the API answers 403 "Upgrade to GitHub Pro". Run with --dry-run first. Needs admin rights (your own gh login on the MacBook).
set -Eeuo pipefail
REPO="${REPO:-samnesvoj/sonya-video-worker-production}"
for f in "$(dirname "$0")"/../.github/rulesets/*.json; do
  name="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['name'])" "$f")"
  if [[ "${1:-}" == "--dry-run" ]]; then echo "[dry-run] would create ruleset: $name"; continue; fi
  gh api -X POST "repos/$REPO/rulesets" --input "$f" --jq '"created: \(.name) (id \(.id))"'
done
