#!/bin/bash
# SONYA self-check beacon on the Mac mini (launchd group.sonya.healthbeacon, every 10 min, background, no windows).
# 1) disk of the data volume -> VPS /var/lib/sonya-health/beacon/mac.json (read by sonya-health.timer there)
# 2) fallback: if the accounting bot itself is down, the VPS lists undelivered alerts in fallback.json -> sent ONCE
#    via ~/.config/pcsony/send.sh (each id remembered in sent.txt)
H="$HOME/SONYA/health"; VPS=root@217.199.252.163
SSH="/usr/bin/ssh -o BatchMode=yes -o ConnectTimeout=15"
PY="$HOME/SONYA/ytrelay/venv/bin/python"; [ -x "$PY" ] || PY=/usr/bin/python3
/usr/sbin/taskpolicy -b "$PY" "$H/health_beacon.py" --disk /System/Volumes/Data \
  | $SSH $VPS 'cat > /var/lib/sonya-health/beacon/mac.tmp && mv /var/lib/sonya-health/beacon/mac.tmp /var/lib/sonya-health/beacon/mac.json' || echo "$(date) beacon push failed"
$SSH $VPS 'cat /var/lib/sonya-health/fallback.json 2>/dev/null' > "$H/fallback.json" 2>/dev/null || exit 0
touch "$H/sent.txt"
"$PY" - "$H/fallback.json" "$H/sent.txt" <<'PYEOF' | while IFS= read -r line; do
import json, sys, time
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
if time.time() - float(d.get("ts") or 0) > 3600:
    sys.exit(0)
seen = set(open(sys.argv[2]).read().split())
for it in d.get("items") or []:
    if it.get("id") and it["id"] not in seen:
        print(json.dumps(it, ensure_ascii=False))
PYEOF
  id=$("$PY" -c 'import json,sys;print(json.loads(sys.argv[1])["id"])' "$line"); txt=$("$PY" -c 'import json,sys;print(json.loads(sys.argv[1])["text"])' "$line")
  "$HOME/.config/pcsony/send.sh" "⚠️ (резервный канал) $txt" >/dev/null && echo "$id" >> "$H/sent.txt"
done
