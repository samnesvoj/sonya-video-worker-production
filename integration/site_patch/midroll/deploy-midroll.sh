#!/bin/bash
# «Мидролл» (2026-11-22): API /api/midroll/*, presets field, claim check; worker release; site Пресеты → Эффекты → Реклама.
# Run on 217.199.252.163 as root. Files staged in /root/midroll-stage. Usage: deploy-midroll.sh <worker main sha>
# Rollback one command: /root/rollback-midroll.sh
set -e
SHA="$1"; [ -n "$SHA" ] || { echo "usage: $0 <sha>"; exit 2; }
S=/root/midroll-stage; W=/var/www/sonya; A=/opt/sonya
# the site files must still be the ones the patch was made on (someone else may have deployed since)
(cd $W && md5sum presets/presets-app.js presets/presets-store.js presets/presets.css index.html app.js app-shell.js) > /tmp/mr-now.md5
diff <(awk '{print $1}' /tmp/mr-now.md5) $S/site-base.md5 >/dev/null || { echo "ABORT: live site files changed since staging"; cat /tmp/mr-now.md5; exit 3; }
cmp -s $A/scripts/prod_generation_api.py $S/api-base/prod_generation_api.py && cmp -s $A/scripts/presets_routes.py $S/api-base/presets_routes.py \
  || { echo "ABORT: live API files changed since staging"; exit 3; }
TS=$(date +%Y%m%d-%H%M%S); B=/root/backups/midroll-$TS; mkdir -p $B/presets
cp -a /etc/sonya-fleet/fleet.env $B/fleet.env
cp -p $A/scripts/prod_generation_api.py $A/scripts/presets_routes.py $B/
cp -p $W/presets/presets-app.js $W/presets/presets-store.js $W/presets/presets.css $B/presets/
for f in index.html app.js app-shell.js opencut.html streamer-mode.html; do cp -p $W/$f $B/; done
cp -p $W/miniapp/index.html $B/miniapp_index.html

# 1. API
install -m0644 $S/api/midroll_routes.py $S/api/presets_routes.py $S/api/prod_generation_api.py $A/scripts/
install -m0644 $S/api/test_midroll_api.py $A/tests/test_midroll_api.py
systemctl restart sonya-api; sleep 6; systemctl is-active sonya-api
curl -s -o /dev/null -w "api health %{http_code}\n" http://127.0.0.1:8000/api/health || true
# 2. worker release pointer (workers switch on their next start / resume)
sed -i "s/^SONYA_RELEASE_SHA=.*/SONYA_RELEASE_SHA=$SHA/" /etc/sonya-fleet/fleet.env
systemctl restart sonya-fleet; sleep 4; systemctl is-active sonya-fleet
# 3. site + Mini App
install -m0644 $S/web/presets/presets-app.js $S/web/presets/presets-store.js $S/web/presets/presets.css $W/presets/
install -m0644 $S/web/index.html $S/web/app.js $S/web/app-shell.js $W/
sed -i 's/app-shell\.js?v=[0-9a-z-]*/app-shell.js?v=20261122-midroll/' $W/opencut.html $W/streamer-mode.html
/root/build-miniapp.sh

cat > /root/rollback-midroll.sh <<EOR
#!/bin/bash
# rollback «Мидролл» (2026-11-22): API, worker release, site, Mini App. Uploaded ads stay in S3 (users/*/midroll/), unused.
set -e
install -m0644 $B/prod_generation_api.py $B/presets_routes.py /opt/sonya/scripts/
rm -f /opt/sonya/scripts/midroll_routes.py /opt/sonya/tests/test_midroll_api.py
systemctl restart sonya-api
cp -a $B/fleet.env /etc/sonya-fleet/fleet.env
systemctl restart sonya-fleet
install -m0644 $B/presets/* /var/www/sonya/presets/
for f in index.html app.js app-shell.js opencut.html streamer-mode.html; do install -m0644 $B/\$f /var/www/sonya/\$f; done
install -m0644 $B/miniapp_index.html /var/www/sonya/miniapp/index.html
sleep 5; systemctl is-active sonya-api sonya-fleet
echo "rolled back midroll: \$(grep ^SONYA_RELEASE_SHA /etc/sonya-fleet/fleet.env)"
EOR
chmod +x /root/rollback-midroll.sh
echo "deployed midroll; backup $B; rollback /root/rollback-midroll.sh"
