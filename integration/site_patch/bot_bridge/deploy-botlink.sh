#!/bin/bash
# @sonya_group_ai «ссылка → режим → подписка → нарезка» (2026-11-20), site VPS part: API bot bridge + site ?tg=connect.
# Run on 217.199.252.163 as root; files staged in /root/botlink-stage. Rollback one command: /root/rollback-botlink.sh
set -e
S=/root/botlink-stage; W=/var/www/sonya; A=/opt/sonya
TS=$(date +%Y%m%d-%H%M%S); B=/root/backups/botlink-$TS; mkdir -p $B
cp -p $A/scripts/prod_generation_api.py $B/
cp -p $W/auth.js $W/index.html $B/; cp -p $W/miniapp/index.html $B/miniapp_index.html
# refuse if the live files changed since they were pulled for this release
echo "$1  $A/scripts/prod_generation_api.py" | md5sum -c --quiet
echo "$2  $W/auth.js" | md5sum -c --quiet
install -m0644 $S/api/bot_bridge_routes.py $A/scripts/bot_bridge_routes.py
install -m0644 $S/api/prod_generation_api.py $A/scripts/prod_generation_api.py
install -m0644 $S/api/test_bot_bridge.py $A/tests/test_bot_bridge.py
systemctl restart sonya-api; sleep 5; systemctl is-active sonya-api
curl -sf -o /dev/null http://127.0.0.1:8000/api/health && echo api health ok || { echo "API DOWN - rolling back"; install -m0644 $B/prod_generation_api.py $A/scripts/prod_generation_api.py; rm -f $A/scripts/bot_bridge_routes.py; systemctl restart sonya-api; exit 1; }
install -m0644 $S/web/auth.js $W/auth.js
sed -i 's#/auth\.js?v=[0-9a-z-]*#/auth.js?v=20261120-tgconnect#' $W/index.html
/root/build-miniapp.sh
cat > /root/rollback-botlink.sh <<EOR
#!/bin/bash
# rollback bot-link (2026-11-20), site VPS: API bridge off + site auth.js/index.html/Mini App as before.
set -e
install -m0644 $B/prod_generation_api.py /opt/sonya/scripts/prod_generation_api.py
rm -f /opt/sonya/scripts/bot_bridge_routes.py /opt/sonya/tests/test_bot_bridge.py
systemctl restart sonya-api
install -m0644 $B/auth.js /var/www/sonya/auth.js
install -m0644 $B/index.html /var/www/sonya/index.html
install -m0644 $B/miniapp_index.html /var/www/sonya/miniapp/index.html
sleep 4; systemctl is-active sonya-api
echo "rolled back bot-link (site VPS)"
EOR
chmod +x /root/rollback-botlink.sh
echo "deployed bot-link (site VPS); backup $B; rollback /root/rollback-botlink.sh"
