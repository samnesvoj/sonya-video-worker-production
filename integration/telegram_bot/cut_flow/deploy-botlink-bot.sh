#!/bin/bash
# @sonya_group_ai «ссылка → режим → подписка → нарезка» (2026-11-20), bot part. Run on 147.45.235.79 as root.
# Files staged in /root/botlink-stage/files. Rollback one command: /root/rollback-botlink.sh
set -e
S=/root/botlink-stage/files; D=/opt/sonya-group-bot
TS=$(date +%Y%m%d-%H%M%S); B=/root/backups/botlink-$TS; mkdir -p $B
tar -C $D --exclude=.venv --exclude=logs --exclude=__pycache__ -czf $B/bot.tgz .
NEW="bot/handlers/cut.py bot/cut_texts.py bot/services/cut_api.py bot/services/cut_flow.py bot/services/cut_runs.py bot/services/media.py tests/test_cut_flow.py"
for f in $NEW; do [ -e $D/$f ] && { echo "refuse: $f already exists"; exit 1; }; done
echo "0eb2976868c3d9d25db27b252a046c1e  $D/main.py
ca9bb5324835639ba127d624a6a6d165  $D/bot/handlers/__init__.py
baa020ce096449dc99b7070248bf82e4  $D/bot/handlers/start.py
ca0babfd46b5375636523edc814b0320  $D/bot/handlers/payment.py" | md5sum -c --quiet
(cd $S && find . -type f) | while read f; do install -o sonya -g sonya -m0644 $S/$f $D/$f; done
systemctl restart sonya-group-bot; sleep 6; systemctl is-active sonya-group-bot
journalctl -u sonya-group-bot --since "-20 sec" --no-pager | grep -E "Cut flow|Start polling|Error|Traceback" | sed 's/token=[^ )]*/token=***/' | tail -5
cat > /root/rollback-botlink.sh <<EOR
#!/bin/bash
# rollback bot-link (2026-11-20), bot server: the bot exactly as before (code from $B/bot.tgz; data/ untouched).
set -e
cd $D && rm -f $NEW
tar -C $D -xzf $B/bot.tgz --exclude=./data --exclude=./.env ./main.py ./bot ./tests
chown -R sonya:sonya $D/bot $D/main.py $D/tests
systemctl restart sonya-group-bot; sleep 4; systemctl is-active sonya-group-bot
echo "rolled back bot-link (bot server)"
EOR
chmod +x /root/rollback-botlink.sh
echo "deployed bot-link (bot); backup $B; rollback /root/rollback-botlink.sh"
