# @sonya_group_ai: «ссылка → режим → подписка → нарезка» (2026-11-20)

Bot repo `samnesvoj/sonya-group-bot` (deployed in /opt/sonya-group-bot on 147.45.235.79). This PAT cannot push
there (403), so the two commits live here as `git format-patch` files and as a bundle on the bot server:
`/root/botlink-stage/sonya-group-bot-feat-cut-flow.bundle` (branch `feat/cut-flow` on top of 5c94ea9).

    cd sonya-group-bot && git am integration/telegram_bot/cut_flow/000*.patch     # or: git fetch <bundle> feat/cut-flow

- 0001 — the channel-subscription gate that has been live since 2026-11-15 but was never committed
- 0002 — the cut flow (bot/handlers/cut.py, bot/services/cut_{api,flow,runs}.py, media.py, cut_texts.py,
  tests/test_cut_flow.py, tools/dryrun/)

Site VPS part (API bridge `/api/telegram/bot/*` + site `/?tg=connect`): `integration/site_patch/bot_bridge/`.

Rollback, one command each: `ssh root@147.45.235.79 /root/rollback-botlink.sh` (bot) and
`ssh root@217.199.252.163 /root/rollback-botlink.sh` (API bridge + site). Off switches without rollback:
`BOT_CUT_FLOW=off` in /opt/sonya-group-bot/.env (+ restart), `SONYA_BOT_BRIDGE=off` in /opt/sonya/.env.local.
