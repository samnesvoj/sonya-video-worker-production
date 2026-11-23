# «Тест-драйв» (test_all) + sonya.group/invite — as deployed 2026-11-23 (VPS 217.199.252.163)

**Test plan (backend only, NOT on sale).** `api/` = the changed /opt/sonya files: pricing.TEST_PLAN (10 processings in
each of Вирал / Хуки / Сторителлинг / Образовательный / Стример / BEEP, 14 days, source ≤60 min, price env
`SONYA_TEST_PLAN_PRICE`, default 1490), migration 026 (plan_mode `test:<slot>`, unique (payment_id, plan_mode)),
payment_store._activate_test_plan (6 user_subscriptions rows), entitlements.test_slot (params.clipType picks the
Вирал/Хуки/Сторителлинг counter), quota_view kind "test", /api/features `test_plan`. Checkout of `test_all` is
refused (403 TEST_PLAN_DISABLED) until `SONYA_TEST_PLAN=on` in /opt/sonya/.env.local + restart sonya-api.
One purchase per account (409 test_plan_used). Cost model: `tools/calc.py`.
Rollback: `/root/rollback-testplan.sh` (DB constraint revert only when no test:* rows exist).

**Invite page.** `web/invite/` → /var/www/sonya/invite/ (static, light theme, real clip frames from admin/loadtest
jobs, headless screenshots of Пресеты→Реклама and Discover). `web/auth.js` adds `?signup=1` (guest → registration
modal); index.html + miniapp/index.html load `auth.js?v=20261123-invite`. OG image rendered by `tools/og.js`.
The «Тест-драйв» card reads /api/features and switches to «Выбрать» by itself once the flag is on.
Rollback: `/root/rollback-invite.sh`.
