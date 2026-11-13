# SONYA Gen (images) - admin only (2026-11-13)
- `gen_admin_api_patch.py` - idempotent patcher for `/opt/sonya/scripts/prod_generation_api.py`: `GET /api/gen/access`, `POST /api/gen/image` (admin = `users.plan_type == 'admin'`, the existing mechanism; non-admin -> 403 `{"error":"coming_soon","message":"скоро"}`; no credits/quota/billing touched, `bypass_quota=True`), generic upload endpoint refuses `sonya_gen`, `/result-url` lists png/jpg.
- `gen/gen-soon.js` + `gen-soon.css.add` (append to `/var/www/sonya/gen/gen-soon.css`) - the Gen window: image tab live for admins, «Скоро» for others; Gen Video stays «Скоро». Bump `?v=3` -> `?v=4` in `app-shell.js`.
- Rollback: restore `*.bak-pre-gen-*` copies (see GEN_VAST_REPORT.md) and `systemctl restart sonya-api`.
