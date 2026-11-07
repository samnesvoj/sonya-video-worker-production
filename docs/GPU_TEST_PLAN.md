# Final GPU test plan (NOT started; needs your explicit OK)
Cost: read-only Vast search (2026-11-07; verified, reliability>0.98, CUDA≥12.4, disk≥80 GB; 64 offers) shows RTX 3090 24 GB from ≈ $0.17/h; 4090/A5000/L4 roughly $0.3–0.6/h. 1–1.5 h ≈ $0.3–1.0 plus traffic. Suggested cap $3, auto-destroy at 90 min.
Blockers: (1) GitHub App created + token minting wired into the dispatcher; (2) base image built in Actions and its digest; (3) `release/models.lock.json` filled; (4) staging/stub backend with its own WORKER_SECRET (not the real user queue); (5) Template created per VAST_TEMPLATE.md.
Steps: 1 instance → bootstrap OK, readiness OK (GPU/CUDA/torch.cuda/NVENC/models) → negatives (wrong SHA refused; Trailer job → `mode_disabled`; SIGTERM requeues) → one short real clip per enabled mode (virality, beep), compare to Mac mini results (ffprobe, contact sheet) → record VRAM/RAM/time/cost, NVENC vs CPU → log leak grep → destroy instance. Opt-in modes only afterwards, one by one.

## Additional checks for idle-stop and accounting (same paid session)
- `SONYA_FLEET_DRY_RUN=false`, one test job: confirm create → running → claim → completed → **stop at ≈60 s** → `exited`; then a second test job → **RESUME** time and whether the GPU is still available; then wait out retention → DESTROY and confirm the instance disappears from `GET /api/v1/instances/`.
- Capture the real shapes of `PUT /asks/{id}/`, `GET /api/v1/instances/`, `GET /api/v0/charges/` and compare with `dispatcher/vast_api.py` and `accounting/vast_billing.py` (they are written from the CLI source, not from live responses).
- Kill test: `vastai stop instance` while a job is `mode_running` → the job must return to `queued` with `WORKER_SHUTDOWN` and complete after RESUME/CREATE.
- Measure: real GPU-minutes per video, model size/download time, stopped-disk billing rate, SIGTERM grace.

---
# Первый GPU-тест — финальный план (v2, 2026-11-07). НЕ запускался; нужно ваше «да».

**Предпосылки (блокеры, см. STATUS):** (1) GitHub App создан по `docs/GITHUB_APP_SETUP.md`; (2) базовый образ опубликован (`publish-base-image`, вручную) → digest; (3) `release/models.lock.json` заполнен (`tools/make_models_lock.py` на машине с доступом к S3) — иначе readiness не откроется; (4) тестовый backend/окно, не живая очередь пользователей; (5) на аккаунте Vast лежит не более $5.

**Стоимость (реальные цены 2026-11-07, курс ЦБ 82,1386; модель `docs/COSTS.md`):**
| GPU | 90 мин GPU | диск 80 ГБ (90 мин) | трафик (≈5 ГБ моделей + клипы) | итого | в рублях |
|---|---|---|---|---|---|
| RTX 3090 (≈ $0.20/ч) | $0.30 | $0.03 | ≈ $0.05 | **≈ $0.38** | **≈ 31 ₽** |
| RTX 4090 (≈ $0.43/ч) | $0.64 | $0.03 | ≈ $0.05 | **≈ $0.72** | **≈ 59 ₽** |
**Безопасный лимит: $3 (≈ 246 ₽) на весь тест**; жёсткий потолок аккаунта — пополнение не более $5 (≈ 411 ₽). Защиты: `SONYA_MAX_DPH=0.60`, `max_instances=1`, `max_creates_per_hour=6`, автоудаление не позднее 90 мин (`vastai destroy instance` вручную как страховка), остановленный диск уничтожается через 30 мин.

**Порядок:** 1) `SONYA_FLEET_DRY_RUN=true` на staging-dispatcher — сверить решения с очередью; 2) `DRY_RUN=false`, один тестовый клип `virality`: create → bootstrap (проверить в логе SHA) → readiness → claim → completed → **stop ≈ 60 с** → второй клип → **RESUME** (замерить время, доступность GPU) → ждать retention → DESTROY; 3) `beep` одним клипом; 4) негативы: неверный SHA отказ, Trailer-задание → `mode_disabled`, `vastai stop instance` в момент `mode_running` → задание вернулось в `queued` и завершилось после RESUME/CREATE; 5) записать: минуты GPU на видео, VRAM/RAM, размер/время моделей, фактическое списание Vast (`/api/v0/charges/`) и сверить с расчётом бота; 6) убедиться, что `vastai show instances` пуст, логи без токенов (grep).
**Остановить немедленно, если:** инстанс не стал `ready` за 15 мин, списание превысило $3, неожиданный режим/задание, любой секрет в логе.
