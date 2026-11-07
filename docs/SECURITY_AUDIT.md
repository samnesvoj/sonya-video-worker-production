# Security audit (legend: ✅ done+tested · 🔎 checked by grep/code · ⚠️ partial · ❌ not done)
| # | Item | Status | Notes |
|---|---|---|---|
| 1 | Secrets in Git/archive/Docker | 🔎 | Pattern scan of tree and full history before first push (see STATUS); CI runs gitleaks on full history. Docker layers not scanned (no Docker on this Mac). `.dockerignore` is deny-all + 4 files. |
| 2 | Keys/tokens/source in startup logs | ✅ | Token redaction (raw/base64) tested; smoke: GitHub token and WORKER_SECRET absent from logs; entrypoint prints no env (the archived entrypoint printed BACKEND_API_URL). |
| 3 | Videos/transcripts/URLs/PII in logs | ⚠️ | Worker logs job_id, mode, filenames, S3 keys (contain user_id). Per-mode loggers not audited line by line. |
| 4 | Worker↔server auth | ⚠️ | Shared static `WORKER_SECRET` Bearer (as production). Per-instance token needs a server change (not done). |
| 5 | Ports | ✅ | none inbound; health on 127.0.0.1 only (tested). |
| 6 | Foreign requests to Worker API | ✅ | worker serves no API. |
| 7 | File rights / user | ✅ by Dockerfile | uid 10001, umask 077; not verified in a running container. |
| 8 | Dependencies | ✅/⚠️ | hashed lock; pip-audit: no known vulnerabilities; torch/torchvision (+cu124) not covered by pip-audit. torch 2.5.1 predates a `torch.load` RCE fix in 2.6 as far as I recall — verify the advisory; models only come from your S3; upgrade after GPU validation. |
| 9 | Injection / traversal / ffmpeg args | 🔎 | no `shell=True`, `os.system`, `eval`, `pickle.load` in scripts/modes; ffmpeg called with argument lists; per-job `mkdtemp`. No fuzzing of params→ffmpeg. |
| 10 | Temp retention | ✅ | per-job `rmtree` in `finally`; SIGTERM path; `sonya_*` wiped on start and shutdown. kill -9 untested (wiped at next start). |
| 11 | GitHub credentials | ✅ | token only in the child git env; never argv/URL/file/.git/config; unset after bootstrap; App key only on trusted hosts. |
| 12 | Failures/restarts | ✅ | idempotent bootstrap, cached verified release, fail-closed, ready-gate, requeue on SIGTERM. |
Host-owner threat: sees code, env, memory, videos; no promise otherwise. Open: per-instance API token, token broker, signed S3 URLs instead of S3 keys on host, trivy scan after first build, egress restriction.

---
# Дополнение: этап «автоостановка + учёт расходов» (2026-11-07)

| Проверка | Результат |
|---|---|
| **Логи/события без секретов** | ✅ тесты: события флота, `redacted(payload)` и ключ Vast (`repr`, ошибки) не содержат GitHub-токен, `WORKER_SECRET`, S3-ключи, ключ Vast; провод между ботом и сервером не содержит Telegram-токена. Транскрипты/видео в логах режимов построчно **не** аудировались. |
| **Долгосрочные секреты только в центре** | ✅ по дизайну: ключ GitHub App, ключ Vast (файл `chmod 600`, проверяется), токен Telegram (bot VPS) — не на Vast. На инстанс уходят: токен GitHub ≤1 ч (read-only, 1 репо), `WORKER_SECRET`, S3-ключи (как в production; целевое улучшение — подписанные URL/токен на задачу). |
| **Порты** | ✅ у воркера входящих нет, health 127.0.0.1; бот по-прежнему только исходящий; новый API бота — тот же паттерн (опрос, HMAC, nonce, окно 120 с; тесты на подпись/повтор/перекос). |
| **Задания не теряются при остановке** | ✅ fenced stop, откат при новом задании, SIGTERM→requeue, восстановление после падения между шагами (тесты `tests/fleet`). Не проверено на реальном Vast (`kill`/`stop` контейнера, grace-период SIGTERM). |
| **Trailer не попадёт в Worker** | ✅ четыре слоя: `HARD_DISABLED_MODES` в воркере; `FleetConfig` отказывается от `trailer*` в `SONYA_POOL_MODES`; create-payload несёт `ENABLED_MODES=virality,beep`; bootstrap отвергает релиз без запрета Trailer. |
| **Старый пул не нарушен** | ✅ патч dispatcher: без `SONYA_FLEET_ENABLED=true` SQL и поведение прежние; прогон `test_gpu_concurrency`, `test_gpu_instance_lifecycle`, `test_gpu_instance_cleanup_durability`, `test_job_store_claim` на копии production `bd8c968`: до патча и после — одинаково `33 passed, 4 errors` (4 errors — тесты с БД, существуют и без патча). Не применялось на VPS. |
| **Уязвимости зависимостей, PyTorch** | ⚠️ `pip-audit`: чисто, но **не видит `torch==2.5.1+cu124`**. OSV по `torch 2.5.1`: **23 advisory**, среди них RCE при загрузке недоверенных чекпойнтов (`torch.load weights_only`, GHSA-53q9-r3pm-6pq6, исправлено в 2.6.0; ещё RCE через чекпойнт GHSA-63cw-57p8-fm3p, исправлено в 2.10.0) и ряд memory-corruption. `torchvision 0.20.1`: 0. Для рассчитанного кандидата `torch 2.10.0+cu126` (lock в `requirements/candidates/`): 2 advisory (jit.script memory corruption — исправление 2.13.0; один без исправления). **Принятый риск, письменно** в `requirements/osv-accepted.json`: эксплуатация требует недоверенного чекпойнта; модели берутся только из приватного бакета и фиксируются sha256 в `release/models.lock.json` (`SONYA_REQUIRE_MODEL_LOCK=1`). Любой новый advisory ломает CI (`tools/osv_check.py`). План: после GPU-теста перейти на кандидат (нужен драйвер хоста с CUDA ≥ 12.6, фильтр `cuda_max_good` в `dispatcher/launcher.py` поднять с 12.4 до 12.6; валидация качества на GPU обязательна). |
| **Gitleaks** | ✅ локально `gitleaks 8.30.1`: история + рабочее дерево — чисто. Единственное срабатывание (generic-api-key, `tests/test_llm_timeweb.py:12`) — тестовая константа-плейсхолдер (содержит слово SECRET; одно использование); узко разрешена в `.gitleaks.toml` (путь + шаблон значения). |
| **CI на GitHub** | ❌→⏳ **Первый запуск CI (коммит f97ee9e) завершился красным:** `test` ✅, `audit-deps` ✅, `secrets-scan` ❌ (то же ложное срабатывание — исправлено `.gitleaks.toml`), `docker-build` ❌ (`torch==2.5.1+cu124` не найден: lock не содержит URL индекса PyTorch — исправлено `--extra-index-url` в Dockerfile). Исправления в этой ветке, **на GitHub не перепроверены** (push не выполнялся). |
