# Статус (2026-11-07, ветка `feat/idle-stop-and-cost-accounting`)

## Готово и проверено локально (CPU, Apple Silicon, без Docker/GPU/Vast)
- Воркер: Trailer запрещён в коде; режимы по умолчанию `virality`,`beep`; SIGTERM-requeue; pinned-SHA bootstrap; readiness; loopback health (см. прежние пункты).
- **Dispatcher/Fleet:** автоостановка через `IDLE_TIMEOUT_SECONDS=60`, fenced stop, RESUME/CREATE/DESTROY, ограничители расходов, чужие инстансы не трогаются — 29 тестов с поддельным Vast.
- **Учёт:** Decimal-журнал, курс ЦБ РФ, подтверждённое/рассчитанное/оценка раздельно, сверка с дельта-корректировками, отчёты 00:05 / 1-е 00:10 МСК с догонкой пропусков, outbox + подписанный API — 22 теста.
- **Бот:** модуль + патч для существующего бота (применяется чисто; 134 теста бота зелёные до/после); интеграция проверена в процессе с поддельным Telegram (повторы, потерянный ack, частичная доставка).
- **Dispatcher-патч** для production-`gpu_dispatcher.py`/`prod_job_store.py` (применяется чисто к `bd8c968`; тесты production без регрессий).
- Тесты репозитория: `pytest -q` — всё зелёное (число в итоговом отчёте). Gitleaks локально чисто. `pip-audit` чисто; OSV по torch — см. SECURITY_AUDIT.

## Только симуляция
`tools/simulate.py` (поддельный Vast, синтетические данные) — примеры отчётов в `docs/SAMPLE_REPORTS.md`.

## НЕ проверено / требует внешних действий
- **GitHub CI после исправлений** (первый запуск был красным; push не делался).
- Сборка Docker-образа и установка lock (нужен Docker/CI), совместимость torch/numpy 2.x/mediapipe.
- Реальный Vast: формат ответов `PUT /asks`, `GET /api/v1/instances/`, `/api/v0/charges/` (нет ключа Vast на этом Mac), поведение SIGTERM при stop, время RESUME/CREATE, ставка диска остановленного.
- GPU: качество и время `virality`/`beep`, размер моделей, NVENC.
- GitHub App, `release/models.lock.json` (пуст → readiness с `SONYA_REQUIRE_MODEL_LOCK=1` не даст воркеру стартовать), права/секреты на VPS, подключение API-роутера бухгалтерии, включение патчей.
- Остальные режимы (`stories`,`educational`,`streamer`,`sonya_gen`) выключены до подтверждения.
