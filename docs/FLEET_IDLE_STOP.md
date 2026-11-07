# Автоостановка GPU: центральный dispatcher (IDLE_TIMEOUT_SECONDS=60)

Решение об остановке принимает **dispatcher на VPS** (`dispatcher/fleet.py`), а не Worker. Worker только берёт задания и при SIGTERM возвращает активное в очередь.

## Схема состояний
```
            demand, нет stopped того же релиза                      Vast: running
 (нет) ──CREATE──▶ creating ───────────────────────────────────────────────▶ running ◀──────────────┐
                      │ >create_timeout (900 с): DESTROY + бан машины               │                  │ Vast: running
                      ▼                                                             │ idle ≥ 60 с      │
                  destroying                                                        ▼ (после последнего terminal-задания)
                      ▲                                                       stop_pending ──новое задание/active──▶ (откат в running, stop_cancelled)
                      │ retention (30 мин) / старый релиз / resume_failures          │ fenced re-check очереди → Vast PUT state=stopped
                      │                                                              ▼
                   stopped ◀──────────── Vast: exited ──────────────────────── stopping
                      │ demand (тот же release_sha)
                      ▼
                   starting ──Vast: running──▶ running
                      │ >resume_timeout (90 с, 'scheduling'): отмена (stop), failure+1; после max_resume_failures: DESTROY → CREATE
```
Таймер 60 с идёт от **последнего завершённого (completed/failed/cancelled) задания этого воркера**; пока есть `queued` задания пула, активные задания (включая `uploading_result`) или активные задания без `worker_id` — остановки нет. Свежий инстанс, не взявший работу, получает `boot_grace_seconds` (600 с), чтобы не быть остановленным до готовности.

## Защита от гонок и потери заданий
1. **CAS-переходы** `(state, generation)` в SQLite (`BEGIN IMMEDIATE`) — повторный/параллельный tick не сделает двойной stop/resume.
2. **Fenced stop:** `running → stop_pending` (CAS) → *свежий* снимок очереди (pending / active по воркеру / unassigned) → только потом вызов Vast. Любое появившееся задание — откат (`stop_cancelled`).
3. Остаточное окно (задание взято между проверкой и фактической остановкой): Vast шлёт SIGTERM → Worker ставит `WORKER_SHUTDOWN` (retry) и возвращает задание в очередь → dispatcher видит `queued` → RESUME/CREATE. Плюс серверный stale-requeue по heartbeat.
4. Падение процесса между CAS и вызовом Vast: `stop_pending` дочищается следующим tick (с тем же re-check).
5. Идемпотентность вызовов Vast (повтор `state=stopped` безопасен), `stop_retry` через 120 с, при зависании — destroy.
6. Ограничители расходов: `max_instances=1`, `max_creates_per_hour=6`, backoff 60 с после ошибки create, `max_running_hours=24`, цена оффера ≤ `SONYA_MAX_DPH`, образ только по `@sha256`, релиз только полный SHA.
7. Остановленный инстанс **того же релиза** — RESUME; другого релиза никогда (уничтожается): работающие/остановленные не меняют код неожиданно.
8. Чужие инстансы (без префикса `sonya-pool-`) не трогаются. Неизвестные наши — «усыновляются» и подчиняются тем же правилам (нет утечек).
9. **Старый пул:** режимы пула (`SONYA_POOL_MODES`, по умолчанию `virality,beep`) скрыты от старого per-job dispatcher патчем (`integration/dispatcher_patch/gpu_dispatcher-fleet.patch`); без `SONYA_FLEET_ENABLED=true` поведение production не меняется ни на байт (SQL без `exclude_modes` прежний, проверено прогоном тестов production).

## Что подтверждает документация Vast (docs.vast.ai, 2026-11-07)
| Вопрос | Ответ |
|---|---|
| Остановленный инстанс тарифицируется по GPU? | **Нет**, «detached from its GPU(s)». |
| Диск остановленного? | **Да, каждую секунду**; ставка у остановленных «может быть выше» (коэффициент не указан). |
| «Loading» (скачивание образа)? | Не тарифицируется. |
| Трафик? | По использованию, в любом состоянии. |
| Ограничения возобновления | `Scheduling` — GPU может быть занят другим арендатором; ожидание может быть бесконечным; Vast советует создать новый инстанс. Истёкший контракт не перезапускается. |
| Нулевой баланс | инстансы останавливаются, диск продолжает списываться; без карты данные удаляются. |

## STOP/RESUME против DESTROY/CREATE
| | STOP + RESUME | DESTROY + CREATE |
|---|---|---|
| Старт | ≈ 1–2 мин (релиз и модели на диске) | ≈ 5–8 мин billed (bootstrap + модели) + pull образа (не billed) |
| Пока ждём | **платим диск** (80 ГБ ≈ 44 ₽/сут на RTX 3090/4090) | 0 ₽ |
| Надёжность | **GPU может оказаться занят** → fallback | всегда есть новые офферы; можно сменить хост |
| Свежесть кода | остаётся старый релиз → уничтожаем при смене SHA | всегда актуальный pinned SHA |
| Безубыточность | выгоден только при ≳ 11–17 пробуждений/сут (`docs/COSTS.md`) | выгоден при малой нагрузке |
**Реализовано гибридно:** STOP через 60 с (быстрые повторы), DESTROY через `SONYA_STOPPED_RETENTION_SECONDS` (по умолчанию 30 мин). Для постоянной высокой нагрузки увеличить retention.
