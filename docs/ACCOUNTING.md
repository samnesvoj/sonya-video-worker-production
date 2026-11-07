# Учёт расходов Vast.ai и отчёты в существующем бухгалтерском боте

**Нового бота нет.** Токен Telegram остаётся на bot VPS (`147.45.235.79`, `sonya-accounting-bot.service`) в `/etc/sonya-accounting-bot/secrets/`. Исходники бота есть локально (`agents/finance/receipt-worker`, коммит `bd8c968`), поэтому интеграция — проверенный патч, а не догадки.

## Поток данных
```
SONYA VPS (центр): dispatcher + журнал инстансов (fleet.db) + ledger.db (Decimal)
   usage.compute_day  → "рассчитано" (по журналу × цены оффера)          ─┐
   vast_billing.import → "подтверждено" (/api/v0/charges, по UTC-суткам)  ├→ отдельные виды, не смешиваются
   usage (traffic)    → "оценка" (НЕ входит в итог)                       ─┘
   cbr.CbrRates       → курс ЦБ РФ на каждый день (хранится)
   outbox.ReportService → report_outbox (00:05 МСК ежедневно; 1-го числа 00:10 МСК месячный; догоняет пропущенные)
        ▲ HMAC-подпись (X-Accounting-*), nonce, окно 120 с
Bot VPS: receipt_worker.vast_reports.VastReportPuller (поток существующего бота, выкл. по умолчанию)
   POST …/accounting/vast/claim → отправляет текст СВОИМ Telegram-клиентом в СВОИ чаты → …/{key}/sent
```
Токен Telegram, ключи Vast/GitHub не пересекают границу; на Vast ничего из этого нет.

## Правила цифр
- Все суммы `Decimal`; в payload — строки с 6 знаками; курс — официальный ЦБ РФ на день отчёта, сохраняется (`fx_rates`); при недоступности ЦБ берётся последний известный и в отчёте стоит пометка (выдуманных курсов нет).
- **ИТОГО = рассчитано + применённые корректировки сверки.** «Подтверждено» и «оценка» показываются отдельными строками; оценка трафика в итог не входит.
- Подтверждённые списания Vast приходят по **UTC-суткам**; в 00:05 МСК закрыты только предыдущие UTC-сутки. Поэтому дневной отчёт опирается на журнал (расчёт), а сверка делается по последним закрытым UTC-суткам; расхождение > max($0.05, 3%) превращается в **дельта-корректировку** (идемпотентна: повторный импорт/сверка не удваивает; пересмотр Vast даёт только новую разницу) и попадает в следующий дневной отчёт.
- Защита от двойного учёта: `report_key` уникален (`vast-daily:YYYY-MM-DD`), готовый отчёт не пересобирается; записи `cost_entries` — upsert по ключу; факты (видео, GPU-часы) замораживаются через час после конца суток.
- Пропущенные дни: `daily_due` строит все отсутствующие отчёты по порядку (до 45 дней назад); сбой Telegram → `failed` → повтор с backoff 1/5/15/60 мин; потерянный ack → бот не шлёт повторно (ключ записан в его state-файл до ack).

## Ограничения (честно)
- Формат строк `/api/v0/charges/` **не проверен на реальном ответе** (нет ключа Vast на этом Mac): парсер защитный и сохраняет сырую строку; проверить на первом реальном ответе.
- Трафик не измерим без счётчиков Vast → только оценка.
- Ставка диска остановленного инстанса неизвестна (в модели ×1; параметр `stopped_storage_multiplier`).
- «Обработано видео» = завершённые задания пула `virality`/`beep` с `worker_id like 'sonya-pool-%'`; задания старого пула в этот учёт не входят.

## Установка (после проверки, по шагам; на production ничего не делалось)
1. Бот: `git apply integration/accounting_bot/accounting-bot-vast-reports.patch` в `agents/finance/receipt-worker` (проверено: применяется чисто, 134 теста бота зелёные до и после). Добавляет `vast_reports.py`, секрет `accounting_api_secret`, хук в цикл простоя. Включение: `VAST_REPORTS_ENABLED=true` в `/etc/sonya-accounting-bot/env`, `python3 -m receipt_worker secret set accounting_api_secret`.
2. Сервер: подключить `accounting.internal_api.fastapi_router(AccountingApi(service, secret))` к API (маршрут `POST /api/internal/accounting/vast/*`); секрет — тот же `accounting_api_secret`.
3. Dispatcher VPS: `SONYA_ACCOUNTING_ENABLED=true`, `SONYA_ACCOUNTING_FIRST_DAY=YYYY-MM-DD`.
