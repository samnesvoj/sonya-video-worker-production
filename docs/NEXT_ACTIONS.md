# Что осталось до первого GPU-теста (точный список)

## Вы
1. **Mac mini:** выполнить команды из `docs/MODELS_FOR_MAC_MINI.md` (scan, lock, model_load_check в двух venv) и ответить на 4 вопроса (ASR эталона, флаги Gemini, тот ли `webcam_detector.pt`, контрольные видео + эталонные результаты). Прислать JSON/вывод (без самих моделей).
2. **GitHub App:** 12 шагов из `docs/GITHUB_APP_SETUP.md`, затем `tools/verify_github_app.py` → пять PASS.
3. **S3:** выделить тестовый ключ (чтение `models/`, запись `fast-start-test/`, `fleet-metrics/`); решить, кто загружает модели (`models_publish.py upload` — сначала dry-run).
4. **Слияние PR:** по `docs/MERGE_ORDER.md` (способ A) — только после ревью; релиз не выпускать.
5. **Опубликовать базовый образ** (Actions → `publish-base-image`, вручную) → digest для теста.
6. Решения: ASR (base/small/medium/large-v3), Gemini on/off в тесте, GitHub Pro (защита `main`).
## Я (после п. 1–3)
- подставить `models.lock.json` и ASR в `mode.yaml`/манифест, обновить PR; прогнать `fast_start_test.py simulate` и `live` БЕЗ `--apply` (preflight);
- по вашей команде — `live --apply` (≤ $3).
## Не требуется для теста
Pro-защита, Nydus/eStargz, кеш-тома Vast, изменения VPS, включение остальных режимов.
