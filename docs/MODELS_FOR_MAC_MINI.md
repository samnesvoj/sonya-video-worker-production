# Модели для первого полноценного GPU-теста (virality + beep) — список для Mac mini

Источник правды: `release/models.required.json` (проверяется тестом на совпадение с `modes/*/mode.yaml`). Все пути — относительно `$MODELS_LOCAL_DIR` (на воркере `/opt/sonya/models`); ключи S3 — `models/<путь>` в бакете `MODELS_S3_BUCKET`.

## Что нужно (7 файлов, из них публично известны 6)
| Ключ | Путь | Версия / источник | Размер | SHA-256 (публичный) | Кто грузит | Что если нет |
|---|---|---|---|---|---|---|
| `yolo11n` | `common/yolo11n.pt` | Ultralytics YOLO11n, assets v8.4.0 · https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt | 5 613 764 Б | `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` | virality: `sonya_enhancer` (YOLO_COMMON, по умолчанию вкл.) | **тихо пропускается** (только предупреждение) → меняется кроп |
| `yolo11n_pose` | `common/yolo11n-pose.pt` | Ultralytics YOLO11n-pose, v8.4.0 | 6 255 593 Б | `869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0` | virality: YOLO_POSE (вкл.) | то же |
| `webcam_detector` | `common/webcam_detector.pt` | **закрытая** модель SONYA (YOLO, дообучена) — есть только на Mac mini | **неизвестен** | **неизвестен** (нужно от вас) | virality: WEBCAM_DETECTOR (вкл.) | то же |
| `whisper_base_model` | `whisper-base/model.bin` | Systran/faster-whisper-base @ `ebe41f70d5b6…` (HF) | 145 217 532 Б | `d01c3014881c9c6f…` (LFS, полный хеш в манифесте) | virality + beep: `Transcriber` (по умолчанию `whisper-1` → faster-whisper **base**) | без неё ASR ищет модель в HuggingFace **во время работы** (сеть, без проверки) |
| `whisper_base_config` | `whisper-base/config.json` | то же | 2 309 Б | в манифесте | то же | то же |
| `whisper_base_tokenizer` | `whisper-base/tokenizer.json` | то же | 2 203 239 Б | в манифесте | то же | то же |
| `whisper_base_vocabulary` | `whisper-base/vocabulary.txt` | то же | 459 861 Б | в манифесте | то же | то же |
Итого известно ≈ 160 МБ + `webcam_detector.pt`. **Скачивание этого набора на хосте — секунды (при 25 МБ/с ≈ 7 с), а не минуты**; прежняя оценка «5 ГБ» относилась к гипотетическому набору. Дороже станет только выбор более крупной ASR (см. ниже).

## Решения, которые нужны от вас/Mac mini ДО теста
1. **Какая ASR давала эталонные результаты?** Код воркера по умолчанию берёт faster-whisper **base** (слабо для русского). На Mac mini по документации Cut использовалась GigaAM (mlx) — на Vast её нет. Варианты (публичные, MIT, с хешами в манифесте → `alternatives_asr`): `small` 486 МБ, `medium` 1 531 МБ, `large-v3` 3 091 МБ. Выбор = `SONYA_WHISPER_MODEL=<имя>` + замена четырёх записей `whisper_base_*` в `modes/*/mode.yaml` и манифесте. Время загрузки: large-v3 при 25 МБ/с ≈ 2 мин.
2. **Gemini:** `GEMINI_LAYOUT_ENABLED` в коде по умолчанию **включён** и требует `GEMINI_API_KEY` на арендованном хосте. В первом тесте предлагаю выключить (нет внешнего секрета на чужом железе), если эталонный прогон шёл без него. Какие флаги стояли у эталона?
3. **Файл `webcam_detector.pt` — тот, что дал проверенные результаты?** (хеш, дата, чем обучен).
4. **Контрольные видео и эталонные результаты** (1–2 на режим): имя, sha256, длительность, параметры/пресет, результат Mac mini (клипы + `selected_segments.json`).

## Команды на Mac mini (без сети, без учётных данных; ничего не загружают)
```bash
cd ~/SONYA-dev/sonya-video-worker-production && git fetch origin && git switch perf/fast-start   # до слияния; после слияния — main
python tools/models_publish.py scan --src ~/путь/к/моделям --src ~/ещё/папка --out ~/models-scan.json
python tools/models_publish.py lock --src ~/путь/к/моделям --out ~/models.lock.candidate.json    # добавьте --accept-custom KEY, если файл отличается от публичного и ЭТОТ файл — эталон
python tools/model_load_check.py --models-dir ~/SONYA-models   # грузит и запускает модели на CPU (в venv с текущим torch и в venv с кандидатом, см. TORCH_UPGRADE.md)
```
`scan` печатает по каждому ключу: `matches_public` / `differs_from_public` / `private_ok` / `missing` / `ambiguous` (если несколько кандидатов — укажите `--map KEY=ПУТЬ`). **Пришлите мне:** `models-scan.json`, `models.lock.candidate.json` (только хеши и размеры), вывод `model_load_check.py`, ответы на пункты 1–4. Сами файлы моделей присылать не нужно, пока не решено, кто и откуда грузит в S3.

## Загрузка в наш S3 (только после вашего согласования)
```bash
python tools/models_publish.py upload --src <папка> --bucket <MODELS_S3_BUCKET>                         # DRY-RUN: покажет, что загрузит
python tools/models_publish.py upload --src <папка> --bucket <MODELS_S3_BUCKET> --apply --confirm-bucket <MODELS_S3_BUCKET>
python tools/models_publish.py lock   --src <папка> --out release/models.lock.json                       # затем PR с готовым lock
```
Загрузка идёт multipart, SHA-256 кладётся в метаданные объекта (`x-amz-meta-sha256`), повторный запуск пропускает уже загруженное, после загрузки сверяются размер и метаданные (повторного скачивания нет — на Mac mini медленный канал). Нужен S3-ключ: лучше отдельный, с правом записи только в `models/` (а не боевой). Секреты читаются из `S3_ENDPOINT_URL`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`, `S3_REGION` и нигде не печатаются.
