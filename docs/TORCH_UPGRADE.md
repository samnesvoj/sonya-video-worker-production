# Обновление PyTorch до версии без известных уязвимостей

## Проблема
Сейчас `torch 2.5.1+cu124`: **23 advisory в OSV** (RCE при загрузке недоверенных чекпойнтов — GHSA-53q9-r3pm-6pq6 → исправлено в 2.6.0, GHSA-63cw-57p8-fm3p → 2.10.0, ряд memory-corruption). Риск принят письменно (`requirements/osv-accepted.json`): чекпойнты берутся только из нашего бакета и фиксируются sha256, недоверенный ввод — только видео. Но надёжнее уйти на исправленную версию.

## Что выяснено (2026-11-07, OSV + индекс PyTorch)
| torch | advisory в OSV | Индексы с колёсами cp311 |
|---|---|---|
| 2.5.1 (сейчас) | 23 | cu124 |
| 2.9.1 / 2.10.0 / 2.11.0 / 2.12.0 | 5 / 2 / 1 / 1 | cu128 (≤2.11), cu126/cu130 (≥2.12) |
| **2.13.0** | **0** | **cu126**, cu130 |
Кандидат: **`torch 2.13.0+cu126` + `torchvision 0.28.0+cu126`** (первая проверенная версия с 0 advisory; выше есть 2.14.x/2.15, не оценивались). Lock с хешами: `requirements/candidates/torch213-cu126.lock.txt` (+ `.in`, разбивка по слоям `…-split/`). `pip-audit` по остальным пакетам чисто; `torch`/`torchvision` проверяет `tools/osv_check.py` (в CI: 0 advisory, ни одно не «принято»). Совместимость при разрешении зависимостей: `numpy 2.4.6`, `mediapipe 1.2.2`, `ultralytics 8.4.206`, `ctranslate2 4.8.5`, `faster-whisper 1.2.1`, `opencv 4.14.0.94` (4.x сохранён), `cuDNN 9.10.2`, `cuBLAS 12.6.4.1`, `triton 3.7.1`; от текущего lock отличаются 41 строка пакетов.

## Что проверяет CI без GPU (джоб `image-candidate-torch213`, не гейт)
Сборка образа на кандидате (индекс cu126); импорт всех зависимостей вместе; `pip check`; bootstrap/Trailer/readiness/health (лок хешем production — только для этого эксперимента); **модели реально запускаются на новом torch**: sha256-проверенные публичные `yolo11n.pt` и `yolo11n-pose.pt` через ultralytics (`torch.load` по новым умолчаниям), NMS из torchvision, `faster-whisper base` (загрузка + распознавание). Результаты — в артефакте `image-report-torch213` и в разделе «Результаты CI» ниже.

## Что CI доказать не может (нужно вам/Mac mini и GPU)
1. **Закрытый `webcam_detector.pt`** при `weights_only=True` по умолчанию с PyTorch ≥ 2.6: запустить на Mac mini `python tools/model_load_check.py --models-dir ~/SONYA-models` в venv с текущим torch и в venv с кандидатом (CPU, ≈ 1 мин). FAIL = файл нужно пересохранить/переобучить до апгрейда.
2. **Драйвер хоста.** Колёса cu126 приносят свой CUDA 12.6 runtime; по NVIDIA (minor-version compatibility) достаточно драйвера семейства 12.x ≥ 525 [источник: README vast-ai/base-image], но PTX-JIT может требовать более нового драйвера. Эксперимент выбирает хосты с `cuda_max_good ≥ 12.6` (и отдельно ≥ 12.4).
3. **Численное совпадение результатов режимов** (новый cuDNN/cuBLAS/torch): те же контрольные видео на текущем и на кандидате, `compare_runs` + просмотр.
4. **Скорость/VRAM** (cuDNN 9.10, новый torch).

## GPU-эксперимент (отдельный, малый; стоит ≈ $0.2–0.5 ≈ 16–41 ₽ на RTX 3090/4090 — добавляется к первому GPU-тесту)
Одна карта, один хост, два образа подряд (текущий и кандидат), одни и те же 2 контрольных видео (`virality`, `beep`):
1. `tools/fast_start_test.py live` с образом A (текущий) → отчёт A; затем с образом B (кандидат; **в Template/launcher для B задать `SONYA_BASE_LOCK_SHA256` из кандидата и релиз, где `requirements/worker.lock.txt` = кандидат** — т. е. отдельная ветка с подменённым lock; нельзя запустить кандидата на релизе с другим lock, bootstrap откажет).
2. Принять кандидата, если: критерии теста B не хуже A; выходные видео B проходят `compare_runs` против A без флагов и глазами не хуже; `nvidia-smi` без ошибок; время обработки B ≤ A × 1,1.
3. Если принят: PR «lock → torch 2.13.0+cu126» (подмена `requirements/worker.lock.txt`, `requirements/split/*`, `TORCH_INDEX_URL` в Dockerfile и CI на cu126, `tools/update_release_manifest.py`), очистить `requirements/osv-accepted.json`, пересобрать и опубликовать образ, повторить Fast Start. Откат — прежний SHA + digest.

## Решение
До прохождения п. 1 (Mac mini) и GPU-эксперимента **production остаётся на 2.5.1** с письменно принятым риском; CI ломается на любом новом advisory для обеих веток зависимостей.
