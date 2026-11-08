# Интерфейс ASR: согласование MacBook ↔ Mac mini (GigaAM v3 — основной, faster-whisper — запасной)

Основание: решение владельца `docs/OWNER_DECISIONS.md`. **Бэкенд GigaAM для воркера пишет Mac mini** (ветка `mac-mini/models-scan`); MacBook отвечает за образ, зависимости, CI и проверки. Этот документ и `scripts/shared/asr_contract.py` — договорённость о границе; ничего из бэкенда здесь не реализовано.

## Что делает каждая сторона
| | MacBook (этот PR) | Mac mini |
|---|---|---|
| Зависимости образа | lock с хешами: `torch/torchvision/torchaudio 2.11.0+cu126` (одна бинарная сборка), `hydra-core`, `omegaconf`, `onnx 1.19`, `onnxruntime 1.23`, `sentencepiece`, `soundfile` | — |
| Пакет `gigaam` | не вендорит | предлагаю: вендорить **как есть** коммит `7447938d791c…` (MIT) в репозиторий (`pip --require-hashes` не принимает git-зависимости); обёртка — отдельным модулем |
| Бэкенд `scripts/shared/asr` | контракт + валидатор + тесты | реализация: загрузка GigaAM на CUDA, чанкинг, постобработка |
| Ключи моделей в `mode.yaml`/манифесте | **добавлены**: `gigaam_v3_e2e_rnnt_ckpt`, `gigaam_v3_e2e_rnnt_tokenizer` (обязательные); whisper — `optional` (запасной) | подтвердить: путь, размер, SHA-256 своей копии (`models_publish.py scan/lock`) |
| CI | совместимость стека, офлайн-проверка (см. ниже) | — |

## Контракт (проверяется `asr_contract.validate_result`)
`scripts/shared/asr.transcribe(audio_path, language="ru") -> dict`:
```
{"text": str, "language": "ru", "backend": "gigaam", "model": "gigaam-v3_e2e_rnnt-ts3",
 "segments": [{"start","end","text","words":[{"word","start","end"}]}],
 "words": [{"word": " слово", "start": 1.23, "end": 1.55}, ...]}      # секунды от начала аудио; пробел в начале слова допустим
```
Это формат эталонного Desktop-бэкенда (`desktop/sonya_desktop/asr/backends.py::_assemble`), расширять его не нужно. Допустимые `backend`: `gigaam`, `faster_whisper_cuda`, `faster_whisper_cpu`. Выбор — `SONYA_ASR_BACKEND` (по умолчанию `gigaam`).
Для `gigaam` имя модели обязано иметь вид `gigaam-v3_e2e_rnnt-tsN`, **N ≥ 3**: тег = версия постобработки (`ts3` = сдвиг начал −150 мс + минимальная длина слова 0,12 с + склейка пунктуационных токенов). Если постобработку меняете — поднимайте N (он же ключ кэша ASR).

## Жёсткие требования к бэкенду
1. **Никаких загрузок.** `gigaam.load_model(...)` в эталоне вызывается **без** `download_root` — тогда библиотека качает веса сама (`~/.cache/gigaam`, CDN `cdn.chatwm.opensmodel.sberdevices.ru`). В воркере: `download_root=<MODELS_LOCAL_DIR>/gigaam`, проверить наличие файлов заранее, при отсутствии — **исключение**, не загрузка. Воркер экспортирует `SONYA_ASR_OFFLINE=1`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` (в `run_worker.sh`, из Template не переопределяются). *Факт:* TLS-сертификат этого CDN (Let's Encrypt) **истёк 2026-11-04**, штатное скачивание сейчас не работает вообще.
2. Целостность: вес берётся из нашего S3 и проверяется **SHA-256** по `release/models.lock.json` загрузчиком воркера (`model_fetch`). Upstream при уже существующем файле возвращает путь **без проверки хеша** (`_download_file`: `if os.path.exists: return`), поэтому оборванный файл прошёл бы молча — наша проверка обязательна; MD5 upstream используется только как дополнительная сверка при `scan`.
3. `flash_attn` не нужен и не ставится: `load_model(..., use_flash=False)` / `fp16_encoder` по таблице профилей эталона.
4. Приватные API upstream, которые использует эталон (`AudioDataset`, `model._decode`) — поэтому коммит upstream **закреплён**; CI проверяет, что они есть в нём.
5. Ни единого сетевого вызова во время распознавания: тест `test_backend_compliance_offline_with_real_weights` блокирует сокеты.

## Найденный риск для Cut (нужно решение Mac mini)
`desktop/…/backends.py::fix_rnnt_word_times` спасает обычные короткие слова, но слово, которое RNNT выдал **в том же 40-мс кадре**, что и предыдущее, после обработки получает нулевую длину (`start=prev_end`, `end` не растёт), и `boundaries.normalize_words` в Cut **молча отбрасывает его** (`end <= start`). Тест-оракул `test_reference_postprocessing_saves_normal_words_but_a_same_frame_word_still_dies_in_cut` фиксирует это на синтетических данных. Реальную долю потерь меряет `tools/ci_gigaam_real.py` (порог ≤ 1 %). Если на реальных данных она больше — распределять слова одного кадра по подинтервалам кадра перед `fix_rnnt_word_times`. Я это не реализую (зона бэкенда).

## Что проверяется и где
| Проверка | Где | Статус |
|---|---|---|
| Контракт и Cut-метрика (валидатор, оракул, офлайн-фикстура) | `tests/test_asr_contract.py` (CI `test`) | зелёные |
| Одна бинарная сборка torch/torchvision/torchaudio; скомпилированные операции torchaudio; импорт `gigaam` с зависимостями из lock; API, нужный эталону; log-mel из upstream; ConformerEncoder (rotary, rel_pos) на torch 2.11 | `tools/ci_gigaam_compat.py` в образе (CI `docker-build`) | см. результаты CI в PR |
| Реальное распознавание с весами **офлайн** (`--network none`), пословные таймкоды, SHA-256/MD5, Cut-метрика до/после постобработки | `tools/ci_gigaam_real.py`: на Mac mini или в CI при секретах `GIGAAM_S3_*` | **не выполнено**: весов на MacBook нет (CDN с истёкшим сертификатом не использую) |
| Тест соответствия бэкенда Mac mini | `test_backend_compliance_offline_with_real_weights` (`GIGAAM_WEIGHTS_DIR`, `GIGAAM_TEST_AUDIO`) | пропускается, пока `scripts/shared/asr` не влит |
Readiness воркера (`SONYA_REQUIRE_ASR=1`, по умолчанию в `run_worker.sh`) не открывается, пока нет обоих файлов GigaAM на диске и модуль `scripts.shared.asr` не импортируется — **пока бэкенд не влит, воркер сознательно не стартует** (это и есть защита от слияния до интеграции).

## Как запустить проверку с весами на Mac mini (CPU, офлайн)
```bash
git clone https://github.com/salute-developers/GigaAM.git /tmp/GigaAM && git -C /tmp/GigaAM checkout 7447938d791c4f3e643386ee22c33777004293a5
python tools/ci_gigaam_real.py --models-dir ~/SONYA-models --audio <речь.wav> --gigaam-src /tmp/GigaAM
```
Ожидается `ALL CHECKS PASSED`; пришлите вывод (там SHA-256 весов для `models.lock.json`).

## Measured image impact (CI run 42421290906, commit 630b907)
| | before (main) | with GigaAM stack | delta |
|---|---|---|---|
| uncompressed | 7.48 GB | 8.50 GB | +1.02 GB (+14 %) |
| gzip (wire) | 3.75 GB | 4.31 GB | +0.56 GB (+15 %) |

Main contributors: torch/cuDNN moving to the cu126 matched set, onnx (85 MB), onnxruntime (53 MB). The layer split is kept (cuda-libs / torch / rest), so a code-only change still pulls only the small top layers.
Real-weights recognition was SKIPPED in CI (GIGAAM_S3_* secrets not configured) - still to be verified together with the Mac mini.
