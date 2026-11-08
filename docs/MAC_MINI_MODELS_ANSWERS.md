# Mac mini: модели, ASR, PyTorch и эталоны для первого GPU-теста (virality + beep)

Ответ на `docs/MODELS_FOR_MAC_MINI.md` (пункт 1 `docs/NEXT_ACTIONS.md`). Машина: Mac mini M4, 16 ГБ. Дата: 2026-11-07.
Ничего не скачивалось из интернета (кроме колёс PyTorch для изолированных venv), ничего не загружалось в S3, Vast/VPS/образ/CI не трогались.
Пути домашней папки записаны как `~`. Видео и клипы в git не кладутся: только имена, размеры и SHA-256.

## Коротко
| Вопрос | Ответ |
|---|---|
| а) Какая ASR давала эталон | **GigaAM v3 `v3_e2e_rnnt`**, постобработка `ts3` (сдвиг начал −150 мс, минимальная длина слова 0,12 с, склейка тире/кавычек), русский. Пакет `gigaam` 0.2.0 @ `7447938d`, torch 2.14.1, MPS. Фильм 1 ч 51 мин распознан за **114,6 с** (≈ 58× быстрее реального времени). Whisper в эталоне Cut не участвовал (whisper large-v3 mlx — только в 24-секундных smoke-прогонах до GigaAM). **Решение владельца: GigaAM основная ASR**; бэкенд перенесён в воркер (`scripts/shared/asr`), на CUDA запускается тот же код. |
| б) Gemini | **Gemini-layout (`GEMINI_LAYOUT_ENABLED`) в эталоне не работал**: во всех 13 записях прогонов virality на Mac mini предупреждение «GEMINI_API_KEY not set — layout_segments skipped». Gemini использовался только как **LLM-редактор** отбора клипов (через OpenRouter, `google/gemini-3.8-flash`, 1 вызов, $0,019) — это другая функция. Рекомендация для первого теста: `GEMINI_LAYOUT_ENABLED=false`, LLM-редактор выключен (`llm` не `on`), никаких ключей на Vast. |
| в) `webcam_detector.pt` | Найден **один** файл: `~/Desktop/for my agent/вся соня/boosta restore/boosta_local_recovered/models/webcam_detector.pt`, 5 441 498 Б, SHA-256 `b011911e…8481b`. Обучен 2025-12-20 (ultralytics 8.3.240, YOLO11n, 2 эпохи, 1 класс `webcam_frame`, mAP50 0,995 на своей валидации). **В эталонных прогонах НЕ использовался** («webcam_detector.pt not found» во всех 13 прогонах). `--accept-custom` не применялся, в lock не внесён — нужно решение владельца. |
| г) Контрольные видео | Virality: `kishlak.mp4` (1 ч 51 мин) → 32 клипа, просмотрены владельцем; быстрый контроль — `kishlak_15m.mp4` (первые 15 мин того же интервью). BEEP: эталона, одобренного владельцем, на реальном исходнике **нет**; подготовлены 2 контрольных прогона кодом воркера (см. ниже). |

## 1. Найденные модели (`tools/models_publish.py scan`)
Полный вывод: `docs/mac_mini/models-scan.json`; кандидат lock: `docs/mac_mini/models.lock.candidate.json`. Манифест — версия из PR #3 (с ключами GigaAM).

| Ключ | Где на Mac mini | Размер, Б | SHA-256 | Статус |
|---|---|---|---|---|
| `gigaam_v3_e2e_rnnt_ckpt` | `~/.cache/gigaam/v3_e2e_rnnt.ckpt` | 448 929 252 | `f60c62fe45902d967000770a12f260de7c5ac4d4b8e5e852f1df8c548e3958b5` | `matches_upstream_md5` (MD5 `2730de75…` = таблица upstream) |
| `gigaam_v3_e2e_rnnt_tokenizer` | `~/.cache/gigaam/v3_e2e_rnnt_tokenizer.model` | 255 336 | `828c12c991019eef952a960661f25a92d6ad279591e2ea466b4aeddf1d20a18a` | `private_ok` (MD5 `3b3bf8370e882885d79731592fc99f98`; публичного хеша нет) |
| `yolo11n` | `…/boosta_local_recovered/yolo11n.pt` (+ байт-в-байт копия в `boosta_standalone_recovered`) | 5 613 764 | `0ebbc80d…44ee1` | `matches_public` |
| `yolo11n_pose` | `…/boosta_local_recovered/yolo11n-pose.pt` (+ копия) | 6 255 593 | `869e83fc…19dc0` | `matches_public` |
| `webcam_detector` | `…/boosta_local_recovered/models/webcam_detector.pt` | 5 441 498 | `b011911e6129fddebbc32a75cc3f7b3c22b5216dd9d6a6f446757cb67aa8481b` | `private_ok`, **не внесён в lock** (см. в) |
| `whisper_base_model` | `~/.cache/huggingface/hub/models--Systran--faster-whisper-base/…/model.bin` | 145 217 532 | `d01c3014…0a6a9` | `matches_public` |
| `whisper_base_tokenizer` / `_vocabulary` | в кэше base их нет; взяты из снапшота `faster-whisper-small` (файлы байт-в-байт равны публичным base) | 2 203 239 / 459 861 | `fb7b6319…` / `34ce3fe1…` | `matches_public` |
| `whisper_base_config` | **нет на Mac mini** | — | — | `missing` (публичный хеш есть в манифесте, проверен в CI) |

Ещё есть локально (не нужны для теста): faster-whisper `small` (486 МБ) и `medium` (1,53 ГБ) — совпадают с ревизиями `alternatives_asr`; mlx-whisper large-v3 (fp16/8bit/4bit) и large-v3-turbo — только для Apple, на CUDA не запускаются.
**Lock:** `models.lock.candidate.json` содержит 7 ключей с реальными хешами; `not_locked`: `whisper_base_config` (нет файла) и `webcam_detector` (нет доказательства, что это эталон). Фиктивных значений нет.

## 2. ASR: эталон, перенос в воркер, сравнение
### 2.1 Как GigaAM работает на Mac mini сейчас
- Код: `sonya-desktop/desktop/sonya_desktop/asr/backends.py` (`GigaAmBackend`, `speech_chunks`, `merge_punct_tokens`, `fix_rnnt_word_times`), дочерний процесс `asr/child.py`, кэш `asr/artifact.py` (ключ: sha256 исходника + профиль + бэкенд + модель + тег постобработки).
- Профиль `quality` = `v3_e2e_rnnt` fp16 на MPS; чанки 14–22 с по самым тихим 30 мс, батч 8; поправка начал −150 мс (`GIGAAM_START_LEAD`, CUT_PLAYBOOK: «RNNT word starts are ~160 ms late (median over 20k words vs Whisper)… Shift starts back by 150 ms»).
- Окружение эталона (`.venv-desktop`): Python 3.11.16, torch **2.14.1**, torchaudio **2.11.0**, gigaam 0.2.0 @ `7447938d791c4f3e643386ee22c33777004293a5`, hydra-core 1.3.7, omegaconf 2.3.1, sentencepiece 0.2.2, soundfile 0.14.0, onnx 1.19.1, numpy 2.4.6.
- Веса: `~/.cache/gigaam/` (скачаны библиотекой 2026-10-13 с CDN upstream). Лицензия кода и весов — MIT (`LICENSE` upstream: «MIT License, Copyright (c) 2024 GigaChat Team»).
- Скорость по логам: `kishlak.mp4` 6 688 с → **114,6 с**; `kishlak_15m.mp4` 900 с → 18,3 с; CUT_PLAYBOOK: 2,5 ч → 154 с.

### 2.2 Перенос в воркер (эта ветка)
- `scripts/shared/asr/gigaam_backend.py` — тот же код чанкинга/декодирования/постобработки (источник указан в файле); веса **только** из `$MODELS_LOCAL_DIR/gigaam/` (`download_root`), при отсутствии файла — исключение, сетевых вызовов нет. Устройство: CUDA → MPS → CPU (`SONYA_ASR_DEVICE` переопределяет), fp16-энкодер не на CPU, `use_flash=False`.
- `scripts/shared/asr/__init__.py` — `transcribe(audio, language="ru")` по контракту `asr_contract.py` (PR #3). `SONYA_ASR_BACKEND=gigaam` по умолчанию; faster-whisper — запасной: явно (`SONYA_ASR_BACKEND=faster_whisper_*`), для не-русского языка, или при `SONYA_ASR_FALLBACK=1` (по умолчанию 0 — без тихой подмены).
- Virality (`sonya_enhancer`) и BEEP (`modes/beep/runner.py`, раньше на воркере без готовых слов всегда падал `BEEP_FAILED`) теперь вызывают эту ASR.
- `scripts/third_party/gigaam/` — upstream @ `7447938d` **без изменений** + `LICENSE` + `VENDORED.md` с SHA-256 файлов (pip `--require-hashes` не принимает git-зависимости).
- Отличие от эталона одно — `ts4`: слово, которое RNNT выдал в том же кадре, что и предыдущее, после `ts3` имело нулевую длину и Cut его выбрасывал. На фильме это **2 из 18 959** слов (0,01 %; риск из PR #3 на реальных данных мал). `ts4` отдаёт такому слову последние 40 мс предыдущего слова (сначала пробовал «половину» — у «Блядь,» отрезалось 360 мс, для BEEP это опасно; исправлено).
- Тесты: `tests/test_asr_backend.py` (сдвиг −150 мс, мин. длина, склейка пунктуации, ts4, нет весов → ошибка без сети, выбор бэкенда/запасной путь, BEEP запускает ASR и громко падает, хеши вендоренного кода, сверка с эталоном при наличии весов). Вместе с `test_asr_contract.py`, `test_beep.py`, `test_models_publish.py`: **57 passed, 2 skipped** (skip — тесты с реальными весами без переменных окружения).

### 2.3 Тот же бэкенд даёт тот же результат, что эталон (Mac mini, исходные mp4)
| Видео | Окружение | Время | Текст | Тайминги |
|---|---|---|---|---|
| `kishlak_15m.mp4` (900 с, 2 479 слов) | torch 2.14.1, MPS (окружение эталона) | 15,9 с | **идентичен** | **все 2 479 совпали до миллисекунды** |
| `kishlak.mp4` (6 688 с, 18 959 слов) | torch 2.14.1, MPS | 113,6 с (эталон 114,6 с) | **идентичен** | 18 955 идентичны; 4 отличаются ≤ 40 мс — это ровно 2 пары слов `ts4` |
| `kishlak_15m.mp4` | torch 2.13.0 (+torchaudio 2.11.0), MPS | 17,0 с | идентичен | все совпали |
| `kishlak_15m.mp4` | torch 2.11.0 (+torchaudio 2.11.0), MPS | 17,9 с | идентичен | 2 478/2 479 (1 конец слова +40 мс) |
| `kishlak_15m.mp4` | torch 2.5.1, MPS | 24,1 с | идентичен | 2 478/2 479 (1 конец слова +40 мс) |
| `kishlak_15m.mp4` | CPU fp32 (torch 2.14.1 / 2.13 / 2.11 / 2.5.1 — одинаково) | 15,1–15,6 с | 1 слово из 2 479: «Я-я» → «Я» | 2 477/2 479, макс. 70 мс |
Вывод: перенос верный; на fp16-ускорителе результат совпадает с эталоном, на CPU fp32 — практически совпадает. На CUDA (fp16) ожидаю такое же поведение, но **NOT MEASURED** (GPU не запускался); тест `test_same_words_and_times_as_the_mac_mini_reference` с допуском 40 мс готов для GPU-прогона. Справочно: на M4 CPU GigaAM так же быстр, как на MPS (≈ 58×).

### 2.4 Сравнение GigaAM и faster-whisper (подтверждение решения владельца, не выбор)
Данные: 3 отрывка по 180 с из видео runtime (интервью `kishlak.mp4` 2169–2349 с; `satir_cian.mp4` 60–240 с; спектакль с матом `exile_kringe.mp4` 5880–6060 с) + `kishlak_15m` (900 с) для влияния на Cut. Вход — WAV 16 кГц. faster-whisper 1.2.1 с настройками воркера (`language=ru`, `beam_size=5`, `vad_filter=True`, `word_timestamps=True`), на Mac только CPU int8; GigaAM — код воркера, MPS. **faster-whisper base не запускался**: в кэше нет `config.json` (скачивание запрещено); `large-v3` для faster-whisper на диске нет (не скачивался), вместо него как ориентир — mlx-whisper large-v3 (тот же вес, другой движок, только Apple).

**Скорость (Mac mini, секунды на распознавание; в скобках — во сколько раз быстрее реального времени):**
| Аудио | GigaAM MPS | GigaAM CPU | fw small CPU int8 | fw medium CPU int8 | mlx large-v3 |
|---|---|---|---|---|---|
| kish_a 180 с | 3,3 (54×) | 3,2 | 23,7 (7,6×) | 80,6 (2,2×) | 40,5 (4,4×) |
| satir 180 с | 3,4 (53×) | 3,2 | 24,2 (7,4×) | 76,1 (2,4×) | 36,4 (4,9×) |
| exile 180 с | 3,2 (56×) | 3,3 | 33,3 (5,4×) | 105,5 (1,7×) | 71,5 (2,5×) |
| k15 900 с | 15,4 (58×) | 14,7 | 154,3 (5,8×) | 330,4 (2,7×) | 243,4 (3,7×) |
На CUDA faster-whisper быстрее, чем на CPU Mac (fp16), но **NOT MEASURED**; GigaAM на CUDA — тоже NOT MEASURED.

**Точность (метод честно):** ручной расшифровки нет и прослушивания не было. Считал WER **между моделями** (одинаковая нормализация: нижний регистр, ё→е, без пунктуации, «чего -то» whisper склеено) и разбирал расхождения по смыслу текста.
| Пара (WER, % слов) | kish_a | satir | exile | k15 |
|---|---|---|---|---|
| GigaAM ↔ fw medium | 18,1 | 7,4 | 30,4 | 62,1* |
| GigaAM ↔ fw small | 26,5 | 11,2 | 37,4 | 19,4 |
| fw small ↔ fw medium | 24,7 | 9,2 | 34,6 | 64,1* |
| GigaAM ↔ mlx large-v3 | 16,2 | 6,6 | 49,5 | 14,6 |
| fw medium ↔ mlx large-v3 | 10,7 | 6,7 | 52,6 | 117,5* |
| fw small ↔ mlx large-v3 | 26,3 | 11,8 | 55,1 | 19,8 |
\* **faster-whisper medium потерял ≈ 90 % речи в первых 9 минутах `kishlak_15m`** (1 269 слов против 2 474 у GigaAM; 16–24 слова в минуту вместо 150–180) — известный сбой whisper на длинном аудио; это реальная ошибка, а не артефакт метода.
Разбор расхождений (kish_a, GigaAM | whisper medium): «Мама просто думала» | «вот маму просто думал»; «долбоёб» | «долбом»; «виню» | «винюю»; «ценность деньгам» | (small: «цена с деньгам»); small: «вырасту дал помелее» вместо «вырасту долбоёб»; satir: «кепки» | «ебки», «ролексы» | «роликсы». Ошибки GigaAM: «20021-го» вместо «2021», «свой расп писание». GigaAM сохраняет слова-паразиты («ну», «вот», «э»), whisper их выкидывает — это часть «расхождения», а не ошибка GigaAM. Пунктуация: у GigaAM 212 концов предложений на k15, у fw small 148, у fw medium 85 — для Cut предложения критичны (кандидаты строятся только из настоящих предложений).

**Таймкоды начал/концов слов** (медиана разницы «модель − fw medium» по совпавшим словам; отрицательное = раньше):
| | kish_a | satir | exile | k15 |
|---|---|---|---|---|
| GigaAM без поправки, начало | −23 мс | −31 | −6 | −25 |
| GigaAM с −150 мс (`ts4`), начало | −131 | −140 | −120 | −120 |
| fw small, начало | −140 | −120 | −80 | −100 |
| Концы слов GigaAM | −125 | −137 | −109 | −110 |
| **vs mlx large-v3 (метод CUT_PLAYBOOK): GigaAM без поправки, начало** | **+165** | **+162** | +211 | **+160** |
| vs mlx large-v3: GigaAM с −150 мс, начало | +65 | +59 | +113 | +63 |
| vs mlx large-v3: fw small / fw medium, начало | +60 / +180 | +80 / +190 | +120 / +220 | +70 / +170 |
Честный вывод: «истинных» таймкодов нет (нет разметки по звуку), а разные whisper расходятся между собой на 80–190 мс. Но тем же методом, каким в CUT_PLAYBOOK получены «~160 мс» (сверка с whisper large-v3), я **воспроизвёл ту же цифру: начала GigaAM запаздывают на +160…+165 мс** на трёх отрывках из четырёх (на спектакле с криком и музыкой +211 мс). После поправки −150 мс остаток ≈ +60 мс — того же порядка, что разброс между самими whisper. Правило −150 мс подтверждено и остаётся (решение владельца). Концы слов GigaAM близки к large-v3 (+50…+80 мс). Относительно faster-whisper medium картина другая (сырые начала GigaAM почти совпадают), потому что faster-whisper medium сам ставит начала на ≈ 180 мс позже large-v3.

**Влияние на выбор клипов Cut** (k15, локальный отбор воркера `select_story_clips(strategy="qa")` + `refine_segments`, 8 клипов; сравнение с реальными клипами эталона Mac mini, попавшими в первые 15 мин — их 5):
| ASR | Кандидатов | Совпало с эталонными клипами (IoU > 0,5) |
|---|---|---|
| **GigaAM (воркер)** | 2 359 | **5 из 5, границы до миллисекунды (IoU 1,0)** |
| fw small | 742 | 2 из 5 |
| fw medium | 437 | 2 из 5; 2 из 8 клипов — рекламная вставка «Ментал Хэлс уже в продаже…» |
| mlx whisper large-v3 (только Apple) | 3 150 | 3 из 5 (IoU 0,73 / 0,56 / 0,85) |

**Итог по ASR.** Лучшая ASR для русского у SONYA — **GigaAM v3 `v3_e2e_rnnt`**: на Mac mini в 7–20 раз быстрее faster-whisper small/medium (CPU) и в 12–16 раз быстрее whisper large-v3 на GPU Apple; по тексту ближе всех к large-v3 (сверка на интервью: 6,6–16 % против 11–26 % у small), ставит пунктуацию, на тех же данных даёт эталонные клипы, а whisper medium на длинном аудио теряет речь. Это подтверждает решение владельца. На Vast запускается **тот же код** (`scripts/shared/asr`, torch CUDA, без mlx) с весами из S3. Оговорки: скорость и совпадение на CUDA ещё не измерены (нужен GPU-тест); нужен torch ≥ 2.6 по метаданным upstream (на 2.5.1 у меня тоже заработало, но upstream это не поддерживает); faster-whisper base как запасной — без оценки качества (нет `config.json`), small — слабее GigaAM по тексту и пунктуации.

## 3. PyTorch (изолированные venv в `~/SONYA/worker-prep/vast-models/venvs/`, рабочие окружения не менялись)
`tools/model_load_check.py --models-dir <stage>` (CPU) + сверка детекций YOLO на 3 кадрах (кадр интервью, кадр TikTok-референса, синтетический кадр с «вебкой» в углу) + GigaAM (2.3).
| venv | torch / torchvision / torchaudio / ultralytics | yolo11n | yolo11n-pose | webcam_detector | whisper-base | GigaAM |
|---|---|---|---|---|---|---|
| t251 (текущий lock) | 2.5.1 / 0.20.1 / 2.5.1 / 8.4.204 | OK | OK | OK | FAIL: нет `config.json` | работает (upstream требует ≥ 2.6) |
| t211 (PR #3) | 2.11.0 / 0.26.0 / 2.11.0 / 8.4.206 | OK | OK | OK | FAIL: нет `config.json` | OK, текст = эталон |
| t213 (кандидат TORCH_UPGRADE) | 2.13.0 / 0.28.0 / **2.11.0** / 8.4.206 | OK | OK | OK | FAIL: нет `config.json` | OK, бит-в-бит = эталон |
Детекции всех трёх YOLO-моделей **одинаковы до 0,1 px и 0,001 conf** на 2.5.1, 2.11 и 2.13; `webcam_detector.pt` находит синтетическую вебку (conf 0,939) на всех трёх. Новые умолчания `torch.load` (weights_only, ≥ 2.6) этот файл не ломают (ultralytics грузит его сам). macOS-колёса: torch 2.13.0 CPU (arm64) скачались за ~30 с; это не сборка cu126 (та только для Linux) — CUDA/драйвер **NOT RUN**. torchaudio для torch ≥ 2.12 нет (последний 2.11), как и пишет PR #3; эталон Mac mini работает на паре torch 2.14.1 + torchaudio 2.11.0 (на macOS), но для Linux/CUDA я бы не полагался на несовпадающую пару — выбор MacBook (2.11 + 2.11) разумен.
FAIL whisper-base — не проблема torch: в кэше Mac mini нет `config.json` faster-whisper-base.

## 4. Gemini
- `GEMINI_LAYOUT_ENABLED`: 13 из 13 записей прогонов virality (`data/runs/*.json`: 11 smoke-прогонов 24 с 12–13 октября на mlx-whisper и 2 прогона «Кишлака» 16–17 октября на GigaAM) — «GEMINI_API_KEY not set — layout_segments skipped». Улучшал ли он результат — **не измерить**: одобренные владельцем клипы сделаны без него.
- LLM-редактор Cut (`selection.llm`): прогон 20261016-233738 — OpenRouter `google/gemini-3.8-flash`, prompt `cut-editor-v2`, 9 253/3 261 токенов, $0,019, 22,6 с; прогон 20261017-041818 — `not_opted_in`. Подписи (`clip_captions.py`) — отдельный вызов на Mac mini, не часть воркера.
- **Первый тест на Vast:** `GEMINI_LAYOUT_ENABLED=false`, `GEMINI_DIRECT_VIDEO_ENABLED=false`, `llm` не `on` → ни одного ключа на чужом железе; YOLO/pose/webcam — по решению ниже. Это и есть конфигурация, которая дала эталон (плюс кроп YuNet).

## 5. `webcam_detector.pt`
- Путь: `~/Desktop/for my agent/вся соня/boosta restore/boosta_local_recovered/models/webcam_detector.pt`; 5 441 498 Б; SHA-256 `b011911e6129fddebbc32a75cc3f7b3c22b5216dd9d6a6f446757cb67aa8481b`; mtime 2026-05-18, в папке восстановления с 2026-10-12.
- Происхождение (метаданные чекпойнта и `MODELS.md` рядом): `best.pt` из `train_webcam_detector/runs/detect/webcam_2epochs`, YOLO11n, ultralytics 8.3.240, обучен 2025-12-20 на MPS, 2 эпохи, датасет `data/merged/dataset.yaml`, класс `webcam_frame`, val precision 0,999 / recall 0,995 / mAP50 0,995 / mAP50-95 0,945. Использовался в старом проекте «boosta» (`gemini_composer.py`), т. е. **до Mac mini** — владелец считает старые версии плохими.
- В эталонных прогонах Mac mini **не использовался** (файл не был в пути — предупреждение в каждом прогоне). Значит, эталонный кроп получен **без** YOLO/pose/webcam (в тех же прогонах: «yolo11n.pt not found», «yolo11n-pose.pt not found»).
- Варианты для владельца: (A) первый тест как эталон — `YOLO_COMMON_ENABLED=false YOLO_POSE_ENABLED=false WEBCAM_DETECTOR_ENABLED=false` (рекомендую: результат сравним с Mac mini 1:1); (B) включить и принять этот файл (`--accept-custom` не нужен — он `private_ok`; нужно только согласие, тогда добавить в lock); (C) переобучить. Я его не принял.

## 6. Эталоны и контрольные видео
| Режим | Исходник | SHA-256 | Длит. | Параметры | Результат Mac mini |
|---|---|---|---|---|---|
| virality | `runtime/dl/kishlak.mp4` (интервью VSRAP × Кишлак, 1920×804, 25 fps, 1 045 878 759 Б) | `0e5821786bd7f2b107aaf371153c824c90a77c2e2f182a697d62853afb865c70` | 6 688,6 с | `cut_file.py`, `{"subtitle_style":"burgundy","strategy":"qa","settings":{"max_clips":32}}`, код sonya-desktop `7ab01c5`, ASR GigaAM ts3, LLM-редактор gemini-3.8-flash | прогон `20261016-233738-0508d0`: 32 клипа 24–68 с (ср. 44,7 с), `selected_segments.json` SHA-256 `6679d01ef49e30ef8e4feec8ced44d6763c3547f6f22f8c375fcdce4cc1a6fae`; весь Cut 494 с (ASR 115 с). Клипы просмотрены владельцем (CUT_PLAYBOOK §3: урок про клип 04 «работа в столовой»). |
| virality (повтор) | то же | то же | — | 1 клип, story 25–75 с, `exclude` всё кроме 3141–3183 с, без LLM | `20261017-041818-d47641`: 1 клип («love mama»), `selected_segments.json` `0e0fedf8adf42fa07811dbb59a645cf8d67869fee3afd5cdf2f21ff3e8629a1d` |
| virality (быстрый) | `runtime/trailer/a/kishlak_15m.mp4` (первые 15 мин) | `e365e10864917005b66fa3fe46b7fc2d3e1c483213704894232fa530fa961b2f` | 900,0 с | для сверки ASR/Cut | эталонный транскрипт GigaAM 2 479 слов; 5 эталонных клипов в этом интервале (144–198, 462–519, 524–591, 676–713, 713–742 с) |
| beep (контроль 1) | `fixtures/beep/beep_fixture_ru.mp4` (синтетическая речь с матом) | `5704babacc953fd02aca6195bc98082b8e7652b6126032465e3b1dbcf76081b6` | 21,0 с | `{"beep":{"subtitles":"srt"}}`, код этой ветки, GigaAM | 43 слова, 5 попаданий («пиздец», «нахуй», «Бля», «заебало», «сука»), 4,22 с запикано, 2,6 с; `beep_report.json` `f193ead1…197d` |
| beep (контроль 2) | отрывок `exile_kringe.mp4` 5880–6060 с (`-c copy`, 12 466 119 Б); исходник `6a0a760c…ffd1` | отрывок `65b45f4fa1cc3f18ed98f692346a52a7c3afa5b7478061a03fdb22cd49c9ac6b` | 180 с | то же | 373 слова, 19 попаданий, 10,63 с запикано, 7,4 с; `beep_report.json` `72c2e716…0b80` |
**BEEP: эталона, одобренного владельцем, нет.** Раньше BEEP на Mac mini работал только как шаг внутри Trailer (голос TTS: «Волк» — 2 слова запикано, отправлено владельцу; «Кишлак», «Сатир» — мата нет). Контрольные прогоны выше — результат кода воркера на Mac mini, владелец их не смотрел. SHA выходных mp4 зависит от версии ffmpeg, сравнивать по `beep_report.json` (слова и интервалы).

## 7. Что нужно MacBook для образа (сам образ не менял)
- Пакет `gigaam` **не ставить из git**: код вендорен в `scripts/third_party/gigaam` (MIT, `7447938d`, без правок).
- Нужны в lock: `torch>=2.6` и `torchaudio>=2.6` одной сборки (PR #3: 2.11.0+cu126 — подходит), `hydra-core==1.3.*` (эталон 1.3.7), `omegaconf==2.3.*` (2.3.1), `sentencepiece` (0.2.2), `soundfile` (0.14.0), `tqdm`, `numpy==2.*`, `onnxruntime==1.23.*` (импортируется в `gigaam.onnx_utils`), `onnx` (upstream объявляет `1.19.*`, в PR #3 стоит 1.22 — для распознавания не используется, но это расхождение с метаданными upstream). `pyannote`, `flash_attn`, `transformers` не нужны (longform/flash не используются).
- `ffmpeg` в образе (уже есть) — бэкенд декодирует звук через него (`SONYA_FFMPEG` учитывается).
- Веса в S3: `models/gigaam/v3_e2e_rnnt.ckpt` и `models/gigaam/v3_e2e_rnnt_tokenizer.model` — хеши выше; загрузка — после согласования.

## 8. Чего не хватает / решения владельца
1. `whisper_base_config` (2 309 Б) — на Mac mini нет; публичный, хеш проверен в CI → MacBook может взять из закреплённого URL. Без него faster-whisper base (запасной путь) не запустится.
2. `webcam_detector.pt` — вариант A/B/C (п. 5).
3. Загрузка в S3 — после согласования (ключ, кто грузит).
4. GPU-проверка: скорость GigaAM на CUDA и совпадение с эталоном (`GIGAAM_REFERENCE_JSON` — транскрипт `kishlak_15m`, могу передать отдельно; это пользовательские данные, в git не кладу).
5. Educational v2: ветка `feat/educational-v2-overlays` ждёт (разработка остановлена владельцем), здесь не трогалась.

## Воспроизведение
Скрипты сверки и сравнения (вне репозитория): `~/SONYA/worker-prep/vast-models/scripts/` (`refcheck.py`, `asr_one.py`, `analyze.py`, `yolo_cmp.py`, `beep_ctrl.sh`); сводка `bench/summary.json` копией в `docs/mac_mini/asr_bench_summary.json` (только числа и короткие примеры расхождений).
