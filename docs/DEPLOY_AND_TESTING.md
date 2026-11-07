# Тестирование и безопасный деплой (ничего из этого ещё не сделано на VPS/Vast)

## Локально (MacBook, без денег)
```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install pytest pyyaml requests pydantic fastapi boto3 numpy pillow python-dotenv httpx scenedetect cryptography
pip install --no-deps opencv-python==4.14.0.94 opencv-contrib-python==4.14.0.94 opencv-python-headless==4.14.0.94
python -m pytest -q                         # воркер + fleet + accounting + интеграция с ботом
python tools/simulate.py                    # сутки на поддельном Vast, примеры отчётов
python tools/cost_model.py                  # пересчёт docs/COSTS.md
gitleaks detect --source . --redact         # секреты (правила: .gitleaks.toml)
```
## Порядок выката (каждый шаг обратим; Vast-расходы только на шагах 4–5)
1. **GitHub:** push ветки → дождаться зелёного CI (тесты, gitleaks, pip-audit, docker build). Первый CI на `main` (f97ee9e) был красный (см. STATUS) — исправления в этой ветке.
2. **GitHub App** (`docs/GITHUB_APP_SETUP.md`), ключ на VPS (`chmod 600`), `release/models.lock.json` через `tools/make_models_lock.py`.
3. **Dry-run на VPS:** выложить пакеты `dispatcher/`, `accounting/`, `tools/`; применить `gpu_dispatcher-fleet.patch` к копии кода в стейдже; `SONYA_FLEET_ENABLED=true SONYA_FLEET_DRY_RUN=true` — dispatcher только пишет `dry_run_*` события (ни одного write-вызова Vast), очередь читает read-only. Сверить решения с реальной очередью.
4. **GPU-тест** (`docs/GPU_TEST_PLAN.md`, с вашего «да»): один инстанс, тестовое задание.
5. **Боевой режим для пула virality+beep:** `SONYA_FLEET_DRY_RUN=false`; старый пул продолжает обслуживать остальные режимы; следить за `fleet_events`.
6. Бот: патч + `VAST_REPORTS_ENABLED=true`, отчёты начнут приходить со следующего 00:05 МСК.
**Откат:** `SONYA_FLEET_ENABLED=false` (пул замирает, режимы возвращаются старому dispatcher), `VAST_REPORTS_ENABLED=false`; `git apply -R` патчей. Работающие инстансы пула уничтожить вручную/`destroy` после остановки.
