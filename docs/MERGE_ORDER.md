# Безопасный порядок слияния PR #1 → PR #2 (ничего не слито)

Структура: `main` (f97ee9e) ← **PR #1** `feat/idle-stop-and-cost-accounting` ← **PR #2** `perf/fast-start` (база PR #2 — ветка PR #1).
Порядок и способ слияния **отрепетированы** в одноразовых копиях репозитория (без push; временные ветки удалены):

| Способ | Результат репетиции |
|---|---|
| **A. PR #1 «Create a merge commit» → затем PR #2 («Create a merge commit»)** | 0 конфликтов, 735 тестов зелёные, `split_lock`/manifest проверки проходят — **рекомендуется** |
| B. PR #1 «Squash and merge» → затем обычное слияние PR #2 | **4 конфликта** (`ci.yml`, `Dockerfile`, `tools/ci_docker_checks.sh`, `tools/mint_release_token.py`): PR #2 содержит исходные коммиты PR #1, а в `main` лежит их «склейка» |
| C. PR #1 squash → PR #2 перебазировать `git rebase --onto main <старый HEAD PR #1> perf/fast-start` + `git push --force-with-lease` | 0 конфликтов, дерево файлов идентично; допустимо, но требует force-push ветки PR #2 |

## Пошагово (способ A)
1. Дождаться зелёного CI **на текущем HEAD PR #1** и PR #2; просмотреть дифф PR #1 (бухгалтерия, флот, dispatcher-патчи — всё выключено по умолчанию).
2. GitHub → PR #1 → «Merge pull request» → **Create a merge commit** (не squash, не rebase). Ветку PR #1 **не удалять** до шага 4.
3. PR #2: Edit → поменять base с `feat/idle-stop-and-cost-accounting` на `main` (или `gh pr edit 2 --base main`). Диф PR #2 после смены базы не должен содержать коммитов PR #1; CI перезапустится — дождаться зелёного.
4. PR #2 → **Create a merge commit** → теперь можно удалить обе ветки.
5. После слияния: `main` содержит всё; **релиза нет** — `SONYA_RELEASE_SHA` в Vast/dispatcher по-прежнему закреплён вами; релиз только после GPU-теста (`docs/dev/RELEASE_AND_ROLLBACK.md`, workflow `release`).

## Что проверить перед слиянием
- Нет `release/models.lock.json` с выдуманными значениями (пока пуст — это нормально; дальше он заполняется PR после загрузки моделей).
- Защита `main` недоступна на бесплатном приватном репозитории (нужен GitHub Pro): слияние выполняете вы вручную; прямые push в `main` блокирует только локальный хук на Mac mini.
- В `main` не попадают `results/`, видео, модели (`.gitignore`, gitleaks).
## Если нужно слить в другом порядке
PR #2 нельзя сливать первым «как есть»: он принесёт в `main` и коммиты PR #1 (его база). Это допустимо технически (результат тот же), но смешивает ревью — лучше способ A.
