# Процесс доработки режима

```
Vast-тест → замечание → задание Claude на Mac mini → ветка mode/<режим>/<кратко> → правка → тесты → контрольное видео vs прошлая версия
→ PR (с report.md) → CI на GitHub → MacBook/вы смотрите и одобряете → merge в main → (отдельно, по вашей команде) релиз → GPU-проверка → SHA в Vast
```
Production **не обновляется** от push и merge: Vast получает только SHA, который вы явно указали.

## Команды (на Mac mini и MacBook)
| Что | Команда |
|---|---|
| Режимы и тесты | `python tools/mode_dev.py list` · `python tools/mode_dev.py deps virality` |
| Ветка под исправление | `python tools/mode_dev.py branch virality fix-crop-jitter` (от `origin/main`, имя `mode/<режим>/<slug>`) |
| Тесты режима | `python tools/mode_dev.py test virality` (общие «gate»-тесты политики/контракта/bootstrap + тесты режима) |
| Сравнение с прошлой версией | `python tools/compare_runs.py --mode virality --input ~/SONYA-controls/ctrl01.mp4 --baseline origin/main` → `results/…/report.md` |
| PR | `git push -u origin mode/virality/fix-crop-jitter` · `tools/sonya-gh pr create --base main --fill` |
| Откат правки | `git revert <sha>` в новой ветке → PR (никогда `reset --hard` на общих ветках) |

## Правила
- **Одна ветка — один режим — одна проблема.** Меняйте только то, что нужно. Файлы в `scripts/shared/*` используются несколькими режимами: `mode_dev deps <режим>` показывает общие пути; после правки shared прогоните тесты **каждого** режима, который его импортирует (`tools/modes_map.json`).
- Проверенные алгоритмы не переписываем «для красоты». Крупный рефакторинг — только по согласованию.
- Не трогать: `scripts/mode_policy.py` (запрет Trailer, список режимов), `release/`, `bootstrap/`, `deploy/`, `dispatcher/`, `accounting/`, `.github/`, `requirements/` — это зона владельца (CODEOWNERS).
- Видео, транскрипты, модели, токены **не коммитятся** (`.gitignore`, gitleaks в CI). В PR прикладываются только `report.md`/`report.json` и описание.
- Результаты видеотестов: папка `results/` (в `.gitignore`) хранит видео локально; `report.json` — маленький отчёт, его прикладывают к PR. Ссылку на сами видео оставляйте локальной (путь на Mac mini).
- Новый режим становится «включённым по умолчанию» только после реального видео + GPU-проверки: отдельный PR, меняющий `mode_policy.DEFAULT_ENABLED_MODES`, `release/manifest.json` и `tools/modes_map.json`.

## Ограничения сравнения
`compare_runs.py` ловит грубые регрессии (число клипов, длительность, разрешение, потеря звука, SSIM, изменения JSON) — **качество всё равно оценивается глазами**. Прежняя версия берётся из git-ref (по умолчанию `origin/main`) во временном worktree; ASR/модели используются локальные (без скачивания).
