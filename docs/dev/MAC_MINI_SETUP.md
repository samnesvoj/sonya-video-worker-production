# Mac mini как станция разработки режимов (делается на Mac mini, вами)

Я готовил это с MacBook: на Mac mini ничего не запускалось и не настраивалось. Скрипты безопасны для существующих проектов: **новая** папка, токен только в Keychain, настройки git только внутри нового клона.

## Шаги
1. **Токен (вы, в браузере).** github.com → Settings → Developer settings → Personal access tokens → **Fine-grained tokens** → Generate:
   Resource owner `samnesvoj` · Repository access **Only select repositories → `sonya-video-worker-production`** · Expiration 30–90 дней ·
   Permissions: **Contents: Read and write**, **Pull requests: Read and write**, **Actions: Read**, Metadata: Read. Больше ничего (ни Administration, ни Workflows, ни Secrets).
2. **Скрипт на Mac mini:** скачайте `tools/macmini_setup.sh` (или `git archive`), сначала `bash macmini_setup.sh --dry-run`, затем `bash macmini_setup.sh`. Токен вводится скрытым вводом и кладётся в Keychain (`sonya-video-worker-github`); в логах, argv и файлах его нет.
3. Скрипт делает **частичный клон** (`--filter=blob:none`, один branch, без тегов) в `~/SONYA-dev/sonya-video-worker-production`, ставит локальные настройки (сильное сжатие, `fetch.prune`, `gc.auto 0`), включает хук `pre-push`, создаёт `~/SONYA-models` и `~/SONYA-controls` (вне Git), venv и запускает два быстрых теста.
4. **Модели без скачивания:** `python tools/link_models.py --from <папки, где модели уже лежат> --dry-run`, затем без `--dry-run` — создаёт симлинки в `~/SONYA-models` (ничего не копирует и не качает; Trailer-модели игнорируются). `export MODELS_LOCAL_DIR=$HOME/SONYA-models`.
5. **Контрольные видео:** положите в `~/SONYA-controls/` (вне Git). Рекомендуемый набор — 1–2 коротких файла на режим (30–120 с). Опишите их локально в `~/SONYA-controls/README.txt` (файл, что проверяет, ожидаемое поведение).
6. **Pull Request** с Mac mini: `tools/sonya-gh pr create --base main` (обёртка берёт токен из Keychain на время одной команды и не меняет другие входы `gh`).

## Что НЕ делается
Существующие проекты, ветки, видео и результаты Mac mini не трогаются. Токен не даёт доступа ни к другим репозиториям, ни к настройкам, ни к секретам. Токен с правом записи технически может писать в `main` (защита веток недоступна на бесплатном приватном репозитории, см. PROTECTION.md) — от случайности защищает хук `pre-push` и процесс PR, а от попадания в Vast — то, что production берёт только SHA, указанный владельцем.
Отзыв доступа: удалить токен в GitHub (Settings → Fine-grained tokens → Revoke) и `security delete-generic-password -s sonya-video-worker-github`.
