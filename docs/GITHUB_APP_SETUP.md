# GitHub App для Vast: что нужно сделать именно вам (я ничего не создавал)

Назначение: dispatcher на вашем VPS выпускает **токен на один инстанс** (только чтение одного репозитория, живёт до часа). Постоянных ключей на Vast нет.

## Шаги (≈10 минут)
1. github.com → **Settings → Developer settings → GitHub Apps → New GitHub App**.
   - Name: `sonya-release-reader` · Homepage URL: любой (например профиль) · **Webhook: снять галку Active**.
   - **Repository permissions → Contents: Read-only** (больше ничего; Metadata: Read-only выставится сам).
   - Where can this app be installed: **Only on this account** → Create.
2. На странице приложения: запишите **App ID**. Внизу **Generate a private key** → скачается `.pem` (его нельзя восстановить).
3. **Install App** (слева) → ваш аккаунт → **Only select repositories → `sonya-video-worker-production`**. После установки в адресе будет `/installations/<число>` — это **Installation ID**.
4. Ключ `.pem` положите **только** на VPS dispatcher: `/etc/sonya-fleet/github-app.pem`, владелец пользователь dispatcher, `chmod 600`. Не в Git, не в чат, не на Vast, не в CI. Локальную копию с Mac удалите или храните в Keychain/менеджере паролей.
5. Переменные dispatcher (не секретные имена, значения — ваши): `GH_APP_ID`, `GH_APP_INSTALLATION_ID`, `GH_APP_PRIVATE_KEY_FILE=/etc/sonya-fleet/github-app.pem`.
6. Проверка без Vast: `python tools/mint_release_token.py --repo sonya-video-worker-production | wc -c` (должно вернуть длину токена; сам токен не показывайте). Затем `curl -sI -H "Authorization: Bearer $(python tools/mint_release_token.py --repo sonya-video-worker-production)" https://api.github.com/repos/samnesvoj/sonya-video-worker-production | head -1` → `200`, а запрос к любому другому вашему репозиторию → `404`.
7. Защита релизов: GitHub Pro + `bash tools/apply_github_protection.sh` (docs/dev/PROTECTION.md).

## Что получает Vast
Токен ≤ 1 ч, `contents:read`, один репозиторий, в `env` создаваемого инстанса. Владелец чужого хоста может прочитать его и код в течение этого часа — это принятое ограничение (код он видит и так). После скачивания bootstrap удаляет токен из окружения, на диске его нет.

## Ротация и отзыв
Скомпрометирован `.pem` → App settings → Private keys → Delete, сгенерировать новый, заменить на VPS. Остановить выдачу токенов мгновенно: Uninstall App.
