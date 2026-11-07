# GitHub App для Vast: точная инструкция (я ничего не создавал)

Цель: dispatcher на вашем VPS выпускает **токен на один инстанс** — только чтение одного репозитория, живёт до часа. Постоянных ключей на Vast нет. Приложение видит **только** `samnesvoj/sonya-video-worker-production`.

## 1. Создать приложение (≈ 5 минут, в браузере)
1. github.com → фото профиля → **Settings** → внизу слева **Developer settings** → **GitHub Apps** → **New GitHub App**.
2. **GitHub App name:** `sonya-release-reader` (если занято — добавьте суффикс). **Homepage URL:** адрес вашего профиля GitHub.
3. **Webhook:** снять галку **Active**.
4. **Permissions → Repository permissions:** выставить **Contents: Read-only**. Больше ничего не трогать (Metadata: Read-only появится сам). Account и Organization permissions — без доступа.
5. **Where can this GitHub App be installed?** → **Only on this account**.
6. **Create GitHub App.**
## 2. Ключ и идентификаторы
7. На странице приложения запишите **App ID** (число вверху).
8. Раздел **Private keys → Generate a private key** → скачается файл `.pem`. Восстановить его нельзя; хранить только как описано ниже.
## 3. Установить приложение на ОДИН репозиторий
9. Слева **Install App** → ваш аккаунт → **Install** → **Only select repositories** → выбрать **`sonya-video-worker-production`** → **Install**.
10. **Installation ID** — число в адресе страницы после установки (`…/settings/installations/<ЧИСЛО>`); либо командой под вашим админским логином: `gh api /user/installations --jq '.installations[] | "\(.id) \(.app_slug)"'`.
## 4. Положить ключ туда, где ему место (не на Vast!)
11. Только на VPS dispatcher: `sudo install -m 600 -o <пользователь-dispatcher> sonya-release-reader.private-key.pem /etc/sonya-fleet/github-app.pem`. Затем удалить скачанный файл с Mac (`rm -P`) или перенести в менеджер паролей. Не в Git, не в чат, не в Docker, не в CI, не в Template.
12. Переменные окружения dispatcher (значения — ваши): `GH_APP_ID`, `GH_APP_INSTALLATION_ID`, `GH_APP_PRIVATE_KEY_FILE=/etc/sonya-fleet/github-app.pem`.
## 5. Проверка, что права действительно минимальны (без расходов, без записи, токен не печатается)
```bash
cd <checkout> && GH_APP_ID=… GH_APP_INSTALLATION_ID=… GH_APP_PRIVATE_KEY_FILE=/etc/sonya-fleet/github-app.pem \
python tools/verify_github_app.py --repo samnesvoj/sonya-video-worker-production --other-repo samnesvoj/sonya-production
```
Ожидается пять строк `PASS`: права ровно `contents:read` (+`metadata:read`) · токен ограничен одним репозиторием · срок жизни ≤ 61 мин · целевой репозиторий читается · **другой ваш репозиторий недоступен (404)**. Любой `FAIL` → не запускать тест, исправить права приложения.
## 6. Эксплуатация
- Токен выпускает dispatcher на каждый инстанс (`tools/mint_release_token.py`); на Vast он живёт ≤ 1 ч в окружении контейнера, bootstrap использует его один раз и удаляет; на диске его нет.
- Ротация ключа: App settings → Private keys → Delete, затем сгенерировать новый и заменить файл на VPS. Мгновенно прекратить выдачу токенов: App settings → **Uninstall** (или Suspend).
- Остаточный риск: владелец арендованного хоста может прочитать токен и код в течение часа (код он видит и так). Улучшение (не сделано): token-broker на VPS.
