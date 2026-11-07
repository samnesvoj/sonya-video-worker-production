# Защита main и release-тегов

**Факт (проверено API 2026-11-07):** rulesets и branch protection на приватном репозитории бесплатного плана недоступны (`403 Upgrade to GitHub Pro or make this repository public`). Поэтому сейчас защита `main` и тегов **не включена и не может быть включена без решения владельца**.

Готово к применению: `.github/rulesets/main.json` (только через PR, четыре обязательных проверки, без force-push/удаления, линейная история) и `release-tags.json` (теги `worker-v*` нельзя менять/удалять), `.github/CODEOWNERS`, `tools/apply_github_protection.sh` (`--dry-run` проверен). Применять после перехода на GitHub Pro (≈ $4/мес) или иного решения: `bash tools/apply_github_protection.sh` под вашим админским логином.

## Что защищает уже сейчас (без Pro)
1. **Vast берёт только явный SHA** (bootstrap отвергает ветки/HEAD); push в `main` с Mac mini не меняет production никак.
2. Релизный workflow создаёт теги только для SHA на `main` с зелёным CI; Vast теги не использует.
3. Токен Mac mini ограничен одним репозиторием; локальный хук `pre-push` блокирует push в `main` и `worker-v*` (не граница безопасности, защита от ошибки).
4. Токен Mac mini **не имеет** прав на Workflows/Secrets/Administration — workflow-файлы с Mac mini изменить нельзя.
Остаточный риск без Pro: держатель токена Mac mini может вручную запушить в `main` (это не попадёт в Vast, но засорит историю). Закрывается Pro-rulesets.
