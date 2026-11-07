# Синхронизация Mac mini ↔ MacBook ↔ Vast

```
Mac mini (режимы)  ──branch+PR──▶  GitHub (private)  ◀──review/merge, release──  MacBook (владелец, релизы, dispatcher/бухгалтерия)
     ▲ fetch (частичный)                 │ CI: pytest · gitleaks · pip-audit+OSV · docker build + in-image checks
     │                                   └──── SHA (только по вашей команде) ──▶ Vast Template/dispatcher ──▶ новые инстансы (bootstrap по SHA)
```
- **Вниз к Mac mini:** `git fetch origin main` (частичный клон: качаются только изменённые объекты). Большие данные (модели, видео) в Git не ходят; они лежат локально и в вашем S3.
- **Вверх:** только ветки `mode/<режим>/<slug>` и PR. На MacBook ветки владельца (`feat/*`, `release/*`) — те же правила, PR → main.
- **К Vast:** только SHA, выбранный владельцем после GPU-проверки; изменения Mac mini туда не «доезжают» сами.
- Конфликт параллельных работ: режимы живут в разных папках; общие `scripts/shared/*` правятся осторожно и по согласованию, `rebase` на свежий `origin/main` перед PR.
