# Security audit (legend: ✅ done+tested · 🔎 checked by grep/code · ⚠️ partial · ❌ not done)
| # | Item | Status | Notes |
|---|---|---|---|
| 1 | Secrets in Git/archive/Docker | 🔎 | Pattern scan of tree and full history before first push (see STATUS); CI runs gitleaks on full history. Docker layers not scanned (no Docker on this Mac). `.dockerignore` is deny-all + 4 files. |
| 2 | Keys/tokens/source in startup logs | ✅ | Token redaction (raw/base64) tested; smoke: GitHub token and WORKER_SECRET absent from logs; entrypoint prints no env (the archived entrypoint printed BACKEND_API_URL). |
| 3 | Videos/transcripts/URLs/PII in logs | ⚠️ | Worker logs job_id, mode, filenames, S3 keys (contain user_id). Per-mode loggers not audited line by line. |
| 4 | Worker↔server auth | ⚠️ | Shared static `WORKER_SECRET` Bearer (as production). Per-instance token needs a server change (not done). |
| 5 | Ports | ✅ | none inbound; health on 127.0.0.1 only (tested). |
| 6 | Foreign requests to Worker API | ✅ | worker serves no API. |
| 7 | File rights / user | ✅ by Dockerfile | uid 10001, umask 077; not verified in a running container. |
| 8 | Dependencies | ✅/⚠️ | hashed lock; pip-audit: no known vulnerabilities; torch/torchvision (+cu124) not covered by pip-audit. torch 2.5.1 predates a `torch.load` RCE fix in 2.6 as far as I recall — verify the advisory; models only come from your S3; upgrade after GPU validation. |
| 9 | Injection / traversal / ffmpeg args | 🔎 | no `shell=True`, `os.system`, `eval`, `pickle.load` in scripts/modes; ffmpeg called with argument lists; per-job `mkdtemp`. No fuzzing of params→ffmpeg. |
| 10 | Temp retention | ✅ | per-job `rmtree` in `finally`; SIGTERM path; `sonya_*` wiped on start and shutdown. kill -9 untested (wiped at next start). |
| 11 | GitHub credentials | ✅ | token only in the child git env; never argv/URL/file/.git/config; unset after bootstrap; App key only on trusted hosts. |
| 12 | Failures/restarts | ✅ | idempotent bootstrap, cached verified release, fail-closed, ready-gate, requeue on SIGTERM. |
Host-owner threat: sees code, env, memory, videos; no promise otherwise. Open: per-instance API token, token broker, signed S3 URLs instead of S3 keys on host, trivy scan after first build, egress restriction.
