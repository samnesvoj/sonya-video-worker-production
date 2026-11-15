# Instance disk size (SONYA_DISK_GB) — 2026-11-15

**Default: 40 GB** (`dispatcher/config.py`, override `SONYA_DISK_GB` in `/etc/sonya-fleet/fleet.env`).

## What lives on the disk
| item | size |
|---|---|
| worker image, unpacked | ~7.3 GB |
| models (S3 cache) + ad banner assets | ~1.1 GB |
| one job, source ≤ 20 min (≤ 1440p, may be VP9 → normalized H.264 copy) | ≤ 2–3 GB |
| one job, source 1 h (> 20 min → fetched at 1080p H.264, no normalize) | ~1 GB source + clips ≈ 2–5 GB |
| one job, admin 4 h source at 1080p H.264 (download + merge peak) | ~14 GB |

Two jobs per server (concurrency 2): worst realistic case = 4 h admin job + a normal job ≈ 8.4 + 14 + 3 ≈ 26 GB → 40 GB leaves ~14 GB headroom.

The 2026-11-15 incident (55-min YouTube, 1440p VP9 3.7 GB + its re-encode on a 30 GB disk) is now prevented by the code, not by a bigger disk:
- sources > 20 min (`SONYA_LONG_SOURCE_SEC=1200`) are fetched at ≤ 1080p H.264 → no normalize copy;
- free-space check before download (`SOURCE_NO_SPACE`) and normalize (`NORMALIZE_NO_SPACE`), clear errors;
- worker does not claim below `SONYA_MIN_FREE_GB_TO_CLAIM` (6 GB); an S3 input bigger than free/2.5 is requeued;
- each job's temp files live in its own folder, removed after the job; folders of dead worker processes are swept.

## Price limits (checked)
- `max_dph` $0.18/h includes our disk: storage_cost × disk_gb / 720. At the storage cap $0.20/GB-month: 40 GB = +$0.011/h, 60 GB = +$0.017/h (fewer offers fit under $0.18).
- Paused warm/POWER-reserve server bills disk only: 40 GB × ≤ $0.20 = ≤ $8 ≈ 656 ₽/month (60 GB: $12 ≈ 984 ₽). POWER reserve cap `SONYA_POWER_RESERVE_MAX_STORAGE_RUB=500` → at 40 GB hosts with storage ≤ $0.152/GB-month qualify (60 GB: ≤ $0.10 — half the hosts drop out).
- Offer filter `disk_space >= disk_gb`: 40 GB is available on practically all RTX 3060 hosts.

60 GB (manual setting of 2026-11-15) works but costs more on every server and narrows the reserve for no gain after the code fixes; recommended value: 40.
