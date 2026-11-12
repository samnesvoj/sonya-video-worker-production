# Fleet: spare offers per class + the paused POWER reserve (2026-11-09)

Both are **OFF by default**. Off = the fleet does exactly what it did before (one offer per create, no extra Vast searches,
no reserve server) - `tests/fleet/test_spares.py::test_options_off_nothing_new_happens` and the older fleet tests.
Code: `dispatcher/fleet.py` (`_refresh_spares`, `_candidates`, `_wake_reserve`, `_prime_reserve`), `dispatcher/launcher.py`
(`rank_offers`, `rank_power_offers`, `reserve_ok`), `dispatcher/store.py` (`fleet_spares`), `dispatcher/status.py`.

## Spare offers (`SONYA_SPARES_ENABLED=true`)

- Every `SONYA_SPARES_REFRESH_SECONDS` (300) the dispatcher ranks the offers Vast has now and keeps the best
  `SONYA_SPARES_COUNT` (5) per class:
  - `warm` - European RTX 3060 class with storage <= `SONYA_MAX_STORAGE_GB_MONTH` (may hold the paused warm);
  - `burst` - any default-class offer (same rule as before: Europe -> 3060 class -> cached image -> time x price -> internet);
  - `power` - multi-GPU offers (only when `SONYA_POWER_ENABLED=true`);
  - `reserve` - power offers whose pause costs <= `SONYA_POWER_RESERVE_MAX_STORAGE_RUB` (only with `SONYA_POWER_RESERVE=true`).
- Banned hosts (host error / create timeout / failed resume / stalled worker, 1 h) are never in a list.
- **Cached image first**: every machine where our container reached "running" is remembered in `fleet.db` (`good_machines`,
  30 days). Within the same region + class such a host comes before a cheaper one (the image pull is skipped / faster).
- A create Vast refuses (offer rented a second ago, `no_such_ask` ...) goes **straight to the next offer of the ranking in the
  same tick** (at most `SONYA_SPARES_COUNT` tries per server; event `create_failed_next`). A warm whose resume call is refused
  is destroyed, its host banned and a new server created in the same tick (the 90 s resume timeout already worked so).
- Where to see the list: event `spares` in `fleet_events` (written when a list changes), the dispatcher log line
  `spares <class>: ...` at every refresh, and `python -m dispatcher.main status` (reads `fleet.db` only: no Vast call, no DB).
- Cost: two read-only Vast searches per 5 min. Nothing is rented for the list.

## POWER reserve (`SONYA_POWER_RESERVE=true`)

One multi-GPU server (owner's choice: 2x A40 Belgium, ~$0.94-1.10/h, storage $0.20/GB-month) kept **stopped** like the warm:
- Primed after the warm exists and nothing is happening: created, boots (image + release + models), stopped after the boot
  grace (10 min, ~$0.18 at $1.10/h). From then on only its disk bills: 30 GB x $0.20 = $6 = ~492 RUB/month.
- Only hosts whose pause costs <= `SONYA_POWER_RESERVE_MAX_STORAGE_RUB` (500) per month (disk_gb x storage price x
  `SONYA_USD_RUB`, default 82 - set the real rate: at $0.20/GB-month the cap holds while 1 USD <= 83 RUB).
- Woken when queued + running jobs >= `SONYA_POWER_RESERVE_WAKE_AT` (0 = `max_instances x jobs_per_instance + 1` = 7 with
  3 x 2: more than the warm RTX 3060 + the allowed bursts carry). It counts its `SONYA_POWER_JOBS_PER_INSTANCE` (8) slots, so
  fewer RTX 3060 bursts are created. The reserve is **outside** `SONYA_FLEET_MAX_INSTANCES`.
- After the rush: idle 60 s -> stopped again (not destroyed).
- Not running 90 s after the wake (`SONYA_RESUME_TIMEOUT_SECONDS`: its GPUs were rented by someone else) or the resume call is
  refused -> destroyed, host banned 1 h, and the **next reserve-eligible power offer is created in the same tick** (event
  `reserve_replaced`). That new server serves the rush and becomes the paused reserve afterwards.
- Stale release -> destroyed and primed again. `SONYA_POWER_RESERVE=false` later -> the stopped reserve is destroyed after
  `SONYA_STOPPED_RETENTION_SECONDS` (30 min).
- Independent of `SONYA_POWER_ENABLED` (that one creates NEW power servers on a rush; both can be on).
- Price cap: `SONYA_POWER_MAX_DPH` (1.00). Belgian 2x A40 cost $1.097/h on 2026-11-09 -> set `SONYA_POWER_MAX_DPH=1.20`, or no
  reserve is found (`reserve_failed` every 10 min).

## Settings

| env (dispatcher) | default | meaning |
|---|---|---|
| `SONYA_SPARES_ENABLED` | `false` | ranked spare offers per class + next offer on a refused create |
| `SONYA_SPARES_COUNT` | `5` | offers kept per class = creates tried per server (1-10) |
| `SONYA_SPARES_REFRESH_SECONDS` | `300` | how often the lists are refreshed |
| `SONYA_POWER_RESERVE` | `false` | one paused power server for rushes |
| `SONYA_POWER_RESERVE_WAKE_AT` | `0` = max_instances x jobs_per_instance + 1 (7) | queued + running jobs that wake it |
| `SONYA_POWER_RESERVE_MAX_STORAGE_RUB` | `500` | pause cost cap per month |
| `SONYA_USD_RUB` | `82` | rate used only for that cap |
| `SONYA_POWER_JOBS_PER_INSTANCE` | `8` | videos at once on a power server (reserve included); 0 = from the offer |
| `SONYA_POWER_MAX_DPH` | `1.00` | USD/h cap for a power offer (2x A40 Belgium: set 1.20) |

Tests: `tests/fleet/test_spares.py` (fake Vast): ranking, next offer on refusal, cached-image priority, reserve wakes /
sleeps / is replaced, dry-run, nothing new when off.
