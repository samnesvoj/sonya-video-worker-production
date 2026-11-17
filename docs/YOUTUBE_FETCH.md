# YouTube links: how a video is fetched (2026-11-17)

## Why
YouTube answers **"Sign in to confirm you're not a bot"** to Vast data-centre IPs. On 2026-11-17 three user jobs failed,
the fleet replaced the host 11 times (~800 ₽, 0 clips). A probe on a fresh Vast host (Norway): every player client
(tv, web_safari, mweb, ios, android_vr, web, tv_simply, web_embedded) with the newest yt-dlp **and** a bgutil PO token
was refused on 3 live links. The IP is what is blocked, so the fix is to download somewhere YouTube trusts.

| path | where | speed (21-min 1080p) | notes |
|---|---|---|---|
| 1. direct | Vast worker | — | refused today; tried first (a few seconds), skipped for 1 h after a refusal |
| 2. relay `bot` | bot server 147.45.235.79 (Timeweb Moscow) | 26 s incl. upload (≈12 MB/s) | priority 0 |
| 3. relay `mac` | owner's Mac mini (Rostelecom, home IP) | ≈1.5 min (16 parallel fragments ≈2–3 MB/s; one connection is throttled to ~120 KB/s) | priority 1, S3 through an ssh tunnel to the VPS (the ISP drops the S3 host at times) |
| 4. cookies | relays, only on a "not a bot" retry | — | separate non-personal account, owner's consent 2026-11-17 |

The VPS 217.199.252.163 cannot reach youtube.com at all.

## Flow
1. Worker (`scripts/gpu_worker.py`): `fetch_source()`; on `SOURCE_BOT_CHECK` it writes `worker-v2/yt-relay/req/<job>.json`
   and waits (`scripts/yt_relay.fetch_via_relay`, heartbeats the job). The host keeps claiming other jobs.
2. Relays (`python -m scripts.yt_relay serve`) poll `req/`; the lowest priority alive relay takes the job, a higher one
   takes over when it failed / did not claim in 25 s / went silent for 4 min. Same code as the worker
   (`source_fetch.fetch_source`: length check first, the user's segment, ≤1080p) → upload to the job's `s3_input_key` →
   `st/<job>/<relay>.json` = done. A user-facing reason (private, too long, …) from any relay ends the job with that text.
3. All relays refused / none alive → `SOURCE_BOT_CHECK` with `retry=True` (bounded by `max_retries`), then a clear error.
4. Fleet: a bot-check host is banned for a week, replaced at most `SONYA_BOTCHECK_REPLACE_PER_DAY` (2) times a day.

## Monitoring
Each relay: heartbeat every 30 s (`hb/<relay>.json`), an hourly canary (probe + download of a 19 s public video by the
same code), the cookie check every 6 h (and right after the file changes), daily `pip -U --pre yt-dlp` + restart.
The dispatcher (`dispatcher/main.watch_youtube` → `accounting/yt_watch.py`) alerts through the accounting bot:
all paths down (every 2 h), a relay down > 15 min, a failed canary, cookies expired.

## Cookies (`tools/yt_cookies.py`, launchd `group.sonya.ytcookies`)
Owner drops `cookies.txt` into `/Users/agent/Business/SONYA/_incoming/yt-cookies/` → checked → login Keychain
(`group.sonya.yt-cookies`, chunks, via `security -i` stdin) → file zeroed + deleted → active entry written 0600 to
`~/SONYA/ytrelay/secret/cookies.txt` and `root@bot:/etc/sonya-ytrelay/cookies.txt`. Relays report "expired" → next pool
entry. Cookies are never printed and never leave our machines (not sent to Vast hosts).

## Operations
- Relay status: `python -m scripts.yt_relay status` (needs S3 env); bot: `journalctl -u sonya-ytrelay`; Mac: `~/SONYA/ytrelay/relay.log`.
- Switches: `SONYA_YT_RELAY=off` (worker: old behaviour, requeue), `SONYA_YT_DIRECT=off` (worker: straight to the relay),
  `SONYA_BOTCHECK_REPLACE_PER_DAY`, `SONYA_YT_WATCH=off` (no alerts).
- Deploy relays: `~/SONYA/ytfix/deploy_relays.sh`.
