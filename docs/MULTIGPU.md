# Multi-GPU: SONYA uses every card on a server

Branch `feat/multi-gpu` (from `release/first-vast-test` fc5c3ac). **Not live-tested on a multi-GPU Vast server yet.** Everything
below is covered by pytest with fake 1/2/4-GPU servers plus a dry-run simulation (`tools/multigpu_sim.py`). The default scheme
(1 warm RTX 3060 + bursts, 2 videos per server) works exactly as before.

## What it does

| | before | now |
|---|---|---|
| cards used on a 2x/4x server | only GPU 0. Every worker process ran on card 0 and the other cards sat idle | **every card**. Each worker process is pinned to ONE card with `CUDA_VISIBLE_DEVICES=<GPU uuid>` + `CUDA_DEVICE_ORDER=PCI_BUS_ID`, spread evenly (difference <= 1) |
| which code follows the pin | - | torch (GigaAM, YOLO), ctranslate2 (faster-whisper fallback) and ffmpeg `h264_nvenc` all open "device 0" through the CUDA driver, so they all land on the worker's card. No code inside a job knows about multi-GPU |
| one visible card | - | nothing is pinned: same environment as before |
| readiness | GPU 0 only: VRAM, torch matmul in-process, one NVENC probe | **every card at the same time**: VRAM >= `SONYA_MIN_VRAM_MB`, a CUDA context + matmul in its own pinned child process (timeout 180 s, a crashing card cannot take readiness down, and the child must see exactly that card), plus a real 0.5 s `h264_nvenc` encode on that card. One log line per card |
| a broken card | readiness failed and the whole server was lost | the card gets **no worker**, the server works on the others. Log: `WARNING ... NOT used: ...`, `working on 1 of 2 GPUs`; timeline mark `gpu_degraded` (uploaded to S3 `fleet-metrics/`, visible to the dispatcher). Readiness fails only when **no** card is usable |
| NVENC broken on one card | - | the card is still used and its clips render with libx264 (the worker trusts the readiness probe: `SONYA_NVENC_PROBED=fail`). With `SONYA_REQUIRE_NVENC=1` such a card is left out instead |
| how many videos at once | `SONYA_WORKER_CONCURRENCY=N` (dispatcher: 2) | still N when N is given. **`auto`** = what the hardware carries (see the budget below), capped by `SONYA_WORKER_CONCURRENCY_MAX` |
| GeForce NVENC session cap | 3/5/8 sessions per system, depending on the driver. Going over the cap failed the clip and with it the job | sessions are counted across all workers of the server (flock slots in `/run/sonya/nvenc`). An encode beyond the cap, or one the driver refuses, **runs with libx264 with the same Desktop CPU args, without an error**. Data-center cards (A40, L40S, A6000, A4000 ...) have no cap and nothing changes for them |
| res_sampler | GPU 0 only | server totals (one card: same fields as before) + `gpus` per card -> summary `per_gpu`: peak VRAM, peak/average utilisation, NVENC, our processes per card |
| dispatcher | single-GPU offers only | optional **POWER class** (multi-GPU offers, e.g. 2x A40 Belgium) for a rush. Off by default |

## Per-video budget (`scripts/ops/gpu_plan.py`, ONE formula for the worker and the dispatcher)

Measured in the load test 2026-11-09 (RTX 3060, kishlak_15m): about 1.5 GB VRAM, 3.5 GB RAM and 3-4 CPU threads per extra video.
The defaults add a margin on top:

| | default | env |
|---|---|---|
| VRAM per video | 2000 MB (+33 %) | `SONYA_VRAM_PER_JOB_MB` |
| VRAM kept free per card | 1024 MB | `SONYA_VRAM_RESERVE_MB` |
| RAM per video | 4096 MB (+17 %) | `SONYA_RAM_PER_JOB_MB` |
| RAM kept free | 4096 MB | `SONYA_RAM_RESERVE_MB` |
| CPU threads per video | 4 | `SONYA_CPU_THREADS_PER_JOB` |
| videos per card max | 8 | `SONYA_JOBS_PER_GPU_MAX` |
| NVENC sessions | GeForce: 3 (driver < 530), 5 (530+), 8 (550+), per system; others: no cap | `SONYA_NVENC_MAX_SESSIONS` (0 = no cap) |

`auto` = min(sum over usable cards of min(8, (free VRAM - 1 GB) / 2 GB), CPU threads / 4, (RAM - 4 GB) / 4 GB, 16), and at least 1.
CPU threads and RAM are read from the container's cgroup (quota / memory.max), not the whole host.
Examples (`python -m scripts.ops.gpu_plan simulate ...`):

- RTX 3060 + 12 threads + 32 GB: carries 3 (CPU-bound). The dispatcher still sends 2, as before.
- 2x A40 + 32 threads + 128 GB: **8 = 4 per card** (CPU-bound).
- 2x RTX 3090 + 32 threads: 8 = 4 per card. GeForce cap of 8 sessions for the server, extra encodes go to the CPU.
- 4x L40S + 64 threads + 256 GB: 16 = 4 per card.

## How to turn it on

Worker (any server, no dispatcher change needed):
- `SONYA_WORKER_CONCURRENCY=auto` (+ `SONYA_WORKER_CONCURRENCY_MAX=8`) sizes the worker from the hardware. A number keeps its
  meaning (exactly N workers, spread over the cards).

Dispatcher, POWER class for rushes (VPS env; default `SONYA_POWER_ENABLED=false` = the current scheme):

| env | default | meaning |
|---|---|---|
| `SONYA_POWER_ENABLED` | false | on/off |
| `SONYA_POWER_GPUS` | A40,L40S,L40,RTX A6000,RTX 6000Ada | Vast `gpu_name` |
| `SONYA_POWER_MIN_GPUS` / `MAX_GPUS` | 2 / 4 | Vast `num_gpus` |
| `SONYA_POWER_MIN_TOTAL_VRAM_GB` | 48 | all cards together |
| `SONYA_POWER_MIN_CPU_CORES` | 16 | `cpu_cores_effective` (EPYC/Xeon allowed here) |
| `SONYA_POWER_MIN_CPU_RAM_GB` | 32 | |
| `SONYA_POWER_MAX_DPH` | 1.00 | USD/h for the whole offer |
| `SONYA_POWER_JOBS_PER_INSTANCE` | 8 (safe until the A40 test gives the final number); 0 = from the offer (same formula) | slots the fleet counts |
| `SONYA_POWER_MIN_GAP` | 0 = jobs_per_instance + 1 (3) | use power when this many jobs are still uncovered |

Rules: the warm is still exactly one European RTX 3060 (a power server is never the warm and is DESTROYED when idle). Demand is
covered in **video slots**: a default server counts `jobs_per_instance` (2, as before), a power server counts its own slots
(stored per instance). When >= `power_gap` jobs are uncovered, the next create is ONE power server. Choice order: Europe first,
then the cached image, then price per video slot per hour, then the fastest internet. No power offer -> default-class bursts as
before. `max_instances` and `max_creates_per_hour` still apply. The power server gets `SONYA_WORKER_CONCURRENCY=auto` +
`SONYA_WORKER_CONCURRENCY_MAX=<slots>`, so it never runs more than the fleet counted, and runs fewer when a card is broken.

Spare offers per class and the paused POWER reserve (both off by default): docs/FLEET_SPARES.md.

Dry run of a rush (real FleetManager + real worker plan, virtual time):

```
python tools/multigpu_sim.py --jobs 12                # default scheme: 3x RTX 3060 class, all done 380 s
                                                      # power on: warm 3060 + 2x A40 (4+4 workers per card), all done 270 s
python tools/multigpu_sim.py --jobs 12 --broken-card  # A40 card 1 broken: 8 workers on card 0, nothing fails
```

## Results unchanged

- Clip selection, crop and subtitles do not look at the GPU index. A render through the new guard is byte-identical to the old
  code (checked with real ffmpeg: same MD5). The NVENC -> CPU fallback is byte-identical to the plain CPU render.
- With no cap (data-center cards), or with NVENC free, the ffmpeg command is exactly the old one.
- On one card nothing is pinned. The added env (`SONYA_GPU_INDEX`, `SONYA_NVENC_*`) does not change any output.

## Not done / honest

- **Not run on a real multi-GPU Vast server.** First live check: rent 2x A40 (or set `SONYA_WORKER_CONCURRENCY=4` on any 2-GPU
  box) and look for the readiness lines `GPU 0 ... cuda OK ..., nvenc: OK` and `GPU 1 ...`, then `plan: 4 worker(s) ... GPU0 x2, GPU1 x2`.
  With `SONYA_RES_SAMPLER=1`, `per_gpu` in the summary must show both cards busy.
- The GeForce session numbers (3/5/8 per system) are conservative published values. If they are wrong the guard still turns a
  refused encode into a CPU encode. On Vast another renter's container may share the host's cap.
- A card that fails AFTER start (Xid error) kills its worker -> run_worker stops the server -> the fleet replaces it (as before).
- Trailer modes (not in the pool) still call ffmpeg directly.
- The power class slot count is an estimate from the offer. If the container gets less (cgroup), the worker runs fewer workers
  and the fleet waits for those slots a bit longer.

## Live test 2x A40 (Belgium, EPYC 7662 128 threads, 2026-11-09)

Both cards used and balanced (readiness `GPU 0/1 ... cuda OK, nvenc: OK`, `GPU0 x12, GPU1 x12`, per-card VRAM equal).
Old code: 16 at once lost 10 of 48 clips, 24 lost 19 of 72 - the container ran out of THREADS (pids.max 31232; ffmpeg /
OpenCV "Resource temporarily unavailable", Python "can't start new thread"), not VRAM (30-40 % used). Fix `0404dd3`:
per-worker CPU window (`SONYA_CPU_AFFINITY`, sched affinity, max(16, 2 x CPUs / workers); off below 32 CPUs, for 1-2
workers, or `SONYA_CPU_WINDOW=0`). With it: 8 -> 269 s, 16 -> 355 s, **32 -> 438 s (263 videos/h, 0 clips lost, all
NVENC)**, ~9.7k threads. `auto` still caps at 16 per server (AUTO_MAX); 32 needs AUTO_MAX + SONYA_JOBS_PER_GPU_MAX raised.
Full report: worker-prep/vast-run/A40_REPORT.md.
