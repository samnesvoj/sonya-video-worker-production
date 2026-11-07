#!/usr/bin/env python3
"""
Start-up TIME MODEL (a calculator, not a measurement). Inputs marked MEASURED come from the CI image report and the real Vast offer
snapshot; everything else is an ASSUMPTION you can override. Prints Markdown tables for docs/FAST_START_REPORT.md.
  python tools/startup_model.py [--wire-gb 3.75] [--unpacked-gb 7.48]
"""
import argparse, json, pathlib, statistics

ROOT = pathlib.Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--wire-gb", type=float, default=3.75)       # MEASURED (CI image report, gzip-compressed layers)
ap.add_argument("--unpacked-gb", type=float, default=7.48)   # MEASURED
ap.add_argument("--extract-mbs", type=float, default=200.0)  # ASSUMPTION: gunzip + untar to host disk
a = ap.parse_args()

offers = json.loads(next(ROOT.glob("data/vast_offers_*.json")).read_text())["offers"]
bw = sorted(o["inet_down"] for o in offers)                    # MEASURED: advertised inet_down Mbit/s of 62 real offers
pct = lambda p: bw[int(p * (len(bw) - 1))]
print(f"Advertised download speed of {len(bw)} real Vast offers (Mbit/s): min {bw[0]:.0f} | p10 {pct(.1):.0f} | p25 {pct(.25):.0f} | median {pct(.5):.0f} | p75 {pct(.75):.0f}\n")

fmt = lambda s: f"{s:.0f} s" if s < 120 else f"{s/60:.1f} min"
print("### Image pull + unpack, time = wire / throughput + unpacked / extract speed  (throughput = what the host really gets from the registry: ASSUMED)")
print("| image | wire GB | unpacked GB | 25 MB/s | 50 MB/s | 100 MB/s | 200 MB/s |\n|---|---|---|---|---|---|---|")
for name, wire, unp in (("current default (3 layers)", a.wire_gb, a.unpacked_gb), ("without triton (-0.56 GB unpacked, ~-0.26 GB wire; candidate)", a.wire_gb - 0.26, a.unpacked_gb - 0.56),
                        ("default + 5 GB models baked in", a.wire_gb + 5, a.unpacked_gb + 5), ("only the 3 top layers if the CUDA base is host-cached (see nvidia report)", None, None)):
    if wire is None:
        continue
    print(f"| {name} | {wire:.2f} | {unp:.2f} | " + " | ".join(fmt(wire * 1000 / r + unp * 1000 / a.extract_mbs) for r in (25, 50, 100, 200)) + " |")
print(f"\nExtraction assumed {a.extract_mbs:.0f} MB/s (adds {fmt(a.unpacked_gb*1000/a.extract_mbs)} for the current image regardless of bandwidth).\n")

print("### Model download (size is UNKNOWN until models.lock.json is filled; S3 is Timeweb, far from many Vast hosts: throughput ASSUMED)")
print("| models total | 10 MB/s | 25 MB/s | 50 MB/s | 100 MB/s |\n|---|---|---|---|---|")
for gb in (1, 3, 5, 10):
    print(f"| {gb} GB | " + " | ".join(fmt(gb * 1000 / r) for r in (10, 25, 50, 100)) + " |")
print("\nSHA-256 verification: tens of seconds at most for 10 GB on one core at ~0.5-1 GB/s (x86, assumed; CI prints the real runner number); skipped on RESUME via the .verified sidecar.\n")
