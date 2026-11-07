#!/usr/bin/env python3
"""
image_report.py - measured layer report of a Docker image (CI). Input: `docker save IMAGE -o img.tar`.
Per layer: uncompressed size, gzip-compressed size (zlib level 6 = what a gzip registry layer costs on the wire, +-few %), the Dockerfile
instruction that created it. Totals are what a Vast host must download (compressed) and extract (uncompressed). Output: Markdown.
"""
import json, sys, tarfile, zlib

CHUNK = 4 << 20


def main(tar_path: str) -> int:
    rows, tot_u, tot_c = [], 0, 0
    with tarfile.open(tar_path) as tf:
        manifest = json.load(tf.extractfile("manifest.json"))[0]
        cfg = json.load(tf.extractfile(manifest["Config"]))
        steps = [h for h in cfg.get("history", []) if not h.get("empty_layer")]
        for i, layer in enumerate(manifest["Layers"]):
            f = tf.extractfile(layer)
            z, comp, raw = zlib.compressobj(6), 0, 0
            while True:
                b = f.read(CHUNK)
                if not b:
                    break
                raw += len(b)
                comp += len(z.compress(b))
            comp += len(z.flush())
            tot_u, tot_c = tot_u + raw, tot_c + comp
            by = (steps[i]["created_by"] if i < len(steps) else "?").replace("/bin/sh -c ", "").replace("#(nop) ", "")
            rows.append((raw, comp, " ".join(by.split())[:110]))
    print("| # | uncompressed MB | gzip MB (wire) | created by |\n|---|---|---|---|")
    for i, (u, c, by) in enumerate(rows):
        if u > 1 << 20:
            print(f"| {i} | {u/1e6:,.0f} | {c/1e6:,.0f} | `{by}` |")
    print(f"\n**Total: {tot_u/1e9:.2f} GB uncompressed, {tot_c/1e9:.2f} GB gzip-compressed ({len(rows)} layers; {sum(1 for r in rows if r[1] > 50e6)} layers > 50 MB compressed).**")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
