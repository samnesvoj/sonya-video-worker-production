#!/usr/bin/env python3
"""
models_publish.py - find the models on a machine, verify them, generate release/models.lock.json, and (only on explicit command) upload
them to our S3 with SHA-256 verification. Nothing is uploaded unless `upload --apply --confirm-bucket <same name>` is given.

  scan    hash local files, compare with the public hashes in release/models.required.json         (no network, no credentials)
  lock    write release/models.lock.json from the local files                                       (no network, no credentials)
  upload  dry-run by default; --apply uploads missing/changed files with a multipart upload and stores the sha256 as object metadata,
          then re-reads the object's size + metadata (no re-download: the Mac mini may have a slow link)

Typical flow:  Mac mini: `scan` then `lock`  ->  send the JSON (hashes only) to the MacBook  ->  whoever holds S3 credentials runs `upload`.
Secrets: S3_ENDPOINT_URL / S3_ACCESS_KEY_ID / S3_SECRET_ACCESS_KEY / S3_REGION come from the environment and are never printed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
from typing import Dict, List, Optional

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "release" / "models.required.json"
CHUNK = 8 << 20


def sha256_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def sha256_md5_file(p: pathlib.Path) -> tuple[str, str]:
    """One pass over the file (the GigaAM checkpoint is ~450 MB): SHA-256 for our lock, MD5 only to cross-check upstream's own hash table."""
    h, m = hashlib.sha256(), hashlib.md5()  # noqa: S324 - MD5 is upstream's checksum, used for comparison only
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b); m.update(b)
    return h.hexdigest(), m.hexdigest()


def load_manifest(path: pathlib.Path = MANIFEST) -> dict:
    return json.loads(path.read_text())


def locate(model: dict, srcs: List[pathlib.Path], explicit: Dict[str, pathlib.Path]) -> tuple[Optional[pathlib.Path], List[pathlib.Path]]:
    """-> (chosen file, all candidates). Prefers the exact relative path; else same file name under a directory whose path mentions the model."""
    if model["key"] in explicit:
        return explicit[model["key"]], [explicit[model["key"]]]
    rel = pathlib.PurePosixPath(model["local_path"])
    cands: List[pathlib.Path] = []
    for s in srcs:
        exact = s / rel
        if exact.is_file():
            cands.append(exact)
    if not cands:
        hint = rel.parent.name if rel.parent.name not in ("", "common") else ""
        for s in srcs:
            for p in s.rglob(rel.name):
                if p.is_file() and (not hint or hint in str(p) or hint.replace("whisper-", "faster-whisper-") in str(p)):
                    cands.append(p)
    uniq = sorted({c.resolve() for c in cands})
    return (uniq[0] if len(uniq) == 1 else None), uniq


def scan(manifest: dict, srcs: List[pathlib.Path], explicit: Dict[str, pathlib.Path]) -> List[dict]:
    rows = []
    for m in manifest["models"]:
        chosen, cands = locate(m, srcs, explicit)
        row = {"key": m["key"], "local_path": m["local_path"], "s3_path": m["s3_path"], "used_by": m["used_by"], "source_type": m["source"]["type"],
               "public_sha256": m.get("public_sha256"), "public_size": m.get("public_size")}
        if chosen is None:
            row.update(status="ambiguous" if len(cands) > 1 else "missing", candidates=[str(c) for c in cands])
        else:
            sha, md5 = sha256_md5_file(chosen) if m.get("upstream_md5") else (sha256_file(chosen), None)
            row.update(path=str(chosen), size=chosen.stat().st_size, sha256=sha)
            if m.get("public_sha256"):
                row["status"] = "matches_public" if sha == m["public_sha256"] else "differs_from_public"
            elif m.get("upstream_md5"):
                row.update(upstream_md5=m["upstream_md5"], md5=md5)
                row["status"] = "matches_upstream_md5" if md5 == m["upstream_md5"] else "differs_from_upstream"
            else:
                row["status"] = "private_ok"
            if m.get("public_size") and row["size"] != m["public_size"]:
                row["status"] = "differs_from_public" if row["status"].startswith("matches") else row["status"]
                row["size_note"] = f"size {row['size']} != expected {m['public_size']}"
        rows.append(row)
    return rows


def make_lock(rows: List[dict], accept_custom: set) -> dict:
    models, problems = {}, []
    for r in rows:
        if r["status"] in ("missing", "ambiguous"):
            problems.append(f"{r['key']}: {r['status']}")
        elif r["status"] in ("differs_from_public", "differs_from_upstream") and r["key"] not in accept_custom:
            problems.append(f"{r['key']}: differs from the {'public' if r['status'].endswith('public') else 'upstream'} file - re-run with --accept-custom {r['key']} if this Mac mini file is the tested reference")
        else:
            models[r["key"]] = {"s3_path": r["s3_path"], "sha256": r["sha256"], "size": r["size"]}
    if problems:
        raise SystemExit("cannot write a complete lock:\n  " + "\n  ".join(problems))
    return {"schema_version": 1, "note": "generated by tools/models_publish.py; hashes of the exact files that were tested", "models": models}


def s3_client():
    import boto3
    return boto3.client("s3", endpoint_url=os.environ["S3_ENDPOINT_URL"], aws_access_key_id=os.environ["S3_ACCESS_KEY_ID"],
                        aws_secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"], region_name=os.environ.get("S3_REGION", "ru-1"))


def upload(rows: List[dict], bucket: str, client, apply: bool) -> List[str]:
    from boto3.s3.transfer import TransferConfig
    cfg = TransferConfig(multipart_threshold=64 << 20, multipart_chunksize=32 << 20, max_concurrency=8)
    log = []
    for r in rows:
        if "sha256" not in r:
            raise SystemExit(f"{r['key']}: no local file; fix `scan` first")
        try:
            head = client.head_object(Bucket=bucket, Key=r["s3_path"])
            same = head["ContentLength"] == r["size"] and head.get("Metadata", {}).get("sha256") == r["sha256"]
        except Exception:  # noqa: BLE001 - not there
            head, same = None, False
        if same:
            log.append(f"skip   {r['s3_path']} (already uploaded, same size + sha256)")
            continue
        if head is not None:
            log.append(f"REPLACE {r['s3_path']} (different content already in the bucket)")
        if not apply:
            log.append(f"would upload {r['s3_path']} ({r['size']/1e6:.1f} MB)")
            continue
        client.upload_file(r["path"], bucket, r["s3_path"], ExtraArgs={"Metadata": {"sha256": r["sha256"]}}, Config=cfg)
        got = client.head_object(Bucket=bucket, Key=r["s3_path"])
        if got["ContentLength"] != r["size"] or got.get("Metadata", {}).get("sha256") != r["sha256"]:
            raise SystemExit(f"{r['s3_path']}: uploaded object does not match (size/metadata)")
        log.append(f"uploaded {r['s3_path']} ({r['size']/1e6:.1f} MB) verified size+sha256 metadata")
    return log


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["scan", "lock", "upload"])
    ap.add_argument("--src", action="append", type=pathlib.Path, required=True, help="directory with the models (repeatable)")
    ap.add_argument("--map", action="append", default=[], metavar="KEY=PATH", help="explicit file for a key (when the search is ambiguous)")
    ap.add_argument("--manifest", type=pathlib.Path, default=MANIFEST)
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--accept-custom", action="append", default=[], metavar="KEY", help="accept a file whose hash differs from the public one")
    ap.add_argument("--bucket"); ap.add_argument("--confirm-bucket"); ap.add_argument("--apply", action="store_true")
    ap.add_argument("--skip", action="append", default=[], metavar="KEY", help="leave a model out of lock/upload on purpose (e.g. webcam_detector: off by owner decision)")
    a = ap.parse_args(argv)
    explicit = {k: pathlib.Path(v).expanduser() for k, v in (m.split("=", 1) for m in a.map)}
    man = load_manifest(a.manifest)
    man = {**man, "models": [m for m in man["models"] if m["key"] not in set(a.skip)]}
    rows = scan(man, [s.expanduser() for s in a.src], explicit)
    if a.cmd == "scan":
        print(json.dumps(rows, indent=1) if a.out is None else f"report -> {a.out}")
        if a.out:
            a.out.write_text(json.dumps(rows, indent=1))
        miss = [r["key"] for r in rows if r["status"] in ("missing", "ambiguous")]
        print(f"\n{len(rows)} models: " + ", ".join(f"{s}={sum(1 for r in rows if r['status']==s)}" for s in sorted({r['status'] for r in rows})), file=sys.stderr)
        return 1 if miss else 0
    if a.cmd == "lock":
        lock = make_lock(rows, set(a.accept_custom))
        out = a.out or ROOT / "release" / "models.lock.json"
        out.write_text(json.dumps(lock, indent=2) + "\n")
        print(f"wrote {out} with {len(lock['models'])} models")
        return 0
    # upload
    if not a.bucket:
        raise SystemExit("--bucket is required")
    make_lock(rows, set(a.accept_custom))                      # same completeness/identity gate as `lock`
    if a.apply and a.confirm_bucket != a.bucket:
        raise SystemExit("refusing to upload: pass --confirm-bucket with the exact bucket name to confirm")
    for line in upload(rows, a.bucket, s3_client() if (a.apply or os.environ.get("S3_ENDPOINT_URL")) else _NoClient(), a.apply):
        print(line)
    if not a.apply:
        print("\nDRY RUN - nothing was uploaded. Add --apply --confirm-bucket", a.bucket, "to upload.")
    return 0


class _NoClient:
    def head_object(self, **kw):
        raise KeyError("no S3 configured (dry run without credentials)")


if __name__ == "__main__":
    sys.exit(main())
