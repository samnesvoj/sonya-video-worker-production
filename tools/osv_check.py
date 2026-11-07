#!/usr/bin/env python3
"""
OSV check for the packages pip-audit cannot see (local-version wheels like torch==2.5.1+cu124 are not on PyPI).
Queries https://api.osv.dev with the base version and FAILS on any advisory that is not listed, with a reason, in
requirements/osv-accepted.json. Accepting an advisory is a reviewed, written decision; a NEW advisory breaks CI.
Usage: python tools/osv_check.py [lock-file] [--write-accepted "reason"]
"""
import json, pathlib, re, sys, urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
ACCEPTED = ROOT / "requirements" / "osv-accepted.json"


def local_version_pins(lock: pathlib.Path):
    return re.findall(r"^([A-Za-z0-9_.-]+)==([0-9][^\s+\\]*)\+[A-Za-z0-9.]+", lock.read_text(), re.M)


def query(name, version, opener=urllib.request.urlopen):
    body = json.dumps({"package": {"name": name, "ecosystem": "PyPI"}, "version": version}).encode()
    req = urllib.request.Request("https://api.osv.dev/v1/query", data=body, headers={"Content-Type": "application/json"})
    with opener(req, timeout=40) as r:
        return json.load(r).get("vulns", [])


def main(argv, opener=urllib.request.urlopen):
    args = [a for a in argv if not a.startswith("--")]
    lock = pathlib.Path(args[0]) if args else ROOT / "requirements" / "worker.lock.txt"
    accepted = json.loads(ACCEPTED.read_text()) if ACCEPTED.exists() else {}
    write_reason = argv[argv.index("--write-accepted") + 1] if "--write-accepted" in argv else None
    bad = []
    for name, ver in local_version_pins(lock):
        vulns = query(name, ver, opener)
        print(f"{name} {ver}: {len(vulns)} advisories")
        for v in vulns:
            if write_reason:
                accepted[v["id"]] = write_reason
            elif v["id"] not in accepted:
                bad.append(f"{name} {ver}: {v['id']} {v.get('summary', '')[:80]}")
    if write_reason:
        ACCEPTED.write_text(json.dumps(accepted, indent=1, sort_keys=True) + "\n")
    for b in bad:
        print("NOT ACCEPTED:", b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
