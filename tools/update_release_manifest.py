#!/usr/bin/env python3
"""Set requirements_lock_sha256 in release/manifest.json (use --check in CI)."""
import hashlib, json, pathlib, sys

root = pathlib.Path(__file__).resolve().parents[1]
mf = root / "release" / "manifest.json"
data = json.loads(mf.read_text())
digest = hashlib.sha256((root / "requirements" / "worker.lock.txt").read_bytes()).hexdigest()
if "--check" in sys.argv:
    sys.exit(0 if data.get("requirements_lock_sha256") == digest else 1)
data["requirements_lock_sha256"] = digest
mf.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
print(digest)
