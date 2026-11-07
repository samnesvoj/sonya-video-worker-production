import json
import pathlib

from tools import osv_check

LOCK = "torch==2.5.1+cu124 \\\n    --hash=sha256:aa\nrequests==2.0.0 \\\n    --hash=sha256:bb\n"


class R:
    def __init__(self, vulns): self.v = vulns
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def read(self): return json.dumps({"vulns": self.v}).encode()


def test_only_local_version_pins_are_checked(tmp_path):
    f = tmp_path / "lock.txt"
    f.write_text(LOCK)
    assert osv_check.local_version_pins(f) == [("torch", "2.5.1")]


def test_unaccepted_advisory_fails_and_accepted_passes(tmp_path, monkeypatch):
    f = tmp_path / "lock.txt"
    f.write_text(LOCK)
    acc = tmp_path / "acc.json"
    monkeypatch.setattr(osv_check, "ACCEPTED", acc)
    opener = lambda req, timeout=0: R([{"id": "GHSA-new", "summary": "new problem"}])
    assert osv_check.main([str(f)], opener) == 1                       # new advisory -> CI red
    acc.write_text(json.dumps({"GHSA-new": "reviewed: untrusted input never reaches torch.load"}))
    assert osv_check.main([str(f)], opener) == 0
