#!/usr/bin/env python3
"""
mode_dev.py - small helper for improving ONE video mode at a time (Mac mini / MacBook). No refactoring, no network.
  python tools/mode_dev.py list                    modes, default-enabled flag, tests available
  python tools/mode_dev.py deps <mode>             what the mode owns / shares / which tests guard it
  python tools/mode_dev.py test <mode> [-k expr]   gate tests + the mode's tests (CPU)
  python tools/mode_dev.py branch <mode> <slug>    git switch -c mode/<mode>/<slug> from origin/main (never commits to main)
Trailer is not a selectable mode here.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
MAP = json.loads((ROOT / "tools" / "modes_map.json").read_text())


def mode_or_die(name: str) -> dict:
    if name.startswith("trailer"):
        sys.exit("Trailer is disabled in this repository and cannot be worked on through this tool.")
    if name not in MAP["modes"]:
        sys.exit(f"unknown mode {name!r}; choose one of: {', '.join(MAP['modes'])}")
    return MAP["modes"][name]


def test_files(mode: str) -> list[str]:
    m = mode_or_die(mode)
    names = list(dict.fromkeys(MAP["gate"] + m["tests"]))
    return [f"tests/{n}.py" for n in names if (ROOT / "tests" / f"{n}.py").exists()]


def branch_name(mode: str, slug: str) -> str:
    mode_or_die(mode)
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,40}", slug):
        sys.exit("slug: lowercase letters, digits and dashes, 2-41 chars")
    return f"mode/{mode}/{slug}"


def cmd_list(_a):
    for k, m in MAP["modes"].items():
        print(f"{k:12s} {m['label']:16s} default_enabled={str(m['default_enabled']):5s} tests={len(m['tests'])}")


def cmd_deps(a):
    m = mode_or_die(a.mode)
    print(json.dumps({"own": m["own"], "legacy": m["legacy"], "shared": m["shared"], "models": m["models"], "tests": test_files(a.mode),
                      "warning": "changing 'shared' paths affects other modes - run `test` for each mode listed in docs/dev/DEV_WORKFLOW.md"}, indent=2))


def cmd_test(a):
    files = test_files(a.mode)
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *files] + (["-k", a.k] if a.k else [])
    print("+", " ".join(cmd))
    return subprocess.call(cmd, cwd=ROOT)


def cmd_branch(a):
    name = branch_name(a.mode, a.slug)
    subprocess.check_call(["git", "fetch", "--quiet", "origin", "main"], cwd=ROOT)
    subprocess.check_call(["git", "switch", "-c", name, "origin/main"], cwd=ROOT)
    print(f"on {name} (from origin/main). Commit here, then: git push -u origin {name} && gh pr create --base main")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest="cmd", required=True)
    s.add_parser("list").set_defaults(fn=cmd_list)
    d = s.add_parser("deps"); d.add_argument("mode"); d.set_defaults(fn=cmd_deps)
    t = s.add_parser("test"); t.add_argument("mode"); t.add_argument("-k"); t.set_defaults(fn=cmd_test)
    b = s.add_parser("branch"); b.add_argument("mode"); b.add_argument("slug"); b.set_defaults(fn=cmd_branch)
    a = p.parse_args(argv)
    return a.fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())
