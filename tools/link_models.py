#!/usr/bin/env python3
"""
link_models.py - make models that already exist on this Mac visible to the worker WITHOUT copying or downloading anything.
Reads every modes/*/mode.yaml `models:` entry, looks for the file by name under the roots you give, and symlinks it into
MODELS_LOCAL_DIR (default ~/SONYA-models) at the path the mode expects. Never touches the network, never overwrites a real file.
  python tools/link_models.py --from ~/Business/SONYA/runtime ~/Downloads/models [--dry-run]
"""
import argparse, os, pathlib, sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]


def wanted() -> dict:
    out = {}
    for cfg in sorted((ROOT / "modes").glob("*/mode.yaml")):
        mode = cfg.parent.name
        if mode.startswith("trailer"):
            continue
        for key, spec in ((yaml.safe_load(cfg.read_text()) or {}).get("models") or {}).items():
            lp = spec.get("local_path") or key
            out.setdefault(lp, {"key": key, "modes": [], "optional": spec.get("optional", True)})["modes"].append(mode)
    return out


def find(name: str, roots) -> pathlib.Path | None:
    for r in roots:
        for p in pathlib.Path(r).expanduser().rglob(name):
            if p.is_file() and not p.is_symlink():
                return p
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="roots", nargs="+", required=True)
    ap.add_argument("--models-dir", default=os.environ.get("MODELS_LOCAL_DIR", str(pathlib.Path.home() / "SONYA-models")))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    mdir = pathlib.Path(a.models_dir).expanduser()
    missing_required = 0
    for lp, info in wanted().items():
        dest = mdir / lp
        src = find(pathlib.Path(lp).name, a.roots)
        tag = f"{lp} (modes: {','.join(info['modes'])}; {'optional' if info['optional'] else 'REQUIRED'})"
        if dest.exists():
            print(f"present   {tag}")
        elif src is None:
            print(f"MISSING   {tag}")
            missing_required += 0 if info["optional"] else 1
        else:
            print(f"link      {tag} -> {src}")
            if not a.dry_run:
                dest.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(src, dest)
    return 1 if missing_required else 0


if __name__ == "__main__":
    sys.exit(main())
