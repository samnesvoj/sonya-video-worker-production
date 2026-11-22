"""Local midroll run: copy fixture clips (+ .ass, selected_segments.json) into out/<name>, apply, print reports.
usage: run_local.py <name> <ad file> '<midroll json>' <clip dir> [clip names...]"""
import json, os, shutil, sys, logging
sys.path.insert(0, os.path.expanduser("~/SONYA/worker-prep/vast-run/repo"))
logging.basicConfig(level=logging.INFO, format="%(message)s")
name, ad, cfg, src = sys.argv[1:5]
names = sys.argv[5:]
out = os.path.expanduser(f"~/SONYA/midroll/out/{name}")
shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
for f in os.listdir(src):
    stem = f.rsplit(".", 1)[0]
    if f == "selected_segments.json" or (f.endswith((".mp4", ".ass")) and not stem.endswith(("_preview",)) and (not names or stem in names)):
        shutil.copy(os.path.join(src, f), out)
os.environ["SONYA_MIDROLL_ASSET"] = ad
os.environ.setdefault("SONYA_X264_PRESET", "veryfast")
from scripts.shared import midroll
params = {"midroll": dict({"key": "users/test/midroll/0123456789abcdef.mp4"}, **json.loads(cfg))}
rep = midroll.apply_to_outputs(out, "virality", params, work_dir=out + "_work")
print(json.dumps(rep, ensure_ascii=False, indent=1))
