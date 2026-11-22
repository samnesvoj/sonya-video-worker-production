"""Check a midroll output: stream durations, loudness of clip-before / ad / clip-after, contact sheet around the ad."""
import json, subprocess, sys, os, re
FF = "/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg"
def sh(c): return subprocess.run(c, capture_output=True, text=True)
def lufs(f, ss, t):
    r = sh(["ffmpeg","-nostats","-ss",f"{ss:.2f}","-t",f"{t:.2f}","-i",f,"-vn","-af","ebur128=framelog=verbose","-f","null","-"])
    m = re.findall(r"I:\s+(-?[\d.]+) LUFS", r.stderr); return float(m[-1]) if m else None
out = sys.argv[1]
rep = json.load(open(os.path.join(out, "selected_segments.json")))
for seg in rep["segments"]:
    mr = seg.get("midroll"); f = os.path.join(out, seg["file"])
    if not mr or mr.get("status") != "ok": print(seg["file"], mr and mr.get("status"), mr and mr.get("reason")); continue
    j = json.loads(sh(["ffprobe","-v","error","-show_entries","stream=codec_type,duration,width,height,r_frame_rate","-show_entries","format=duration","-of","json",f]).stdout)
    durs = {s["codec_type"]: round(float(s["duration"]),2) for s in j["streams"]}
    A = mr["ad_s"]; place = mr["place"]
    s = mr.get("at") or 0.0
    if place == "pre": s = 0.0
    if place == "post": s = mr["clip_s"] - 0.2
    ad0 = s + (mr.get("xfade") or 0.2)/2 if place == "mid" else s
    L = (lufs(f, max(0, ad0-8), 7.5) if place != "pre" else None, lufs(f, ad0+0.3, A-0.6), lufs(f, ad0+A+0.3, 7.5) if place != "post" else None)
    print(seg["file"], place, "streams", durs, "reported", mr["total_s"], "| LUFS before/ad/after", L, "| gain", mr.get("gain_db"), "fit", mr.get("fit"))
    ts = [ad0-1.0, ad0-0.1, ad0+0.05, ad0+0.4, ad0+A/2, ad0+A-0.15, ad0+A+0.1, ad0+A+1.0]
    ts = [t for t in ts if 0 <= t < durs["video"]-0.05]
    sheet = os.path.join(out, seg["file"].replace(".mp4", "_sheet.jpg"))
    tmpd = sheet + ".d"; os.makedirs(tmpd, exist_ok=True)
    for i, t in enumerate(ts):
        sh([FF,"-y","-loglevel","error","-ss",f"{t:.3f}","-i",f,"-frames:v","1","-vf",
            f"scale=270:-2,drawtext=text='{t:.2f}':x=6:y=6:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.6",f"{tmpd}/{i:02d}.jpg"])
    sh(["ffmpeg","-y","-loglevel","error","-i",f"{tmpd}/%02d.jpg","-vf",f"tile={len(ts)}x1","-frames:v","1",sheet])
    print("  sheet", sheet, [round(t,2) for t in ts])
