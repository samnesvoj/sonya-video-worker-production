"""
storyfx.py — style «story», v2 (tester feedback 2026-11-05: «скучно, мелко, не хватает динамики»).

Free, local helpers only (no network, no Gemini):

  * canvas / crop geometry: the film is CROPPED to 1:1 (or 4:5, 3:4) and fills the whole output frame (the samples are 1:1 and 3:4
    videos, not letterboxed 9:16); the crop window follows the faces (the same YuNet detector as the Cut / Trailer smart crop);
  * a per-episode scan (faces + optical flow at 5 fps, cached) → a score per candidate shot: big faces and motion win, empty rooms
    and static wide shots lose;
  * shot rhythm: slower at the start, 1.2–1.8 s towards the climax, the strongest picture first;
  * slow push-in (3–6 %) on static shots only;
  * variant B sound: the film's own sound far under the narrator (ducked further where the film's characters talk) plus quiet
    accents (whoosh / impact / riser) taken ONLY from the owner's own sound pack.
"""
from __future__ import annotations

import bisect
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import media

CANVAS_W = 1080
ASPECTS = {"1:1": (1, 1), "4:5": (4, 5), "3:4": (3, 4)}
DEFAULT_ASPECT = "1:1"
SCAN_FPS = 5.0
SCAN_VERSION = "story-scan-v1"
PUSH_IN = 0.05                                  # 5 % slow push-in on a static shot (brief: 3–6 %)
SFX_DEFAULT_DIR = "/Users/agent/Desktop/rils/Саунд пак"


# ── geometry ────────────────────────────────────────────────────────────────

def canvas_size(aspect: str = DEFAULT_ASPECT) -> Tuple[int, int]:
    if aspect not in ASPECTS:
        raise ValueError(f"crop must be one of {sorted(ASPECTS)}")
    bw, bh = ASPECTS[aspect]
    return CANVAS_W, int(round(CANVAS_W * bh / bw / 2)) * 2


def crop_window(src_w: int, src_h: int, aspect: str = DEFAULT_ASPECT) -> Tuple[int, int]:
    """The largest even window of the given aspect inside the source frame (a wide film → full height, a horizontal slide)."""
    bw, bh = ASPECTS[aspect]
    ar = bw / bh
    if src_w / src_h >= ar:
        ch = src_h - src_h % 2
        cw = int(ch * ar) // 2 * 2
    else:
        cw = src_w - src_w % 2
        ch = int(cw / ar) // 2 * 2
    return max(2, min(cw, src_w)), max(2, min(ch, src_h))


# ── per-episode scan: faces + motion, cached ────────────────────────────────

def scan_episode(src: str, a: float, b: float, cache: Optional[Path] = None, fps: float = SCAN_FPS,
                 det_width: int = 640) -> Dict[str, Any]:
    """One sequential pass over [a, b]: every 1/fps s → faces (YuNet from the Cut crop engine, boxes in source pixels) and motion
    (Farneback optical flow on a 128-px grey frame, mean magnitude; a hard cut between two samples is flagged, not measured)."""
    if cache and cache.is_file():
        try:
            d = json.loads(cache.read_text())
            if d.get("meta", {}).get("version") == SCAN_VERSION:
                return d
        except (OSError, ValueError):
            pass
    import cv2
    from scripts.shared.crop import clip_analysis as ca
    st = ca.model_ok()
    if not st["ok"]:
        raise RuntimeError(st["reason"])
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError("cannot open source for the scan")
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, a - 1.0) * 1000)
    scale = min(1.0, det_width / W)
    dw, dh = int(round(W * scale)), int(round(H * scale))
    det = cv2.FaceDetectorYN.create(str(ca.MODEL_PATH), "", (dw, dh), ca.SCORE_MIN, 0.3, 50)
    samples: List[Dict[str, Any]] = []
    prev = None
    next_t = a
    gw = 128
    gh = max(8, int(round(H * gw / W)))
    while True:
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t > b:
            break
        if t + 1e-6 < next_t:
            if not cap.grab():
                break
            continue
        ok, frame = cap.read()
        if not ok:
            break
        next_t = max(next_t + 1.0 / fps, t + 0.5 / fps)
        small = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA) if scale < 1 else frame
        _, faces = det.detect(small)
        boxes = [[int(f[0] / scale), int(f[1] / scale), int(f[2] / scale), int(f[3] / scale), round(float(f[14]), 3)]
                 for f in (faces if faces is not None else [])]
        boxes.sort(key=lambda x: -x[2] * x[3])
        g = cv2.cvtColor(cv2.resize(frame, (gw, gh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        motion, cut = None, False
        if prev is not None:
            if float(cv2.absdiff(prev, g).mean()) > 30.0:
                cut = True
            else:
                fl = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 2, 9, 2, 5, 1.1, 0)
                motion = float(cv2.cartToPolar(fl[..., 0], fl[..., 1])[0].mean())
        prev = g
        samples.append({"t": round(t, 3), "motion": None if motion is None else round(motion, 4), "cut": cut, "faces": boxes})
    cap.release()
    data = {"meta": {"version": SCAN_VERSION, "width": W, "height": H, "fps": fps, "start": a, "end": b, "samples": len(samples),
                     "detector": st["detector"]}, "samples": samples}
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data))
    return data


def _pct(vals: Sequence[float], q: float) -> float:
    s = sorted(vals)
    return s[min(len(s) - 1, int(q * len(s)))] if s else 0.0


class Scan:
    """Window features + shot score over the scan samples. Pure (no I/O)."""

    FACE_REF = 0.38                                   # face height / frame height that counts as a full close-up

    def __init__(self, data: Dict[str, Any]):
        self.s = data["samples"]
        self.t = [x["t"] for x in self.s]
        self.W = int(data["meta"]["width"])
        self.H = int(data["meta"]["height"])
        mot = [x["motion"] for x in self.s if x["motion"] is not None]
        self.ref = max(0.05, _pct(mot, 0.8))          # «lively» = the 80th percentile of this episode
        self.low = _pct(mot, 0.35)                    # «empty room» motion level = the lowest third
        self.calm = _pct(mot, 0.55)                   # «calm shot» (gets the slow push-in) = below the 55th percentile

    def window(self, t0: float, t1: float, crop: Optional[Tuple[float, float, int]] = None) -> Dict[str, Any]:
        """Features of [t0, t1). With crop = (x_start, x_end, cw) only the faces INSIDE the crop window count (a face elsewhere in
        the wide film frame is not in the picture)."""
        i, j = bisect.bisect_left(self.t, t0), bisect.bisect_left(self.t, t1)
        ws = self.s[i:j]
        if not ws:
            return {"n": 0, "presence": 0.0, "face": 0.0, "motion": 0.0, "cuts": 0}
        lo = hi = None
        if crop:
            lo, hi = min(crop[0], crop[1]) + 0.12 * crop[2], max(crop[0], crop[1]) + 0.88 * crop[2]

        def inside(f):
            return crop is None or lo <= f[0] + f[2] / 2.0 <= hi

        fh = []
        for x in ws:
            fs = [f for f in x["faces"] if inside(f)]
            if fs:
                fh.append(max(f[3] for f in fs) / self.H)
        mots = [x["motion"] for x in ws if x["motion"] is not None]
        return {"n": len(ws), "presence": len(fh) / len(ws), "face": sum(fh) / len(fh) if fh else 0.0,
                "motion": sum(mots) / len(mots) if mots else 0.0, "cuts": sum(1 for x in ws[1:] if x["cut"])}

    def score(self, f: Dict[str, Any]) -> float:
        if f["n"] == 0:
            return -9.0
        face = min(f["face"] / self.FACE_REF, 1.0) * f["presence"]
        mot = min(f["motion"] / self.ref, 1.5) * (0.35 + 0.65 * f["presence"])        # motion without a face is worth little
        sc = 2.0 * face + 1.2 * mot
        if f["presence"] < 0.5:
            sc -= 1.5 + (1.0 if f["motion"] <= self.low else 0.0)       # no face in the picture: a flag, trees, a back of a head
        elif f["face"] < 0.12:
            sc -= 0.5                                                   # a face lost in a wide shot
        return sc - 3.0 * f["cuts"]

    def is_static(self, f: Dict[str, Any]) -> bool:
        return f["motion"] <= max(self.calm, 0.12 * self.ref)

    def crop_xs(self, t0: float, t1: float, cw: int) -> Tuple[int, int]:
        """Window x at the start and at the end of the shot: the area-weighted centre of the main face in the first / last third
        (the drift is limited so the shot never reads as a pan-cut). No face → the middle of the frame."""
        mid = (self.W - cw) // 2

        def cx(a: float, b: float) -> Optional[float]:
            i, j = bisect.bisect_left(self.t, a), bisect.bisect_right(self.t, b)
            num = den = 0.0
            for x in self.s[i:j]:
                if x["faces"]:
                    f = x["faces"][0]
                    w = f[2] * f[3] * f[4]
                    num += (f[0] + f[2] / 2.0) * w
                    den += w
            return num / den if den else None

        d = t1 - t0
        c0, c1 = cx(t0 - 0.2, t0 + d / 3), cx(t1 - d / 3, t1 + 0.2)
        whole = cx(t0, t1)
        c0 = c0 if c0 is not None else whole
        c1 = c1 if c1 is not None else whole
        if c0 is None and c1 is None:
            return mid // 2 * 2, mid // 2 * 2
        c0 = c0 if c0 is not None else c1
        c1 = c1 if c1 is not None else c0
        x0 = min(max(0.0, c0 - cw / 2.0), self.W - cw)
        x1 = min(max(0.0, c1 - cw / 2.0), self.W - cw)
        x1 = x0 + max(-0.18 * cw, min(0.18 * cw, x1 - x0))          # slow drift only
        return int(x0) // 2 * 2, int(min(max(0.0, x1), self.W - cw)) // 2 * 2

    def face_cx_rel(self, t0: float, t1: float, x: float, cw: int) -> float:
        """Where the main face sits inside the crop window, 0..1 (the push-in keeps that point still)."""
        i, j = bisect.bisect_left(self.t, t0), bisect.bisect_right(self.t, t1)
        cs = [(f[0] + f[2] / 2.0) for x_ in self.s[i:j] for f in x_["faces"][:1]]
        if not cs:
            return 0.5
        return min(0.8, max(0.2, (sum(cs) / len(cs) - x) / cw))


# ── rhythm ──────────────────────────────────────────────────────────────────

def shot_range(progress: float) -> Tuple[float, float]:
    """Allowed shot length by the position in the video: calm start, 1.2–1.8 s in the last third (the climax)."""
    if progress < 0.35:
        return 2.0, 3.0
    if progress < 0.68:
        return 1.6, 2.4
    return 1.2, 1.8


def shot_durs(span: float, progress: float, first: bool = False, hook_first: float = 1.9) -> List[float]:
    """Pure: shot lengths for one voice segment, each inside shot_range(progress) when possible, summing to `span`. The very first
    shot of the video is short (the strongest picture in the first 1–2 s)."""
    lo, hi = shot_range(progress)
    head: List[float] = []
    if first and span > hook_first + 0.9:
        head, span = [hook_first], span - hook_first
    if span <= hi:
        body = [span]
    else:
        n = max(1, int(math.ceil(span / hi)))
        while n > 1 and span / n < lo:
            n -= 1
        body = [span / n] * n
    return [round(x, 4) for x in head + body]


# ── shot choice ─────────────────────────────────────────────────────────────

def choose_window(scan: Scan, region: Tuple[float, float], d: float, cuts: Sequence[float], used: Sequence[Tuple[float, float]],
                  bonus: float = 1.0, step: float = 0.5, pad: float = 0.4, cw: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """The best cut-free, unused window of length d inside region: (src_start, features, score) or None. Pure."""
    lo, hi = region
    best = None
    s = lo
    while s + d <= hi + 1e-6:
        e = s + d
        if not any(s + 0.08 < c < e - 0.08 for c in cuts) and all(e + pad <= u0 or s >= u1 + pad for u0, u1 in used):
            f = scan.window(s, e, (*scan.crop_xs(s, e, cw), cw) if cw else None)
            if f["n"] and f["cuts"] == 0:
                sc = scan.score(f) * bonus
                if best is None or sc > best["score"]:
                    best = {"src_start": round(s, 3), "feat": f, "score": round(sc, 3)}
        s += step
    return best


def plan_shots_v2(script: Dict[str, Any], vt: Dict[str, Any], events: Sequence[Dict[str, Any]], a: float, b: float, cuts_in,
                  scan: Scan, cw: int, speech_in=lambda t0, t1: False) -> Dict[str, Any]:
    """Shots for every voice segment: windows with big faces / motion inside the part of the episode the segment talks about
    (events ± margin; the hook looks further, its first shot is the strongest picture of that region)."""
    from . import compose
    ev = {e["id"]: e for e in events}
    spans = compose.segment_spans(vt)
    total = vt["total"]
    shots, warns, used = [], [], []
    for si, seg in enumerate(script["segments"]):
        sa, sb = spans[seg["id"]]
        durs = shot_durs(sb - sa, sa / total, first=(si == 0))
        evs = [ev[e] for e in seg["event_ids"] if e in ev]
        lo0 = min(e["start"] for e in evs)
        hi0 = max(e["end"] for e in evs)
        hook = seg.get("function") == "hook"
        t0 = sa
        for i, d in enumerate(durs):
            t1 = sb if i == len(durs) - 1 else compose.q(t0 + d)
            d = round(t1 - t0, 4)
            best = None
            for margin in ((12.0, 40.0) if hook else (6.0, 18.0, 45.0)):
                region = (max(a, lo0 - margin), min(b, hi0 + margin))
                cuts = cuts_in(max(a, region[0] - 0.5), min(b, region[1] + 0.5))
                best = choose_window(scan, region, d, cuts, used, bonus=1.25 if (hook and i == 0) else 1.0, cw=cw)
                if best:
                    break
            if best is None:                                  # nothing cut-free and unused anywhere near: report, take the middle
                s0 = min(max(a, (lo0 + hi0) / 2 - d / 2), b - d)
                best = {"src_start": round(s0, 3), "feat": scan.window(s0, s0 + d, (*scan.crop_xs(s0, s0 + d, cw), cw)), "score": -9.0}
                warns.append(f"{seg['id']} shot {i + 1}: no cut-free unused window — taken as is")
            s0 = best["src_start"]
            used.append((s0, s0 + d))
            f = best["feat"]
            x0, x1 = scan.crop_xs(s0, s0 + d, cw)
            static = scan.is_static(f)
            shots.append({"n": len(shots) + 1, "segment_id": seg["id"], "candidate_id": f"{seg['id']}.{i + 1}", "src_start": s0,
                          "src_end": round(s0 + d, 3), "t0": round(t0, 4), "t1": round(t1, 4), "duration": d,
                          "why": "face/motion score", "fallback": False, "scene_shot": "-",
                          "speaking": "film speech" if speech_in(s0, s0 + d) else "no speech",
                          "film_speech": bool(speech_in(s0, s0 + d)), "score": best["score"],
                          "face": round(f["face"], 3), "presence": round(f["presence"], 2), "motion": round(f["motion"], 3),
                          "static": static, "push_in": PUSH_IN if static else 0.0, "crop_x0": x0, "crop_x1": x1,
                          "fx": round(scan.face_cx_rel(s0, s0 + d, (x0 + x1) / 2.0, cw), 3)})
            t0 = t1
    return {"shots": shots, "total": total, "fps": compose.FPS, "warnings": warns}


def cut_stats_v2(tl: Dict[str, Any]) -> Dict[str, Any]:
    d = [s["t1"] - s["t0"] for s in tl["shots"]]
    tot = tl["total"]
    early = [s["t1"] - s["t0"] for s in tl["shots"] if s["t0"] < 0.35 * tot]
    late = [s["t1"] - s["t0"] for s in tl["shots"] if s["t0"] >= 0.68 * tot]
    out = [round(x, 2) for x in d if x < 1.15 or x > 3.45]
    return {"shots": len(d), "min": round(min(d), 2), "max": round(max(d), 2), "avg": round(sum(d) / len(d), 2),
            "avg_first_third": round(sum(early) / max(1, len(early)), 2), "avg_last_third": round(sum(late) / max(1, len(late)), 2),
            "out_of_range": out, "static_push_in": sum(1 for s in tl["shots"] if s["push_in"]),
            "faces_avg": round(sum(s["face"] for s in tl["shots"]) / len(d), 3),
            "no_face_shots": sum(1 for s in tl["shots"] if s["presence"] < 0.3)}


# ── render ──────────────────────────────────────────────────────────────────

def shot_filter(sh: Dict[str, Any], src_w: int, cw: int, ch: int, out_w: int, out_h: int, fps: int = 30) -> str:
    """The crop window (x drifting slowly from crop_x0 to crop_x1) cut from the film upscaled ×k, then — on a static shot — a slow
    push-in around the face (zoompan on the already-cropped picture, so the pan and the zoom do not fight)."""
    k = max(1, int(round(out_h / ch)))
    d = max(0.04, sh["src_end"] - sh["src_start"])
    x0, x1 = sh["crop_x0"] * k, sh["crop_x1"] * k
    xexpr = f"{x0}+({x1}-{x0})*min(1\\,t/{d:.3f})"
    f = [f"scale={src_w * k}:-2:flags=lanczos", f"crop={cw * k}:{ch * k}:'{xexpr}':0", f"fps={fps}"]
    n = max(2, int(round(d * fps)))
    if sh.get("push_in"):
        z = f"1+{sh['push_in']:.3f}*on/{n}"
        fx = sh.get("fx", 0.5)
        f.append(f"zoompan=z='{z}':x='(iw-iw/zoom)*{fx:.3f}':y='(ih-ih/zoom)*0.4':d=1:s={out_w}x{out_h}:fps={fps}")
    else:
        f.append(f"scale={out_w}:{out_h}:flags=lanczos")
    f += ["setsar=1", "format=yuv420p"]
    return ",".join(f)


def render_shot_v2(src: str, sh: Dict[str, Any], out_mp4: str, out_wav: str, src_w: int, cw: int, ch: int, out_w: int, out_h: int,
                   fps: int = 30) -> None:
    s, d = sh["src_start"], sh["src_end"] - sh["src_start"]
    vf = shot_filter(sh, src_w, cw, ch, out_w, out_h, fps)
    r = subprocess.run([media.ffmpeg_bin(), "-y", "-hide_banner", "-nostdin", "-loglevel", "error", "-ss", f"{s:.3f}", "-t", f"{d + 0.1:.3f}",
                        "-i", src, "-an", "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "17", out_mp4],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"shot {sh['n']} failed: {r.stderr[-300:]}")
    r = subprocess.run([media.ffmpeg_bin(), "-y", "-hide_banner", "-nostdin", "-loglevel", "error", "-ss", f"{s:.3f}", "-t", f"{d:.3f}",
                        "-i", src, "-vn", "-af", "aresample=48000,aformat=channel_layouts=stereo,"
                        f"afade=t=in:d=0.03,afade=t=out:st={max(0.0, d - 0.03):.3f}:d=0.03,apad,atrim=duration={d:.3f}",
                        "-c:a", "pcm_s16le", out_wav], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"bed audio of shot {sh['n']} failed: {r.stderr[-300:]}")


# ── sound: accents from the owner's pack ────────────────────────────────────

# fixed picks (deterministic): short whooshes, soft impacts, short risers. Peak levels (dBFS) are deliberately quiet.
SFX_WHOOSH = ["whoosh/Вуши 02.wav", "whoosh/Вуши 09.wav", "whoosh/Быстрые Под вылеты/Вуши быстрые 02.wav", "whoosh/Вуши 01.wav"]
SFX_IMPACT = {"hook": "Riser Impact Drop/Impact/Impact 08.wav", "end": "Riser Impact Drop/Impact/Impact 05.wav"}
SFX_RISER = "Riser Impact Drop/Riser/Riser 03.wav"
SFX_PEAK_DB = {"whoosh": -26.0, "impact": -20.0, "riser": -25.0}


def peak_time(path: str) -> float:
    """Time of the loudest moment of a sound file (the accent is placed so that this moment lands on the cut)."""
    import numpy as np
    r = subprocess.run([media.ffmpeg_bin(), "-v", "error", "-nostdin", "-i", path, "-ac", "1", "-ar", "8000", "-f", "s16le", "-"],
                       capture_output=True)
    x = np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32)
    if x.size < 80:
        return 0.0
    env = np.convolve(np.abs(x), np.ones(80) / 80, mode="same")            # 10 ms
    return float(int(env.argmax())) / 8000.0


def plan_accents(tl: Dict[str, Any], script: Dict[str, Any], vt: Dict[str, Any], pack: str, max_whoosh: int = 7) -> List[Dict[str, Any]]:
    """Where the quiet accents go: impact on the turn from the hook into the story, whooshes on the segment turns into twist /
    cliffhanger and on some cuts of the climax, a riser into the break and an impact right after the last word. Pure but for file checks."""
    from . import compose
    root = Path(pack)
    spans = compose.segment_spans(vt)
    out: List[Dict[str, Any]] = []
    segs = script["segments"]
    shots = tl["shots"]
    cuts = [s["t0"] for s in shots[1:]]
    if len(segs) > 1:
        out.append({"kind": "impact", "tag": "hook→story", "file": SFX_IMPACT["hook"], "at": spans[segs[1]["id"]][0], "keep": 1.6})
    turn = {spans[s["id"]][0] for s in segs if s["function"] in ("twist", "cliffhanger")}
    climax = [c for c in cuts if c >= 0.68 * tl["total"]]
    whoosh_at = sorted(turn | set(climax[1::2]))
    whoosh_at = [t for t in whoosh_at if t > 1.0 and t < tl["total"] - 1.5]
    if len(whoosh_at) > max_whoosh:                                             # keep the turns, thin the rest evenly
        keep = sorted(turn & set(whoosh_at))
        rest = [t for t in whoosh_at if t not in keep]
        step = max(1, len(rest) // max(1, max_whoosh - len(keep)))
        whoosh_at = sorted(keep + rest[::step][:max(0, max_whoosh - len(keep))])
    for i, t in enumerate(whoosh_at):
        out.append({"kind": "whoosh", "tag": "cut", "file": SFX_WHOOSH[i % len(SFX_WHOOSH)], "at": t, "keep": 1.2})
    last_end = min(spans[segs[-1]["id"]][1], tl["total"] - compose.TAIL)          # end of the last WORD (the last span runs on into the tail)
    out.append({"kind": "riser", "tag": "into the break", "file": SFX_RISER, "at": last_end + 0.05, "keep": 3.0})
    out.append({"kind": "impact", "tag": "the break", "file": SFX_IMPACT["end"], "at": last_end + 0.08, "keep": 1.5})
    out = [x for x in out if (root / x["file"]).is_file()]
    for x in out:
        x["path"] = str(root / x["file"])
        x["gain_db_peak"] = SFX_PEAK_DB[x["kind"]]
    return sorted(out, key=lambda x: x["at"])


def accent_filters(accents: Sequence[Dict[str, Any]], first_input: int, total: float) -> Tuple[List[str], List[str]]:
    """ffmpeg filter pieces for the accents: (filters, labels). Each accent: stereo 48 k, cut to `keep` s with a fade, scaled so its
    own peak is SFX_PEAK_DB, delayed so that its loudest moment lands on `at`."""
    parts, labels = [], []
    for i, x in enumerate(accents):
        pk = peak_time(x["path"])
        delay = max(0.0, x["at"] - pk)
        keep = min(max(x["keep"], pk + 0.3), max(0.2, total - delay))
        gain = x["gain_db_peak"] - sample_peak_db(x["path"])
        x["peak_s"], x["start_s"], x["gain_db"] = round(pk, 3), round(delay, 3), round(gain, 2)
        lab = f"sx{i}"
        parts.append(f"[{first_input + i}:a]aresample=48000,aformat=channel_layouts=stereo,atrim=duration={keep:.3f},"
                     f"afade=t=out:st={max(0.0, keep - 0.5):.3f}:d=0.5,volume={gain:.2f}dB,"
                     f"adelay={int(delay * 1000)}|{int(delay * 1000)},apad,atrim=duration={total:.3f}[{lab}]")
        labels.append(f"[{lab}]")
    return parts, labels


def sample_peak_db(path: str) -> float:
    m = media.loudness(path)
    return m["sample_peak_db"] if m.get("sample_peak_db") is not None else -6.0


def build_bed(bed_wavs: Sequence[str], speech: Sequence[bool], duck_db: float, out_wav: str) -> None:
    """The film's own sound of every shot, concatenated; shots where the film's characters talk are ducked by duck_db more
    (nothing intelligible may compete with the narrator)."""
    n = len(bed_wavs)
    ins: List[str] = []
    for p in bed_wavs:
        ins += ["-i", str(Path(p).resolve())]
    fc = ";".join(f"[{i}:a]volume={duck_db if speech[i] else 0.0:.1f}dB[a{i}]" for i in range(n)) + ";" + \
        "".join(f"[a{i}]" for i in range(n)) + f"concat=n={n}:v=0:a=1[o]"
    r = subprocess.run([media.ffmpeg_bin(), "-y", "-hide_banner", "-nostdin", "-loglevel", "error", *ins, "-filter_complex", fc,
                        "-map", "[o]", "-c:a", "pcm_s16le", out_wav], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"bed concat failed: {r.stderr[-300:]}")
