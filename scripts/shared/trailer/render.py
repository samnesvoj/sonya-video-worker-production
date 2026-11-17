"""
Trailer render: shots → 1080×1920 canvas (the film cropped to 3:4 around the main subject, big in the middle, over its own
blurred copy), the voice on top of the film's own sound (ducked under the voice, up in the gaps), a whoosh + white flash
after the hook, a low hit on the cliffhanger cut, subtitles in the site style ON the picture, the hook title on top for
3 s, and a small SONYA end card. Loudness −14 LUFS like every SONYA clip.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .edit import FPS

logger = logging.getLogger(__name__)

W, H = 1080, 1920
ASPECTS = {"3:4": (3, 4), "1:1": (1, 1), "4:5": (4, 5)}
SR = 48000
BED_DB = -9.0            # the film's sound when nobody narrates
DUCK_DB = -24.0          # under the narrator
TEXT_BED_DB = -3.0       # no voice (subtitles narrate): the film's sound is the soundtrack
TEXT_FADE = 0.045        # longer fades at cuts when the film's sound is up front (no clicks / chopped syllables)
TITLE_TOP = 300
SUB_MARGIN_V = 400       # subtitle bottom 400 px above the frame bottom: on the picture, clear of the TikTok UI
SAFE_W = 1080 - 2 * 80


def geometry(src_w: int, src_h: int, aspect: str = "3:4") -> Dict[str, int]:
    bw, bh = ASPECTS.get(aspect, ASPECTS["3:4"])
    ph = int(round(W * bh / bw / 2)) * 2
    ar = bw / bh
    if src_w / src_h >= ar:
        ch = src_h - src_h % 2
        cw = int(ch * ar) // 2 * 2
    else:
        cw = src_w - src_w % 2
        ch = int(cw / ar) // 2 * 2
    return {"pw": W, "ph": ph, "py": (H - ph) // 2, "cw": cw, "ch": ch, "src_w": src_w, "src_h": src_h}


def shot_filter(sh: Dict[str, Any], g: Dict[str, int]) -> str:
    d = max(0.04, sh["t1"] - sh["t0"])
    x0, x1 = sh["crop_x0"], sh["crop_x1"]
    xexpr = f"{x0}+({x1}-{x0})*min(1\\,t/{d:.3f})" if x0 != x1 else str(x0)
    fg = [f"crop={g['cw']}:{g['ch']}:'{xexpr}':0", f"scale={g['pw']}:{g['ph']}:flags=lanczos"]
    if sh.get("push_in"):
        n = max(2, int(round(d * FPS)))
        fg.append(f"zoompan=z='1+{sh['push_in']:.3f}*on/{n}':x='(iw-iw/zoom)*{sh.get('fx', 0.5):.3f}':"
                  f"y='(ih-ih/zoom)*{sh.get('fy', 0.4):.3f}':d=1:s={g['pw']}x{g['ph']}:fps={FPS}")
    bg = [f"scale={W // 4}:{H // 4}:force_original_aspect_ratio=increase", f"crop={W // 4}:{H // 4}",
          "boxblur=12:2", "eq=brightness=-0.10:saturation=1.15", f"scale={W}:{H}"]
    return (f"[0:v]fps={FPS},setpts=PTS-STARTPTS,split=2[a][b];[a]{','.join(fg)}[fg];[b]{','.join(bg)}[bg];"
            f"[bg][fg]overlay=0:{g['py']},setsar=1,format=yuv420p,trim=duration={d:.4f}[v]")


def render_shots(src: str, shots: Sequence[Dict[str, Any]], g: Dict[str, int], work: str, *, ffmpeg: str = "ffmpeg",
                 enc: Sequence[str] = (), workers: int = 3, log: Callable[[str], None] = logger.info) -> List[str]:
    """Each shot → its own short H.264 piece (same params → concat by copy). Cached by the shot's content."""
    wd = Path(work)
    wd.mkdir(parents=True, exist_ok=True)
    enc = list(enc) or ["-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p"]

    def one(sh):
        key = f"{sh['src_start']:.3f}_{sh['t1'] - sh['t0']:.4f}_{sh['crop_x0']}_{sh['crop_x1']}_{sh.get('push_in', 0)}"
        out = wd / f"shot_{abs(hash(key)) % 10**10}.mp4"
        if out.is_file() and out.stat().st_size > 1000:
            return str(out)
        d = sh["t1"] - sh["t0"]
        cmd = [ffmpeg, "-y", "-v", "error", "-nostdin", "-ss", f"{sh['src_start']:.3f}", "-t", f"{d + 0.25:.3f}", "-i", src,
               "-filter_complex", shot_filter(sh, g), "-map", "[v]", "-an", "-r", str(FPS), *enc, "-g", "60",
               "-video_track_timescale", "15360", str(out) + ".tmp.mp4"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode != 0 and "h264_nvenc" in cmd:
            cmd = [c for c in cmd]
            i = cmd.index("-c:v")
            j = cmd.index("-g")
            cmd[i:j] = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p"]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            raise RuntimeError(f"shot {sh['i']} render failed: {r.stderr[-300:]}")
        os.replace(str(out) + ".tmp.mp4", out)
        return str(out)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        return list(ex.map(one, shots))


# ── audio ─────────────────────────────────────────────────────────────────────────────────────────────────────────

def _read(ffmpeg: str, src: str, ss: float, d: float, ch: int = 2):
    import numpy as np
    r = subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-ss", f"{max(0.0, ss):.3f}", "-t", f"{d:.3f}", "-i", src, "-vn",
                        "-ac", str(ch), "-ar", str(SR), "-f", "f32le", "-"], capture_output=True, timeout=120)
    x = np.frombuffer(r.stdout, dtype=np.float32)
    x = x[: x.size // ch * ch].reshape(-1, ch)
    n = int(round(d * SR))
    if x.shape[0] < n:
        x = np.vstack([x, np.zeros((n - x.shape[0], ch), dtype=np.float32)])
    return x[:n].copy()


def _db(v: float) -> float:
    return 10 ** (v / 20.0)


def whoosh(sec: float = 0.45, seed: int = 7):
    """Band-passed noise swell, fast rise, short tail (the same family as the Hooks whoosh)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    n = int(sec * SR)
    x = rng.standard_normal(n).astype(np.float32)
    # crude band-pass by differencing + smoothing (≈ 800–5000 Hz)
    x = np.convolve(np.diff(x, prepend=0.0), np.ones(4) / 4, mode="same")
    t = np.linspace(0, 1, n)
    env = np.where(t < 0.7, (t / 0.7) ** 2.2, np.exp(-(t - 0.7) * 14))
    return (x * env / (np.max(np.abs(x * env)) + 1e-9) * _db(-20)).astype(np.float32)


def hit(sec: float = 1.1):
    """A low cinematic hit: 48 Hz sine + a click, exponential decay."""
    import numpy as np
    t = np.arange(int(sec * SR)) / SR
    f = 48 + 30 * np.exp(-t * 18)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 3.2)
    x[: int(0.004 * SR)] += np.hanning(int(0.008 * SR))[: int(0.004 * SR)] * 0.6
    return (x / np.max(np.abs(x)) * _db(-12)).astype(np.float32)


def mix(src: str, tl: Dict[str, Any], shots: Sequence[Dict[str, Any]], out_wav: str, *, ffmpeg: str = "ffmpeg",
        target_lufs: float = -14.0) -> Dict[str, Any]:
    """Voice + ducked film bed (+ whoosh after the hook, a hit on the end cut) → stereo 48 kHz WAV at −14 LUFS.
    Segments without a WAV (narration.reading: subtitles narrate) → the film's own sound up front, no ducking."""
    import numpy as np
    total = tl["total"]
    n = int(round(total * SR))
    voiced = any(s.get("wav") for s in tl["segments"])
    bed = np.zeros((n, 2), dtype=np.float32)
    fade = int((0.012 if voiced else TEXT_FADE) * SR)
    for sh in shots:
        d = sh["t1"] - sh["t0"]
        x = _read(ffmpeg, src, sh["src_start"], d)
        if x.shape[0] > 2 * fade:
            x[:fade] *= np.linspace(0, 1, fade)[:, None]
            x[-fade:] *= np.linspace(1, 0, fade)[:, None]
        a = int(round(sh["t0"] * SR))
        b = min(n, a + x.shape[0])
        bed[a:b] += x[: b - a]
    voice = np.zeros(n, dtype=np.float32)
    for s in tl["segments"]:
        if not s.get("wav"):
            continue
        from .voice import read_mono
        y, sr = read_mono(s["wav"])
        if sr != SR:
            raise RuntimeError("voice WAV must be 48 kHz")
        a = int(round(s["t0"] * SR))
        b = min(n, a + y.size)
        voice[a:b] += y[: b - a]
    # ducking envelope from the voiced intervals (attack 60 ms before a phrase, release 300 ms after)
    gain = np.full(n, _db(BED_DB if voiced else TEXT_BED_DB), dtype=np.float32)
    for s in (tl["segments"] if voiced else []):
        a = int(max(0.0, s["t0"] - 0.06) * SR)
        b = int(min(total, s["t1"] + 0.05) * SR)
        gain[a:b] = _db(DUCK_DB)
    k = int(0.15 * SR)
    ker = np.ones(k, dtype=np.float32) / k
    gain = np.convolve(gain, ker, mode="same").astype(np.float32)
    card0, _ = tl["end_card"]
    c0 = int(card0 * SR)
    if c0 < n:                                         # the film fades out under the end card
        gain[c0:] *= np.linspace(1.0, 0.0, n - c0) ** 2
    out = bed * gain[:, None] + voice[:, None]
    fx = np.zeros(n, dtype=np.float32)
    segs = tl["segments"]
    if len(segs) > 1 and segs[0]["fn"] == "hook":
        w = whoosh()
        a = max(0, int((segs[1]["t0"] - 0.38) * SR))
        fx[a:a + w.size] += w[: max(0, min(w.size, n - a))]
    h = hit()
    a = int(card0 * SR)
    if a < n:
        fx[a:a + h.size] += h[: n - a]
    out += fx[:, None]
    tmp = out_wav + ".pre.wav"
    _write(tmp, out)
    gain_db = 0.0
    try:
        from ..loudness import measure
        m = measure(ffmpeg, tmp)
        if m is not None:
            gain_db = max(-12.0, min(12.0, target_lufs - m))
    except Exception as exc:                           # noqa: BLE001
        logger.warning("[trailer] loudness measure failed: %s", exc)
    out = out * _db(gain_db)
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    lim = _db(-1.5)
    if peak > lim:                                     # soft limiter: tanh knee above −1.5 dBFS
        out = np.where(np.abs(out) > lim * 0.8, np.sign(out) * (lim * 0.8 + (lim * 0.2) * np.tanh((np.abs(out) - lim * 0.8) / (lim * 0.2))), out)
    _write(out_wav, out)
    os.unlink(tmp)
    return {"gain_db": round(gain_db, 2), "bed_db": BED_DB if voiced else TEXT_BED_DB, "duck_db": DUCK_DB if voiced else None,
            "voice": voiced}


def _write(path: str, x) -> None:
    import wave
    import numpy as np
    y = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(y.tobytes())


# ── subtitles + title + end card ──────────────────────────────────────────────────────────────────────────────────

def ass(tl: Dict[str, Any], title: str, preset: str, font: Optional[str] = None) -> str:
    from ..boundaries import Word
    from ..subtitles import SubtitleStyle
    from ..subtitle_styles import to_ass
    words = [Word(w, a, b) for s in tl["segments"] for w, a, b in s["words"]]
    # attach punctuation from the text so the cue groups split on sentence ends
    words = _with_punct(words, tl)
    total = tl["total"]
    card0, _ = tl["end_card"]
    text = to_ass(words, preset, card0, base=SubtitleStyle(margin_v=SUB_MARGIN_V), hook=title or None,
                  font=font, hook_top=TITLE_TOP)
    card_style = "Style: Card,Montserrat Black,86,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,2,0,1,0,0,5,0,0,0,204"
    text = re.sub(r"^(Style: Hook,[^\n]*)$", lambda m: m.group(1) + "\n" + card_style, text, count=1, flags=re.M)
    end = [f"Dialogue: 2,{_t(card0 + 0.12)},{_t(total)},Card,,0,0,0,,{{\\an5\\pos({W // 2},{H // 2 - 30})\\fad(120,0)}}SONYA",
           f"Dialogue: 2,{_t(card0 + 0.25)},{_t(total)},Card,,0,0,0,,{{\\an5\\pos({W // 2},{H // 2 + 50})\\fad(150,0)"
           f"\\fs38\\c&H00D0D0D0&}}sonya.group"]
    return text.rstrip("\n") + "\n" + "\n".join(end) + "\n"


def _with_punct(words, tl):
    from ..boundaries import Word
    out = []
    for s in tl["segments"]:
        toks = re.findall(r"[\w]+(?:[-'][\w]+)*[^\w\s]*", s["text"], re.U)
        ws = [w for w in words if s["t0"] - 0.01 <= w.start <= s["t1"] + 0.01]
        for k, w in enumerate(ws):
            t = toks[k] if k < len(toks) and re.sub(r"[^\w]", "", toks[k]) == re.sub(r"[^\w]", "", w.text) else w.text
            t = re.sub(r"[—–-]+$", "", t)
            out.append(Word(t, w.start, w.end))
    return out


def _t(t: float) -> str:
    cs = int(round(max(0.0, t) * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def ass_widths(text: str) -> List[Tuple[str, float]]:
    """Widest rendered line of every subtitle/title event (glyph widths of the style's font at its size × the active-word
    grow), for the «never outside the frame» check."""
    from ..subtitle_styles import _text_width
    styles = {}
    for line in text.splitlines():
        if line.startswith("Style:"):
            p = line[6:].split(",")
            styles[p[0].strip()] = (p[1].strip(), int(float(p[2])), float(p[12]) if len(p) > 12 else 100.0)
    out = []
    for line in text.splitlines():
        if not line.startswith("Dialogue:"):
            continue
        parts = line.split(",", 9)
        st = styles.get(parts[3].strip())
        if not st or "SONYA" in parts[9] or "sonya.group" in parts[9]:
            continue
        raw = parts[9]
        sc = re.search(r"\\fscx(\d+)", raw)
        scale = (int(sc.group(1)) / 100.0) if sc and parts[3].strip() == "Hook" else 1.0
        clean = re.sub(r"\{[^}]*\}", "", raw)
        grow = 1.12 if "\\fscx1" in raw and parts[3].strip() != "Hook" else 1.0
        for row in clean.split("\\N"):
            wdt = _text_width(row.upper() if st[2] == 1 else row, st[0], st[1]) * grow * scale
            out.append((row, wdt))
    return out


def fit_ass(text: str) -> Tuple[str, Dict[str, Any]]:
    """Shrink the subtitle style until every line fits SAFE_W (never outside the frame)."""
    ws = ass_widths(text)
    widest = max((w for _, w in ws), default=0.0)
    info = {"widest_px": round(widest, 1), "safe_px": SAFE_W, "scaled": 1.0}
    if widest > SAFE_W:
        k = SAFE_W / widest
        def fix(m):
            p = m.group(0).split(",")
            p[2] = str(int(int(float(p[2])) * k))
            return ",".join(p)
        text = re.sub(r"^Style: Sonya,[^\n]*", fix, text, flags=re.M)
        info["scaled"] = round(k, 3)
        info["widest_px"] = round(max((w for _, w in ass_widths(text)), default=0.0), 1)
    return text, info


# ── the final encode ──────────────────────────────────────────────────────────────────────────────────────────────

def final(pieces: Sequence[str], tl: Dict[str, Any], audio_wav: str, ass_path: str, out: str, work: str, *,
          ffmpeg: str = "ffmpeg", enc: Sequence[str] = (), fonts_dir: Optional[str] = None,
          flash_at: Optional[float] = None) -> Dict[str, Any]:
    wd = Path(work)
    lst = wd / (Path(out).stem + ".concat.txt")
    lst.write_text("".join(f"file '{Path(p).resolve()}'\n" for p in pieces))
    total = tl["total"]
    card0, _ = tl["end_card"]
    fdir = fonts_dir or str(Path(__file__).resolve().parent.parent / "fonts")
    vf = [f"tpad=stop_mode=clone:stop_duration={total - card0 + 0.5:.3f}", f"trim=duration={total:.4f}",
          f"boxblur=18:2:enable='gte(t,{card0:.3f})'", f"eq=brightness=-0.32:enable='gte(t,{card0:.3f})'"]
    if flash_at is not None:
        vf.append(f"drawbox=x=0:y=0:w=iw:h=ih:color=white@0.85:t=fill:enable='between(t,{flash_at:.3f},{flash_at + 0.1:.3f})'")
    vf.append(f"subtitles='{_esc(ass_path)}':fontsdir='{_esc(fdir)}'")
    vf.append("format=yuv420p")
    enc = list(enc) or ["-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p"]

    def make(e):
        return [ffmpeg, "-y", "-v", "error", "-nostdin", "-f", "concat", "-safe", "0", "-i", str(lst), "-i", audio_wav,
                "-vf", ",".join(vf), "-map", "0:v", "-map", "1:a", "-r", str(FPS), *e, "-c:a", "aac", "-b:a", "192k",
                "-ar", str(SR), "-t", f"{total:.4f}", "-movflags", "+faststart", out]
    try:
        from ..cut_profile import run_encode
        r, used, sec = run_encode(make, enc, timeout=max(300.0, 10 * total), what="trailer")
    except ImportError:
        r = subprocess.run(make(enc), capture_output=True, text=True)
        used, sec = "cpu", 0.0
    if r.returncode != 0:
        raise RuntimeError(f"trailer encode failed: {(r.stderr or '')[-400:]}")
    return {"encoder": used, "encode_s": sec}


def _esc(p: str) -> str:
    return str(p).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def probe(path: str) -> Dict[str, Any]:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height",
                        "-of", "json", path], capture_output=True, text=True)
    d = json.loads(r.stdout or "{}")
    v = next((s for s in d.get("streams", []) if s.get("codec_type") == "video"), {})
    return {"duration": float((d.get("format") or {}).get("duration") or 0), "width": v.get("width"),
            "height": v.get("height"), "audio": any(s.get("codec_type") == "audio" for s in d.get("streams", []))}


def contact_sheet(video: str, out_jpg: str, *, ffmpeg: str = "ffmpeg", every: float = 3.0, cols: int = 8) -> str:
    dur = probe(video)["duration"]
    n = max(1, int(dur / every))
    rows = (n + cols - 1) // cols
    subprocess.run([ffmpeg, "-y", "-v", "error", "-nostdin", "-i", video, "-vf",
                    f"fps=1/{every},scale=216:384,drawtext=text='%{{pts\\:hms}}':x=4:y=4:fontsize=18:fontcolor=yellow:box=1:boxcolor=black@0.6,"
                    f"tile={cols}x{rows}:padding=4", "-frames:v", "1", "-q:v", "4", out_jpg], capture_output=True)
    return out_jpg
