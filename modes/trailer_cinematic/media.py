"""
media.py — ffmpeg side of trailer_cinematic: probe, fast-run source cut, scene cuts, per-fragment 9:16
render (the existing Cut crop engine), final assembly (concat + ducking + voice + loudnorm/limiter +
narration subtitles), loudness/peak check. No frames go to the LLM in v1.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from scripts.shared.crop.smart_crop_adapter import _env_encoder_args, ffmpeg_bin

SCENES_VERSION = "scdet-v1"


def ffprobe_bin() -> str:
    ff = ffmpeg_bin()
    d = os.path.dirname(ff)
    return os.path.join(d, "ffprobe") if d else "ffprobe"


def probe(path: str) -> Dict[str, Any]:
    out = subprocess.run([ffprobe_bin(), "-v", "error", "-show_entries",
                          "format=duration:stream=codec_type,width,height", "-of", "json", path],
                         capture_output=True, text=True, check=True)
    d = json.loads(out.stdout)
    v = next((s for s in d.get("streams", []) if s.get("codec_type") == "video"), {})
    return {"duration": float(d["format"]["duration"]), "width": int(v.get("width") or 0),
            "height": int(v.get("height") or 0),
            "has_audio": any(s.get("codec_type") == "audio" for s in d.get("streams", []))}


def scene_cuts(path: str, cache: Optional[str] = None, threshold: float = 0.32) -> List[float]:
    """Hard cuts from ffmpeg's scene score on a small decode (CPU; ~1–2 min per 90 min of 480p)."""
    if cache and os.path.isfile(cache):
        try:
            d = json.loads(Path(cache).read_text())
            if d.get("version") == SCENES_VERSION and d.get("threshold") == threshold:
                return d["cuts"]
        except (OSError, ValueError, KeyError):
            pass
    cmd = [ffmpeg_bin(), "-hide_banner", "-nostdin", "-loglevel", "error", "-skip_frame", "noref",
           "-i", path, "-an", "-sn",
           "-vf", f"scale=192:-2,select='gt(scene,{threshold})',metadata=print:file=-", "-f", "null", "-"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"scene detection failed: {r.stderr[-300:]}")
    cuts = [round(float(m), 3) for m in re.findall(r"pts_time:([0-9.]+)", r.stdout)]
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        Path(cache).write_text(json.dumps({"version": SCENES_VERSION, "threshold": threshold, "cuts": cuts}))
    return cuts


def cut_head(src: str, minutes: float, out: str) -> str:
    """Fast run: the first `minutes` of the source, stream copy (no re-encode)."""
    if os.path.isfile(out) and os.path.getsize(out) > 0:
        return out
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    tmp = out + ".partial.mp4"
    r = subprocess.run([ffmpeg_bin(), "-y", "-hide_banner", "-nostdin", "-loglevel", "error", "-i", src,
                        "-t", f"{minutes * 60:.3f}", "-map", "0:v:0", "-map", "0:a:0", "-c", "copy",
                        "-movflags", "+faststart", tmp], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"fast-run cut failed: {r.stderr[-300:]}")
    os.replace(tmp, out)
    return out


def extract_wav16k(src: str, out: str) -> str:
    r = subprocess.run([ffmpeg_bin(), "-y", "-hide_banner", "-nostdin", "-loglevel", "error", "-i", src,
                        "-vn", "-ac", "1", "-ar", "16000", out], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"audio extract failed: {r.stderr[-300:]}")
    return out


def loudness(path: str) -> Dict[str, Any]:
    """Integrated loudness / true peak (ebur128) + sample peak (volumedetect) of the final mix."""
    r = subprocess.run([ffmpeg_bin(), "-hide_banner", "-nostdin", "-i", path, "-vn",
                        "-af", "ebur128=peak=true,volumedetect", "-f", "null", "-"], capture_output=True, text=True)
    err = r.stderr
    def last(rx):
        m = re.findall(rx, err)
        return float(m[-1]) if m else None
    return {"integrated_lufs": last(r"I:\s+(-?[0-9.]+) LUFS"), "true_peak_dbfs": last(r"Peak:\s+(-?[0-9.]+) dBFS"),
            "sample_peak_db": last(r"max_volume: (-?[0-9.]+) dB")}


# ── per-fragment render: the Cut crop engine (faces, cuts, split) ──────────

def render_fragments(src: str, frags: List[Dict[str, Any]], out_dir: Path, *, width: int, height: int,
                     layout: str = "smart", cache_dir: Optional[str] = None,
                     source_sha256: Optional[str] = None) -> List[str]:
    from scripts.shared.crop import clip_analysis as ca
    from scripts.shared.crop.smart_crop_adapter import crop_track_v2, fit_track, render_tracked

    def one(f: Dict[str, Any]) -> str:
        out = str(out_dir / f"frag_{f['id']}.mp4")
        s, d = f["render_start"], f["render_end"] - f["render_start"]
        if layout in ("fit", "blur"):
            track = fit_track(width, height, d, "blur" if layout == "blur" else "black")
        else:
            an = None
            key = None
            if cache_dir and source_sha256:
                key = os.path.join(cache_dir, f"{source_sha256[:16]}-{s:.3f}-{d:.3f}-{ca.CLIP_ANALYSIS_VERSION}.json")
                if os.path.isfile(key):
                    try:
                        an = json.loads(Path(key).read_text())
                    except (OSError, ValueError):
                        an = None
            if an is None and ca.model_ok()["ok"]:
                try:
                    an = ca.analyze_clip(src, s, d)
                    if key:
                        os.makedirs(cache_dir, exist_ok=True)
                        Path(key).write_text(json.dumps(an))
                except Exception as exc:                       # noqa: BLE001 — centred crop, recorded
                    f["crop_error"] = str(exc)[:200]
            track = crop_track_v2(an, d, width=width, height=height)
        f["crop_status"] = track.get("crop_status")
        f["crop_fallback_reason"] = track.get("fallback_reason")
        render_tracked(src, out, s, d, track)
        return out

    workers = max(1, int(os.environ.get("SONYA_CLIP_WORKERS", "2")))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, frags))


# ── voice track + final assembly ────────────────────────────────────────────

def voice_track(frags: Sequence[Dict[str, Any]], wavs: Dict[str, str], length: float, out_wav: str,
                sr: int = 48000) -> None:
    import numpy as np
    buf = np.zeros(int((length + 0.5) * sr), dtype=np.float32)
    for f in frags:
        p = wavs.get(f["id"])
        if not p or not f.get("voice_sec"):
            continue
        with wave.open(p, "rb") as w:
            data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
            if w.getframerate() != sr:                       # nearest-neighbour resample (voice only)
                idx = (np.arange(int(len(data) * sr / w.getframerate())) * w.getframerate() / sr).astype(int)
                data = data[np.clip(idx, 0, len(data) - 1)]
        a = int((f["t0"] + f["voice_offset"]) * sr)
        b = min(len(buf), a + len(data))
        buf[a:b] += data[:b - a]
    pcm = (np.clip(buf, -1, 1) * 32767).astype("<i2")
    with wave.open(out_wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def assemble(frag_files: Sequence[str], frags: Sequence[Dict[str, Any]], voice_wav: str, duck: str,
             ass_path: Optional[str], out_path: str, *, fade_out: float = 0.6) -> List[str]:
    """One ffmpeg pass: concat fragments (40 ms audio fades at every join) → original audio ducked under
    the voice → voice (light compression) → loudnorm −14 LUFS → limiter −1.5 dBFS (no clipping) →
    burned ASS subtitles → fade out. No transitions."""
    n = len(frag_files)
    total = sum(f["render_end"] - f["render_start"] for f in frags)
    inputs: List[str] = []
    for p in frag_files:
        inputs += ["-i", str(Path(p).resolve())]
    inputs += ["-i", str(Path(voice_wav).resolve())]
    parts = []
    for i, f in enumerate(frags):
        d = f["render_end"] - f["render_start"]
        parts.append(f"[{i}:a]aresample=48000,aformat=channel_layouts=stereo,"
                     f"afade=t=in:d=0.04,afade=t=out:st={max(0.0, d - 0.04):.3f}:d=0.04[a{i}]")
        parts.append(f"[{i}:v]settb=AVTB,fps=30,format=yuv420p[v{i}]")
    parts.append("".join(f"[v{i}][a{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=1[vc][ac]")
    parts.append(f"[ac]volume='{duck}':eval=frame[orig]")
    parts.append(f"[{n}:a]aresample=48000,aformat=channel_layouts=stereo,highpass=f=70,"
                 "acompressor=threshold=-20dB:ratio=3:attack=5:release=120,volume=1.7[vo]")
    parts.append(f"[orig][vo]amix=inputs=2:normalize=0:duration=first,"
                 f"loudnorm=I=-14:TP=-1.5:LRA=11,alimiter=limit=0.84:level=disabled,afade=t=out:st={max(0.0, total - fade_out):.3f}:d={fade_out}[aout]")
    vf = "[vc]"
    cwd = None
    if ass_path:
        from scripts.shared.subtitle_styles import FONTS_DIR
        sub = Path(ass_path).resolve()
        cwd = str(sub.parent)
        fd = str(FONTS_DIR).replace(":", "\\:").replace("'", "")
        vf += f"ass=filename={sub.name}:fontsdir='{fd}',"
    vf += f"fade=t=out:st={max(0.0, total - fade_out):.3f}:d={fade_out}[vout]"
    parts.append(vf)
    enc = _env_encoder_args() or ["-c:v", "libx264", "-crf", "20", "-preset", "medium"]
    cmd = [ffmpeg_bin(), "-y", "-hide_banner", "-nostdin", "-loglevel", "error", *inputs,
           "-filter_complex", ";".join(parts), "-map", "[vout]", "-map", "[aout]", *enc,
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart",
           str(Path(out_path).resolve())]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if r.returncode != 0:
        raise RuntimeError(f"assembly failed: {r.stderr[-600:]}")
    return cmd


# ── v2 recap: silent shots, narration-only assembly ─────────────────────────

def has_audio(path: str) -> bool:
    return probe(path)["has_audio"]


def strip_audio(src_mp4: str, out_mp4: str) -> str:
    """Remux without any audio stream (video is copied, nothing re-encoded)."""
    r = subprocess.run([ffmpeg_bin(), "-y", "-hide_banner", "-nostdin", "-loglevel", "error", "-i", src_mp4,
                        "-map", "0:v:0", "-an", "-c:v", "copy", "-movflags", "+faststart", out_mp4],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"strip audio failed: {r.stderr[-300:]}")
    return out_mp4


def steady_track(track: Dict[str, Any], d: float) -> Optional[float]:
    """One fixed 9:16 window per recap shot (≤ 5.5 s): the duration-weighted median x of the tracker's keyframes.
    The tracker jumps inside a shot when it sees a false cut or switches between two faces; in the final that jump
    reads as an extra cut (observed: timeline_match found 3 "unplanned cuts" that were crop jumps, not source cuts).
    Returns the x used, or None when the track has no keyframes (fit / fallback layouts are left alone)."""
    keys = track.get("keyframes")
    if not keys or len(keys) < 2:
        return None
    segs = [(max(0.0, min(t1, d) - max(t0, 0.0)), (x0 + x1) / 2) for (t0, x0), (t1, x1) in zip(keys, keys[1:])]
    segs = [(w, x) for w, x in segs if w > 0] or [(1.0, keys[0][1])]
    segs.sort(key=lambda z: z[1])
    half, acc, xm = sum(w for w, _ in segs) / 2, 0.0, segs[-1][1]
    for w, x in segs:
        acc += w
        if acc >= half:
            xm = x
            break
    xm = float(int(xm) // 2 * 2)
    track["keyframes"] = [[0.0, xm], [round(d, 3), xm]]
    return xm


class DamagedSource(RuntimeError):
    """The source cannot be decoded in this shot's window, even error-tolerantly. `candidate_id` → pick another."""

    def __init__(self, msg: str, candidate_id: str):
        super().__init__(msg)
        self.candidate_id = candidate_id


def render_shots(src: str, shots: List[Dict[str, Any]], out_dir: Path, *, width: int, height: int,
                 cache_dir: Optional[str] = None, source_sha256: Optional[str] = None,
                 workers: Optional[int] = None) -> List[str]:
    """One 9:16 clip per shot with ONE framing style for the whole video: the Cut smart crop that follows the main
    face inside a scene (jumps only at cuts). No split screen, no close-up inset, no blurred-background layout —
    v1's chaotic switching is gone by construction. Every clip is written WITHOUT an audio stream."""
    from scripts.shared.crop import clip_analysis as ca
    from scripts.shared.crop.smart_crop_adapter import crop_track_v2, render_tracked

    def one(sh: Dict[str, Any]) -> str:
        out = str(out_dir / f"shot_{sh['n']:02d}.mp4")
        s, d = sh["src_start"], sh["src_end"] - sh["src_start"]
        an, key = None, None
        if cache_dir and source_sha256:
            key = os.path.join(cache_dir, f"{source_sha256[:16]}-{s:.3f}-{d:.3f}-{ca.CLIP_ANALYSIS_VERSION}.json")
            if os.path.isfile(key):
                try:
                    an = json.loads(Path(key).read_text())
                except (OSError, ValueError):
                    an = None
        if an is None and ca.model_ok()["ok"]:
            try:
                an = ca.analyze_clip(src, s, d)
                if key:
                    os.makedirs(cache_dir, exist_ok=True)
                    Path(key).write_text(json.dumps(an))
            except Exception as exc:                            # noqa: BLE001 — centred crop, recorded
                sh["crop_error"] = str(exc)[:200]
        track = crop_track_v2(an, d, width=width, height=height)
        track.pop("split", None)
        track.pop("closeup", None)
        sh["crop_status"] = track.get("crop_status")
        sh["crop_steady"] = steady_track(track, d)
        tmp = out + ".withaudio.mp4"
        try:
            render_tracked(src, tmp, s, d, track)
        except RuntimeError as exc:
            # a long source can hold a damaged stretch the decoder aborts on (observed on a 3-hour film): cut that
            # piece out error-tolerantly first, then crop the clean copy with the same track
            rescue = out + ".rescue.mp4"
            r = subprocess.run([ffmpeg_bin(), "-y", "-hide_banner", "-nostdin", "-loglevel", "error",
                                "-err_detect", "ignore_err", "-fflags", "+discardcorrupt+genpts", "-ss", f"{max(0.0, s - 1.0):.3f}",
                                "-i", src, "-ss", f"{min(1.0, s):.3f}", "-t", f"{d + 0.2:.3f}", "-an", "-c:v", "libx264",
                                "-preset", "veryfast", "-crf", "16", rescue], capture_output=True, text=True)
            if r.returncode != 0 or not os.path.isfile(rescue):
                raise DamagedSource(f"shot {sh['n']}: source damaged at {s:.2f}s and rescue failed: {exc}",
                                    sh.get("candidate_id", "")) from exc
            sh["render_rescue"] = True
            render_tracked(rescue, tmp, 0.0, d, track)
            os.remove(rescue)
        strip_audio(tmp, out)
        os.remove(tmp)
        return out

    n = workers or max(1, int(os.environ.get("SONYA_CLIP_WORKERS", "2")))
    with ThreadPoolExecutor(max_workers=n) as pool:
        return list(pool.map(one, shots))


def voice_gain_db(voice_wav: str, target_lufs: float = -16.0, peak_ceiling_db: float = 2.5) -> Dict[str, float]:
    """Static gain that brings the narrator to target loudness. Speech has a high crest factor, so the sample peak after
    gain may reach +peak_ceiling_db; the alimiter in assemble_recap (−1.5 dBFS) shaves only those few peaks (deterministic;
    the validator still compares the final audio with the voice track)."""
    m = loudness(voice_wav)
    gain = target_lufs - (m["integrated_lufs"] if m["integrated_lufs"] is not None else target_lufs)
    if m["sample_peak_db"] is not None:
        gain = min(gain, peak_ceiling_db - m["sample_peak_db"])
    return {"gain_db": round(gain, 2), "measured_lufs": m["integrated_lufs"], "sample_peak_db": m["sample_peak_db"]}


def voice_chain(gain_db: float) -> str:
    """The ONLY processing the narrator track gets (shared by the assembly and by the validator's reference)."""
    return f"highpass=f=70,volume={gain_db:.2f}dB,alimiter=limit=0.84:level=disabled"


def assemble_recap(shot_files: Sequence[str], voice_wav: str, gain_db: float, total: float, ass_path: Optional[str],
                   out_path: str, *, fade_out: float = 0.6, fps: int = 30,
                   shot_frames: Optional[Sequence[int]] = None) -> List[str]:
    """Concat the silent shots → burn narration subtitles → fade out; audio = ONLY the narrator track
    (highpass + static gain + safety limiter). The shot files carry no audio, so the source sound cannot
    reach the mix: there is no input to mix it from."""
    n = len(shot_files)
    inputs: List[str] = []
    for p in shot_files:
        inputs += ["-i", str(Path(p).resolve())]
    inputs += ["-i", str(Path(voice_wav).resolve())]
    # shot_frames (optional): exact frame count per shot on the output grid. Source clips are cut at the SOURCE frame rate
    # (e.g. 25 fps) and lose up to one source frame each; without this the cuts drift earlier than the plan.
    fit = (lambda i: f",tpad=stop_mode=clone:stop_duration=1,trim=end_frame={int(shot_frames[i])},setpts=PTS-STARTPTS"
           if shot_frames is not None and len(shot_frames) == n else (lambda i: ""))
    parts = [f"[{i}:v]settb=AVTB,fps={fps}{fit(i)},format=yuv420p[v{i}]" for i in range(n)]
    parts.append("".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[vc]")
    vf = "[vc]"
    cwd = None
    if ass_path:
        from scripts.shared.subtitle_styles import FONTS_DIR
        sub = Path(ass_path).resolve()
        cwd = str(sub.parent)
        fd = str(FONTS_DIR).replace(":", "\\:").replace("'", "")
        vf += f"ass=filename={sub.name}:fontsdir='{fd}',"
    vf += f"fade=t=out:st={max(0.0, total - fade_out):.3f}:d={fade_out},trim=duration={total:.3f}[vout]"
    parts.append(vf)
    parts.append(f"[{n}:a]aresample=48000,aformat=channel_layouts=stereo,{voice_chain(gain_db)},"
                 f"atrim=duration={total:.3f},asetpts=PTS-STARTPTS[aout]")
    enc = _env_encoder_args() or ["-c:v", "libx264", "-crf", "20", "-preset", "medium"]
    cmd = [ffmpeg_bin(), "-y", "-hide_banner", "-nostdin", "-loglevel", "error", *inputs,
           "-filter_complex", ";".join(parts), "-map", "[vout]", "-map", "[aout]", *enc, "-pix_fmt", "yuv420p",
           "-r", str(fps), "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart",
           str(Path(out_path).resolve())]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if r.returncode != 0:
        raise RuntimeError(f"recap assembly failed: {r.stderr[-600:]}")
    return cmd


def contact_sheet(src: str, items: Sequence[Dict[str, Any]], out_jpg: str, *, cols: int = 4, tile_w: int = 270,
                  tile_h: int = 480) -> str:
    """Contact sheet of the FINISHED video: one frame (mid-shot) per shot with its label — for the Creative Judge."""
    import cv2
    import numpy as np
    tiles = []
    for it in items:
        t = it["t"]
        tmp = out_jpg + f".{it['n']}.png"
        r = subprocess.run([ffmpeg_bin(), "-y", "-hide_banner", "-nostdin", "-loglevel", "error", "-ss", f"{t:.3f}",
                            "-i", src, "-frames:v", "1", "-vf", f"scale={tile_w}:{tile_h}", tmp],
                           capture_output=True, text=True)
        img = cv2.imread(tmp) if r.returncode == 0 and os.path.isfile(tmp) else None
        if os.path.isfile(tmp):
            os.remove(tmp)
        if img is None:
            img = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
        cv2.rectangle(img, (0, 0), (tile_w, 26), (0, 0, 0), -1)
        cv2.putText(img, it["label"], (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img)
    while len(tiles) % cols:
        tiles.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(out_jpg, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 70])
    return out_jpg


def frame_cuts(src: str, start: float, end: float, cache_dir: Optional[str] = None, sha: str = "") -> List[float]:
    """Frame-accurate scene cuts (absolute source times) in [start, end] — every frame decoded (clip analysis), unlike the
    keyframe-only pass of scene_cuts(), which can miss cuts."""
    from scripts.shared.crop import clip_analysis as ca
    key = os.path.join(cache_dir, f"cuts-{sha[:16]}-{start:.2f}-{end:.2f}-{ca.CLIP_ANALYSIS_VERSION}.json") if cache_dir and sha else None
    if key and os.path.isfile(key):
        try:
            return json.loads(Path(key).read_text())
        except (OSError, ValueError):
            pass
    an = ca.analyze_clip(src, start, end - start, face_fps=1.0)
    cuts = [round(start + c, 3) for c in an.get("cuts") or []]
    if key:
        os.makedirs(cache_dir, exist_ok=True)
        Path(key).write_text(json.dumps(cuts))
    return cuts
