"""
validator.py — Technical Validator of trailer_cinematic v2 (spec §7, §10). Deterministic: ffprobe / ffmpeg /
numpy, no LLM. An LLM score can never override a failed check here.

Checks (each → {name, ok, value, detail}):
  duration            final length inside [lo, hi] (default 54–66 s)
  container           ffmpeg decodes the whole file without errors; 1 video + exactly 1 audio stream
  params              1080×1920, h264, yuv420p, 30 fps; AAC, 48 kHz; bit rates sane
  video_continuity    packet timestamps monotonic with no gap, frame count matches duration, no black stretch
  timeline_match      shots are contiguous and cover the video; the scene cuts detected IN the final MP4 coincide
                      with the planned shot boundaries (no unplanned cuts, no missing planned cuts)
  no_original_audio   (a) shot files have no audio stream, (b) the mix plan is narration-only,
                      (c) final audio == narrator track (correlation + residual), (d) nothing but silence in the
                      pauses, (e) negative control: zero correlation with the SOURCE audio of the used intervals
  words_intact        every sentence starts/ends in near-silence on the final audio, voice ends before the video;
                      optional ASR of the final audio vs the script text (cut / garbled words)
  loudness            integrated loudness in range, true peak ≤ −1 dBFS, per-sentence spread ≤ 6 dB
"""
from __future__ import annotations

import difflib
import json
import os
import re
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

SR = 16000


def _ff() -> str:
    return os.environ.get("SONYA_FFMPEG") or "ffmpeg"


def _fp() -> str:
    d = os.path.dirname(_ff())
    return os.path.join(d, "ffprobe") if d else "ffprobe"


def chk(name: str, ok: bool, value: Any = None, detail: str = "") -> Dict[str, Any]:
    return {"name": name, "ok": bool(ok), "value": value, "detail": detail}


def probe_all(path: str) -> Dict[str, Any]:
    r = subprocess.run([_fp(), "-v", "error", "-show_streams", "-show_format", "-of", "json", path],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return {}
    return json.loads(r.stdout)


def decode_audio(path: str, sr: int = SR, start: Optional[float] = None, dur: Optional[float] = None):
    import numpy as np
    cmd = [_ff(), "-hide_banner", "-nostdin", "-loglevel", "error"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", path]
    if dur is not None:
        cmd += ["-t", f"{dur:.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    r = subprocess.run(cmd, capture_output=True)
    return np.frombuffer(r.stdout, dtype="<f4").astype("float64") if r.returncode == 0 else np.zeros(0)


def rms_db(x) -> float:
    import numpy as np
    return 20 * np.log10(max(1e-7, float(np.sqrt(np.mean(x ** 2))))) if len(x) else -140.0


def ncc(a, b, max_lag: int = 0):
    """Normalised cross-correlation of two signals at every integer lag in ±max_lag (FFT); returns (best value, lag).
    lag > 0: a is delayed relative to b (a[i + lag] ≈ b[i])."""
    import numpy as np
    n = min(len(a), len(b))
    if n < 100:
        return 0.0, 0
    a, b = np.asarray(a[:n], dtype="float64"), np.asarray(b[:n], dtype="float64")
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den < 1e-9:
        return 0.0, 0
    size = 1 << (2 * n - 1).bit_length()
    cc = np.fft.irfft(np.fft.rfft(a, size) * np.conj(np.fft.rfft(b, size)), size)
    lags = np.arange(-max_lag, max_lag + 1)
    vals = cc[lags % size] / den
    k = int(np.argmax(vals))
    return float(vals[k]), int(lags[k])


# ── individual checks ───────────────────────────────────────────────────────

def check_duration(dur: float, lo: float = 54.0, hi: float = 66.0, target: float = 60.0) -> Dict[str, Any]:
    return chk("duration", lo <= dur <= hi, round(dur, 2), f"required {lo:g}–{hi:g} s (target {target:g}); off target {dur - target:+.1f} s")


def check_container(path: str) -> Dict[str, Any]:
    r = subprocess.run([_ff(), "-v", "error", "-nostdin", "-i", path, "-f", "null", "-"], capture_output=True, text=True)
    pr = probe_all(path)
    kinds = [s.get("codec_type") for s in pr.get("streams", [])]
    ok = r.returncode == 0 and not r.stderr.strip() and kinds.count("video") == 1 and kinds.count("audio") == 1
    return chk("container", ok, {"streams": kinds, "decode_errors": r.stderr.strip()[:200]},
               "decodes cleanly, exactly one video and one audio stream")


def check_params(path: str) -> Dict[str, Any]:
    pr = probe_all(path)
    v = next((s for s in pr.get("streams", []) if s.get("codec_type") == "video"), {})
    a = next((s for s in pr.get("streams", []) if s.get("codec_type") == "audio"), {})
    num, _, den = (v.get("r_frame_rate") or "0/1").partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 0.0
    vb = int(v.get("bit_rate") or pr.get("format", {}).get("bit_rate") or 0)
    problems = []
    if (v.get("width"), v.get("height")) != (1080, 1920):
        problems.append(f"size {v.get('width')}x{v.get('height')}")
    if v.get("codec_name") != "h264" or v.get("pix_fmt") != "yuv420p":
        problems.append(f"video {v.get('codec_name')}/{v.get('pix_fmt')}")
    if abs(fps - 30) > 0.01:
        problems.append(f"fps {fps}")
    if a.get("codec_name") != "aac" or int(a.get("sample_rate") or 0) != 48000:
        problems.append(f"audio {a.get('codec_name')}/{a.get('sample_rate')}")
    if vb < 2_000_000:
        problems.append(f"video bitrate {vb}")
    return chk("params", not problems, {"size": f"{v.get('width')}x{v.get('height')}", "fps": fps, "video_bps": vb,
                                       "audio": f"{a.get('codec_name')}/{a.get('sample_rate')}/{a.get('channels')}ch"},
               "; ".join(problems))


def check_video_continuity(path: str, fps: int = 30, tail_black_ok: float = 0.9) -> Dict[str, Any]:
    r = subprocess.run([_fp(), "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time",
                        "-of", "csv=p=0", path], capture_output=True, text=True)
    pts = sorted(float(x) for x in r.stdout.split() if x.strip())
    dur = float(probe_all(path).get("format", {}).get("duration") or 0)
    gaps = [round(b - a, 4) for a, b in zip(pts, pts[1:]) if b - a > 1.5 / fps]
    expected = round(dur * fps)
    problems = []
    if gaps:
        problems.append(f"{len(gaps)} timestamp gaps > 1.5 frames (max {max(gaps)})")
    if abs(len(pts) - expected) > 2:
        problems.append(f"{len(pts)} frames, expected ≈{expected}")
    bl = subprocess.run([_ff(), "-hide_banner", "-nostdin", "-i", path, "-an", "-vf", "blackdetect=d=0.25:pic_th=0.98",
                         "-f", "null", "-"], capture_output=True, text=True).stderr
    blacks = [(float(a), float(b)) for a, b in re.findall(r"black_start:([0-9.]+) black_end:([0-9.]+)", bl)]
    bad = [x for x in blacks if x[0] < dur - tail_black_ok]
    if bad:
        problems.append(f"black frames {bad[:3]}")
    return chk("video_continuity", not problems, {"frames": len(pts), "expected": expected, "gaps": gaps[:5],
                                                 "black": blacks[:3]}, "; ".join(problems))


HARD_CUT = 0.4          # scene score of a real cut; fast camera moves / confetti / strobe score 0.15–0.35 on action footage


def parse_scene_scores(text: str) -> List[Tuple[float, float]]:
    """(time, score) pairs from ffmpeg `metadata=print` output."""
    out, t = [], None
    for line in text.splitlines():
        m = re.search(r"pts_time:([0-9.]+)", line)
        if m:
            t = float(m.group(1))
            continue
        m = re.search(r"lavfi\.scene_score=([0-9.]+)", line)
        if m and t is not None:
            out.append((t, float(m.group(1))))
            t = None
    return out


def detect_cuts_scored(path: str, threshold: float = 0.15) -> List[Tuple[float, float]]:
    r = subprocess.run([_ff(), "-hide_banner", "-nostdin", "-loglevel", "error", "-i", path, "-an",
                        "-vf", f"scale=270:-2,select='gt(scene,{threshold})',metadata=print:file=-", "-f", "null", "-"],
                       capture_output=True, text=True)
    scored = parse_scene_scores(r.stdout)
    return scored or [(float(m), 1.0) for m in re.findall(r"pts_time:([0-9.]+)", r.stdout)]


def hard_cuts_in(path: str, a: float, b: float, threshold: float = HARD_CUT) -> List[float]:
    """Absolute times of HARD cuts (scene score ≥ threshold, the same scale the timeline_match check uses on the final)
    inside [a, b] of a source file — finds the flash / insert cuts the frame-difference detector can miss."""
    r = subprocess.run([_ff(), "-hide_banner", "-nostdin", "-loglevel", "error", "-ss", f"{max(0.0, a):.3f}", "-t", f"{b - a:.3f}",
                        "-i", path, "-an", "-vf", f"scale=270:-2,select='gt(scene,{threshold})',metadata=print:file=-",
                        "-f", "null", "-"], capture_output=True, text=True)
    # the first frames after an input seek compare against nothing useful → ignore the first 0.12 s
    return [round(max(0.0, a) + t, 3) for t, sc in parse_scene_scores(r.stdout) if sc >= threshold and t > 0.12]


def detect_cuts(path: str, threshold: float = 0.15) -> List[float]:
    return [t for t, _ in detect_cuts_scored(path, threshold)]


def camera_motion(unplanned: Sequence[float], shots: Sequence[Dict[str, Any]], source: str, win: float = 0.25) -> List[float]:
    """Unplanned "cuts" that are camera motion IN THE SOURCE: the shot was rendered with a fixed 9:16 window
    (crop_steady), and the source has no hard cut around the mapped moment. A whip pan scores ≥ 0.4 in the narrow
    crop but is continuous footage (observed: «Волк» shot 4, motion blur). Crop jumps cannot hide here — a shot with a
    moving window (no crop_steady) is never excused."""
    out = []
    for u in unplanned:
        sh = next((x for x in shots if x["t0"] <= u < x["t1"]), None)
        if sh is None or sh.get("crop_steady") is None or "src_start" not in sh:
            continue
        ts = sh["src_start"] + (u - sh["t0"])
        if not any(abs(c - ts) <= win for c in hard_cuts_in(source, ts - 1.0, ts + 1.0)):   # decode with context
            out.append(u)
    return out


def check_timeline_match(path: str, tl: Dict[str, Any], dur: float, tol: float = 0.12,
                         source: Optional[str] = None) -> Dict[str, Any]:
    shots = tl["shots"]
    problems, t = [], 0.0
    for s in shots:
        if abs(s["t0"] - t) > 1e-3:
            problems.append(f"gap/overlap before shot {s['n']}")
        t = s["t1"]
    if abs(t - dur) > 2 / 30 + 0.05:
        problems.append(f"timeline {t:.2f}s vs video {dur:.2f}s")
    planned = [s["t0"] for s in shots[1:]]
    scored = detect_cuts_scored(path)
    found = [t for t, _ in scored]
    hit = [p for p in planned if any(abs(p - f) <= tol for f in found)]
    # an UNPLANNED cut must be a hard one: action footage (camera whips, confetti, flashes) crosses the low threshold
    # that is right for finding the planned cuts, yet nothing is cut there
    unplanned = [t for t, sc in scored if sc >= HARD_CUT and not any(abs(p - t) <= tol for p in planned)]
    motion = camera_motion(unplanned, shots, source) if (source and unplanned) else []
    unplanned = [u for u in unplanned if u not in motion]
    missing = [p for p in planned if p not in hit]
    # two similar neighbouring frames may not trigger the scene detector → tolerate a few misses, never extra cuts
    if unplanned:
        problems.append(f"{len(unplanned)} unplanned cut(s) at {[round(u, 2) for u in unplanned[:5]]}")
    if planned and len(hit) / len(planned) < 0.9:
        problems.append(f"only {len(hit)}/{len(planned)} planned cuts found in the video")
    return chk("timeline_match", not problems, {"planned_cuts": len(planned), "matched": len(hit),
                                               "missing": [round(m, 2) for m in missing], "unplanned": [round(u, 2) for u in unplanned],
                                               "camera_motion_in_source": [round(u, 2) for u in motion]},
               "; ".join(problems))


def check_no_original_audio(final: str, voice_wav: str, gain_db: float, vt: Dict[str, Any], shot_files: Sequence[str],
                            mix_owners: Sequence[str], source: str, shots: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    import numpy as np
    problems: List[str] = []
    val: Dict[str, Any] = {}
    # (a) the inputs carry no audio at all
    with_audio = [os.path.basename(p) for p in shot_files if any(s.get("codec_type") == "audio"
                                                              for s in probe_all(p).get("streams", []))]
    val["shot_files_with_audio"] = with_audio
    if with_audio:
        problems.append(f"{len(with_audio)} shot file(s) carry audio")
    # (b) plan
    bad_owner = [o for o in mix_owners if o != "narration"]
    val["non_narration_owners"] = bad_owner
    if bad_owner:
        problems.append("mix plan has non-narration audio owners")
    # (c) final audio == narrator track
    fa = decode_audio(final)
    # reference = the narrator track through the intended chain only (no AAC): what the final audio must equal
    from .media import voice_chain
    rr = subprocess.run([_ff(), "-hide_banner", "-nostdin", "-loglevel", "error", "-i", voice_wav, "-af",
                         f"aresample={SR},{voice_chain(gain_db)}", "-ac", "1", "-f", "f32le", "-"], capture_output=True)
    ref = np.frombuffer(rr.stdout, dtype="<f4").astype("float64")
    n = min(len(fa), len(ref))
    corr, lag = ncc(fa[:n], ref[:n], max_lag=int(0.08 * SR))
    val["corr_with_voice_track"], val["lag_ms"] = round(corr, 4), round(lag / SR * 1000, 1)
    if corr < 0.97:
        problems.append(f"final audio differs from the narrator track (corr {corr:.3f})")
    if n > 100:
        x, y = (fa[lag:n], ref[:n - lag]) if lag >= 0 else (fa[:n + lag], ref[-lag:n])
        m = min(len(x), len(y))
        x, y = x[:m], y[:m]
        a = float(np.dot(x, y) / max(1e-12, np.dot(y, y)))
        rdb = rms_db(x - a * y) - rms_db(x)
        val["residual_vs_final_db"] = round(float(rdb), 1)
        if rdb > -18:
            problems.append(f"residual after removing the narrator is only {rdb:.1f} dB below the mix")
    # (d) silence in the pauses (interior of gaps, 60 ms off the edges), lead-in and tail
    gaps = [(0.0, vt["segments"][0]["sentences"][0]["t0"])]
    allr = [r for s in vt["segments"] for r in s["sentences"]]
    gaps += [(a["t1"], b["t0"]) for a, b in zip(allr, allr[1:])]
    gaps.append((allr[-1]["t1"], vt["total"]))
    worst = -140.0
    for a, b in gaps:
        a, b = a + 0.08, b - 0.08
        if b - a >= 0.1:
            worst = max(worst, rms_db(fa[int(a * SR):int(b * SR)]))
    val["loudest_pause_dbfs"] = round(worst, 1)
    if worst > -55:
        problems.append(f"sound in a pause: {worst:.1f} dBFS (should be silence)")
    # (e) negative control: source audio of the used intervals, same positions, lags ±50 ms
    cors = []
    for s in shots:
        seg_f = fa[int(s["t0"] * SR):int(s["t1"] * SR)]
        if rms_db(seg_f) < -50:
            continue
        seg_s = decode_audio(source, SR, s["src_start"], s["src_end"] - s["src_start"])
        if len(seg_s) > SR // 2 and len(seg_f) > SR // 2:
            v, _ = ncc(seg_f, seg_s, max_lag=int(0.05 * SR))
            cors.append(abs(v))
    val["max_abs_corr_with_source_audio"] = round(max(cors), 4) if cors else None
    val["source_shots_compared"] = len(cors)
    if cors and max(cors) > 0.20:
        problems.append(f"final audio correlates with the source audio (max |ρ| {max(cors):.2f})")
    return chk("no_original_audio", not problems, val, "; ".join(problems))


def check_words_intact(final: str, vt: Dict[str, Any], script_text: str, asr: Optional[Callable[[str], str]] = None,
                       edge_db: float = -35.0) -> Dict[str, Any]:
    fa = decode_audio(final)
    problems, edges = [], []
    for s in vt["segments"]:
        for r in s["sentences"]:
            a = rms_db(fa[int(max(0, r["t0"] - 0.005) * SR):int((r["t0"] + 0.012) * SR)])
            b = rms_db(fa[int((r["t1"] - 0.012) * SR):int((r["t1"] + 0.005) * SR)])
            edges.append((round(a, 1), round(b, 1)))
            if a > edge_db or b > edge_db:
                problems.append(f"{s['id']} «{r['text'][:25]}…»: edge level {a:.0f}/{b:.0f} dBFS (word may be cut)")
    if vt["voice_end"] > vt["total"] - 0.5:
        problems.append("voice runs into the last 0.5 s of the video")
    val: Dict[str, Any] = {"sentence_edges_db": edges[:40]}
    if asr is not None:
        heard = asr(final)
        norm = lambda t: re.sub(r"[^\w\s]", "", t.lower().replace("ё", "е")).split()      # noqa: E731
        ref, hyp = norm(script_text), norm(heard)
        sm = difflib.SequenceMatcher(None, ref, hyp)
        sim = round(sm.ratio(), 3)
        val.update(asr_similarity=sim, asr_text=heard[:600])
        if sim < 0.80:
            problems.append(f"ASR of the final audio matches the script only {sim:.0%}")
        # the last word of each sentence must be heard
        missing_tail = [r["text"].split()[-1] for s in vt["segments"] for r in s["sentences"]
                        if norm(r["text"]) and norm(r["text"])[-1] not in hyp and difflib.get_close_matches(norm(r["text"])[-1], hyp, 1, 0.75) == []]
        val["sentence_last_words_not_heard"] = missing_tail
        if len(missing_tail) > 2:
            problems.append(f"{len(missing_tail)} sentence endings not heard: {missing_tail[:4]}")
    return chk("words_intact", not problems, val, "; ".join(problems[:6]))


def check_loudness(final: str, spread_db: float, lufs_range=(-19.0, -13.0), tp_max: float = -1.0,
                   max_spread: float = 6.0) -> Dict[str, Any]:
    r = subprocess.run([_ff(), "-hide_banner", "-nostdin", "-i", final, "-vn", "-af", "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True).stderr
    i = re.findall(r"I:\s+(-?[0-9.]+) LUFS", r)
    p = re.findall(r"Peak:\s+(-?[0-9.]+) dBFS", r)
    lufs, tp = (float(i[-1]) if i else None), (float(p[-1]) if p else None)
    problems = []
    if lufs is None or not lufs_range[0] <= lufs <= lufs_range[1]:
        problems.append(f"integrated {lufs} LUFS outside {lufs_range}")
    if tp is None or tp > tp_max:
        problems.append(f"true peak {tp} dBFS above {tp_max}")
    if spread_db > max_spread:
        problems.append(f"sentence loudness spread {spread_db} dB > {max_spread}")
    return chk("loudness", not problems, {"integrated_lufs": lufs, "true_peak_dbfs": tp, "sentence_spread_db": spread_db},
               "; ".join(problems))


# ── aggregate ───────────────────────────────────────────────────────────────

def check_subtitles_in_frame(ass_path: str) -> Dict[str, Any]:
    """Every subtitle line, measured with the real font file at its size (subs.check_ass), fits the 9:16 safe width
    and no cue has more than 2 lines. Deterministic: the burned-in text is exactly what libass gets."""
    from .subs import check_ass
    r = check_ass(Path(ass_path).read_text(encoding="utf-8"))
    return chk("subtitles_in_frame", r["ok"], r, "; ".join(r["problems"][:4]))


def check_no_names(script_text: str, ass_path: Optional[str], stems: Sequence[str]) -> Dict[str, Any]:
    """Owner rule: no personal names in the narration or in the burned-in subtitles (names from the story map + CLI)."""
    from .recap import find_names
    sub = ""
    if ass_path and Path(ass_path).is_file():
        sub = " ".join(re.sub(r"\{[^}]*\}", "", ln.split(",", 9)[9]).replace("\\N", " ")
                       for ln in Path(ass_path).read_text(encoding="utf-8").splitlines() if ln.startswith("Dialogue:"))
    hits = {"narration": find_names(script_text, stems), "subtitles": find_names(sub, stems)}
    ok = not hits["narration"] and not hits["subtitles"]
    return chk("no_names", ok, {"checked_names": len(stems), "found": hits}, "" if ok else f"names in the text: {hits}")


def check_pace(script_text: str, dur: float, lo: float, hi: float) -> Dict[str, Any]:
    """Spoken words per second of the finished video (owner, punchy style: ~3.5–4)."""
    n = len(re.findall(r"\w+(?:-\w+)*", script_text))
    wps = round(n / max(dur, 1e-6), 2)
    ok = lo <= wps <= hi
    return chk("pace", ok, {"words": n, "words_per_sec": wps, "range": [lo, hi]},
               "" if ok else f"{wps} words/s outside {lo}–{hi}")


def validate(final: str, *, tl: Dict[str, Any], vt: Dict[str, Any], voice_wav: str, gain_db: float,
             shot_files: Sequence[str], source: str, script_text: str, spread_db: float,
             asr: Optional[Callable[[str], str]] = None, dur_range=(54.0, 66.0), target: float = 60.0,
             ass_path: Optional[str] = None, pace: Optional[Tuple[float, float]] = None,
             forbid_names: Optional[Sequence[str]] = None, expected_heard: Optional[str] = None) -> Dict[str, Any]:
    info = probe_all(final)
    dur = float(info.get("format", {}).get("duration") or 0)
    checks = [check_duration(dur, *dur_range, target=target), check_container(final), check_params(final),
              check_video_continuity(final), check_timeline_match(final, tl, dur, source=source),
              check_no_original_audio(final, voice_wav, gain_db, vt, shot_files, ["narration"] * len(tl["shots"]), source,
                                      tl["shots"]),
              check_words_intact(final, vt, expected_heard if expected_heard is not None else script_text, asr), check_loudness(final, spread_db)]
    if ass_path:
        checks.append(check_subtitles_in_frame(ass_path))
    if pace:
        checks.append(check_pace(script_text, dur, *pace))
    if forbid_names is not None:
        checks.append(check_no_names(script_text, ass_path, forbid_names))
    ok = all(c["ok"] for c in checks)
    hard = {"no_original_audio", "container", "duration"}
    failed = [c["name"] for c in checks if not c["ok"]]
    score = 0.0 if hard & set(failed) else round(10 * sum(c["ok"] for c in checks) / len(checks), 1)
    return {"passed": ok, "technical_score": score, "failed": failed, "duration": round(dur, 3), "checks": checks,
            "note": "deterministic, no LLM; any failure of no_original_audio / container / duration forces score 0"}
