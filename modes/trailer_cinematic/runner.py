"""
runner.py — trailer_cinematic v1 pipeline (docs/TRAILER_V1_SPEC.md, first milestone). CLI-only for now:
no daemon / worker / registry wiring yet (shared modules untouched).

    [fast run: first N min]  → ASR (existing backend seam, word timestamps) + scene cuts
    → Story Analyst map (chunks 15–20 min, bounded memory) → refine (global story map)
    → Director (beats) → exact bounds (words / sentences / cuts / pauses) → dedupe → fit to target
    → TTS calibration → voice rooms → Writer → Judge (+ max one rewrite + Judge)
    → TTS per phrase (measured) → place voice → 9:16 render (existing crop) → concat + mix + ASS → final.mp4

Run dir: story_map.json, beats.json, narration.json, judge.json, metrics.json, mix_plan.json,
fragments.json, narration.ass, final.mp4, _work/.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import budget, media, story, timeline as tl

MODE = "trailer_cinematic"
VERSION = "trailer-cinematic-v1.0"
ROOT = Path(__file__).resolve().parents[2]
LLM_ENV_FILE = Path.home() / ".config" / "sonya" / "openrouter.env"


def _load_llm_env(path: Path = LLM_ENV_FILE) -> None:
    """OPENROUTER_API_KEY from the local env file into os.environ (never printed)."""
    if os.environ.get("OPENROUTER_API_KEY") or not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        k, _, v = line.strip().partition("=")
        if k == "OPENROUTER_API_KEY" and v:
            os.environ[k] = v.strip().strip('"').strip("'")


def _desktop_media_env() -> Dict[str, Any]:
    """Same as the Desktop worker does for Cut: an ffmpeg build with libass (burned ASS) and the
    hardware encoder at 6 Mb/s. Only fills env vars that are not set already."""
    sys.path.insert(0, str(ROOT / "desktop"))
    out: Dict[str, Any] = {}
    if not os.environ.get("SONYA_FFMPEG"):
        from sonya_desktop.media.ffmpeg import find_ffmpeg_with_filter
        full = find_ffmpeg_with_filter("ass")
        if full:
            os.environ["SONYA_FFMPEG"] = full
    out["ffmpeg"] = os.environ.get("SONYA_FFMPEG") or "ffmpeg"
    if not os.environ.get("SONYA_RENDER_ARGS"):
        try:
            from sonya_desktop.compute import detect_hardware
            from sonya_desktop.media.encoders import cached_probe, select_encoder
            hw = detect_hardware(probe_torch=False)
            os.environ["SONYA_RENDER_ARGS"] = json.dumps(
                select_encoder(cached_probe(), hw.os, hw.gpu_vendors).args("quality", bitrate="6M"))
        except Exception:                                   # noqa: BLE001 — libx264 default
            pass
    out["render_args"] = os.environ.get("SONYA_RENDER_ARGS")
    return out


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _dump(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)


def run_asr(src: str, cache_dir: Path, sha: str, language: str = "ru") -> Dict[str, Any]:
    """Existing desktop ASR seam (GigaAM for Russian → Whisper), cached by source sha."""
    cache = cache_dir / f"asr-{sha[:16]}-{language}.json"
    if cache.is_file():
        d = json.loads(cache.read_text(encoding="utf-8"))
        d.setdefault("meta", {})["cached"] = True
        return d
    sys.path.insert(0, str(ROOT / "desktop"))
    from sonya_desktop.asr.backends import select_backend
    from sonya_desktop.compute import detect_hardware
    hw = detect_hardware(probe_torch=False)
    backend = select_backend(hw, "quality", language=language)
    wav = media.extract_wav16k(src, str(cache_dir / f"asr-{sha[:16]}.wav"))
    t0 = time.time()
    d = backend.transcribe(wav, "quality", language=language, word_timestamps=True)
    d["meta"] = {"backend": d.get("backend"), "model": d.get("model"), "language": d.get("language"),
                 "elapsed_sec": round(time.time() - t0, 2), "words": len(d.get("words") or []), "cached": False}
    os.remove(wav)
    _dump(cache, d)
    return d


def forecast_run(parts_chars: List[int], model: str) -> Dict[str, Any]:
    """Whole-run worst case before the first request: every map chunk + refine + director + writer ×2 +
    judge ×2 (one rewrite), each with its output cap, plus one schema repair per step."""
    pin, pout = budget.price(model)
    tok = lambda c: c / budget.CHARS_PER_TOKEN + 50
    calls = [(tok(c + 3000), 8000) for c in parts_chars]
    events_chars = 260 * 12 * len(parts_chars)
    calls += [(tok(events_chars + 2000), 6000), (tok(events_chars * 2 + 6000), 6000),
              (tok(events_chars + 3000), 4000), (tok(events_chars + 5000), 4000),
              (tok(events_chars + 4000), 4000), (tok(events_chars + 6000), 4000)]
    usd = sum(i * pin + o * pout for i, o in calls)
    return {"calls": len(calls), "usd": round(usd, 4), "usd_with_repairs": round(usd * 1.5, 4)}


def run(input_path: str, duration: float = 60.0, *, run_dir: Optional[str] = None, source_minutes: float = 15.0,
        budget_usd: float = 1.0, tts_voice: str = "aidar", layout: str = "smart", language: str = "ru",
        chat=None) -> Dict[str, Any]:
    t_all = time.time()
    rid = time.strftime("%Y%m%d-%H%M%S")
    out = Path(run_dir or (ROOT.parent / "runtime" / "trailer" / rid)).resolve()
    out.mkdir(parents=True, exist_ok=True)
    work = out / "_work"
    work.mkdir(exist_ok=True)
    cache_dir = ROOT.parent / "runtime" / "trailer" / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    T: Dict[str, float] = {}
    M: Dict[str, Any] = {"mode": MODE, "version": VERSION, "timeline_version": tl.VERSION, "run_dir": str(out),
                         "source_path": str(Path(input_path).resolve()), "target_duration": duration,
                         "budget_usd": budget_usd, "warnings": [], "timings_sec": T,
                         "limitations": ["Story Analyst / Director / Writer / Judge saw ONLY the ASR transcript "
                                         "(with timestamps) and scene-cut times — no video frames."]}

    def tick(k, t0):
        T[k] = round(time.time() - t0, 2)

    def save():
        T["total"] = round(time.time() - t_all, 2)
        _dump(out / "metrics.json", M)

    # ── source (fast run) ───────────────────────────────────────────────────
    t0 = time.time()
    M["media_env"] = _desktop_media_env()
    full = media.probe(input_path)
    M["source_full_duration"] = round(full["duration"], 2)
    src = input_path
    if source_minutes and full["duration"] > source_minutes * 60 + 5:
        src = media.cut_head(input_path, source_minutes,
                             str(cache_dir / f"{Path(input_path).stem}_{int(source_minutes)}m.mp4"))
        M["fast_run"] = {"source_minutes": source_minutes, "cut_path": src}
    info = media.probe(src)
    dur = info["duration"]
    M["source_duration"] = round(dur, 2)
    sha = _sha256(src)
    M["source_sha256"] = sha
    tick("source", t0)

    # ── ASR + scenes ────────────────────────────────────────────────────────
    t0 = time.time()
    asr = run_asr(src, cache_dir, sha, language)
    M["asr"] = {k: asr["meta"].get(k) for k in ("backend", "model", "language", "words", "cached")}
    M["asr_backend"] = asr["meta"].get("backend")
    M["asr_seconds"] = asr["meta"].get("elapsed_sec")
    tick("asr", t0)
    words, sents = tl.sentences(asr.get("words") or [])
    sid = {s["id"]: s for s in sents}
    t0 = time.time()
    cuts = media.scene_cuts(src, str(cache_dir / f"scenes-{sha[:16]}.json"))
    shot_list = tl.shots(cuts, dur)
    tick("scene_detection", t0)
    M["scene_detection_seconds"] = T["scene_detection"]
    M["scenes"] = {"cuts": len(cuts), "shots": len(shot_list)}
    chunks = tl.chunks(dur, sents, shot_list)
    M["number_of_chunks"] = len(chunks)
    M["chunks"] = [{"start": round(c["start"], 1), "end": round(c["end"], 1), "sentences": len(c["sentences"])}
                   for c in chunks]

    # ── LLM budget ──────────────────────────────────────────────────────────
    _load_llm_env()
    from scripts.shared.llm_provider import chat_json, configured_providers
    provs = configured_providers()
    if chat is None and not provs:
        raise RuntimeError("no LLM provider configured (OPENROUTER_API_KEY)")
    model = provs[0].model if provs else "fake"
    M["llm_provider"] = f"{provs[0].name}/{model}" if provs else "fake"
    llm = budget.LLM(chat or chat_json, budget_usd, model)
    M["llm_calls"] = llm.calls
    parts_chars = [len(story.transcript_lines(c["sentences"])) for c in chunks]
    fc = forecast_run(parts_chars, model)
    M["forecast"] = fc
    if fc["usd_with_repairs"] > budget_usd:
        M["stopped"] = f"forecast ${fc['usd_with_repairs']:.3f} > budget ${budget_usd:.2f}"
        save()
        raise budget.BudgetExceeded(M["stopped"])

    try:
        # Story Analyst: map (bounded memory) → refine
        t0 = time.time()
        memory: Dict[str, Any] = {}
        parts = []
        for c in chunks:
            part = llm.ask(f"analyst_map_{c['index'] + 1}", story.map_messages(c, memory, language),
                           story.MAP_VERSION, lambda d, c=c: story.validate_map(d, c), max_out=8000)
            parts.append(part)
            memory = story.update_memory(memory, part)
        story_map = llm.ask("analyst_refine", story.refine_messages(parts, dur, language), story.REFINE_VERSION,
                            lambda d: story.validate_story_map(d, parts), max_out=6000)
        tick("story_analyst", t0)
        _dump(out / "story_map.json", {"version": story.REFINE_VERSION, "chunks": len(parts),
                                       "story_map": story_map, "chunk_parts": parts})

        # Director
        t0 = time.time()
        cands = story.candidates(story_map, sents, shot_list)
        beats = llm.ask("director", story.director_messages(story_map, cands, duration), story.DIRECTOR_VERSION,
                        lambda d: story.validate_beats(d, story_map, dur), temperature=0.3, max_out=6000)
        tick("director", t0)

        # exact bounds
        frags = [tl.refine(tl.beat_to_fragment(b, sents), words, sid, cuts, dur) for b in beats]
        for f in frags:
            if f.get("downgraded"):
                M["warnings"].append(f"{f['id']}: {f['downgraded']}")
        frags, dropped = tl.dedupe(frags)
        frags, dropped2 = tl.fit_to_target(frags, duration)
        M["dropped_beats"] = dropped + dropped2
        M["warnings"] += tl.voice_rooms(frags, words)
        for f in frags:
            if f.get("lines"):
                f["original_line"] = " ".join(sid[k]["text"] for k in f["lines"])

        # TTS calibration → Writer budgets
        from scripts.shared.tts import SileroProvider
        tts = SileroProvider(tts_voice)
        if not tts.available():
            raise RuntimeError("Silero model is not in the torch-hub cache")
        t0 = time.time()
        rate = tts.calibrate(str(cache_dir))
        T["tts_calibration"] = round(time.time() - t0, 2)
        for f in frags:
            if f.get("voice_room_sec"):
                f["max_chars"] = tts.max_chars(f["voice_room_sec"])
        M["tts"] = dict(tts.info(), rate=rate)
        _dump(out / "beats.json", {"version": story.DIRECTOR_VERSION, "beats_raw": beats, "fragments": frags,
                                   "dropped": M["dropped_beats"]})

        # Writer → Judge (+ max one rewrite)
        t0 = time.time()
        wb = [dict(f, beat_id=f["id"], audio_owner=f["audio"]) for f in frags]
        phrases = llm.ask("writer", story.writer_messages(story_map, wb, duration), story.WRITER_VERSION,
                          lambda d: story.validate_phrases(d, wb, story_map), temperature=0.8, effort="medium",
                          max_out=4000)
        tick("writer", t0)
        t0 = time.time()
        est = lambda ph: {p["beat_id"]: tts.estimate_sec(p["text"]) for p in ph}
        verdict = llm.ask("judge", story.judge_messages(story_map, wb, phrases, est(phrases)), story.JUDGE_VERSION,
                          story.validate_judge, temperature=0.0, max_out=4000)
        tick("judge", t0)
        attempts = [{"phrases": phrases, "verdict": verdict, "rewrite_reasons": story.needs_rewrite(verdict)}]
        if attempts[0]["rewrite_reasons"]:
            t0 = time.time()
            phrases2 = llm.ask("writer_rewrite", story.writer_messages(story_map, wb, duration, previous=phrases,
                                                                       rewrite_instructions=verdict["rewrite_instructions"]
                                                                       or "; ".join(verdict["factual_errors"])),
                               story.WRITER_VERSION + "+rewrite", lambda d: story.validate_phrases(d, wb, story_map),
                               temperature=0.7, effort="medium", max_out=4000)
            T["writer"] += round(time.time() - t0, 2)
            t0 = time.time()
            verdict2 = llm.ask("judge_rewrite", story.judge_messages(story_map, wb, phrases2, est(phrases2)),
                               story.JUDGE_VERSION, story.validate_judge, temperature=0.0, max_out=4000)
            T["judge"] += round(time.time() - t0, 2)
            attempts.append({"phrases": phrases2, "verdict": verdict2, "rewrite_reasons": story.needs_rewrite(verdict2)})
            phrases, verdict = phrases2, verdict2          # spec: the rewrite is the answer (max 1)
        _dump(out / "judge.json", {"version": story.JUDGE_VERSION, "final": verdict, "attempts": attempts,
                                   "rewrites": len(attempts) - 1})
    finally:
        M["llm"] = llm.summary()
        M["cost_usd"] = llm.spent
        save()

    # ── TTS per phrase (measured) → voice placement ────────────────────────
    t0 = time.time()
    lines = {p["beat_id"]: p["text"] for p in phrases}
    wavs, secs = {}, {}
    for bid, text in lines.items():
        wavs[bid], secs[bid] = tts.synthesize(text, str(work / f"voice_{bid}.wav"))
    for f in frags:
        if f["audio"] == "narration" and f["id"] not in lines:     # a silent narration beat: keep it short
            f["render_end"] = round(min(f["render_end"], f["render_start"] + 2.5), 3)
    M["warnings"] += tl.place_narration(frags, secs, words, cuts, dur)
    total_len = tl.layout(frags)
    tick("tts", t0)
    narration = [dict(p, measured_sec=round(secs[p["beat_id"]], 2), estimated_sec=round(tts.estimate_sec(p["text"]), 2),
                      tts_license=tts.license) for p in phrases]
    _dump(out / "narration.json", {"version": story.WRITER_VERSION, "tts": M["tts"], "phrases": narration})

    # ── render ──────────────────────────────────────────────────────────────
    t0 = time.time()
    files = media.render_fragments(src, frags, work, width=info["width"], height=info["height"], layout=layout,
                                   cache_dir=str(cache_dir / "clips"), source_sha256=sha)
    tick("render_fragments", t0)
    t0 = time.time()
    plan = tl.mix_plan(frags)
    _dump(out / "mix_plan.json", plan)
    if plan["voice_share"] > 0.65:
        M["warnings"].append(f"voice covers {plan['voice_share']:.0%} of the trailer (too continuous)")
    voice = str(work / "voice.wav")
    media.voice_track([f for f in frags if not f.get("voice_dropped")], wavs, total_len, voice)
    duck = tl.duck_expr(tl.voice_windows(frags))
    from scripts.shared.subtitle_styles import to_ass
    ass = out / "narration.ass"
    ass.write_text(to_ass(tl.subtitle_words(frags, lines), "burgundy", total_len, font="dela"), encoding="utf-8")
    final = out / "final.mp4"
    media.assemble(files, frags, voice, duck, str(ass), str(final))
    tick("render_assemble", t0)
    res = media.probe(str(final))
    M["final_duration"] = round(res["duration"], 2)
    M["final_output_path"] = str(final)
    M["final_size_mb"] = round(final.stat().st_size / 1e6, 1)
    M["audio_check"] = media.loudness(str(final))
    M["selected_timestamps"] = [{"beat_id": f["id"], "audio_owner": f["audio"], "story_function": f["story_function"],
                                 "event_ids": f["event_ids"], "source_start": f["render_start"],
                                 "source_end": f["render_end"], "trailer_t0": f["t0"], "trailer_t1": f["t1"],
                                 "boundary_reasons": f.get("boundary_reasons"), "crop": f.get("crop_status")}
                                for f in frags]
    M["narration"] = [{"beat_id": p["beat_id"], "text": p["text"], "measured_sec": p["measured_sec"]} for p in narration]
    _dump(out / "fragments.json", frags)
    save()
    return M


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="trailer_cinematic", description="SONYA trailer_cinematic v1 (dev CLI)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--duration", type=float, default=60.0, help="target trailer length, s")
    ap.add_argument("--source-minutes", type=float, default=15.0, help="fast run: first N minutes (0 = all)")
    ap.add_argument("--budget", type=float, default=1.0, help="OpenRouter budget per run, USD")
    ap.add_argument("--run-dir")
    ap.add_argument("--voice", default="aidar")
    ap.add_argument("--layout", default="smart", choices=("smart", "fit", "blur"))
    a = ap.parse_args(argv)
    m = run(a.input, a.duration, run_dir=a.run_dir, source_minutes=a.source_minutes, budget_usd=a.budget,
            tts_voice=a.voice, layout=a.layout)
    print(json.dumps({k: m.get(k) for k in ("final_output_path", "final_duration", "cost_usd", "timings_sec",
                                            "warnings")}, ensure_ascii=False, indent=1))
    return 0
