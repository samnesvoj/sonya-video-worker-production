"""
recap_runner.py — trailer_cinematic v2: AI Cinematic Story Recap (docs/TRAILER_V2_RECAP_SPEC.md). CLI only.

Source → ASR + scene detection → Story Analyst → Visual Analyst (frames → vision LLM) → Writer (one script,
sized by the MEASURED voice) → TTS → Visual Director → Timeline Composer → render (silent shots, one framing
style) + narration-only audio → Technical Validator (deterministic) → Creative Judge (+ at most ONE rewrite).

Reuses v1 infrastructure: ASR / scene cache, budget.LLM (forecast + hard stop; + response cache), VideoToolbox
encoder env, the Cut crop engine, subtitle utilities. LLM and vision calls go ONLY through
scripts/shared/llm_provider.chat_json.

Run dir: story_map.json, visual_analysis.json, script.json, narration.json, director_plan.json, timeline.json,
mix_plan.json, validation.json, judge.json, metrics.json, narration.ass, final.mp4, _work/.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import budget, censor, compose, media, recap, story, subs, timeline as tl, validator, visual
from .runner import ROOT, _desktop_media_env, _dump, _load_llm_env, _sha256, run_asr

MODE = "trailer_cinematic"
VERSION = "trailer-recap-v2.1"
PACE_PUNCHY = (3.3, 4.4)                   # words per second of the finished video (owner: ~3.5–4)
TARGET_LO, TARGET_HI = 54.0, 66.0          # spec: acceptable final length
FIT_LO, FIT_HI = 57.5, 62.5                # what we aim for before accepting (closer to 60)
MAX_REQUEST_BYTES = 8_000_000              # Timeweb: refuse a larger body. Live: 5.0 MB answered in 5.6 s, 19.7 MB in 156 s (no hard error found)


class LLMUnavailable(RuntimeError):
    """The chosen LLM provider is switched off / not configured. v2 cannot run without an LLM — say so, do not fall back."""


def select_provider(llm_provider: str, *, timeweb_off: bool = False, env=None, keychain: bool = True):
    """('timeweb', cfg) → ONE explicit provider, no silent fallback; ('auto', None) → the previous behaviour
    (llm_provider.configured_providers(), unchanged). Raises LLMUnavailable with a plain-language reason."""
    from scripts.shared import llm_provider as lp
    if llm_provider != "timeweb":
        return "auto", None
    if timeweb_off or not lp.timeweb_enabled(env):
        raise LLMUnavailable("Провайдер Timeweb выключен (SONYA_LLM_TIMEWEB_ENABLED=false или --no-timeweb). "
                             "Трейлер v2 без LLM не работает; запустите без --llm-provider timeweb (прежний путь) "
                             "или включите Timeweb.")
    cfg = lp.timeweb_provider(env, keychain=keychain)
    if cfg is None:
        raise LLMUnavailable("Ключ Timeweb не найден (Keychain group.sonya.timeweb-ai / api_key или TIMEWEB_GEMINI_API_KEY).")
    return "timeweb", cfg


def forecast(parts_chars: List[int], model: str, n_vision_batches: int, n_cards: int, vision: str = "frames") -> Dict[str, Any]:
    pin, pout = budget.price(model)
    tok = lambda c: c / budget.CHARS_PER_TOKEN + 50                # noqa: E731
    img = budget.IMAGE_TOKENS
    calls = [(tok(c + 3000), 8000) for c in parts_chars]
    ev = 260 * 12 * len(parts_chars)
    calls += [(tok(ev * 2 + 5000), 6000)]                                            # refine
    per_batch = 18 * img if vision != "clips" else 6 * budget.VIDEO_TOKENS            # 6 windows × (3 frames | 1 clip)
    calls += [(tok(3500) + per_batch, 4000)] * n_vision_batches                     # vision
    calls += [(tok(ev + 6000), 4000)] * 4                                            # writer + fit ×2 + rewrite
    calls += [(tok(n_cards * 520 + 6000), 4000)] * 3                                 # director ×2 + repairs
    calls += [(tok(ev + 8000) + 3 * 3 * img, 4000)] * 2                              # judge ×2 (sheets ≈ 3 images each)
    usd = sum(i * pin + budget.forecast_out(o) * pout for i, o in calls)           # calibrated: typical, not cap, output
    return {"calls": len(calls), "usd": round(usd, 4), "usd_with_repairs": round(usd * 1.4, 4)}


def asr_text_fn(language: str, cache_dir: Path):
    """ASR of the FINAL audio (validator: cut / garbled words). Same desktop backend as the pipeline."""
    def run(path: str) -> str:
        import sys
        sys.path.insert(0, str(ROOT / "desktop"))
        from sonya_desktop.asr.backends import select_backend
        from sonya_desktop.compute import detect_hardware
        backend = select_backend(detect_hardware(probe_torch=False), "quality", language=language)
        wav = media.extract_wav16k(path, str(cache_dir / "final-audio-asr.wav"))
        d = backend.transcribe(wav, "quality", language=language, word_timestamps=True)
        os.remove(wav)
        return " ".join(str(w.get("word") or w.get("text") or "") for w in d.get("words") or [])
    return run


def window_text(words, c, limit: int = 170) -> str:
    t = " ".join(w.text for w in words if w.end > c["start"] and w.start < c["end"])
    return t[:limit]


def ask_length_bound(llm, step: str, msgs, version: str, sm, lo_c: int, hi_c: int, *, strict: float, loose: float,
                     temperature: float, style: str = "calm") -> Dict[str, Any]:
    """A script with a length window. A small model may miss it twice (observed on a 3-hour film: 1080 chars for
    727–836, then 644 for 754–817); that must not kill the run, because the measured-TTS fit loop corrects the length.
    So: the strict window first, then ONE more ask with a loose length check (everything else stays strict)."""
    loose_cp = llm._cache_path(step + "_loose", msgs, version + "+loose", temperature, 4000) if hasattr(llm, "_cache_path") else None
    if loose_cp is not None and loose_cp.is_file():
        # an earlier run already fell through to the loose ask for this very request: the strict one failed then and
        # its failures are not cached — asking it again only pays for the same miss
        return llm.ask(step + "_loose", msgs, version + "+loose", lambda d: recap.validate_script(d, sm, lo_c, hi_c, loose, style=style),
                       temperature=temperature, effort="medium", max_out=4000)
    try:
        return llm.ask(step, msgs, version, lambda d: recap.validate_script(d, sm, lo_c, hi_c, strict, style=style),
                       temperature=temperature, effort="medium", max_out=4000)
    except budget.LLMStop:
        raise
    except RuntimeError:
        return llm.ask(step + "_loose", msgs, version + "+loose", lambda d: recap.validate_script(d, sm, lo_c, hi_c, loose, style=style),
                       temperature=temperature, effort="medium", max_out=4000)


def ask_writer(llm, sm, duration: float, lo_c: int, hi_c: int, rate, *, loose: float = 0.5, style: str = "calm",
               evidence=None) -> Dict[str, Any]:
    step = "writer" if style == "calm" else f"writer_{style}"
    return ask_length_bound(llm, step, recap.writer_messages(sm, duration, lo_c, hi_c, rate, style=style, evidence=evidence),
                            recap.versions(style)["writer"], sm, lo_c, hi_c, strict=0.12, loose=loose, temperature=0.8,
                            style=style)


def span_text(words, a: float, b: float, limit: int = 240) -> str:
    """Real transcript of [a, b] (evidence for the punchy analyst / writer: concrete details, not summaries)."""
    t = " ".join(w.text for w in words if w.end > a and w.start < b)
    return t[:limit]


def speed_up(src_wav: str, out_wav: str, tempo: float) -> str:
    """Faster narration WITHOUT a pitch shift: ffmpeg atempo (time-stretch, timbre kept), then the edge silence the
    stretch leaves is trimmed. Owner (punchy style) lifted the spec's "never speed the voice up"; words stay whole —
    the validator re-checks every sentence edge and ASR of the final."""
    import subprocess
    from scripts.shared.tts import trim_silence
    if os.path.isfile(out_wav) and os.path.getsize(out_wav) > 44:
        return out_wav
    tmp = out_wav + ".tmp.wav"
    r = subprocess.run([media.ffmpeg_bin(), "-y", "-hide_banner", "-nostdin", "-loglevel", "error", "-i", src_wav,
                        "-af", f"atempo={tempo:.3f}", "-c:a", "pcm_s16le", tmp], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"atempo failed: {r.stderr[-300:]}")
    trim_silence(tmp)
    soften_edges(tmp)
    os.replace(tmp, out_wav)
    return out_wav


def soften_edges(path: str, fade: float = 0.015, pad: float = 0.04) -> None:
    """The stretch + trim leaves the breath tail at ≈ −45 dBFS right at the file edge; after the loudness gain that
    edge reads −31 dBFS and looks like a cut word. A 15 ms fade over that noise-floor tail + 40 ms of silence on both
    sides makes the edge honestly silent (speech itself ends ≥ 30 ms earlier: trim keeps 30 ms above −45 dBFS)."""
    import wave
    import numpy as np
    with wave.open(path, "rb") as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        y = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32)
    k = min(len(y) // 4, int(fade * sr))
    if k > 0:
        ramp = np.linspace(0.0, 1.0, k, dtype=np.float32)
        y[:k] *= ramp
        y[-k:] *= ramp[::-1]
    z = np.zeros(int(pad * sr), np.float32)
    y = np.concatenate([z, y, z])
    with wave.open(path, "wb") as w:
        w.setnchannels(ch), w.setsampwidth(sw), w.setframerate(sr)
        w.writeframes(np.clip(y, -32768, 32767).astype("<i2").tobytes())


def forecast_creative(model: str, n_events: int, n_cards: int, style: str, n_story: int = 30,
                      refine_cached: bool = False, writer_cached: bool = False, out_tok: int = 2500,
                      vision_batches: int = 5, stage: str = "all", director_cached: bool = False,
                      rewrite: bool = True) -> Dict[str, Any]:
    """stage "head" = refine + supplementary vision + writer chain; "tail" = director + judge + ONE rewrite
    (writer + director + judge); "all" = both."""
    """Worst case of the steps after the (cached) analysis: punchy refine, writer + 3 fits (+ loose retries),
    director ×2, judge ×2, rewrite; every call at its full output cap."""
    pin, pout = budget.price(model)
    tok = lambda c: c / budget.CHARS_PER_TOKEN + 50                  # noqa: E731
    # out_tok: the most a creative answer took on this film was ≈ 2.1k tokens (writer); the per-call guard still uses
    # each call's full max_out, so a longer answer can never push the spend over the limit
    calls = [(tok(520 * n_events * 1.6 + 6000), out_tok)] if style == "punchy" and not refine_cached else []  # refine
    ev = 700 * n_story                                                                # writer / judge: map + evidence
    if style != "calm":
        calls += [(tok(3500) + 6 * budget.VIDEO_TOKENS, out_tok)] * vision_batches    # supplementary vision (uncached)
    if stage == "tail":
        calls = []
    # answer sizes measured on «Волк с Уолл-стрит» (2026-11-04): writer ≈ 1.5–2.1k tokens, director ≈ 0.7–1k,
    # judge ≈ 0.6–1k → forecast with ≈ 2× margin per kind (out_tok for the writer / analyst)
    out_dir, out_judge = int(out_tok * 0.8), int(out_tok * 0.6)
    if stage in ("all", "head"):
        calls += [(tok(ev + 9000), out_tok)] * (0 if writer_cached else 4)              # writer, fit ×2, one retry
    if stage in ("all", "tail"):
        calls += [(tok(ev + 9000), out_tok)] * int(rewrite)                              # rewrite (writer)
        calls += [(tok(n_cards * 560 + 7000), out_dir)] * ((0 if director_cached else 2) + int(rewrite))  # director a (+repair) / b
        calls += [(tok(ev + 9000) + 3 * 3 * budget.IMAGE_TOKENS, out_judge)] * (1 + int(rewrite))         # judge (+ rewrite)
    usd = sum(i * pin + budget.forecast_out(o) * pout for i, o in calls)          # calibrated: typical, not cap, output
    # the repair retry is one of the explicit calls above; every call is still guarded on its own before it is sent
    return {"calls": len(calls), "usd": round(usd, 4), "usd_with_repairs": round(usd, 4)}


def run(input_path: str, duration: float = 60.0, *, run_dir: Optional[str] = None, source_minutes: float = 15.0,
        budget_usd: Optional[float] = None, tts_voice: str = "aidar", tts_provider: str = "silero", language: str = "ru",
        subtitles: bool = True, max_candidates: int = 56, cold: bool = False, chat=None, tts=None,
        llm_provider: str = "auto", timeweb_off: bool = False, vision: str = "frames",
        day_limit_usd: Optional[float] = None, style: str = recap.DEFAULT_STYLE,
        video_limit_rub: Optional[float] = None, video_seed_usd: float = 0.0, max_rewrites: int = 1,
        debug_budget: bool = False, forbid_names: Sequence[str] = (),
        script_file: Optional[str] = None, censor_profanity: Optional[bool] = None,
        director_plan_file: Optional[str] = None, skip_judge: bool = False,
        no_extra_vision: bool = False, allow_words: Sequence[str] = ()) -> Dict[str, Any]:
    t_all = time.time()
    if video_limit_rub is None:       # production default 20 ₽/video; --debug-budget lifts the hard stop to the 100 ₽ ceiling
        video_limit_rub = budget.DEBUG_VIDEO_LIMIT_RUB if debug_budget else budget.PROD_VIDEO_LIMIT_RUB
    scfg = recap.style_cfg(style)
    ver = recap.versions(style)
    if censor_profanity is None:        # BEEP step: on by default for the energetic style
        censor_profanity = style == "punchy"
    kind, tw = select_provider(llm_provider, timeweb_off=timeweb_off) if chat is None else ("auto", None)
    if budget_usd is None:
        budget_usd = float(os.environ.get("SONYA_TRAILER_LLM_RUN_USD") or budget.DEFAULT_RUN_USD) if tw else 1.0
    if day_limit_usd is None:
        day_limit_usd = float(os.environ.get("SONYA_TRAILER_LLM_DAY_USD") or budget.DEFAULT_DAY_USD)
    rid = time.strftime("%H%M%S")
    out = Path(run_dir or (ROOT.parent / "runtime" / "trailer" / f"v2-{rid}")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    work = out / "_work"
    work.mkdir(exist_ok=True)
    cache_dir = ROOT.parent / "runtime" / "trailer" / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # cold run: LLM answers, frames, clip analyses and TTS sentences live in this run's own folder (nothing reused);
    # only ASR and scene detection of the source stay in the shared cache (their cost is reported from the cache meta)
    ccache = work / "cache" if cold else cache_dir
    T: Dict[str, float] = {}
    M: Dict[str, Any] = {"mode": MODE, "version": VERSION, "run_dir": str(out), "source_path": str(Path(input_path).resolve()),
                         "target_duration": duration, "accepted_range": [TARGET_LO, TARGET_HI], "budget_usd": budget_usd,
                         "warnings": [], "timings_sec": T, "cold": cold, "style": style, "style_cfg": scfg,
                         "limitations": []}

    def tick(k: str, t0: float) -> None:
        T[k] = round(T.get(k, 0.0) + time.time() - t0, 2)

    def save() -> None:
        T["total"] = round(time.time() - t_all, 2)
        _dump(out / "metrics.json", M)

    def say(msg: str) -> None:
        print(f"[recap {time.time() - t_all:6.1f}s] {msg}", flush=True)

    all_events: List[Dict[str, Any]] = []

    def creative_gate(args: Dict[str, Any], vision_batches: int = 0, stage: str = "head") -> None:
        """Owner: a forecast BEFORE the calls, above the per-video limit → STOP with the forecast. Cached steps cost 0.
        Two gates: "head" before the analysis / writer calls, "tail" right before the director (montage, judge and
        the one allowed rewrite). Every single call is additionally guarded with its full output cap."""
        fc2 = forecast_creative(model, len(all_events), max_candidates + 30, style, vision_batches=vision_batches,
                                stage=stage, **args)
        fc2.update(args, vision_batches=vision_batches, stage=stage)
        M.setdefault("forecast_gates", []).append(fc2)
        heads = [x for x in (ledger if isinstance(ledger, list) else [ledger]) if isinstance(x, budget.VideoLedger)]
        if not heads:
            return
        spent_v = heads[0].spent()
        say(f"forecast ({stage}) {budget.rub(fc2['usd_with_repairs']):.1f} ₽; this video so far "
            f"{budget.rub(spent_v):.1f} ₽ of {heads[0].limit_rub:.0f} ₽")
        if spent_v + fc2["usd_with_repairs"] > heads[0].limit:
            M["stopped"] = (f"forecast {budget.rub(fc2['usd_with_repairs']):.1f} ₽ + spent {budget.rub(spent_v):.1f} ₽ "
                            f"> limit {heads[0].limit_rub:.0f} ₽")
            raise budget.BudgetExceeded(M["stopped"])

    # ── source ──────────────────────────────────────────────────────────────
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

    # ── ASR + scene detection ───────────────────────────────────────────────
    t0 = time.time()
    asr = run_asr(src, cache_dir, sha, language)
    M["asr"] = {k: asr["meta"].get(k) for k in ("backend", "model", "language", "words", "cached")}
    M["asr_seconds"] = asr["meta"].get("elapsed_sec")
    tick("asr", t0)
    words, sents = tl.sentences(asr.get("words") or [])
    t0 = time.time()
    cuts = media.scene_cuts(src, str(cache_dir / f"scenes-{sha[:16]}.json"))
    shot_list = tl.shots(cuts, dur)
    tick("scene_detection", t0)
    M["scenes"] = {"cuts": len(cuts), "shots": len(shot_list)}
    chunks = tl.chunks(dur, sents, shot_list)
    M["number_of_chunks"] = len(chunks)

    # ── LLM: provider, budget, forecast ─────────────────────────────────────
    _load_llm_env()
    from scripts.shared.llm_provider import certifi_transport, chat_json, configured_providers
    ledger = chat_kwargs = None
    if tw is not None:                                   # Timeweb only: hard run + day limits, retries, no other provider
        provs = [tw]
        chat_kwargs = dict(providers=provs, retries=2, max_request_bytes=MAX_REQUEST_BYTES, transport=certifi_transport())
        ledger = budget.DailyLedger(str(ROOT.parent / "runtime" / "trailer" / "_cache" / "llm_spend_timeweb.json"), day_limit_usd)
        M["day_limit_usd"], M["day_spent_before_usd"] = day_limit_usd, round(ledger.spent(), 5)
        if video_limit_rub:                              # owner: ≤ N ₽ for THIS video over all its runs (seeded with earlier spend)
            vled = budget.VideoLedger(str(cache_dir / f"spend_video-{sha[:16]}.json"), video_limit_rub, seed_usd=video_seed_usd)
            left = max(0.0, vled.limit - vled.spent())
            M["video_budget"] = {"limit_rub": video_limit_rub, "spent_before_rub": budget.rub(vled.spent()),
                                 "left_rub": budget.rub(left), "rub_per_usd": budget.RUB_PER_USD, "ledger": str(vled.path)}
            budget_usd = min(budget_usd, left) if budget_usd else left
            M["budget_usd"] = round(budget_usd, 4)
            ledger = [ledger, vled]
    else:
        provs = configured_providers()
    if chat is None and not provs:
        raise LLMUnavailable("no LLM provider configured (OPENROUTER_API_KEY / Timeweb)")
    model = provs[0].model if provs else "fake"
    M["llm_provider"] = f"{provs[0].name}/{model}" if provs else "fake"
    M["vision_input"] = vision
    llm = budget.LLM(chat or chat_json, budget_usd, model, cache_dir=None if cold else str(cache_dir / "llm"),
                     ledger=ledger, chat_kwargs=chat_kwargs, reasoning=tw is None)
    parts_chars = [len(story.transcript_lines(c["sentences"])) for c in chunks]
    n_batches = (max_candidates + 5) // 6
    fc = forecast(parts_chars, model, n_batches, max_candidates, vision)
    M["forecast"] = fc
    say(f"forecast ${fc['usd']:.3f} (with repairs ${fc['usd_with_repairs']:.3f}), budget ${budget_usd:.2f}")
    # with a per-video ledger the full forecast assumes NO cache (the analysis of this film is cached); the analysis
    # calls are guarded one by one, and the creative steps get their own forecast below, before any of them runs
    if fc["usd_with_repairs"] > budget_usd and not video_limit_rub:
        M["stopped"] = f"forecast ${fc['usd_with_repairs']:.3f} > budget ${budget_usd:.2f}"
        save()
        raise budget.BudgetExceeded(M["stopped"])

    if tts is None:
        from scripts.shared.tts import get_provider
        tts = get_provider(tts_provider, tts_voice if tts_provider == "silero" else None)
    if not tts.available():
        raise RuntimeError(f"TTS provider {tts.name} is not available")
    rate = tts.calibrate(str(cache_dir))
    tempo = float(scfg["tempo"])
    if tempo != 1.0:                                   # writer sizes the text by the voice as it will be heard
        rate = dict(rate, chars_per_sec=round(rate["chars_per_sec"] * tempo, 3), base_sec=round(rate["base_sec"] / tempo, 4),
                    tempo=tempo)
    M["tts"] = dict(tts.info(), rate=rate, tempo=tempo)
    M["tts_note"] = ("TEST ONLY — non-commercial voice; not for production" if tts.test_only else "production voice")

    try:
        # ── Story Analyst ───────────────────────────────────────────────────
        t0 = time.time()
        memory: Dict[str, Any] = {}
        parts = []
        gate_args: Dict[str, Any] = {}
        for c in chunks:
            part = llm.ask(f"analyst_map_{c['index'] + 1}", story.map_messages(c, memory, language), story.MAP_VERSION,
                           lambda d, c=c: story.validate_map(d, c), max_out=8000)
            parts.append(part)
            memory = story.update_memory(memory, part)
        all_events[:] = [e for p_ in parts for e in p_["events"]]
        ev_text = {e["id"]: span_text(words, e["start"], e["end"]) for e in all_events}
        if style == "calm":
            sm = llm.ask("analyst_refine", recap.refine_messages(parts, dur, language), recap.REFINE_VERSION,
                         lambda d: recap.validate_recap_map(d, parts), max_out=6000)
        else:
            # the map parts are cached; only the global refine is redone for the new tone (+ real transcript evidence)
            rmsg = recap.refine_messages(parts, dur, language, style=style, evidence=ev_text)
            cp = llm._cache_path(f"analyst_refine_{style}", rmsg, ver["refine"], 0.2, 6000)
            refine_cached = bool(cp is not None and cp.is_file())
            writer_cached = False
            if refine_cached:                       # the same map → the same writer request: is its answer cached too?
                try:                                    # the cache holds the raw answer; validation is deterministic
                    sm0 = recap.validate_recap_map(json.loads(cp.read_text(encoding="utf-8")), parts)
                except (ValueError, budget.SchemaError):
                    sm0 = None
                if sm0 is not None:
                    oh0 = float(scfg["overhead_sec"])
                    lo0 = int((duration - oh0) * rate["chars_per_sec"] * 0.93)
                    hi0 = int((duration - oh0) * rate["chars_per_sec"] * 1.07)
                    ev0 = {e["id"]: [ev_text[e["id"]]] for e in sm0["events"] if ev_text.get(e["id"])}
                    wm = recap.writer_messages(sm0, duration, lo0, hi0, rate, style=style, evidence=ev0)
                    writer_cached = any(x is not None and x.is_file() for x in (
                        llm._cache_path(f"writer_{style}", wm, ver["writer"], 0.8, 4000),
                        llm._cache_path(f"writer_{style}_loose", wm, ver["writer"] + "+loose", 0.8, 4000)))
            gate_args = dict(refine_cached=refine_cached, writer_cached=writer_cached)
            if not refine_cached:            # the refine itself is a new call: gate now, counting every later step
                creative_gate(dict(gate_args), vision_batches=5, stage="all")
            sm = llm.ask(f"analyst_refine_{style}", rmsg,
                         ver["refine"], lambda d: recap.validate_recap_map(d, parts), max_out=6000)
        tick("story_analyst", t0)
        _dump(out / "story_map.json", {"version": ver["refine"], "chunks": len(parts), "story_map": sm,
                                       "chunk_parts": parts})
        if style != "calm":       # owner rule: no personal names in narration / subtitles (code check + Judge criterion)
            sm["forbid_names"] = recap.forbidden_names(sm, forbid_names)
            sm["forbid_names"] = [n for n in sm["forbid_names"] if n.rstrip("$") not in set(allow_words)]   # e.g. a role word the analyst took for a name
            say(f"names the narrator must not say: {sm['forbid_names']}")
            sm["source_crime_stems"] = recap.crime_stems_in(" ".join(w.text for w in words))
        say(f"story analyst done: {len(sm['events'])} events; hero={sm['characters'][0]['name'] if sm['characters'] else '?'}")
        hero = sm["characters"][0]["name"] if sm["characters"] else "главный герой"

        # ── Visual Analyst ──────────────────────────────────────────────────
        t0 = time.time()
        # the Visual Analyst sees the DOCUMENTARY story map (calm refine): its answers for this film are cached and the
        # windows it picks do not depend on the narration tone; a different map would re-pay the whole vision pass
        vsm = sm if style == "calm" else llm.ask("analyst_refine", recap.refine_messages(parts, dur, language),
                                                 recap.REFINE_VERSION, lambda d: recap.validate_recap_map(d, parts), max_out=6000)
        va = visual.analyze(llm, src, shot_list, words, vsm["events"], vsm["characters"], dur, sha,
                            ccache / "frames", max_candidates=max_candidates, vision=vision,
                            speech_by_window=lambda c: window_text(words, c), log=say)
        if style != "calm" and not no_extra_vision:     # (--no-extra-vision: only already analysed windows, 0 ₽)
            # pictures for what the NEW script talks about: wild moments + events without any analysed window
            covered = lambda e: any(c["end"] > e["start"] and c["start"] < e["end"] for c in va["candidates"])  # noqa: E731
            wild = {w["event_id"] for w in sm.get("wild_moments", [])}
            focus = [e for e in sm["events"] if e["id"] in wild or not covered(e)]
            if gate_args.get("refine_cached"):   # deferred gate: now we know which supplementary batches are cached
                dry = visual.supplement(llm, src, shot_list, words, focus, vsm["characters"], dur, sha, ccache / "frames",
                                        va["candidates"], vision=vision, speech_by_window=lambda c: window_text(words, c),
                                        log=say, dry_run=True)
                creative_gate(dict(gate_args), vision_batches=dry["uncached_batches"], stage="head")
            extra = visual.supplement(llm, src, shot_list, words, focus, vsm["characters"], dur, sha, ccache / "frames",
                                      va["candidates"], vision=vision, speech_by_window=lambda c: window_text(words, c), log=say)
            va["candidates"] += extra["candidates"]
            va["stage1"]["supplement"] = dict(extra["stage1"], focus_events=[e["id"] for e in focus])
        for c in va["candidates"]:
            c["event_ids"] = [e["id"] for e in sm["events"] if e["end"] > c["start"] and e["start"] < c["end"]]
        cands = {c["id"]: c for c in va["candidates"]}
        va["summary"] = {"analysed": len(cands), "usable": sum(1 for c in cands.values() if c["usable"]),
                         "speaking_states": {s: sum(1 for c in cands.values() if c["speaking"]["state"] == s)
                                             for s in ("not_speaking", "likely_listening", "likely_speaking", "uncertain")},
                         "advertisement_or_graphic": sum(1 for c in cands.values() if c["kind"] in ("advertisement", "title_graphic"))}
        _dump(out / "visual_analysis.json", va)
        tick("visual_analyst", t0)
        M["visual"] = dict(va["stage1"], **va["summary"])
        say(f"visual analyst done: {va['summary']}")

        # ── Writer → TTS (measured) → fit ───────────────────────────────────
        oh = float(scfg["overhead_sec"])
        script_chars = lambda sec: (int((sec - oh) * rate["chars_per_sec"] * 0.93), int((sec - oh) * rate["chars_per_sec"] * 1.07))  # noqa: E731
        lo_c, hi_c = script_chars(duration)
        evidence = {e["id"]: [ev_text[e["id"]]] for e in sm["events"] if ev_text.get(e["id"])} if style != "calm" else None
        t0 = time.time()
        if script_file:       # hand-corrected narration (saves a Writer pass); still passes every script check
            raw = json.loads(Path(script_file).read_text(encoding="utf-8"))
            script = recap.validate_script(raw.get("script", raw), sm, lo_c, hi_c, 0.5, style=style)
            M["warnings"].append(f"script taken from {script_file} (hand-corrected), not written by the model")
        else:
            script = ask_writer(llm, sm, duration, lo_c, hi_c, rate, style=style, evidence=evidence)
        tick("writer", t0)
        say(f"writer: {sum(len(s['text']) for s in script['segments'])} chars in {len(script['segments'])} segments")

        def synth(scr: Dict[str, Any], tag: str):
            """Sentence-level TTS (cached), measured durations → voice timeline."""
            wavs, secs = {}, {}
            for seg in scr["segments"]:
                secs[seg["id"]] = []
                for j, sent in enumerate(recap.sentences_of(seg["text"])):
                    h = hashlib.sha256(f"{tts.name}|{tts.voice}|{sent}".encode()).hexdigest()[:20]
                    p = ccache / "tts" / f"{h}.wav"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    if not p.is_file():
                        tts.synthesize(sent, str(p))
                    if tempo != 1.0:
                        p = Path(speed_up(str(p), str(p.with_suffix(f".x{tempo:.2f}e.wav")), tempo))
                    from scripts.shared.tts import wav_duration
                    wavs[(seg["id"], j)] = str(p)
                    secs[seg["id"]].append(wav_duration(str(p)))
            return wavs, compose.voice_timeline(scr, secs, pauses=scfg["pauses"])

        t0 = time.time()
        wavs, vt = synth(script, "w1")
        tick("tts", t0)
        fit_log = []
        for it in range(3):
            fit_log.append({"iteration": it, "total_sec": vt["total"], "voice_sec": vt["voice_sec"]})
            if FIT_LO <= vt["total"] <= FIT_HI:
                break
            per_seg = {s["id"]: s["t1"] - s["t0"] for s in vt["segments"]}
            instr = recap.fit_instructions(vt["total"], FIT_LO + 0.5, FIT_HI - 0.5, per_seg, rate)
            say(f"fit #{it + 1}: total {vt['total']:.1f}s → rewrite for length")
            t0 = time.time()
            cur_chars = sum(len(s["text"]) for s in script["segments"])
            tgt = cur_chars * ((FIT_LO + FIT_HI) / 2 - 1.7) / max(1.0, vt["total"] - 1.7)
            lo2, hi2 = int(tgt * 0.96), int(tgt * 1.04)
            fit_step = f"writer_fit_{it + 1}" + ("" if style == "calm" else f"_{style}")
            try:
                script = ask_length_bound(llm, fit_step,
                                          recap.writer_messages(sm, duration, lo2, hi2, rate, previous=script, instructions=instr,
                                                                style=style, evidence=evidence),
                                          ver["writer"] + "+fit", sm, lo2, hi2, strict=0.05, loose=0.3, temperature=0.5, style=style)
            except budget.LLMStop:
                raise
            except RuntimeError as e:          # e.g. the provider's content filter refused the previous text: write anew
                M["warnings"].append(f"{fit_step}: {str(e)[:120]} — script written anew for the corrected length")
                say(f"{fit_step} refused ({str(e)[:80]}): writing the script anew for {lo2}–{hi2} chars")
                script = ask_length_bound(llm, fit_step + "_anew",
                                          recap.writer_messages(sm, duration, lo2, hi2, rate, style=style, evidence=evidence),
                                          ver["writer"] + "+anew", sm, lo2, hi2, strict=0.05, loose=0.3, temperature=0.7, style=style)
            tick("writer", t0)
            t0 = time.time()
            wavs, vt = synth(script, f"w{it + 2}")
            tick("tts", t0)
        # last resort inside natural limits: scale the pauses a little (never speed the voice up)
        if not FIT_LO <= vt["total"] <= FIT_HI:
            for sc in (1.25, 1.1, 0.85, 0.7):
                alt = compose.voice_timeline(script, {s["id"]: [r["sec"] for r in s["sentences"]] for s in vt["segments"]},
                                             pause_scale=sc, pauses=scfg["pauses"])
                if abs(alt["total"] - duration) < abs(vt["total"] - duration):
                    vt = alt
                if FIT_LO <= vt["total"] <= FIT_HI:
                    break
            fit_log.append({"pause_scaled_total": vt["total"]})
        M["script_fit"] = fit_log
        _dump(out / "script.json", {"version": ver["writer"], "style": style, "script": script,
                                    "chars": sum(len(s["text"]) for s in script["segments"])})

        # ── produce: Director → Composer → render → validate ────────────────
        def produce(scr: Dict[str, Any], wavs, vt, label: str, instr: Optional[str] = None, prev: Optional[Dict[str, Any]] = None):
            t0 = time.time()
            seg_sec = compose.segment_seconds(vt)
            cards = [dict(recap.candidate_card(c), hero_in_frame=recap.hero_visible(dict(c, people_raw=c["people"]), hero))
                     for c in cands.values() if c["usable"]]
            dmsg = recap.director_messages(scr, seg_sec, cards, hero, prev, instr)
            if label == "a" and gate_args.get("refine_cached") and not director_plan_file:
                dcp = llm._cache_path(f"director_{label}", dmsg, ver["director"], 0.3, 4000)
                creative_gate({"writer_cached": True, "director_cached": bool(dcp is not None and dcp.is_file()),
                               "rewrite": max_rewrites > 0}, stage="tail")
            if director_plan_file:           # a montage plan made earlier for the same segments: no Director call (0 ₽)
                old_va = Path(director_plan_file).with_name("visual_analysis.json")
                if old_va.is_file():             # windows analysed in that run (e.g. supplementary X*) are reused as they are
                    for c in json.loads(old_va.read_text(encoding="utf-8")).get("candidates", []):
                        if c["id"] not in cands:
                            c["event_ids"] = [e["id"] for e in sm["events"] if e["end"] > c["start"] and e["start"] < c["end"]]
                            cands[c["id"]] = c
            if director_plan_file:           # a montage plan made earlier for the same segments: no Director call (0 ₽)
                plan = recap.validate_director(json.loads(Path(director_plan_file).read_text(encoding="utf-8")), scr, seg_sec, cands, hero)
                M["warnings"].append(f"montage plan taken from {director_plan_file}, not asked from the Director")
            else:
                plan = llm.ask(f"director_{label}", dmsg, ver["director"] + ("+rewrite" if instr else ""),
                               lambda d: recap.validate_director(d, scr, seg_sec, cands, hero), temperature=0.3, max_out=4000)
            tick("visual_director", t0)
            wd = work / label
            wd.mkdir(exist_ok=True)
            damaged: List[str] = []
            for _attempt in range(4):
                t0 = time.time()
                timeline = compose.compose(plan, cands, vt)
                t1 = time.time()
                spare = [c["id"] for c in sorted(cands.values(), key=lambda c: -(c["visual_interest"] * 10 + c["event_weight"]))
                         if c["usable"] and c["speaking"]["state"] in ("not_speaking", "likely_listening", "uncertain")]
                compose.verify_sources(timeline, cands, lambda a, b: sorted(set(media.frame_cuts(src, a, b, str(ccache / "clips"), sha))
                                                                            | set(validator.hard_cuts_in(src, a, b))), dur, spare)
                tick("source_verification", t1)
                problems = compose.check_plan(timeline)
                if problems:
                    raise RuntimeError(f"composer produced an invalid timeline: {problems}")
                tick("timeline_composer", t0)
                t0 = time.time()
                try:
                    files = media.render_shots(src, timeline["shots"], wd, width=info["width"], height=info["height"],
                                               cache_dir=str(ccache / "clips"), source_sha256=sha)
                except media.DamagedSource as e:     # undecodable source window: drop it, recompose (no LLM call), retry
                    if not e.candidate_id or e.candidate_id not in cands:
                        raise
                    cands[e.candidate_id]["usable"] = False
                    damaged.append(e.candidate_id)
                    say(f"{e} → {e.candidate_id} marked unusable, recomposing")
                    continue
                finally:
                    tick("render_shots", t0)
                break
            else:
                raise RuntimeError(f"too many damaged source windows: {damaged}")
            if damaged:
                timeline["warnings"] = list(timeline.get("warnings", [])) + [f"damaged source window, replaced: {d}" for d in damaged]
            t0 = time.time()
            voice = str(wd / "voice.wav")
            compose.write_voice(vt, wavs, voice)
            stab = compose.stability(vt, voice)
            g = media.voice_gain_db(voice)
            ass = None
            sw, _bounds = censor.subtitle_words(vt)
            cplan = None
            mix_voice = voice
            if censor_profanity:                 # SONYA BEEP: bleep the narrator, mask the subtitles («пи###ц»)
                t1 = time.time()
                try:
                    asr_w = censor.asr_words_fn(language, wd)(voice)
                except Exception as e:           # local ASR is only a refinement of the timing
                    asr_w = None
                    M["warnings"].append(f"beep: ASR of the voice failed ({str(e)[:100]}), word times are estimated")
                cplan = censor.plan(vt, asr_w)
                if cplan.intervals:
                    mix_voice = censor.beep_voice(voice, cplan.intervals, str(wd / "voice_beep.wav"), cplan.cfg)
                    sw = cplan.words
                M["beep"] = cplan.report()
                tick("beep", t1)
            if subtitles:
                ass = str(out / ("narration.ass" if label == "a" else f"narration_{label}.ass"))
                Path(ass).write_text(subs.to_ass(sw, vt["total"]), encoding="utf-8")
            final = str(out / ("final.mp4" if label == "a" else f"final_{label}.mp4"))
            media.assemble_recap(files, mix_voice, g["gain_db"], vt["total"], ass, final,
                                 shot_frames=[round((s["t1"] - s["t0"]) * 30) for s in timeline["shots"]])
            tick("render_assemble", t0)
            return {"plan": plan, "timeline": timeline, "files": files, "voice": voice, "mix_voice": mix_voice, "gain": g, "stab": stab, "final": final,
                    "ass": ass}

        def validate_now(pr, scr, vt):
            t0 = time.time()
            script_text = " ".join(s["text"] for s in scr["segments"])
            heard = censor.without_profanity(script_text) if censor_profanity else None   # bleeped words are not heard
            v = validator.validate(pr["final"], tl=pr["timeline"], vt=vt, voice_wav=pr["mix_voice"], gain_db=pr["gain"]["gain_db"],
                                   shot_files=pr["files"], source=src, script_text=script_text, spread_db=pr["stab"]["spread_db"],
                                   asr=asr_text_fn(language, work), dur_range=(TARGET_LO, TARGET_HI), target=duration,
                                   ass_path=pr.get("ass"), pace=PACE_PUNCHY if style == "punchy" else None,
                                   forbid_names=sm.get("forbid_names") if style == "punchy" else None, expected_heard=heard)
            tick("validator", t0)
            return v

        def judge_now(pr, scr, vt, step):
            t0 = time.time()
            shots = pr["timeline"]["shots"]
            items = [{"n": s["n"], "t": (s["t0"] + s["t1"]) / 2, "label": f"{s['segment_id']} #{s['n']}"} for s in shots]
            sheets = []
            for k in range(0, len(items), 12):
                sp = str(work / f"sheet_{step}_{k // 12}.jpg")
                media.contact_sheet(pr["final"], items[k:k + 12], sp)
                sheets.append(visual.image_part(sp))
            verdict = llm.ask(step if style == "calm" else f"{step}_{style}",
                              recap.judge_messages(sm, scr, pr["plan"], cands, sheets, style=style), ver["judge"],
                              lambda d: recap.validate_judge(d, style), temperature=0.0, max_out=4000)
            tick("creative_judge", t0)
            return verdict

        pr = produce(script, wavs, vt, "a")
        say(f"render done: {pr['final']}")
        val = validate_now(pr, script, vt)
        say(f"validator: passed={val['passed']} failed={val['failed']}")
        if skip_judge:                       # no Creative Judge call (0 ₽); creative scores are NOT available
            verdict = {"scores": {}, "weak_beats": [], "factual_errors": [], "visual_problems": [], "rewrite_instructions": "",
                       "style": style, "skipped": "Creative Judge not run (--skip-judge)"}
            M["warnings"].append("Creative Judge skipped (--skip-judge): no creative scores")
        else:
            verdict = judge_now(pr, script, vt, "judge")
        say(f"judge: {verdict['scores']}")
        attempts = [{"script": script, "validation": val, "verdict": verdict,
                     "rewrite_reasons": [] if skip_judge else recap.needs_rewrite(verdict)}]
        final_script, final_vt, final_pr, final_val = script, vt, pr, val
        scope = None if skip_judge else recap.rewrite_scope(verdict)
        attempts[0]["rewrite_scope"] = scope
        if scope and max_rewrites < 1:
            M["warnings"].append(f"judge asked for a {scope} rewrite; rewrites disabled (--max-rewrites 0, budget)")
            attempts[0]["rewrite_skipped"] = "max_rewrites=0"
            scope = None
        if scope:
            say(f"judge requests ONE rewrite ({scope}): {attempts[0]['rewrite_reasons']}")
            instr = verdict["rewrite_instructions"] + "\n" + "\n".join(
                f"{w['segment_id']} ({w['severity']}, {w['kind']}): {w['problem']} → {w['fix']}" for w in verdict["weak_beats"])
            instr += "\n" + "\n".join(f"Фактическая ошибка: {e}" for e in verdict["factual_errors"])
            vis = "Визуальные замечания: " + "; ".join(verdict["visual_problems"])
            script2, wavs2, vt2 = script, wavs, vt
            if scope == "text":
                t0 = time.time()
                cur = sum(len(x["text"]) for x in script["segments"])           # keep the FITTED length, not the first guess
                try:
                    script2 = ask_length_bound(llm, "writer_rewrite" + ("" if style == "calm" else f"_{style}"),
                                               recap.writer_messages(sm, duration, int(cur * 0.95), int(cur * 1.05), rate, previous=script,
                                                                     instructions=instr + "\n" + vis, style=style, evidence=evidence),
                                               ver["writer"] + "+rewrite", sm, int(cur * 0.95), int(cur * 1.05), strict=0.03,
                                               loose=0.12, temperature=0.6, style=style)
                    wavs2, vt2 = synth(script2, "r")
                except (budget.BudgetExceeded, RuntimeError) as e:   # optional step: the first version stays
                    M["warnings"].append(f"rewrite (text) not done: {str(e)[:200]}; keeping the first version")
                    attempts[0]["rewrite_failed"] = str(e)[:300]
                    script2, wavs2, vt2 = script, wavs, None
                tick("writer", t0)
            if vt2 is None:
                pass
            elif not TARGET_LO <= vt2["total"] <= TARGET_HI:
                M["warnings"].append(f"rewrite is {vt2['total']:.1f}s — outside the accepted range; keeping the first version")
                attempts[0]["rewrite_rejected"] = f"duration {vt2['total']:.1f}s"
            else:
                try:
                    pr2 = produce(script2, wavs2, vt2, "b", instr=instr + "\n" + vis, prev=pr["plan"])
                    val2 = validate_now(pr2, script2, vt2)
                    verdict2 = judge_now(pr2, script2, vt2, "judge_rewrite")
                except budget.BudgetExceeded as e:  # the money ran out inside the optional rewrite: keep the first version
                    M["warnings"].append(f"rewrite stopped by the budget ({str(e)[:160]}); keeping the first version")
                    attempts[0]["rewrite_failed"] = str(e)[:300]
                    scope = None
                except budget.LLMStop:
                    raise
                except RuntimeError as e:         # the rewrite is optional: a bad answer must not lose the finished first version
                    M["warnings"].append(f"rewrite failed ({str(e)[:160]}); keeping the first version")
                    attempts[0]["rewrite_failed"] = str(e)[:300]
                    scope = None
                else:
                    attempts.append({"script": script2, "validation": val2, "verdict": verdict2, "scope": scope,
                                     "rewrite_reasons": recap.needs_rewrite(verdict2)})
                    better = val2["passed"] and (not val["passed"] or sum(verdict2["scores"].values()) >= sum(verdict["scores"].values()))
                    attempts[-1]["chosen"] = bool(better)
                    if better:
                        final_script, final_vt, final_pr, final_val, verdict = script2, vt2, pr2, val2, verdict2
                        os.replace(pr2["final"], str(out / "final.mp4"))
                        final_pr["final"] = str(out / "final.mp4")
                        if pr2.get("ass"):
                            os.replace(pr2["ass"], str(out / "narration.ass"))
                            final_pr["ass"] = str(out / "narration.ass")
                        _dump(out / "script.json", {"version": ver["writer"] + ("+rewrite" if scope == "text" else ""), "style": style,
                                                    "script": script2, "chars": sum(len(x["text"]) for x in script2["segments"]),
                                                    "first_version": script})
        _dump(out / "judge.json", {"version": ver["judge"], "final": verdict, "attempts": [
            {k: v for k, v in a.items() if k != "script"} for a in attempts], "rewrites": len(attempts) - 1,
            "note": "creative scores only; technical result is validation.json"})
        _dump(out / "validation.json", final_val)
        _dump(out / "director_plan.json", final_pr["plan"])
        _dump(out / "timeline.json", dict(final_pr["timeline"], reuse=compose.reuse_report(final_pr["timeline"])))
        _dump(out / "mix_plan.json", {"audio_owner": "narration only", "original_audio": "removed (shots rendered without an audio stream)",
                                      "gain_db": final_pr["gain"], "loudness_spread": final_pr["stab"],
                                      "voice_sec": final_vt["voice_sec"], "voice_share": round(final_vt["voice_sec"] / final_vt["total"], 3)})
        _dump(out / "narration.json", {"tts": M["tts"], "tts_license": tts.license, "segments": [
            dict(s, text=next(x["text"] for x in final_script["segments"] if x["id"] == s["id"]),
                 function=next(x["function"] for x in final_script["segments"] if x["id"] == s["id"]))
            for s in final_vt["segments"]]})
        M["final_output_path"] = str(out / "final.mp4")
        M["final_duration"] = final_val["duration"]
        M["technical_passed"] = final_val["passed"]
        M["technical_failed"] = final_val["failed"]
        M["judge_scores"] = verdict["scores"]
        M["rewrites"] = len(attempts) - 1
        M["shots"] = len(final_pr["timeline"]["shots"])
        M["composer_warnings"] = final_pr["timeline"]["warnings"]
        M["selected_timestamps"] = [{k: s[k] for k in ("n", "segment_id", "candidate_id", "src_start", "src_end", "t0", "t1")}
                                    for s in final_pr["timeline"]["shots"]]
        M["final_size_mb"] = round(Path(M["final_output_path"]).stat().st_size / 1e6, 1)
        fs = " ".join(x["text"] for x in final_script["segments"])
        M["pace"] = {"words": len(fs.split()), "words_per_sec": round(len(fs.split()) / max(1e-6, final_val["duration"]), 2),
                     "tempo": tempo, "voice_share": round(final_vt["voice_sec"] / final_vt["total"], 3)}
        M["invented_gags"] = [dict(g, segment_id=x["id"]) for x in final_script["segments"] for g in x.get("gags", [])]
    finally:
        M["llm"] = llm.summary()
        M["cost_usd"] = llm.spent
        M["cost_rub"] = budget.rub(llm.spent)
        if M.get("video_budget"):
            vl = [x for x in llm.ledgers if isinstance(x, budget.VideoLedger)]
            if vl:
                M["video_budget"]["spent_after_rub"] = budget.rub(vl[0].spent())
        save()
    return M


def parse_window(text: Optional[str]) -> tuple:
    """'1:22:25-1:43:08' / '4945-6188' → (4945.0, 6188.0)."""
    def sec(x: str) -> float:
        v = 0.0
        for part in x.strip().split(":"):
            v = v * 60 + float(part)
        return v
    a, _, b = (text or "").partition("-")
    if not a or not b or sec(b) <= sec(a):
        raise SystemExit(f"--window START-END expected (got {text!r})")
    return sec(a), sec(b)


def economy_enabled(economy: str, style: str, window: Optional[str]) -> bool:
    """Economy (lite) path: default for the energetic style when the source is ONE fragment (--window)."""
    if economy == "off":
        return False
    if economy == "on":
        if not window:
            raise SystemExit("--economy on needs --window START-END (one fragment)")
        return True
    return style == "punchy" and bool(window)


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="trailer_cinematic", description="SONYA trailer_cinematic v2 — AI Cinematic Story Recap (dev CLI)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--source-minutes", type=float, default=15.0, help="fast run: first N minutes (0 = whole video)")
    ap.add_argument("--budget", type=float, default=None, help="LLM budget per run, USD (default 1.0; with --llm-provider timeweb 0.50)")
    ap.add_argument("--llm-provider", choices=("auto", "timeweb"), default="auto",
                    help="auto = as before (env providers); timeweb = Gemini via Timeweb ONLY (key from Keychain), no fallback")
    ap.add_argument("--no-timeweb", action="store_true", help="kill switch: Timeweb provider off (also SONYA_LLM_TIMEWEB_ENABLED=false)")
    ap.add_argument("--vision", choices=("frames", "clips"), default="frames",
                    help="what the vision model gets per window: 3 JPEG frames (default) or one short silent clip")
    ap.add_argument("--day-limit", type=float, default=None, help="Timeweb: spend limit per day over all runs, USD (default 2.00)")
    ap.add_argument("--run-dir")
    ap.add_argument("--voice", default="aidar")
    ap.add_argument("--tts", default="silero", choices=("silero", "xtts", "elevenlabs"))
    ap.add_argument("--cold", action="store_true", help="no LLM/frame/clip/TTS cache: honest cost and timings")
    ap.add_argument("--no-subtitles", action="store_true")
    ap.add_argument("--max-candidates", type=int, default=56)
    ap.add_argument("--style", choices=sorted(recap.STYLES) + ["story"], default=recap.DEFAULT_STYLE,
                    help="punchy = fast cheeky trailer voice (default); calm = the v2 documentary recap; story = ONE episode told by the "
                         "hero in the first person, serious, film cropped to 1:1 / 4:5 / 3:4 around the faces, variants A (voice) + B (voice + film sound + accents), storystyle.py")
    ap.add_argument("--video-limit-rub", type=float, default=None,
                    help="Timeweb: spend limit for THIS video over all its runs, roubles (ledger _cache/spend_video-<sha>.json)")
    ap.add_argument("--debug-budget", action="store_true",
                    help=f"hard stop at the debugging ceiling {budget.DEBUG_VIDEO_LIMIT_RUB:.0f} ₽/video instead of the production "
                         f"limit {budget.PROD_VIDEO_LIMIT_RUB:.0f} ₽ (SONYA_TRAILER_VIDEO_LIMIT_RUB)")
    ap.add_argument("--video-seed-usd", type=float, default=0.0,
                    help="money already spent on this video before the ledger existed (booked once, on creation)")
    ap.add_argument("--forbid-names", default="",
                    help="extra names (comma-separated) the narrator must never say, besides the story-map characters")
    ap.add_argument("--script-file", default=None,
                    help="use this script.json (hand-corrected) instead of asking the Writer; all script checks still apply")
    ap.add_argument("--no-censor", action="store_true", help="no BEEP step (default for punchy: profanity bleeped and masked)")
    ap.add_argument("--director-plan", default=None, help="reuse this director_plan.json instead of asking the Director (0 ₽)")
    ap.add_argument("--no-extra-vision", action="store_true", help="skip the supplementary vision pass (0 ₽): only windows already analysed")
    ap.add_argument("--allow-words", default="", help="comma-separated stems removed from the forbidden-names list (role words mistaken for names)")
    ap.add_argument("--skip-judge", action="store_true", help="do not call the Creative Judge (0 ₽, no creative scores)")
    ap.add_argument("--max-rewrites", type=int, default=1, choices=(0, 1),
                    help="1 = the Judge may trigger ONE rewrite (spec); 0 = report only (e.g. when the video budget is nearly spent)")
    ap.add_argument("--window", default=None, help="one fragment of the source, START-END (seconds or h:mm:ss), e.g. 1:22:25-1:43:08")
    ap.add_argument("--economy", choices=("auto", "on", "off"), default="auto",
                    help="economy path (lite.py: 1 Story+Writer call, 12–20 stills, no Judge/Director, stop 5 ₽). auto = ON for the "
                         "energetic style when --window is given")
    ap.add_argument("--stop-rub", type=float, default=None, help="economy: hard stop in roubles (default 5)")
    ap.add_argument("--frames", type=int, default=14, help="economy: stills sent to the model (12–20)")
    ap.add_argument("--bed-db", type=float, default=None, help="story: film sound level under the narrator where nobody talks, dB (variant B; default −17, −23 where the film's heroes talk)")
    ap.add_argument("--crop", choices=["1:1", "4:5", "3:4"], default="1:1", help="story: the film is cropped to this aspect around the faces")
    ap.add_argument("--sfx-dir", default=None, help="story: the owner's sound pack folder for the accents (default: Desktop/rils/Саунд пак)")
    ap.add_argument("--fix-notes", default=None, help="story: with --script-file, a text file of fact-check notes (one per line) for ONE more text-only fix call")
    ap.add_argument("--no-sfx", action="store_true", help="story: variant B without whoosh/impact/riser accents")
    ap.add_argument("--pick", default=None, help="story: take local candidate C1…C5 without asking the model which episode")
    a = ap.parse_args(argv)
    if a.no_timeweb:
        os.environ["SONYA_LLM_TIMEWEB_ENABLED"] = "false"
    if a.style == "story":
        from . import storystyle
        try:
            m = storystyle.run_story(a.input, run_dir=a.run_dir, n_frames=a.frames, stop_rub=a.stop_rub or storystyle.STORY_STOP_RUB,
                                     tts_voice=a.voice, tts_provider=a.tts, day_limit_usd=a.day_limit,
                                     bed_db=a.bed_db if a.bed_db is not None else storystyle.BED_DB,
                                     forbid_names=[x.strip() for x in a.forbid_names.split(",") if x.strip()],
                                     window=parse_window(a.window) if a.window else None, script_file=a.script_file, pick=a.pick, crop=a.crop, fix_notes=a.fix_notes,
                                     sfx_dir="" if a.no_sfx else a.sfx_dir)
        except LLMUnavailable as e:
            print(f"LLM недоступен: {e}")
            return 2
        except budget.LLMStop as e:
            print(f"СТОП: {e}")
            return 3
        print(json.dumps({k: m.get(k) for k in ("episode", "final_A", "final_B", "final_duration", "pace", "cuts", "technical_passed",
                                                "technical_failed", "variant_B_failed", "cost_rub", "timings_sec", "warnings")},
                         ensure_ascii=False, indent=1))
        return 0
    if economy_enabled(a.economy, a.style, a.window):
        from . import lite
        try:
            m = lite.run_lite(a.input, parse_window(a.window), duration=a.duration, run_dir=a.run_dir, n_frames=a.frames,
                              stop_rub=a.stop_rub or lite.LITE_STOP_RUB, tts_voice=a.voice, tts_provider=a.tts,
                              subtitles=not a.no_subtitles, censor_profanity=not a.no_censor, day_limit_usd=a.day_limit,
                              forbid_names=[x.strip() for x in a.forbid_names.split(",") if x.strip()], script_file=a.script_file)
        except LLMUnavailable as e:
            print(f"LLM недоступен: {e}")
            return 2
        except budget.LLMStop as e:
            print(f"СТОП: {e}")
            return 3
        print(json.dumps({k: m.get(k) for k in ("final_output_path", "final_duration", "technical_passed", "technical_failed",
                                                "cost_usd", "cost_rub", "timings_sec", "warnings")}, ensure_ascii=False, indent=1))
        return 0
    try:
        m = run(a.input, a.duration, run_dir=a.run_dir, source_minutes=a.source_minutes, budget_usd=a.budget, tts_voice=a.voice,
                tts_provider=a.tts, subtitles=not a.no_subtitles, max_candidates=a.max_candidates, cold=a.cold,
                llm_provider=a.llm_provider, timeweb_off=a.no_timeweb, vision=a.vision, day_limit_usd=a.day_limit,
                style=a.style, video_limit_rub=a.video_limit_rub, video_seed_usd=a.video_seed_usd,
                max_rewrites=a.max_rewrites, debug_budget=a.debug_budget,
                forbid_names=[x.strip() for x in a.forbid_names.split(",") if x.strip()],
                script_file=a.script_file, censor_profanity=False if a.no_censor else None,
                director_plan_file=a.director_plan, skip_judge=a.skip_judge, no_extra_vision=a.no_extra_vision,
                allow_words=[x.strip() for x in a.allow_words.split(",") if x.strip()])
    except LLMUnavailable as e:
        print(f"LLM недоступен: {e}")
        return 2
    except budget.LLMStop as e:
        print(f"СТОП: {e}")
        return 3
    print(json.dumps({k: m.get(k) for k in ("final_output_path", "final_duration", "technical_passed", "technical_failed",
                                            "judge_scores", "pace", "cost_usd", "cost_rub", "video_budget", "timings_sec", "warnings")}, ensure_ascii=False, indent=1))
    return 0
