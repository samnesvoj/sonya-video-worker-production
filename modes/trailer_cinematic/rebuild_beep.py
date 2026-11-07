"""
rebuild_beep.py — re-assemble a FINISHED trailer run with the BEEP step, without any LLM call.

Reads the run folder (timeline.json, narration.json, mix_plan.json, _work/<label>/shot_NN.mp4 + voice.wav), finds the
profanity in the narration, bleeps the voice, masks the subtitles («пи###ц»), assembles and re-validates the result.

    python -m modes.trailer_cinematic.rebuild_beep --run-dir runtime/trailer/kishlak_dz --work a --source runtime/dl/kishlak.mp4 \\
        --out runtime/trailer/kishlak_dz/kishlak_BEEP.mp4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import censor, media, subs, validator
from .recap_runner import asr_text_fn
from .runner import _desktop_media_env


def rebuild(run_dir: str, work: str, source: str, out: str, language: str = "ru") -> dict:
    _desktop_media_env()
    rd = Path(run_dir)
    tl = json.loads((rd / "timeline.json").read_text(encoding="utf-8"))
    nar = json.loads((rd / "narration.json").read_text(encoding="utf-8"))
    mix = json.loads((rd / "mix_plan.json").read_text(encoding="utf-8"))
    script = json.loads((rd / "script.json").read_text(encoding="utf-8"))["script"]
    wd = rd / "_work" / work
    segs = nar["segments"]
    total = float(tl["shots"][-1]["t1"])
    vt = {"segments": segs, "voice_end": segs[-1]["t1"], "total": total, "voice_sec": mix.get("voice_sec")}
    files = [str(wd / f"shot_{s['n']:02d}.mp4") for s in tl["shots"]]
    voice = str(wd / "voice.wav")
    asr_w = censor.asr_words_fn(language, wd)(voice)
    cp = censor.plan(vt, asr_w, total)
    mix_voice = censor.beep_voice(voice, cp.intervals, str(wd / "voice_beep.wav"), cp.cfg) if cp.intervals else voice
    ass = str(Path(out).with_suffix(".ass"))
    Path(ass).write_text(subs.to_ass(cp.words if cp.intervals else censor.subtitle_words(vt)[0], total), encoding="utf-8")
    gain = mix["gain_db"]["gain_db"]
    media.assemble_recap(files, mix_voice, gain, total, ass, out, shot_frames=[round((s["t1"] - s["t0"]) * 30) for s in tl["shots"]])
    text = " ".join(s["text"] for s in segs)
    val = validator.validate(out, tl=tl, vt=vt, voice_wav=mix_voice, gain_db=gain, shot_files=files, source=source, script_text=text,
                             spread_db=0.0, asr=asr_text_fn(language, wd), dur_range=(54.0, 66.0), target=60.0, ass_path=ass,
                             pace=(3.3, 4.4), expected_heard=censor.without_profanity(text))
    rep = {"beep": cp.report(), "validation": val}
    Path(out).with_suffix(".beep.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--work", default="a")
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = rebuild(a.run_dir, a.work, a.source, a.out)
    print(json.dumps({"beep": r["beep"], "passed": r["validation"]["passed"], "failed": r["validation"]["failed"]}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
