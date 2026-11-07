#!/usr/bin/env python3
"""
Real recognition with the real weights, OFFLINE, using the upstream gigaam package at the pinned commit (NOT the Mac mini backend, so this is
an independent check of weights + torch stack + word timestamps + Cut compatibility). Runs wherever the weights exist: the Mac mini, or CI when
the repository secrets for the test bucket are configured (the CI step is skipped, and reported as skipped, otherwise).
  python tools/ci_gigaam_real.py --models-dir ~/SONYA-models --audio speech.wav [--gigaam-src /path/to/pinned/GigaAM] [--device cpu]
Runs the load inside a socket guard: any network attempt = FAIL. Checks: SHA-256 vs release/models.lock.json, upstream MD5, transcription text
(Cyrillic, non-empty), word timestamps (inside the audio, monotonic), and Cut's own normalisation (dropped-word ratio) before and after the
reference post-processing (-150 ms start lead, minimum word length).
"""
import argparse, hashlib, json, pathlib, socket, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ap = argparse.ArgumentParser(); ap.add_argument("--models-dir", required=True, type=pathlib.Path); ap.add_argument("--audio", required=True)
ap.add_argument("--gigaam-src"); ap.add_argument("--device", default="cpu")
a = ap.parse_args()
if a.gigaam_src:
    sys.path.insert(0, a.gigaam_src)

from scripts.shared import asr_contract as ac

problems = []
man = {m["key"]: m for m in json.loads((ROOT / "release/models.required.json").read_text())["models"]}
lock = json.loads((ROOT / "release/models.lock.json").read_text()).get("models", {})
gdir = a.models_dir / "gigaam"
for key, rel in ac.GIGAAM_FILES.items():
    f = a.models_dir / rel
    if not f.is_file():
        problems.append(f"missing {rel}"); continue
    h, md5 = hashlib.sha256(), hashlib.md5()
    with open(f, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b); md5.update(b)
    if key in lock and lock[key]["sha256"] != h.hexdigest():
        problems.append(f"{rel}: sha256 does not match models.lock.json")
    if man[key].get("upstream_md5") and md5.hexdigest() != man[key]["upstream_md5"]:
        problems.append(f"{rel}: md5 does not match upstream")
    print(f"{rel}: sha256={h.hexdigest()[:16]}… size={f.stat().st_size}", "(pinned in lock)" if key in lock else "(not pinned yet)")
if problems:
    print("\n".join("FAIL " + p for p in problems)); sys.exit(1)

real_connect, real_gai = socket.socket.connect, socket.getaddrinfo
def deny(*x, **k):
    raise AssertionError("NETWORK ACCESS while ASR must be offline")
socket.socket.connect, socket.getaddrinfo = deny, deny
import gigaam
model = gigaam.load_model("v3_e2e_rnnt", device=a.device, fp16_encoder=False, download_root=str(gdir))
res = model.transcribe(a.audio, word_timestamps=True)
socket.socket.connect, socket.getaddrinfo = real_connect, real_gai
dur = float(json.loads(subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", a.audio], capture_output=True, text=True).stdout)["format"]["duration"])
words = [{"word": " " + w.text, "start": float(w.start), "end": float(w.end)} for w in (res.words or [])]
print(f"text: {res.text[:160]!r}\nwords: {len(words)}; audio {dur:.1f}s")
if not any("а" <= ch.lower() <= "я" for ch in res.text):
    problems.append("transcript has no Cyrillic text")
raw = ac.cut_compat(words)
shifted = [dict(w) for w in words]
prev = 0.0
for i, w in enumerate(shifted):                                   # reference post-processing (desktop/sonya_desktop/asr/backends.py fix_rnnt_word_times)
    s, e = w["start"], w["end"]; s = max(prev, s - ac.RNNT_START_SHIFT_SEC)
    nxt = shifted[i + 1]["start"] - ac.RNNT_START_SHIFT_SEC if i + 1 < len(shifted) else e + ac.MIN_WORD_SEC
    if e - s < ac.MIN_WORD_SEC:
        e = max(e, min(s + ac.MIN_WORD_SEC, nxt - 0.01))
    w["start"], w["end"] = round(s, 3), round(max(e, s), 3); prev = w["end"]
fixed = ac.cut_compat(shifted)
print("Cut normalisation, RAW upstream words :", raw)
print("Cut normalisation, after post-processing:", fixed)
res_dict = {"text": res.text, "language": "ru", "backend": "gigaam", "model": "gigaam-v3_e2e_rnnt-ts3", "segments": [{"start": 0, "end": dur, "text": res.text, "words": shifted}], "words": shifted}
problems += ac.validate_result(res_dict, dur)
if not fixed["ok"]:
    problems.append(f"post-processed words still lose {fixed['dropped']} words in Cut")
print("\n".join(("FAIL " + p) for p in problems) or "ALL CHECKS PASSED (offline load, real recognition, word timestamps, Cut compatibility)")
sys.exit(1 if problems else 0)
