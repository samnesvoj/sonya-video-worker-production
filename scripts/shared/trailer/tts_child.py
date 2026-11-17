"""
Qwen3-TTS child process (runs in the TTS venv, never imported by the worker): voice clone of a fixed SONYA narrator
voice (reference WAV + its text) → one WAV per request. Apache-2.0 model (Qwen/Qwen3-TTS-12Hz-*-Base).

stdin: one JSON {"model": dir, "ref_wav", "ref_text", "device", "items": [{"text", "out", "seed"}], "batch": 4}
stdout: one JSON line per item {"out", "sec", "ok", "error"?} then {"done": true, "load_s", "gen_s"}
"""
from __future__ import annotations

import json
import sys
import time


def main() -> int:
    req = json.loads(sys.stdin.read())
    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel
    dev = req.get("device") or ("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if dev.startswith("cuda") else torch.float32
    t0 = time.time()
    kw = {"device_map": dev, "dtype": dtype}
    if dev.startswith("cuda"):
        try:
            import flash_attn  # noqa: F401
            kw["attn_implementation"] = "flash_attention_2"
        except ImportError:
            kw["attn_implementation"] = "sdpa"
    m = Qwen3TTSModel.from_pretrained(req["model"], **kw)
    prompt = m.create_voice_clone_prompt(ref_audio=req["ref_wav"], ref_text=req["ref_text"], x_vector_only_mode=False)
    load_s = time.time() - t0
    t1 = time.time()
    items = req["items"]
    bs = max(1, int(req.get("batch") or 4))
    gen = {k: req[k] for k in ("temperature", "top_p", "top_k", "repetition_penalty") if k in req}
    for k in range(0, len(items), bs):
        chunk = items[k:k + bs]
        torch.manual_seed(int(chunk[0].get("seed") or 0))
        try:
            wavs, sr = m.generate_voice_clone(text=[x["text"] for x in chunk], language=["Russian"] * len(chunk),
                                              voice_clone_prompt=prompt * len(chunk) if isinstance(prompt, list) else prompt,
                                              max_new_tokens=int(req.get("max_new_tokens") or 1400), **gen)
            for x, w in zip(chunk, wavs):
                w = np.asarray(w, dtype=np.float32)
                sf.write(x["out"], w, sr, subtype="PCM_16")
                print(json.dumps({"out": x["out"], "sec": round(len(w) / sr, 3), "ok": True}), flush=True)
        except Exception as exc:                     # noqa: BLE001 - report per item, the parent decides
            for x in chunk:
                print(json.dumps({"out": x["out"], "ok": False, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}),
                      flush=True)
    print(json.dumps({"done": True, "load_s": round(load_s, 1), "gen_s": round(time.time() - t1, 1), "device": dev}),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
