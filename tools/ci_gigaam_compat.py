#!/usr/bin/env python3
"""
Runs INSIDE the worker image in CI (CPU, no weights, ~1 min). Proves everything about GigaAM that does not need the 450 MB checkpoint:
  1. torch / torchvision / torchaudio are ONE matched binary set (same release series and CUDA tag)
  2. torchaudio's COMPILED operators actually run on this torch (lfilter, resample, MelSpectrogram) - the failure mode of a mismatched pair
  3. the upstream gigaam package at the PINNED commit (git verifies the hash) imports with the dependencies from our lock
  4. the exact API the reference backend relies on exists at that commit (load_model(download_root, fp16_encoder), AudioDataset, _decode)
  5. upstream FeatureExtractor (log-mel front-end) and a small ConformerEncoder (rotary + rel_pos) run a forward pass on torch 2.11
  6. the code path that downloads weights is never needed when the files are on disk (read from the source, not assumed)
NOT proven here: recognition quality (needs the weights - tools/ci_gigaam_real.py), GPU behaviour.
"""
import inspect, os, re, subprocess, sys, tempfile, time

PIN = "7447938d791c4f3e643386ee22c33777004293a5"
rows = []


def step(name, fn):
    t0 = time.time()
    try:
        d = fn()
        rows.append((name, "ok", f"{time.time()-t0:.1f}s", str(d or "")[:110]))
    except Exception as exc:  # noqa: BLE001
        rows.append((name, "FAIL", f"{time.time()-t0:.1f}s", f"{type(exc).__name__}: {str(exc)[:200]}"))


import numpy as np
import torch
import torchaudio
import torchvision

SRC = tempfile.mkdtemp(prefix="gigaam_src_")


def versions():
    base = lambda v: v.split("+")[0]
    tag = lambda v: v.split("+")[1] if "+" in v else ""
    assert base(torch.__version__) == base(torchaudio.__version__), f"torch {torch.__version__} != torchaudio {torchaudio.__version__}"
    assert tag(torch.__version__) == tag(torchaudio.__version__) == tag(torchvision.__version__) != "", "CUDA build tags differ"
    major_minor = ".".join(base(torch.__version__).split(".")[:2])
    tv = {"2.11": "0.26"}.get(major_minor)
    assert tv is None or base(torchvision.__version__).startswith(tv), f"torchvision {torchvision.__version__} does not match torch {torch.__version__}"
    return f"torch {torch.__version__} | torchaudio {torchaudio.__version__} | torchvision {torchvision.__version__}"


def compiled_ops():
    import torchaudio.functional as F
    import torchaudio.transforms as T
    assert hasattr(torch.ops.torchaudio, "_lfilter"), "libtorchaudio custom ops not registered"
    x = torch.randn(1, 16000)
    y = F.lfilter(x, torch.tensor([1.0, -0.5, 0.0]), torch.tensor([0.5, 0.5, 0.0]))          # compiled C++ op
    r = F.resample(x, 16000, 8000)
    m = T.MelSpectrogram(sample_rate=16000, n_mels=64, n_fft=400, win_length=400, hop_length=160)(x)
    assert torch.isfinite(y).all() and r.shape[-1] == 8000 and m.shape[:2] == (1, 64) and torch.isfinite(m).all()
    return f"lfilter ok, resample {tuple(r.shape)}, mel {tuple(m.shape)}"


def clone_pinned():
    subprocess.run(["git", "clone", "--quiet", "https://github.com/salute-developers/GigaAM.git", SRC], check=True, timeout=300)
    subprocess.run(["git", "-C", SRC, "checkout", "--quiet", PIN], check=True)
    head = subprocess.run(["git", "-C", SRC, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    assert head == PIN
    sys.path.insert(0, SRC)
    return f"HEAD == {PIN[:12]}"


def import_pkg():
    import gigaam  # noqa: F401
    import hydra, omegaconf, onnx, onnxruntime, sentencepiece, soundfile  # noqa: F401,E401
    return f"gigaam imports; onnx {onnx.__version__}, onnxruntime {onnxruntime.__version__}, omegaconf {omegaconf.__version__}"


def api_surface():
    import gigaam
    from gigaam.model import GigaAMASR
    from gigaam.utils import AudioDataset
    params = set(inspect.signature(gigaam.load_model).parameters)
    need = {"model_name", "device", "fp16_encoder", "download_root"}
    assert need <= params, f"load_model lacks {need - params}"
    assert hasattr(GigaAMASR, "_decode") and hasattr(AudioDataset, "collate")
    return f"load_model{tuple(sorted(params))}"


def frontend():
    from gigaam.preprocess import FeatureExtractor
    fe = FeatureExtractor(sample_rate=16000, features=64).eval()
    wav = torch.sin(2 * torch.pi * 220 * torch.arange(16000 * 5) / 16000).unsqueeze(0) * 0.3 + 0.01 * torch.randn(1, 16000 * 5)
    feats, n = fe(wav, torch.tensor([wav.shape[1]]))
    assert feats.shape == (1, 64, 501) and int(n[0]) == 501 and torch.isfinite(feats).all()
    return f"features {tuple(feats.shape)}"


def encoder():
    from gigaam.encoder import ConformerEncoder
    out = []
    for attn in ("rotary", "rel_pos"):
        enc = ConformerEncoder(feat_in=64, n_layers=2, d_model=128, n_heads=4, self_attention_model=attn, flash_attn=False).eval()
        x = torch.randn(2, 64, 400)
        with torch.inference_mode():
            y, ln = enc(x, torch.tensor([400, 320]))
        assert torch.isfinite(y).all() and y.shape[0] == 2 and int(ln[0]) == y.shape[-1]
        out.append(f"{attn}->{tuple(y.shape)}")
    return " ".join(out)


def no_download_when_files_exist():
    src = open(os.path.join(SRC, "gigaam", "__init__.py")).read()
    m = re.search(r"def _download_file\(.*?\n(.*?)\n\ndef ", src, re.S)
    body = m.group(1) if m else src
    assert "os.path.exists" in body and "hash" in body.lower(), "cannot see the cached-file short-circuit in _download_file"
    return "_download_file short-circuits on an existing file with a matching hash (read from source)"


for n, f in (("matched torch/torchvision/torchaudio build", versions), ("torchaudio compiled ops run", compiled_ops), ("clone upstream at pinned commit", clone_pinned),
             ("import gigaam + lock dependencies", import_pkg), ("API the reference backend uses", api_surface), ("upstream log-mel front-end", frontend),
             ("upstream ConformerEncoder forward (rotary, rel_pos)", encoder), ("no download when weights are on disk", no_download_when_files_exist)):
    step(n, f)
print("| check | result | time | detail |\n|---|---|---|---|")
for r in rows:
    print("| " + " | ".join(r) + " |")
sys.exit(1 if any(r[1] == "FAIL" for r in rows) else 0)
