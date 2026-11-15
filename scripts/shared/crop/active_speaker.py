"""
Audio-visual active speaker detection for the fill_9x16 crop (owner 2026-11-15: «показывает того, кто слушает»).

Light-ASD (Liao et al., CVPR 2023, https://github.com/Junhua-Liao/Light-ASD, MIT — see models/LICENSE-light-asd):
a 1 M-parameter network that looks at a face's lower half (25 fps, 112×112 gray) together with the audio
(MFCC, 100 fps) and says, per video frame, whether THIS face is the one producing the voice. Unlike plain
mouth motion it is not fooled by a listener who smiles/nods, and it handles profiles and sunglasses.

score_tracks(video, start, s0, s1, tracks) → (times, [scores per track]); a score > 0 means «talking».
Everything here is optional: no torch / no weights / any error → None, and the crop uses its old cue.
Weights: models/light_asd_talkset.model (finetuning_TalkSet.model of the repo, the «in the wild» set), sha256-pinned,
loaded with weights_only=True (no pickle code).
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

ASD_VERSION = "light-asd-talkset-v1"
WEIGHTS = Path(__file__).resolve().parent / "models" / "light_asd_talkset.model"
WEIGHTS_SHA256 = "efc375833887eefa9d209dc92810e18519b04c3c73ea35a549f2a7f40b7d94d5"
FPS = 25
DURATIONS = (1, 2, 3, 4, 6)        # chunk lengths (s) whose scores are averaged (the authors' recipe, trimmed)
CROP_SCALE = 0.4                   # the authors' face crop: box size × (1 + 2·0.4), shifted towards the chin
MAX_GAP_SEC = 0.5                  # a face not detected for longer than this → no score there

_model = None
_model_err: Optional[str] = None
_lock = threading.Lock()          # clips are rendered in worker threads


# ── model (vendored from Light-ASD/model, inference only) ────────────────────────────────────────────

def _build():
    import torch
    from torch import nn

    class AudioBlock(nn.Module):
        def __init__(self, i, o):
            super().__init__()
            self.relu = nn.ReLU()
            bn = lambda c: nn.BatchNorm2d(c, momentum=0.01, eps=0.001)     # noqa: E731
            self.m_3, self.bn_m_3 = nn.Conv2d(i, o, (3, 1), padding=(1, 0), bias=False), bn(o)
            self.t_3, self.bn_t_3 = nn.Conv2d(o, o, (1, 3), padding=(0, 1), bias=False), bn(o)
            self.m_5, self.bn_m_5 = nn.Conv2d(i, o, (5, 1), padding=(2, 0), bias=False), bn(o)
            self.t_5, self.bn_t_5 = nn.Conv2d(o, o, (1, 5), padding=(0, 2), bias=False), bn(o)
            self.last, self.bn_last = nn.Conv2d(o, o, 1, bias=False), bn(o)

        def forward(self, x):
            a = self.relu(self.bn_t_3(self.t_3(self.relu(self.bn_m_3(self.m_3(x))))))
            b = self.relu(self.bn_t_5(self.t_5(self.relu(self.bn_m_5(self.m_5(x))))))
            return self.relu(self.bn_last(self.last(a + b)))

    class VisualBlock(nn.Module):
        def __init__(self, i, o, down=False):
            super().__init__()
            self.relu = nn.ReLU()
            st = (1, 2, 2) if down else (1, 1, 1)
            bn = lambda c: nn.BatchNorm3d(c, momentum=0.01, eps=0.001)     # noqa: E731
            self.s_3, self.bn_s_3 = nn.Conv3d(i, o, (1, 3, 3), stride=st, padding=(0, 1, 1), bias=False), bn(o)
            self.t_3, self.bn_t_3 = nn.Conv3d(o, o, (3, 1, 1), padding=(1, 0, 0), bias=False), bn(o)
            self.s_5, self.bn_s_5 = nn.Conv3d(i, o, (1, 5, 5), stride=st, padding=(0, 2, 2), bias=False), bn(o)
            self.t_5, self.bn_t_5 = nn.Conv3d(o, o, (5, 1, 1), padding=(2, 0, 0), bias=False), bn(o)
            self.last, self.bn_last = nn.Conv3d(o, o, 1, bias=False), bn(o)

        def forward(self, x):
            a = self.relu(self.bn_t_3(self.t_3(self.relu(self.bn_s_3(self.s_3(x))))))
            b = self.relu(self.bn_t_5(self.t_5(self.relu(self.bn_s_5(self.s_5(x))))))
            return self.relu(self.bn_last(self.last(a + b)))

    class VisualEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.block1 = VisualBlock(1, 32, True)
            self.pool1 = nn.MaxPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1))
            self.block2 = VisualBlock(32, 64)
            self.pool2 = nn.MaxPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1))
            self.block3 = VisualBlock(64, 128)
            self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        def forward(self, x):
            x = self.block3(self.pool2(self.block2(self.pool1(self.block1(x)))))
            x = x.transpose(1, 2)
            B, T, C, W, H = x.shape
            return self.maxpool(x.reshape(B * T, C, W, H)).view(B, T, C)

    class AudioEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.block1 = AudioBlock(1, 32)
            self.pool1 = nn.MaxPool3d((1, 1, 3), (1, 1, 2), (0, 0, 1))
            self.block2 = AudioBlock(32, 64)
            self.pool2 = nn.MaxPool3d((1, 1, 3), (1, 1, 2), (0, 0, 1))
            self.block3 = AudioBlock(64, 128)

        def forward(self, x):
            x = self.block3(self.pool2(self.block2(self.pool1(self.block1(x)))))
            return torch.mean(x, dim=2, keepdim=True).squeeze(2).transpose(1, 2)

    class BGRU(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.gru_forward = nn.GRU(c, c, 1, bias=True, batch_first=True)
            self.gru_backward = nn.GRU(c, c, 1, bias=True, batch_first=True)
            self.gelu = nn.GELU()

        def forward(self, x):
            x = self.gelu(self.gru_forward(x)[0])
            x = torch.flip(self.gru_backward(torch.flip(x, dims=[1]))[0], dims=[1])
            return self.gelu(x)

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.visualEncoder, self.audioEncoder, self.GRU = VisualEncoder(), AudioEncoder(), BGRU(128)

    class FC(nn.Module):
        def __init__(self):
            super().__init__()
            self.FC = nn.Linear(128, 2)

    class ASD(nn.Module):
        def __init__(self):
            super().__init__()
            self.model, self.lossAV = Inner(), FC()

        def embed(self, audio, video):          # audio (1, 4T, 13) mfcc, video (1, T, 112, 112) gray 0..255
            m = self.model
            a = m.audioEncoder(audio.unsqueeze(1).transpose(2, 3))
            v = m.visualEncoder(((video / 255.0 - 0.4161) / 0.1688).unsqueeze(1))
            return a + v

        def head(self, x):                       # fused embedding chunk (1, T, 128) → logit of «this face talks»
            return self.lossAV.FC(self.model.GRU(x).reshape(-1, 128))[:, 1]

        def forward(self, audio, video):
            return self.head(self.embed(audio, video))

    return ASD


def load() -> Tuple[Any, Any]:
    """(model, device) or (None, None); cached. SONYA_ASD=0 turns it off."""
    with _lock:
        return _load()


def _load() -> Tuple[Any, Any]:
    global _model, _model_err
    if _model is not None or _model_err is not None:
        return _model if _model is not None else (None, None)
    try:
        if os.environ.get("SONYA_ASD", "1") == "0":
            raise RuntimeError("disabled by SONYA_ASD=0")
        import torch
        if not WEIGHTS.is_file():
            raise RuntimeError(f"weights missing: {WEIGHTS.name}")
        if hashlib.sha256(WEIGHTS.read_bytes()).hexdigest() != WEIGHTS_SHA256:
            raise RuntimeError("weights sha256 mismatch")
        sd = torch.load(str(WEIGHTS), map_location="cpu", weights_only=True)
        sd = {k: v for k, v in sd.items() if not k.startswith("lossV.")}
        net = _build()()
        net.load_state_dict(sd, strict=True)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        net = net.to(dev).eval()
        _model = (net, dev)
        return _model
    except Exception as exc:                 # noqa: BLE001 — optional component
        _model_err = f"{type(exc).__name__}: {exc}"[:200]
        logger.warning("[crop] active speaker model unavailable (%s) — mouth-motion cue", _model_err)
        return None, None


def status() -> Dict[str, Any]:
    net, dev = load()
    return {"ok": net is not None, "version": ASD_VERSION, "device": dev, "error": _model_err}


# ── features ─────────────────────────────────────────────────────────────────────────────────────────

def mfcc(sig: np.ndarray, sr: int = 16000) -> np.ndarray:
    """python_speech_features.mfcc(sig, sr, numcep=13, winlen=0.025, winstep=0.01) — the features the model
    was trained on (rectangular window, 26 mel filters, NFFT 512, preemph 0.97, lifter 22, log-energy in c0)."""
    sig = np.asarray(sig, dtype=np.float64)
    if sig.size == 0:
        return np.zeros((0, 13))
    sig = np.append(sig[0], sig[1:] - 0.97 * sig[:-1])
    flen, step, nfft, nfilt = int(round(0.025 * sr)), int(round(0.01 * sr)), 512, 26
    n = 1 if len(sig) <= flen else 1 + int(math.ceil((len(sig) - flen) / step))
    sig = np.concatenate([sig, np.zeros((n - 1) * step + flen - len(sig))])
    idx = np.arange(flen)[None, :] + step * np.arange(n)[:, None]
    frames = sig[idx]
    pspec = (np.abs(np.fft.rfft(frames, nfft)) ** 2) / nfft
    energy = pspec.sum(1)
    energy = np.where(energy == 0, np.finfo(float).eps, energy)
    hz2mel = lambda h: 2595 * np.log10(1 + h / 700.0)                     # noqa: E731
    mel2hz = lambda m: 700 * (10 ** (m / 2595.0) - 1)                      # noqa: E731
    pts = np.linspace(hz2mel(0), hz2mel(sr / 2), nfilt + 2)
    bins = np.floor((nfft + 1) * mel2hz(pts) / sr)
    fb = np.zeros((nfilt, nfft // 2 + 1))
    for j in range(nfilt):
        a, b, c = int(bins[j]), int(bins[j + 1]), int(bins[j + 2])
        for i in range(a, b):
            fb[j, i] = (i - bins[j]) / (bins[j + 1] - bins[j])
        for i in range(b, c):
            fb[j, i] = (bins[j + 2] - i) / (bins[j + 2] - bins[j + 1])
    feat = pspec @ fb.T
    feat = np.log(np.where(feat == 0, np.finfo(float).eps, feat))
    k = np.arange(nfilt)
    dct = np.cos(np.pi * k[:13, None] * (2 * k[None, :] + 1) / (2 * nfilt)) * np.sqrt(2.0 / nfilt)
    dct[0] *= np.sqrt(0.5)
    feat = feat @ dct.T
    feat *= 1 + 11.0 * np.sin(np.pi * np.arange(13) / 22.0)
    feat[:, 0] = np.log(energy)
    return feat


def _audio(video: str, t0: float, dur: float) -> np.ndarray:
    from .smart_crop_adapter import ffmpeg_bin
    cmd = [ffmpeg_bin(), "-v", "error", "-ss", f"{max(0.0, t0):.3f}", "-t", f"{dur:.3f}", "-i", video,
           "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "pipe:1"]
    out = subprocess.run(cmd, capture_output=True, timeout=120, check=True).stdout
    return np.frombuffer(out, dtype=np.int16).astype(np.float64)


def _interp(boxes: Sequence[Tuple[float, Sequence[float]]], ts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Face boxes (10 fps samples) → per-frame (cx, cy, half size) at `ts` + mask of frames near a detection."""
    bt = np.array([t for t, _ in boxes], dtype=float)
    bx = np.array([[b[0] + b[2] / 2.0, b[1] + b[3] / 2.0, max(b[2], b[3]) / 2.0] for _, b in boxes], dtype=float)
    o = np.argsort(bt)
    bt, bx = bt[o], bx[o]
    # median filter over ~0.5 s of samples (the authors smooth the crop the same way)
    k = 2
    sm = np.array([np.median(bx[max(0, i - k):i + k + 1], axis=0) for i in range(len(bx))])
    out = np.stack([np.interp(ts, bt, sm[:, j]) for j in range(3)], axis=1)
    pos = np.searchsorted(bt, ts)
    near = np.minimum(np.abs(ts - bt[np.clip(pos - 1, 0, len(bt) - 1)]), np.abs(ts - bt[np.clip(pos, 0, len(bt) - 1)]))
    return out, near <= MAX_GAP_SEC / 2.0 + 1e-6


def _crop(gray: np.ndarray, cx: float, cy: float, bs: float) -> np.ndarray:
    """The authors' crop (pad 110, rows cy-bs … cy+bs·1.8, cols cx±bs·1.4 → 224², centre 112²), done as one resize."""
    import cv2
    bs = max(bs, 4.0)
    cs = CROP_SCALE
    # full crop is [cy - bs, cy + bs(1+2cs)] × [cx - bs(1+cs), cx + bs(1+cs)] (side 2.8·bs); its centre half:
    side = bs * (2 + 2 * cs)
    fy0, fx0 = cy - bs, cx - bs * (1 + cs)
    y0, x0 = fy0 + side / 4.0, fx0 + side / 4.0
    y1, x1 = y0 + side / 2.0, x0 + side / 2.0
    H, W = gray.shape
    iy0, ix0, iy1, ix1 = int(math.floor(y0)), int(math.floor(x0)), int(math.ceil(y1)), int(math.ceil(x1))
    pad = max(0, -iy0, -ix0, iy1 - H, ix1 - W)
    if pad:
        gray = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=110)
        iy0, ix0, iy1, ix1 = iy0 + pad, ix0 + pad, iy1 + pad, ix1 + pad
    return cv2.resize(gray[iy0:iy1, ix0:ix1], (112, 112), interpolation=cv2.INTER_AREA)


def score_tracks(video: str, start: float, s0: float, s1: float,
                 tracks: List[Sequence[Tuple[float, Sequence[float]]]]) -> Optional[Tuple[np.ndarray, List[np.ndarray]]]:
    """Per 25 fps frame of [s0, s1) (clip time, the clip starts at `start` in the source): one score per track
    (NaN where the face is not seen). None when the model is unavailable or anything fails."""
    net, dev = load()
    if net is None or s1 - s0 < 0.5 or not tracks:
        return None
    try:
        import cv2
        import torch
        n = int((s1 - s0) * FPS)
        ts = s0 + np.arange(n) / FPS
        geo = [_interp(tr, ts) for tr in tracks]
        crops = [np.zeros((n, 112, 112), dtype=np.uint8) for _ in tracks]
        cap = cv2.VideoCapture(video)
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, start + s0 - 0.2) * 1000.0)
        k, cur = 0, None
        try:
            while k < n:
                ok, frame = cap.read()
                if not ok:
                    break
                t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 - start
                g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # every target frame up to this decoded frame's time uses the latest frame ≤ it
                while k < n and ts[k] < t - 1e-6 and cur is not None:
                    for j, (bx, _) in enumerate(geo):
                        crops[j][k] = _crop(cur, *bx[k])
                    k += 1
                cur = g
        finally:
            cap.release()
        while k < n and cur is not None:          # tail of the range (last decoded frame)
            for j, (bx, _) in enumerate(geo):
                crops[j][k] = _crop(cur, *bx[k])
            k += 1
        if k < n:
            n = k
            ts = ts[:n]
            crops = [c[:n] for c in crops]
        if n < FPS // 2:
            return None
        aud = mfcc(_audio(video, start + s0, n / FPS + 0.05))
        length = min((aud.shape[0] - aud.shape[0] % 4) / 100.0, n / FPS)
        na, nv = int(round(length * 100)), int(round(length * FPS))
        aud = aud[:na]
        out: List[np.ndarray] = []
        with torch.no_grad():
            A = torch.from_numpy(aud.astype(np.float32)).to(dev)
            for j, c in enumerate(crops):
                V = torch.from_numpy(c[:nv].astype(np.float32)).to(dev)
                m = min(na // 4, nv)
                # the front-ends are local (temporal kernels ≤ 5) → run once; only the GRU sees the chunks
                x = net.embed(A[:4 * m].unsqueeze(0), V[:m].unsqueeze(0))
                allsc = []
                for d in DURATIONS:
                    sc = torch.cat([net.head(x[:, i:i + d * FPS]) for i in range(0, m, d * FPS)])
                    sc = sc.float().cpu().numpy()
                    allsc.append(np.concatenate([sc, np.full(nv - len(sc), np.nan)]))
                s = np.nanmean(np.stack(allsc), axis=0)
                full = np.full(n, np.nan)
                full[:nv] = s
                full[~geo[j][1][:n]] = np.nan
                out.append(full)
        return ts, out
    except Exception as exc:                 # noqa: BLE001 — never fail a render because of the speaker cue
        logger.warning("[crop] active speaker scoring failed: %s", str(exc)[:200])
        return None
