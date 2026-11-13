"""
The ONE SONYA Gen model and its ComfyUI workflow.

FLUX.1-schnell (Black Forest Labs, Apache-2.0) as GGUF Q4_K_S (city96, Apache-2.0)
+ T5-XXL encoder GGUF Q5_K_M (Apache-2.0) + CLIP-L (MIT, via comfyanonymous,
Apache-2.0 repo) + FLUX VAE ae.safetensors (Apache-2.0; byte-identical to the
black-forest-labs file, sha256 pinned). Fits Apple M4 16 GB. Commercial use allowed.

schnell is guidance-distilled: CFG = 1, so a negative prompt has NO effect on the
image. SONYA keeps it in the metadata and says so (negative_applied: false)
instead of pretending.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Tuple

WORKFLOW_VERSION = "flux-schnell-gguf-v1"
MODEL = {
    "id": "flux1-schnell-q4ks",
    "name": "FLUX.1-schnell (GGUF Q4_K_S)",
    "license": "Apache-2.0",
    "commercial_use": True,
    "files": {
        "unet": ("flux1-schnell-Q4_K_S.gguf", "4fd16477b3a5296d0cf722c4b92a9fd7f30d09ac7495826e4465d8de9c9fd973"),
        "t5": ("t5-v1_1-xxl-encoder-Q5_K_M.gguf", "b51cbb10b1a7aac6dd1c3b62f0ed908bfd06e0b42d2f3577d43e061361f51dae"),
        "clip_l": ("clip_l.safetensors", "660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd"),
        "vae": ("ae.safetensors", "afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38"),
    },
    "sources": {
        "unet": "https://huggingface.co/city96/FLUX.1-schnell-gguf",
        "t5": "https://huggingface.co/city96/t5-v1_1-xxl-encoder-gguf",
        "clip_l": "https://huggingface.co/comfyanonymous/flux_text_encoders",
        "vae": "https://huggingface.co/black-forest-labs/FLUX.1-schnell (mirror: unsloth/FLUX.1-schnell, same sha256)",
    },
    "negative_supported": False,
    "default_steps": 4,
}

# aspect -> (w, h) at ~1 MP, multiples of 16 (FLUX latent /8, patch 2)
ASPECTS: Dict[str, Tuple[int, int]] = {
    "1:1": (1024, 1024), "4:5": (896, 1120), "3:4": (880, 1168), "2:3": (832, 1248),
    "9:16": (768, 1360), "16:9": (1360, 768), "3:2": (1248, 832), "5:4": (1120, 896),
}
SIZES = {"draft": 0.5, "standard": 1.0}          # area scale vs ~1 MP


def dims(aspect: str, size: str = "standard") -> Tuple[int, int]:
    if aspect not in ASPECTS:
        raise ValueError(f"aspect must be one of {sorted(ASPECTS)}")
    if size not in SIZES:
        raise ValueError(f"size must be one of {sorted(SIZES)}")
    w, h = ASPECTS[aspect]
    k = SIZES[size] ** 0.5
    return int(round(w * k / 16)) * 16, int(round(h * k / 16)) * 16


def build(prompt: str, seed: int, width: int, height: int, steps: int, prefix: str) -> Dict[str, Any]:
    """ComfyUI API-format workflow (node id -> {class_type, inputs})."""
    f = MODEL["files"]
    return {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": f["unet"][0]}},
        "2": {"class_type": "DualCLIPLoaderGGUF",
              "inputs": {"clip_name1": f["clip_l"][0], "clip_name2": f["t5"][0], "type": "flux"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": f["vae"][0]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}},   # cfg 1: unused
        "6": {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "7": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0],
                         "seed": int(seed), "steps": int(steps), "cfg": 1.0, "sampler_name": "euler",
                         "scheduler": "simple", "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}},
    }


def workflow_hash(wf: Dict[str, Any]) -> str:
    """Hash of the workflow minus the output prefix (same params -> same hash)."""
    w = json.loads(json.dumps(wf))
    w["9"]["inputs"]["filename_prefix"] = ""
    return hashlib.sha256(json.dumps(w, sort_keys=True).encode()).hexdigest()[:16]
