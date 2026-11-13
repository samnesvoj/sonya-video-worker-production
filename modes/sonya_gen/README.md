# sonya_gen (images)

Text -> image on the Vast GPU worker. Model: FLUX.1-schnell Q4_K_S (Apache-2.0). Runner starts a private ComfyUI
(pinned release, installed once into a venv under `$SONYA_COMFY_DIR`, default `/work/comfy`), runs the same workflow as the Desktop Gen,
writes `gen_<n>.png` + `gen_meta.json`, stops ComfyUI.

params: `prompt` (required, <=1000 chars), `aspect` (1:1 4:5 3:4 2:3 9:16 16:9 3:2 5:4, default 1:1), `size` (draft|standard),
`seed`, `count` (1..4), `steps` (1..8, default 4). schnell is CFG=1: a negative prompt has no effect.
Access is restricted to admins on the API side; the worker itself does no billing.
Gen Video is not part of this mode.
