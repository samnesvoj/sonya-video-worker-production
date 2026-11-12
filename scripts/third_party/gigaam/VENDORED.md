# Vendored: GigaAM (salute-developers/GigaAM)

- Upstream: https://github.com/salute-developers/GigaAM
- Commit: `7447938d791c4f3e643386ee22c33777004293a5` (package `gigaam` 0.2.0)
- License: MIT (`LICENSE` in this folder, unchanged)
- Copied **as is** (no edits) from the installed package that produced the Mac mini reference
  transcripts (`sonya-desktop/.venv-desktop`, same files byte for byte).
- Why vendored: the worker image installs with `pip --require-hashes`, which does not accept git
  dependencies; the SONYA backend also uses private upstream APIs (`AudioDataset`, `model._decode`),
  so the commit must stay pinned. See `docs/ASR_INTERFACE.md`.
- Runtime dependencies (must be in the image lock): `torch>=2.6`, `torchaudio>=2.6`, `hydra-core==1.3.*`,
  `omegaconf==2.3.*`, `sentencepiece`, `soundfile`, `tqdm`, `numpy==2.*` (+ `onnx`, `onnxruntime` as
  declared by upstream; not used for inference).
- Weights are NOT downloaded by SONYA: `scripts/shared/asr/gigaam_backend.py` passes
  `download_root=$MODELS_LOCAL_DIR/gigaam` and refuses to start if a file is missing, so the
  upstream download code in `__init__.py` is never reached.

SHA-256 of the vendored files (to compare with a fresh checkout of the commit):
```
9d772461e35b9b3caf2f94fd5343261f732395c409f7a7957fa2bf03704b5a5c  __init__.py
fe6b82f8ea48e5a4628414601a160cb62398216ed3d9a9703bb06c0d721795a4  decoder.py
0f77d20748cf27a4fb5f60bd8d0b30a219fd40faf19144788b09d9b76c32bd84  decoding.py
1eac8600faf1b842899c0944b8ebb9d60b41dd6f35d826c41514765dbf18cd3d  encoder.py
38b79ada59e2a8764f3bf64fe10cd44198ab95a2168915f4c521416ea6331236  model.py
5830d6d5bfc34675b9a97a9ee90ed15c63d4996c43bf352167f78c1ea74acd51  onnx_utils.py
d4cd47b7c07664c0aab682148155a1fb2eab4829a91730cb14b1c75dddfcc22f  preprocess.py
9c2f590107508c36c11ae98d45c1a4f757decd7e19c0329509f29a7e6c527c70  timestamps_utils.py
ea6e9f83993f927fce7f73fd7d97fb78d0735ef6429954f7d1502b7a051af765  types.py
a6366d63c304b162202bf97a3c2da05a453741c475ab3bf160fc0428957282f9  utils.py
b5fdb0589eceda795161bee9721b841d41354608057e4a2718fe2debb513cbf5  vad_utils.py
f00de6715714c7a63d08639cdbfaa40224eefc407302614bd19f1a8b98c875aa  LICENSE
```
