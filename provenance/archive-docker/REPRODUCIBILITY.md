# Worker reproducibility

Every worker run must be traceable to the exact code, image and models that produced it.

1. **Commit in the image.** CI and the local build scripts pass `--build-arg SONYA_GIT_SHA=<commit>`. The Dockerfiles store it as `ENV SONYA_GIT_SHA` (`.git` is not in the image). If no build arg is given, it stays `unknown`.
2. **Digest, not tag, for paid runs.** `:fast` and `:latest` move. Resolve them first:

   ```
   python scripts/ghcr_digest.py ghcr.io/samnesvoj/sonya-worker:fast
   # -> ghcr.io/samnesvoj/sonya-worker:fast@sha256:…
   ```

   This is read-only (token + HEAD of the manifest). A private package needs `GHCR_TOKEN` with `read:packages`; visibility is never changed. `VastRunner.run()` refuses an unpinned image unless `allow_unpinned_image=True`. `plan()` shows `image_pinned`.
3. **The instance knows its image.** The runner adds `SONYA_IMAGE=<pinned ref>` to the instance env.
4. **Manifest at worker start.** `worker_entrypoint.sh` runs `python -m scripts.shared.worker_manifest` after preflight. It writes `/tmp/sonya_worker_manifest.json` (or `$SONYA_MANIFEST_FILE`), best-effort and non-fatal. The manifest records:
   - commit and image digest;
   - Python, Torch, CUDA, cuDNN, GPU names;
   - FFmpeg, OpenCV, and whether the Haar cascade is present;
   - versions of key packages, plus a sha256 of every installed `name==version`;
   - stage versions (candidates, ranking, boundaries, crop, subtitles, editor prompt);
   - model file ids (size + sha256 of the first 16 MB);
   - non-secret `SONYA_*` / `WORKER_*` / `APP_*` flags. Secret-looking names are masked, and other keys such as `OPENROUTER_API_KEY` are never read.

Not done yet (needs a paid or real run, or owner approval):
- uploading the manifest with job results in `gpu_worker.py`;
- one real pinned smoke on Vast.

Images built before this change have no `SONYA_GIT_SHA`.
