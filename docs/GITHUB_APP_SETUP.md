# GitHub App: temporary access, no permanent keys on Vast
1. Repo `samnesvoj/sonya-video-worker-production` stays PRIVATE.
2. GitHub → Settings → Developer settings → New GitHub App `sonya-release-reader`; webhook off; **Repository → Contents: Read-only** only; this account only.
3. Install on the single repo; note App ID and Installation ID. Generate the private key (.pem): keep it ONLY on the VPS/Mac (chmod 600), never on Vast, in an image, Template, CI log or Git.
4. Per instance, on the VPS: `GH_APP_ID=… GH_APP_INSTALLATION_ID=… GH_APP_PRIVATE_KEY_FILE=… python tools/mint_release_token.py` → put the output into that instance's `GITHUB_RELEASE_TOKEN` (expires in 1 h; contents:read; one repo).
5. Bootstrap passes it to git through the child's environment only, then unsets it; no `.git`, no credentials remain on disk.
6. Rulesets: protect `main` (PR + required check `ci`) and tags `worker-v*` (no rewrite).
Residual risk: token readable by the host owner for ≤1 h. Better: a token-broker on the VPS (not done; needs VPS change).
