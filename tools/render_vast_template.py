#!/usr/bin/env python3
"""
Prints (does NOT execute) the exact `vastai create template` command and the Template JSON for a given image digest and release SHA.
Nothing secret is ever put in a Template: the dispatcher adds GITHUB_RELEASE_TOKEN, WORKER_SECRET, S3_* at instance creation.
Usage: python tools/render_vast_template.py --image ghcr.io/OWNER/REPO-base@sha256:<64 hex> --sha <40 hex>
"""
import argparse, json, re, shlex, sys

DIGEST = re.compile(r"^[a-z0-9./_-]+@sha256:[0-9a-f]{64}$")
SHA = re.compile(r"^[0-9a-f]{40}$")
REPO = "samnesvoj/sonya-video-worker-production"


def render(image: str, sha: str) -> dict:
    if not DIGEST.match(image):
        raise SystemExit("image must be pinned by @sha256 digest (no tags)")
    if not SHA.match(sha):
        raise SystemExit("sha must be a full 40-hex commit SHA (no branches)")
    env = {"SONYA_RELEASE_SHA": sha, "SONYA_RELEASE_REPO": REPO, "WORKER_BACKEND_MODE": "api", "SONYA_MIN_VRAM_MB": "20000",
           "SONYA_REQUIRE_MODEL_LOCK": "1", "ENABLED_MODES": "virality,beep"}
    docker_opts = " ".join(f"-e {k}={v}" for k, v in env.items())
    cmd = ["vastai", "create", "template", "--name", f"sonya-video-worker-{sha[:8]}", "--image", image.split("@")[0], "--image_tag", "@" + image.split("@")[1],
           "--env", docker_opts, "--disk_space", "80", "--desc", f"SONYA video worker, pinned release {sha[:12]} (no secrets)", "--no-default"]
    return {"env": env, "disk_gb": 80, "ports": [], "launch_mode": "docker entrypoint (no ssh/jupyter, no onstart)", "command": " ".join(shlex.quote(c) for c in cmd),
            "note": "image_tag form for a digest must be confirmed on the first real Template creation; the dispatcher does not depend on the Template (it creates instances from image+env directly)"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--image", required=True); ap.add_argument("--sha", required=True)
    a = ap.parse_args()
    print(json.dumps(render(a.image, a.sha), indent=2))
