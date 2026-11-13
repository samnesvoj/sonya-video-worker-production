import re,sys
p='/opt/sonya/scripts/prod_generation_api.py'; s=open(p).read()
assert 'GEN_IMAGE_ASPECTS' not in s

# 1) generic upload endpoint: sonya_gen goes only through /api/gen/image
s=s.replace('''    # Mode validation
    if mode not in ALLOWED_MODES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "invalid_mode", "allowed": sorted(ALLOWED_MODES), "trace_id": trace_id},
        )

    # Feature flag: Trailer''','''    # Mode validation
    if mode not in ALLOWED_MODES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "invalid_mode", "allowed": sorted(ALLOWED_MODES), "trace_id": trace_id},
        )

    # SONYA Gen (images) has no upload and is admin-only for now: use POST /api/gen/image.
    if mode == "sonya_gen":
        _require_gen_admin(user, trace_id)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail={"error": "use_gen_endpoint", "message": "POST /api/gen/image", "trace_id": trace_id})

    # Feature flag: Trailer''',1)

# 2) result-url: images too
s=s.replace('''and str(f.get("s3_key") or "").lower().endswith((".mp4", ".mov", ".webm"))]
        for n, f in enumerate(sorted(outs, key=lambda f: str(f.get("s3_key"))), 1):
            clips.append({"url": generate_presigned_get_url(f["s3_key"], expires_in=expires),
                          "download_url": generate_presigned_get_url(f["s3_key"], expires_in=expires,
                                                                     download_name="sonya_clip_%d.mp4" % n)})''','''and str(f.get("s3_key") or "").lower().endswith((".mp4", ".mov", ".webm", ".png", ".jpg"))]
        for n, f in enumerate(sorted(outs, key=lambda f: str(f.get("s3_key"))), 1):
            _ext = str(f["s3_key"]).lower().rsplit(".", 1)[-1]
            _nm = ("sonya_gen_%d.%s" % (n, _ext)) if _ext in ("png", "jpg") else ("sonya_clip_%d.mp4" % n)
            clips.append({"url": generate_presigned_get_url(f["s3_key"], expires_in=expires),
                          "download_url": generate_presigned_get_url(f["s3_key"], expires_in=expires,
                                                                     download_name=_nm)})''',1)

# 3) admin check + endpoints, before "# ── Public: Job creation"
new='''# ── SONYA Gen (images): ADMIN ONLY for now (no tariff has it) ─────────────────────
# Admin = the existing mechanism: users.plan_type == 'admin' (same as auth.js / _PLAN_PRIORITY). No credits, no quota, no billing tables touched.
GEN_IMAGE_ASPECTS = ("1:1", "4:5", "3:4", "2:3", "9:16", "16:9", "3:2", "5:4")


def _is_gen_admin(user: dict) -> bool:
    return str(user.get("plan_type") or "").lower().strip() == "admin"


def _require_gen_admin(user: dict, trace_id: str) -> None:
    if not _is_gen_admin(user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail={"error": "coming_soon", "message": "скоро", "trace_id": trace_id})


class GenImageRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000)
    aspect: str = "1:1"
    size: str = "standard"
    count: int = Field(1, ge=1, le=4)
    seed: Optional[int] = Field(None, ge=0, le=2**31 - 1)


@app.get("/api/gen/access")
async def gen_access(user: dict = Depends(get_current_user)):
    """Frontend: is the SONYA Gen (images) tab live for this user? Only {"image": bool}."""
    return {"image": _is_gen_admin(user)}


@app.post("/api/gen/image", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_gen_image_job(
    body: GenImageRequest,
    request: Request,
    user: dict = Depends(get_current_user),
    _origin: None = Depends(verify_browser_origin),
    _rl: None = Depends(_upload_limiter),
):
    trace_id = new_trace_id()
    _require_gen_admin(user, trace_id)
    if body.aspect not in GEN_IMAGE_ASPECTS or body.size not in ("draft", "standard"):
        raise HTTPException(status_code=400, detail={"error": "invalid_params", "trace_id": trace_id})
    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail={"error": "invalid_params", "trace_id": trace_id})
    user_id = str(user["id"])
    job_id = str(uuid.uuid4())
    params = {"prompt": prompt, "aspect": body.aspect, "size": body.size, "count": body.count}
    if body.seed is not None:
        params["seed"] = body.seed
    try:
        result = create_job_with_quota(
            job_id=job_id, user_id=user_id, mode="sonya_gen", params=params,
            s3_input_key=f"gen/{user_id}/{job_id}/none",     # text -> image: no input object exists
            idempotency_key=None, idempotency_fingerprint=None,
            queue_priority=_PLAN_PRIORITY["admin"],
            bypass_quota=True,                                # admin: no credits, nothing debited
            subscription_id=None,
        )
    except Exception as exc:
        logger.error("[api] gen_create_failed trace_id=%s: %s", trace_id, exc)
        raise safe_error("db_error", 500, trace_id)
    logger.info("[api] gen_job_created job_id=%s user_id=%s count=%d", job_id, user_id, body.count)
    job = get_job(job_id)
    return JobResponse(job_id=job_id, status=JOB_STATUS_QUEUED, mode="sonya_gen",
                       created_at=str(job.get("created_at")) if job else None)


# ── Public: Job creation ————'''
marker='# ── Public: Job creation ————'
assert s.count(marker)==1, s.count(marker)
s=s.replace(marker,new,1)
open(p,'w').write(s)
