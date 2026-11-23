/**
 * SONYA Auth layer
 * – API client (credentials:include for HttpOnly cookies)
 * – Auth state (refreshed from /api/auth/me, never from localStorage)
 * – Account button + dropdown
 * – Auth modal (login / registration with 3 consents) — Glass UI
 * – Paywall modal (FREE_PLAN_USED stub)
 * – Legal placeholder modals
 */

/* ─────────────────────────────────────────────
   CONFIG
───────────────────────────────────────────── */
// In production point this to your API domain.
// Empty string = same origin (works when backend serves the frontend too).
const SONYA_API_BASE = window.SONYA_API_BASE || '';

/* ─────────────────────────────────────────────
   AUTH STATE  (single source of truth)
   Never write subscription / limits here from
   frontend code — always refresh from /api/me.
───────────────────────────────────────────── */
const authState = {
  initialized: false,
  user: null,       // null = not logged in
  // user = { user_id, email, plan_type, plan_status, plan_active_until,
  //          free_video_limit, free_video_used, telegram_linked }
};

/* ─────────────────────────────────────────────
   API CLIENT
───────────────────────────────────────────── */
const DEFAULT_NETWORK_ERROR_MESSAGES = {
  timeout: 'Превышено время ожидания ответа сервера. Проверьте соединение и попробуйте снова.',
  aborted: 'Запрос отменён.',
  network: 'Сервер недоступен. Проверьте соединение.',
};

function _classifyFetchError(err, { timedOut }) {
  if (err && err.name === 'AbortError') return timedOut ? 'timeout' : 'aborted';
  return 'network';
}

async function apiFetch(path, options = {}) {
  // timeoutMs: opt-in per call -- omitting it preserves the exact previous
  // behavior (no deadline at all). timeoutMessage: only used for the
  // 'timeout' classification, lets a caller give a more specific message
  // than the generic default (see apiCreateVideoJob below). signal: an
  // optional caller-supplied AbortSignal, composed with our own timeout
  // controller so either one can cancel the request.
  const { timeoutMs, timeoutMessage, signal: callerSignal, ...fetchOptions } = options;
  const url = SONYA_API_BASE + path;
  const defaults = {
    credentials: 'include',  // send HttpOnly session cookie
    headers: { 'Content-Type': 'application/json' },
  };

  const controller = new AbortController();
  let timedOut = false;
  let timer = null;
  if (timeoutMs) {
    timer = setTimeout(() => { timedOut = true; controller.abort(); }, timeoutMs);
  }
  let onCallerAbort = null;
  if (callerSignal) {
    if (callerSignal.aborted) controller.abort();
    else {
      onCallerAbort = () => controller.abort();
      callerSignal.addEventListener('abort', onCallerAbort, { once: true });
    }
  }

  const startedAt = Date.now();
  try {
    const headers = { ...defaults.headers, ...(fetchOptions.headers || {}) };
    if (typeof FormData !== 'undefined' && fetchOptions.body instanceof FormData) {
      delete headers['Content-Type'];
    }
    const res = await fetch(url, { ...defaults, ...fetchOptions, headers, signal: controller.signal });
    return res;
  } catch (e) {
    const kind = _classifyFetchError(e, { timedOut });
    const elapsedMs = Date.now() - startedAt;
    // Diagnostic only: path + classification + timing -- never headers,
    // cookies, the request body, or any token/secret.
    console.error(`[SONYA API] ${kind} error path=${path} elapsedMs=${elapsedMs}` +
      (timeoutMs ? ` timeoutMs=${timeoutMs}` : ''));
    if (typeof window !== 'undefined') {
      window.SONYA_LAST_NETWORK_ERROR = { path, kind, timeoutMs: timeoutMs || null, elapsedMs, timestamp: Date.now() };
    }
    const detail = (kind === 'timeout' && timeoutMessage) || DEFAULT_NETWORK_ERROR_MESSAGES[kind];
    return { ok: false, status: 0, _networkError: true, _networkErrorKind: kind,
      json: async () => ({ detail }) };
  } finally {
    if (timer) clearTimeout(timer);
    if (callerSignal && onCallerAbort) callerSignal.removeEventListener('abort', onCallerAbort);
  }
}

// Timeout model constants -- see docs/patches/generation-upload-timeout-plan.md §2
// for the full rationale and the explicit large-file/slow-connection trade-off.
const UPLOAD_BASE_TIMEOUT_MS = 20_000;
const UPLOAD_MIN_THROUGHPUT_BYTES_PER_SEC = 500 * 1024; // ~4 Mbps floor
const UPLOAD_MAX_TIMEOUT_MS = 20 * 60 * 1000; // matches apiCreateVideoJobFromUrl's existing ~20min ceiling
const AUTH_CHECK_TIMEOUT_MS = 15_000;

function computeUploadTimeoutMs(fileSizeBytes) {
  const size = Math.max(0, Number(fileSizeBytes) || 0);
  const sizeBudgetMs = (size / UPLOAD_MIN_THROUGHPUT_BYTES_PER_SEC) * 1000;
  return Math.min(UPLOAD_BASE_TIMEOUT_MS + sizeBudgetMs, UPLOAD_MAX_TIMEOUT_MS);
}

async function apiGetMe() {
  return apiFetch('/auth/me', { timeoutMs: AUTH_CHECK_TIMEOUT_MS });
}

// Single funnel for "a request that was NOT the expected 401-guest-state
// failed" -- 5xx / network errors on auth checks must be visible in
// diagnostics, never silently treated the same as an anonymous visitor.
function reportAuthError(context, res) {
  console.error(`[SONYA Auth] ${context}: unexpected status ${res?.status}`, res);
}


function normalizeApiAuthPurpose(purpose) {
  return purpose === 'registration' ? 'register' : purpose;
}

function normalizeApiConsents(consents) {
  if (!consents) {
    return {};
  }
  if (Array.isArray(consents)) {
    return {
      terms: consents.includes('terms'),
      privacy: consents.includes('privacy'),
      personal_data: consents.includes('personal_data') || consents.includes('personalData'),
    };
  }
  return consents;
}

async function apiRequestCode(email, purpose) {
  return apiFetch('/auth/request-code', {
    method: 'POST',
    body: JSON.stringify({ email, purpose: normalizeApiAuthPurpose(purpose) }),
  });
}

async function apiVerifyCode(payload) {
  // payload: { email, code, purpose, consents? }
  return apiFetch('/auth/verify-code', {
    method: 'POST',
    body: JSON.stringify({ ...payload, purpose: normalizeApiAuthPurpose(payload.purpose), consents: normalizeApiConsents(payload.consents) }),
  });
}

async function apiLogout() {
  return apiFetch('/auth/logout', { method: 'POST' });
}


/* ─────────────────────────────────────────────
   IDEMPOTENCY KEY  (POST /api/generation/jobs)
   One key per logical submission attempt, generated once via
   crypto.randomUUID() the first time apiCreateVideoJob() actually sends
   it. Kept across a network-error retry of the SAME attempt (the whole
   point of an idempotency key -- avoid a duplicate job if the client
   doesn't know whether the first request reached the server). Cleared
   after any real HTTP response (success or a definitive rejection like
   401/402/400) or an explicit form reset, so the next click is always a
   fresh "new run" with its own key.
───────────────────────────────────────────── */
let _jobIdempotencyKey = null;

function _getOrCreateJobIdempotencyKey() {
  if (!_jobIdempotencyKey) {
    _jobIdempotencyKey = crypto.randomUUID();
  }
  return _jobIdempotencyKey;
}

function clearJobIdempotencyKey() {
  _jobIdempotencyKey = null;
}

/* Trailer feature flag. The single switch lives in app-shell.js
   (window.SONYA_FEATURES.trailer); anything other than an explicit `true`
   — including app-shell.js not having loaded — counts as OFF (fail-closed).
   The backend enforces the same thing (SONYA_TRAILER_ENABLED). */
function _featureOn(name) {
  return Boolean(window.SONYA_FEATURES && window.SONYA_FEATURES[name] === true);
}

const TRAILER_JOB_MODE = 'trailer_film_breaker';
const TRAILER_UNAVAILABLE_MESSAGE = 'Режим Trailer временно недоступен.';

function normalizeMode(clipType) {
  const v = String(clipType || '').toLowerCase();

  // New Trailer (2026-11-17, admins only until the owner approves): clipType trailer runs on the virality pool
  // (worker scripts/shared/trailer); the API refuses it for everyone else.
  if (v === 'trailer') return 'virality';
  if (['filmbreaker', 'cinematic', 'cinematic_trailer', 'cinematic trailer'].includes(v)) {
    return 'trailer_film_breaker';
  }
  if (['viral', 'virality'].includes(v)) return 'virality';
  // Storytelling runs on the virality pool (worker: clipType=storytelling -> story strategy); mode 'stories'
  // has no fleet pool - such jobs were charged and never picked up (2026-11-16).
  if (['storytelling', 'stories'].includes(v)) return 'virality';
  if (['educational', 'education'].includes(v)) return 'educational';
  if (['streamer', 'stream'].includes(v)) return 'streamer';
  if (['sonya_gen', 'gen', 'gen-1'].includes(v)) return 'sonya_gen';
  if (v === 'hooks') return 'virality';

  return 'trailer_film_breaker';
}

// URL-mode job creation: POST /generation/jobs/from-url starts server-side
// download+upload (checking -> downloading -> uploading), then this polls
// GET /generation/jobs/from-url/{id} until it resolves to a real job_id
// (or fails) — the exact same S3-upload / create_job_idempotent code path
// as apiCreateVideoJob() below, just fed by a downloaded file instead of a
// browser upload (see scripts/prod_generation_api.py, scripts/url_ingest.py).
//
// Returns a Response-shaped object (ok/status/json()) so checkAndCreateVideoJob()
// below needs no branching beyond picking which of these two functions to call.
const URL_INGEST_STATUS_TEXT = {
  checking: 'Проверяем ссылку…',
  downloading: 'Получаем видео…',
  uploading: 'Загружаем видео…',
};

function _errorResponse(status, message, code) {
  return { ok: false, status, json: async () => ({ detail: code ? { message, code } : { message } }) };
}

// Server entitlement refusals that are solved by picking a plan
// (scripts/entitlements.py) -- they open the plan picker on the right mode.
const PAYWALL_CODES = ['FREE_PLAN_USED', 'MODE_NOT_IN_PLAN', 'SUBSCRIPTION_EXPIRED', 'PLAN_LIMIT_REACHED'];

/* ─────────────────────────────────────────────
   SOURCE LENGTH: check before uploading, offer "first N minutes" instead of a bare refusal.
   Limit comes from GET /api/generation/limits (tariff / free / admin). Never trusts the browser for the real
   check -- the API re-measures the received file (ffprobe) before the queue.
───────────────────────────────────────────── */
window.SONYA_TRIM = null;   // {start, end} seconds: the segment chosen for a too-long video

async function apiGetSourceLimit(mode) {
  try {
    const r = await apiFetch('/generation/limits?mode=' + encodeURIComponent(mode || 'virality'));
    if (r.ok) { const d = await safeJson(r); return Number(d.max_source_sec) || null; }
  } catch (e) { /* offline: the server check still applies */ }
  return null;
}

function probeVideoDurationSec(file) {
  return new Promise(resolve => {
    if (!file || !(/^video\//.test(file.type || '') || /\.(mp4|mov|mkv|webm|avi|m4v)$/i.test(file.name || ''))) return resolve(null);
    const v = document.createElement('video');
    const url = URL.createObjectURL(file);
    let done = false;
    const fin = x => { if (done) return; done = true; clearTimeout(t); v.removeAttribute('src'); URL.revokeObjectURL(url); resolve(x); };
    const t = setTimeout(() => fin(null), 8000);
    v.preload = 'metadata';
    v.onloadedmetadata = () => fin(isFinite(v.duration) && v.duration > 0 ? v.duration : null);
    v.onerror = () => fin(null);
    v.src = url;
  });
}

function _fmtMinutes(min) {
  const n = Math.round(min), n10 = n % 10, n100 = n % 100;
  return n + ' ' + (n10 === 1 && n100 !== 11 ? 'минута' : (n10 >= 2 && n10 <= 4 && (n100 < 12 || n100 > 14)) ? 'минуты' : 'минут');
}

function _fmtHms(sec) {
  sec = Math.max(0, Math.round(sec));
  const h = Math.floor(sec / 3600), m = Math.floor(sec / 60) % 60, s = sec % 60, p = n => String(n).padStart(2, '0');
  return (h ? h + ':' + p(m) : String(m)) + ':' + p(s);
}
function _parseHms(txt) {
  const parts = String(txt || '').trim().split(':').map(x => x.trim());
  if (!parts.length || parts.length > 3 || parts.some(x => !/^\d+$/.test(x))) return null;
  return parts.reduce((a, x) => a * 60 + Number(x), 0);
}

// The video is longer than the tariff allows: the user picks the segment (start..end, length <= maxSec) to process.
// durSec may be unknown (null). Resolves {start, end} in seconds, or null = choose another video.
function askTrimDialog(durSec, maxSec) {
  maxSec = Math.max(1, Math.floor(maxSec));
  const known = !!(durSec && durSec > maxSec);
  const total = Math.ceil(known ? durSec : Math.max(maxSec * 3, 6 * 3600));
  return new Promise(resolve => {
    document.getElementById('sonya-trim-modal')?.remove();
    const ov = document.createElement('div');
    ov.className = 'sonya-modal-overlay is-open';
    ov.id = 'sonya-trim-modal';
    ov.setAttribute('role', 'dialog'); ov.setAttribute('aria-modal', 'true'); ov.setAttribute('aria-label', 'Выбор отрезка');
    const box = document.createElement('div');
    box.className = 'sonya-modal paywall-modal';
    const head = known ? `Видео длится ${_fmtHms(durSec)}, а максимум — ${_fmtMinutes(maxSec / 60)}.` : `Видео длиннее допустимого: максимум — ${_fmtMinutes(maxSec / 60)}.`;
    box.innerHTML = `
      <button class="sonya-modal-close" type="button" data-k="x" aria-label="Закрыть"><i class="fa-solid fa-xmark"></i></button>
      <div class="paywall-view is-active">
      <h2 class="paywall-title plans-title">Выберите отрезок</h2>
      <p class="trim-lead">${head} Обработаем только выбранную часть — до ${_fmtMinutes(maxSec / 60)}.${known ? '' : ' Если конец позже конца видео — обработаем до конца.'}</p>
      <div class="trim-card">
        <div class="trim-head"><span class="trim-name">Отрезок</span><span class="trim-price" data-k="sum"></span></div>
        <div class="trim-range" data-k="rng">
          <div class="trim-fill" data-k="fill"></div>
          <input type="range" data-k="s" min="0" max="${total}" step="1" aria-label="Начало отрезка">
          <input type="range" data-k="e" min="0" max="${total}" step="1" aria-label="Конец отрезка">
        </div>
        <div class="trim-meta">
          <label>Начало <input class="trim-field" type="text" inputmode="numeric" data-k="st"></label>
          <label>Конец <input class="trim-field" type="text" inputmode="numeric" data-k="et"></label>
        </div>
      </div></div>`;
    const q = k => box.querySelector(`[data-k="${k}"]`);
    const rs = q('s'), re = q('e'), ts = q('st'), te = q('et'), sum = q('sum');
    let st = 0, en = Math.min(maxSec, total);
    const render = () => {
      rs.value = st; re.value = en; ts.value = _fmtHms(st); te.value = _fmtHms(en);
      ts.classList.remove('bad'); te.classList.remove('bad');
      sum.textContent = _fmtMinutes((en - st) / 60);
      const fill = q('fill'); fill.style.left = (st / total * 100) + '%'; fill.style.right = (100 - en / total * 100) + '%';
    };
    // Moving one end drags the other so that 1 s <= length <= maxSec.
    const setStart = v => { st = Math.max(0, Math.min(Math.round(v), total - 1)); en = Math.min(Math.max(en, st + 1), st + maxSec, total); render(); };
    const setEnd = v => { en = Math.max(1, Math.min(Math.round(v), total)); st = Math.max(Math.min(st, en - 1), en - maxSec, 0); render(); };
    rs.oninput = () => setStart(Number(rs.value)); re.oninput = () => setEnd(Number(re.value));
    ts.onchange = () => { const v = _parseHms(ts.value); v === null ? ts.classList.add('bad') : setStart(v); };
    te.onchange = () => { const v = _parseHms(te.value); v === null ? te.classList.add('bad') : setEnd(v); };
    const mk = (txt, primary) => { const b = document.createElement('button'); b.type = 'button'; b.textContent = txt;
      b.className = primary ? 'trim-go' : 'paywall-later-btn'; return b; };
    const yes = mk('Обработать отрезок', true), no = mk('Выбрать другое видео', false);
    const close = v => { document.removeEventListener('keydown', onKey); ov.remove(); resolve(v); };
    const onKey = e => { if (e.key === 'Escape') close(null); };
    yes.onclick = () => { ts.onchange(); te.onchange(); close({ start: st, end: en }); };
    no.onclick = () => close(null);
    box.querySelector('[data-k="x"]').onclick = () => close(null);
    ov.addEventListener('click', e => { if (e.target === ov) close(null); });
    document.addEventListener('keydown', onKey);
    render();
    box.querySelector('.paywall-view').append(yes, no); ov.append(box); document.body.append(ov);
  });
}

function _trimParams() {
  const t = window.SONYA_TRIM;
  if (!t) return {};
  // trim_first_sec too (= the same thing when the segment starts at 0): understood by older API/worker releases.
  return { trim_start_sec: t.start, trim_end_sec: t.end, trim_first_sec: t.start ? undefined : t.end };
}

// Called by app.js when a job failed with SOURCE_TOO_LONG (YouTube links: the length is only known on the GPU worker).
window.sonyaOfferTrimAfterFail = async function (job) {
  const mode = normalizeMode(job && job.mode);
  const maxSec = await apiGetSourceLimit(mode);
  if (!maxSec) return;
  const m = String((job && job.error) || '').match(/(\d+)\s*мин/);
  const seg = await askTrimDialog(m ? Number(m[1]) * 60 : null, maxSec);
  if (seg) {
    window.SONYA_TRIM = seg;
    if (typeof submitGenerationJob === 'function') submitGenerationJob();
  }
};

async function apiCreateVideoJobFromUrl(formData) {
  const sourceUrl = String(formData?.source?.url || '').trim();
  const mode = normalizeMode(formData?.clipType || formData?.mode);

  // Trailer is switched off: no request, and no "processing" screen either.
  if (mode === TRAILER_JOB_MODE && !_featureOn('trailer')) {
    return _errorResponse(403, TRAILER_UNAVAILABLE_MESSAGE, 'TRAILER_DISABLED');
  }

  if (typeof showPage === 'function') showPage('processing');
  if (typeof window.sonyaSetProcessingText === 'function') window.sonyaSetProcessingText('Проверяем ссылку…');
  if (typeof window.sonyaSetProcessingProgress === 'function') window.sonyaSetProcessingProgress(4);

  const startRes = await apiFetch('/generation/jobs/from-url', {
    method: 'POST',
    body: JSON.stringify({
      url: sourceUrl,
      mode,
      params: {
        ...formData,
        source: { ...(formData?.source || {}), url: sourceUrl },
        frontend: 'legacy_static_sonya',
        production_endpoint: '/api/generation/jobs/from-url',
        ..._trimParams()
      }
    }),
    headers: { 'Idempotency-Key': _getOrCreateJobIdempotencyKey() }
  });

  if (startRes._networkError || !startRes.ok) return startRes;

  const startData = await safeJson(startRes);
  const ingestId = startData.ingest_id;
  if (!ingestId) {
    return _errorResponse(502, 'Сервер не вернул идентификатор загрузки. Попробуйте снова.');
  }

  // ~20 minutes ceiling at 3s/poll — matches the generous cap pollJob()
  // (app.js) already uses for the job-status phase that follows this.
  for (let i = 0; i < 400; i++) {
    const pollRes = await apiFetch('/generation/jobs/from-url/' + ingestId);

    if (pollRes.ok) {
      const data = await safeJson(pollRes);
      const st = data.status;

      const text = URL_INGEST_STATUS_TEXT[st];
      if (text && typeof window.sonyaSetProcessingText === 'function') window.sonyaSetProcessingText(text);
      if (typeof data.percent === 'number' && typeof window.sonyaSetProcessingProgress === 'function') {
        window.sonyaSetProcessingProgress(Math.max(4, Math.min(99, data.percent)));
      }

      if (st === 'queued' && data.job_id) {
        return { ok: true, status: 202, json: async () => ({ job_id: data.job_id, status: 'queued', mode }) };
      }
      if (st === 'failed') {
        if (PAYWALL_CODES.includes(data.error)) return _errorResponse(402, data.message, data.error);
        if (data.error === 'SOURCE_TOO_LONG') {
          const seg = (data.can_trim && data.max_source_minutes && !window.SONYA_TRIM)
            ? await askTrimDialog(data.source_minutes ? data.source_minutes * 60 : null, data.max_source_minutes * 60) : null;
          if (seg) {
            window.SONYA_TRIM = seg;
            clearJobIdempotencyKey();
            return apiCreateVideoJobFromUrl(formData);
          }
          return _errorResponse(413, data.message, data.error);
        }
        return _errorResponse(502, data.message || 'Не удалось скачать видео по ссылке.');
      }
      // checking / downloading / uploading — keep polling.
    }
    // A single failed poll (network hiccup, 5xx) doesn't abort the whole
    // flow — same tolerance pollJob() gives the job-status endpoint.

    await new Promise(r => setTimeout(r, 3000));
  }

  return _errorResponse(504, 'Загрузка видео по ссылке заняла слишком много времени. Попробуйте ещё раз.');
}

/* ─────────────────────────────────────────────
   DIRECT UPLOAD → S3  (2026-11-16)
   The file goes from the browser straight to the bucket in 8–16 MB parts (presigned multipart,
   4 at a time). Every part is retried on its own; a lost network / Telegram Mini App sent to the
   background just waits and continues; pressing «Создать» again with the same file resumes from
   the parts already uploaded (localStorage, 3 days). The API only signs URLs and, after complete,
   checks the file (magic bytes, length via ffprobe) and creates the job. null = not available
   (old API) -> the caller uses the old multipart POST.
───────────────────────────────────────────── */
const DU_PARALLEL = 4;
const DU_STORE = 'sonya_direct_upload_v1';
const DU_MAX_TRIES = 12;            // per part / per API call, not counting time offline or in background
let _duWakers = [];
function _duWake() { const w = _duWakers; _duWakers = []; w.forEach(f => f()); }
if (typeof window !== 'undefined') {
  window.addEventListener('online', _duWake);
  document.addEventListener('visibilitychange', () => { if (!document.hidden) _duWake(); });
}
function _duSleep(ms) {
  return new Promise(resolve => {
    const done = () => { clearTimeout(t); _duWakers = _duWakers.filter(x => x !== done); resolve(); };
    const t = setTimeout(done, ms);
    _duWakers.push(done);
  });
}
// Offline or app in background: wait without spending a retry.
async function _duWaitUsable(onWait) {
  let waited = false;
  while ((typeof navigator !== 'undefined' && navigator.onLine === false) || document.hidden) {
    if (!waited && onWait) onWait(navigator.onLine === false ? 'offline' : 'hidden');
    waited = true;
    await _duSleep(15000);
  }
  return waited;
}
function _duResp(status, data) { return { ok: status >= 200 && status < 300, status, json: async () => data }; }

async function _duApi(path, body, { headers, timeoutMs, onWait } = {}) {
  for (let tries = 0; ; tries++) {
    await _duWaitUsable(onWait);
    const res = await apiFetch('/generation/uploads/' + path, {
      method: 'POST', body: JSON.stringify(body), headers: headers || {}, timeoutMs: timeoutMs || 45000 });
    const retryable = res._networkError || res.status === 429 || res.status === 502 || res.status === 504 ||
      (res.status >= 500 && res.status !== 503 && res.status !== 507);
    if (!retryable || tries >= DU_MAX_TRIES - 1) {
      if (res._networkError) return res;
      return _duResp(res.status, await safeJson(res));
    }
    await _duSleep(Math.min(30000, 1000 * 2 ** Math.min(tries, 5)));
  }
}

function _duPut(url, blob, onProgress) {
  return new Promise(resolve => {
    const xhr = new XMLHttpRequest();
    let last = Date.now();
    const stall = setInterval(() => { if (Date.now() - last > 90000) xhr.abort(); }, 5000);
    const end = (st) => { clearInterval(stall); resolve({ ok: st >= 200 && st < 300, status: st }); };
    xhr.open('PUT', url, true);
    xhr.upload.onprogress = (e) => { last = Date.now(); onProgress(e.loaded || 0); };
    xhr.onload = () => end(xhr.status);
    xhr.onerror = () => end(0);
    xhr.onabort = () => end(0);
    xhr.ontimeout = () => end(0);
    xhr.send(blob);
  });
}

function _duFmtMb(b) {
  const mb = b / 1048576;
  return mb >= 1024 ? (mb / 1024).toFixed(1).replace('.', ',') + ' ГБ' : Math.round(mb) + ' МБ';
}

async function _directUploadJob(file, mode, formData, sourceUrl) {
  if (typeof XMLHttpRequest === 'undefined' || !file || typeof file.slice !== 'function' || !file.size) return null;
  const setText = (t) => { if (typeof window.sonyaSetProcessingText === 'function') window.sonyaSetProcessingText(t); };
  const setProg = (p) => { if (typeof window.sonyaSetProcessingProgress === 'function') window.sonyaSetProcessingProgress(p); };
  const onWait = (why) => setText(why === 'offline' ? 'Нет связи — ждём сеть, загрузка продолжится сама…'
    : 'Загрузка на паузе — вернитесь в приложение, она продолжится.');
  const fp = [mode, file.name, file.size, file.lastModified].join('|');

  let token = null, n = 0, ps = 0;
  const done = new Set();
  let saved = null;
  try { saved = JSON.parse(localStorage.getItem(DU_STORE) || 'null'); } catch (_) { saved = null; }
  if (saved && saved.fp === fp && Date.now() - saved.ts < 2.5 * 86400e3) {
    const st = await _duApi('status', { token: saved.token });
    if (st.ok) {
      const d = await st.json();
      token = saved.token; n = d.part_count; ps = saved.ps;
      (d.done || []).forEach(x => done.add(Number(x)));
    }
  }
  if (!token) {
    const r = await _duApi('init', { mode, filename: file.name || 'video.mp4', size: file.size,
      content_type: file.type || '', params: _trimParams() });
    if (r._networkError) return r;
    const d = await r.json();
    if (r.status === 404 || r.status === 405 || (r.status === 503 && d?.detail?.code === 'DIRECT_UPLOAD_OFF')) return null;
    if (!r.ok) return r;
    token = d.token; n = d.part_count; ps = d.part_size;
    try { localStorage.setItem(DU_STORE, JSON.stringify({ fp, token, ps, n, ts: Date.now() })); } catch (_) {}
  }

  const prevPage = (typeof appState !== 'undefined' && appState.currentPage) || null;
  if (typeof showPage === 'function') showPage('processing');
  const goBack = () => { if (prevPage && typeof showPage === 'function') showPage(prevPage); };
  const partLen = (i) => (i < n ? ps : file.size - ps * (n - 1));
  let doneBytes = 0;
  done.forEach(i => { doneBytes += partLen(i); });
  const loaded = {};
  const t0 = Date.now(), b0 = doneBytes;
  let lastPaint = 0;
  const paint = (force) => {
    const now = Date.now();
    if (!force && now - lastPaint < 250) return;
    lastPaint = now;
    let cur = doneBytes;
    Object.values(loaded).forEach(v => { cur += v; });
    const pct = Math.min(99, Math.floor(cur / file.size * 100));
    let eta = '';
    const speed = (cur - b0) / Math.max(1, (now - t0) / 1000);
    if (now - t0 > 6000 && speed > 0) {
      const min = Math.ceil((file.size - cur) / speed / 60);
      eta = min > 1 ? ` · осталось ~${min} мин` : ' · меньше минуты';
    }
    setText(`Загружаем видео: ${pct}% (${_duFmtMb(cur)} из ${_duFmtMb(file.size)})${eta}`);
    setProg(Math.max(2, Math.min(95, pct)));
    if (typeof window.sonyaSetGenerateButtonText === 'function') window.sonyaSetGenerateButtonText(`Загрузка ${pct}%`);
  };
  paint(true);

  const urls = {};
  let fatal = null;
  const runUpload = async (queue) => {
    const getUrls = async (nums) => {
      const r = await _duApi('parts', { token, parts: nums }, { onWait });
      if (!r.ok) { fatal = fatal || r; return; }
      Object.assign(urls, (await r.json()).urls || {});
    };
    const worker = async () => {
      while (queue.length && !fatal) {
        const i = queue.shift();
        for (let tries = 0; !fatal; ) {
          if (await _duWaitUsable(onWait)) paint(true);
          if (!urls[i]) await getUrls([i, ...queue.slice(0, 15)].filter(x => !urls[x]));
          if (fatal) break;
          const blob = file.slice((i - 1) * ps, (i - 1) * ps + partLen(i));
          const r = await _duPut(urls[i], blob, (l) => { loaded[i] = l; paint(); });
          loaded[i] = 0;
          if (r.ok) { doneBytes += partLen(i); done.add(i); paint(); break; }
          if (r.status === 404) { fatal = _errorResponse(410, 'Сессия загрузки устарела. Нажмите «Создать» ещё раз.', 'UPLOAD_EXPIRED'); break; }
          if (r.status === 403 || r.status === 400) delete urls[i];   // signature expired -> ask again
          if (document.hidden || navigator.onLine === false) continue;  // not the network's fault, no retry spent
          tries++;
          if (tries >= DU_MAX_TRIES) {
            fatal = _errorResponse(0, 'Связь всё время обрывается. Нажмите «Создать» ещё раз — загрузка продолжится с того же места.', 'UPLOAD_NETWORK');
            break;
          }
          setText('Связь прервалась — повторяем…');
          await _duSleep(Math.min(30000, 1000 * 2 ** Math.min(tries, 5)));
        }
      }
    };
    await Promise.all(Array.from({ length: Math.min(DU_PARALLEL, queue.length) }, worker));
  };

  const params = {
    ...formData,
    source: { ...(formData?.source || {}), url: sourceUrl, fileName: file.name || formData?.source?.fileName || null, platform: 'upload' },
    frontend: 'legacy_static_sonya',
    production_endpoint: '/api/generation/uploads',
  };
  for (let round = 0; round < 3; round++) {
    const queue = [];
    for (let i = 1; i <= n; i++) if (!done.has(i)) queue.push(i);
    await runUpload(queue);
    if (fatal) { goBack(); return fatal; }
    paint(true);
    setText('Проверяем видео…');
    const r = await _duApi('complete', { token, params: { ...params, ..._trimParams() } },
      { headers: { 'Idempotency-Key': _getOrCreateJobIdempotencyKey() }, timeoutMs: 15 * 60 * 1000, onWait });
    if (r._networkError) { goBack(); return r; }
    const d = await r.json();
    if (r.ok) {
      try { localStorage.removeItem(DU_STORE); } catch (_) {}
      setProg(5);
      return _duResp(r.status, d);
    }
    const code = d?.detail?.code || d?.detail?.error;
    if (r.status === 409 && code === 'UPLOAD_INCOMPLETE') {          // a part got lost on the way -> send it again
      (d.detail.missing || []).forEach(x => done.delete(Number(x)));
      doneBytes = 0; done.forEach(i => { doneBytes += partLen(i); });
      continue;
    }
    if (r.status === 413 && code === 'SOURCE_TOO_LONG' && d.detail.can_trim && !window.SONYA_TRIM) {
      const seg = await askTrimDialog(d.detail.source_minutes ? d.detail.source_minutes * 60 : null, d.detail.max_source_minutes * 60);
      if (seg) { window.SONYA_TRIM = seg; clearJobIdempotencyKey(); round--; continue; }  // same uploaded file, just the segment
      _duApi('abort', { token });
      try { localStorage.removeItem(DU_STORE); } catch (_) {}
      goBack();
      return _duResp(499, { detail: { message: 'Отменено.', code: 'cancelled' } });
    }
    if (r.status >= 400 && r.status < 500 && r.status !== 429 && r.status !== 409) {
      try { localStorage.removeItem(DU_STORE); } catch (_) {}
    }
    goBack();
    return _duResp(r.status, d);
  }
  goBack();
  return _errorResponse(502, 'Файл не удалось загрузить полностью. Нажмите «Создать» ещё раз — загрузка продолжится.', 'UPLOAD_INCOMPLETE');
}

async function apiCreateVideoJob(formData) {
  const sourceUrl = String(formData?.source?.url || '').trim();

  // appState is defined in app.js, loaded before auth.js.
  const uploadedFile =
    formData?.file ||
    formData?.source?.file ||
    (typeof appState !== 'undefined' ? appState.uploadedFile : null) ||
    (typeof window !== 'undefined' && window.appState ? window.appState.uploadedFile : null);

  if (!uploadedFile) {
    return {
      ok: false,
      status: 400,
      json: async () => ({
        detail: {
          error: 'file_required',
          message: 'Выберите видеофайл для генерации.'
        }
      })
    };
  }

  const mode = normalizeMode(formData?.clipType || formData?.mode);
  if (mode === TRAILER_JOB_MODE && !_featureOn('trailer')) {
    return _errorResponse(403, TRAILER_UNAVAILABLE_MESSAGE, 'TRAILER_DISABLED');
  }
  // 2026-11-16: files go straight to S3 (presigned multipart, see _directUploadJob); the old
  // multipart POST through nginx (<= 500 MB, read into the API's memory) is only the fallback.
  const directRes = await _directUploadJob(uploadedFile, mode, formData, sourceUrl);
  if (directRes) return directRes;
  const MAX_FILE_MB = 500;
  if (uploadedFile.size && uploadedFile.size > MAX_FILE_MB * 1024 * 1024) {
    const gb = (uploadedFile.size / 1024 / 1024 / 1024).toFixed(1).replace('.', ',');
    return _errorResponse(413,
      `Файл слишком большой (${gb} ГБ) для запасного способа загрузки (до ${MAX_FILE_MB} МБ). ` +
      'Попробуйте позже или вставьте ссылку на это видео.', 'file_too_large');
  }
  const fd = new FormData();
  fd.append('mode', mode);
  fd.append('file', uploadedFile, uploadedFile.name || 'input.mp4');
  fd.append('params', JSON.stringify({
    ...formData,
    source: {
      ...(formData?.source || {}),
      url: sourceUrl,
      fileName: uploadedFile.name || formData?.source?.fileName || null,
      platform: 'upload'
    },
    frontend: 'legacy_static_sonya',
    production_endpoint: '/api/generation/jobs',
    ..._trimParams()
  }));

  return apiFetch('/generation/jobs', {
    method: 'POST',
    body: fd,
    headers: { 'Idempotency-Key': _getOrCreateJobIdempotencyKey() },
    timeoutMs: computeUploadTimeoutMs(uploadedFile.size),
    timeoutMessage: 'Загрузка видео заняла слишком много времени. Проверьте соединение и попробуйте снова.',
  });
}

async function apiGetSubscriptionStatus() {
  return apiFetch('/billing/subscription-status');
}

/* ─────────────────────────────────────────────
   REFRESH AUTH STATE  (call after login/logout)
───────────────────────────────────────────── */
async function refreshAuthState() {
  const res = await apiGetMe();
  if (res.status === 401) {
    // Expected guest state -- not logged in, not an error.
    authState.user = null;
  } else if (!res.ok) {
    // Real failure (5xx / network) -- fall back to "no user" (we can't
    // assume logged-in) but, unlike the 401 case, this is NOT expected,
    // so it must stay visible instead of being silently swallowed.
    reportAuthError('refreshAuthState', res);
    authState.user = null;
  } else {
    try {
      authState.user = await res.json();
    } catch {
      authState.user = null;
    }
  }
  authState.initialized = true;
  renderAccountButton();
  // 2026-11-22 «Мой лимит»: the glass count next to «Запустить AI» follows the account (quota-glass.js)
  if (window.SonyaQuota) window.SonyaQuota.onAuth(authState.user);
  return authState.user;
}

/* ─────────────────────────────────────────────
   ACCOUNT BUTTON RENDERING
───────────────────────────────────────────── */
function renderAccountButton() {
  const btn = document.getElementById('sonya-account-btn');
  if (!btn) return;

  const user = authState.user;
  if (!user) {
    btn.innerHTML = '<i class="fa-solid fa-user"></i>';
    btn.setAttribute('aria-label', 'Аккаунт');
    btn.classList.remove('is-logged-in');
  } else {
    // Show first letter of email as avatar
    const initials = user.email ? user.email[0].toUpperCase() : '?';
    btn.innerHTML = `<span class="acct-initials">${initials}</span>`;
    btn.setAttribute('aria-label', user.email);
    btn.classList.add('is-logged-in');
  }
}

/* ─────────────────────────────────────────────
   ACCOUNT DROPDOWN
───────────────────────────────────────────── */
// "Нарезка · Pro — осталось 12 из 20 · до 03.11.2026"
function _subscriptionLine(sub) {
  const mode = SONYA_PRICING.modes.find(m => m.id === sub.mode);
  const plan = mode && mode.plans.find(p => p.id === sub.plan_id);
  const name = `${mode ? mode.label : sub.mode} · ${plan ? plan.name : sub.plan_id}`;
  const until = sub.period_end ? new Date(sub.period_end).toLocaleDateString('ru-RU') : '';
  return escHtml(`${name} — осталось ${sub.ops_remaining} из ${sub.ops_limit}${until ? ` · до ${until}` : ''}`);
}
function openAccountDropdown() {
  const drop = document.getElementById('sonya-account-drop');
  if (!drop) return;

  const user = authState.user;
  if (!user) {
    drop.innerHTML = `
      <button class="acct-drop-item" id="acct-drop-login">
        <i class="fa-solid fa-arrow-right-to-bracket"></i> Войти
      </button>
      <button class="acct-drop-item" id="acct-drop-register">
        <i class="fa-solid fa-user-plus"></i> Зарегистрироваться
      </button>
      <div class="acct-drop-divider"></div>
      <button class="acct-drop-item" id="acct-drop-plans">
        <i class="fa-solid fa-bolt"></i> Тарифы
      </button>`;
    drop.querySelector('#acct-drop-login').onclick = () => { closeAccountDropdown(); openAuthModal('login'); };
    drop.querySelector('#acct-drop-register').onclick = () => { closeAccountDropdown(); openAuthModal('registration'); };
    // Prices are public: guests browse the picker freely, auth is only
    // asked for at the pay CTA (see startCheckout).
    drop.querySelector('#acct-drop-plans').onclick = () => { closeAccountDropdown(); openPaywallModal(); };
  } else {
    const emailLower = String(user.email || '').toLowerCase();
    // Active per-mode plans from /api/auth/me (server entitlements).
    const subscriptions = Array.isArray(user.subscriptions) ? user.subscriptions : [];
    const planLabel =
      emailLower === 'elcuevimran@gmail.com' && user.plan_type === 'admin' ? 'CEO' :
      emailLower === 'mironowism@gmail.com' && user.plan_type === 'admin' ? 'Admin' :
      user.plan_type === 'admin' ? 'Admin' :
      user.plan_type === 'pro' || subscriptions.length ? 'Подписка ✦' :
      'Free Plan';
    const freeLeft = user.free_video_limit - user.free_video_used;
    const until = user.plan_active_until
      ? new Date(user.plan_active_until).toLocaleDateString('ru-RU')
      : null;

    // A plan_type of 'pro' with a plan_active_until in the past is a real,
    // common state (nothing demotes plan_type back to 'free' when a
    // subscription lapses) -- must not be shown identically to an active
    // subscription (stale "до <past date>" with no way to renew).
    const isProActive = _isProActive(user);
    const isExpiredPro = user.plan_type === 'pro' && !isProActive;

    const statusLine =
      isProActive ? (until ? `Активна до ${until}` : 'Активна') :
      subscriptions.length ? subscriptions.map(_subscriptionLine).join('<br>') :
      isExpiredPro ? 'Подписка закончилась' :
      user.plan_type === 'free' ? 'Подписка не активна' :
      null;

    // Buying and renewing both just open the same paywall modal (plan
    // picker) -- there is only ever one checkout UI.
    const showBuyCta = user.plan_type === 'free';
    const showRenewCta = isExpiredPro;

    // 2026-11-22 «Мой лимит»: one glass card — plan, «Осталось N из M видео», progress, reset date,
    // «Продлить» / «Повысить» (quota-glass.js, numbers from GET /api/me/quota). The old text lines stay
    // as the fallback when the card script is missing.
    if (window.SonyaQuota) {
      drop.innerHTML = `
      <div class="acct-drop-user has-sq">
        <span class="acct-drop-email">${escHtml(user.email)}</span>
        <div id="sq-acct-card"></div>
      </div>
      <div class="acct-drop-divider"></div>
      <button class="acct-drop-item" id="acct-drop-plans">
        <i class="fa-solid fa-bolt"></i> Тарифы
      </button>
      <button class="acct-drop-item acct-drop-logout" id="acct-drop-logout">
        <i class="fa-solid fa-right-from-bracket"></i> Выйти
      </button>`;
      window.SonyaQuota.mountCard(drop.querySelector('#sq-acct-card'), 'virality');
      drop.querySelector('#acct-drop-plans').onclick = () => { closeAccountDropdown(); openPaywallModal(); };
    } else {
    drop.innerHTML = `
        <div class="acct-drop-user">
          <span class="acct-drop-email">${escHtml(user.email)}</span>
          <span class="acct-drop-plan ${user.plan_type === 'pro' || user.plan_type === 'admin' || subscriptions.length ? 'is-pro' : ''}">${planLabel}</span>
          ${statusLine ? `<span class="acct-drop-until">${statusLine}</span>` : ''}
          ${user.plan_type === 'free'
            ? `<span class="acct-drop-free">Бесплатных видео: ${freeLeft} / ${user.free_video_limit}</span>`
            : ''}
        </div>
        <div class="acct-drop-divider"></div>
        ${showBuyCta
          ? `<button class="acct-drop-item acct-drop-upgrade" id="acct-drop-upgrade">
               <i class="fa-solid fa-bolt"></i> Выбрать тариф
             </button>`
          : ''}
        ${showRenewCta
          ? `<button class="acct-drop-item acct-drop-upgrade" id="acct-drop-upgrade">
               <i class="fa-solid fa-bolt"></i> Продлить подписку
             </button>`
          : ''}
        <button class="acct-drop-item acct-drop-logout" id="acct-drop-logout">
          <i class="fa-solid fa-right-from-bracket"></i> Выйти
        </button>`;
  
      if (showBuyCta || showRenewCta) {
        drop.querySelector('#acct-drop-upgrade').onclick = () => { closeAccountDropdown(); openPaywallModal(); };
      }
    }
    drop.querySelector('#acct-drop-logout').onclick = handleLogout;
  }

  drop.classList.add('is-open');

  // Close on outside click
  setTimeout(() => {
    document.addEventListener('click', onOutsideDropdown, { once: true });
  }, 0);
}

function closeAccountDropdown() {
  const drop = document.getElementById('sonya-account-drop');
  if (drop) drop.classList.remove('is-open');
}

function onOutsideDropdown(e) {
  const drop = document.getElementById('sonya-account-drop');
  const btn  = document.getElementById('sonya-account-btn');
  if (drop && !drop.contains(e.target) && e.target !== btn) {
    closeAccountDropdown();
  }
}

async function handleLogout() {
  closeAccountDropdown();
  await apiLogout();
  await refreshAuthState();
}

/* ─────────────────────────────────────────────
   AUTH MODAL  (Glass UI, login / registration / code)
───────────────────────────────────────────── */
let _authPurpose  = 'login';   // 'login' | 'registration'
let _authView     = 'login';   // 'login' | 'register' | 'code'
let _authEmail    = '';        // remembered email used to request code
let _authConsents = null;      // remembered consents array (registration only)

/* Resend code cooldown (UI only — backend has its own throttling) */
const RESEND_COOLDOWN_SECONDS = 45;
let _resendDeadline = 0;
let _resendTimerId  = null;

function startResendCooldown() {
  _resendDeadline = Date.now() + RESEND_COOLDOWN_SECONDS * 1000;
  refreshResendButton();
  if (_resendTimerId) clearInterval(_resendTimerId);
  _resendTimerId = setInterval(refreshResendButton, 1000);
}
function stopResendCooldown() {
  if (_resendTimerId) { clearInterval(_resendTimerId); _resendTimerId = null; }
}
function refreshResendButton() {
  const btn = document.getElementById('sonya-auth-resend');
  if (!btn) return;
  const remaining = Math.max(0, Math.ceil((_resendDeadline - Date.now()) / 1000));
  if (remaining > 0) {
    btn.disabled = true;
    btn.textContent = `Отправить код повторно (${remaining}s)`;
  } else {
    btn.disabled = false;
    btn.textContent = 'Отправить код повторно';
    stopResendCooldown();
  }
}

/* ── helpers: CTA text + loading (preserve SVG icons) ── */
function setCtaText(btn, text) {
  if (!btn) return;
  const span = btn.querySelector('span');
  if (span) span.textContent = text;
  else      btn.textContent  = text;
}

function setCtaLoading(btn, loading) {
  if (!btn) return;
  btn.disabled = !!loading;
  btn.classList.toggle('is-loading', !!loading);
}

/* ── open / close ── */
function openAuthModal(mode = 'login', message = '', { resumeCheckout = null } = {}) {
  const overlay = document.getElementById('sonya-auth-overlay');
  if (!overlay) return;

  // Only the paywall's guest CTA passes a checkout to resume; any other
  // way into the auth modal drops a stale one (e.g. guest dismissed the
  // login, then signed in later from the account menu).
  _pendingCheckout = resumeCheckout;

  // Normalize mode: callers may use 'registration' (legacy) or 'register'.
  const view = (mode === 'registration' || mode === 'register') ? 'register' : 'login';
  setAuthView(view);
  setAuthBanner(message);
  clearAuthError();

  overlay.inert = false;
  overlay.setAttribute('aria-hidden', 'false');
  overlay.classList.add('is-open');
  document.body.classList.add('sonya-auth-open');
  document.body.style.overflow = 'hidden';

  setTimeout(() => {
    const id = view === 'register' ? 'sonya-register-email' : 'sonya-login-email';
    document.getElementById(id)?.focus();
  }, 80);
}

function closeAuthModal() {
  const overlay = document.getElementById('sonya-auth-overlay');
  if (!overlay) return;
  overlay.classList.remove('is-open');
  const focused = document.activeElement;
  if (focused && overlay.contains(focused)) {
    focused.blur();
  }

  overlay.inert = true;
  overlay.setAttribute('aria-hidden', 'true');
  document.body.classList.remove('sonya-auth-open');
  document.body.style.overflow = '';
  clearAuthError();
  setAuthBanner('');
  stopResendCooldown();
}

/* ── view switching ── */
function setAuthView(view) {
  _authView = view;

  // Tabs reflect login/register; in 'code' view neither is active.
  document.querySelectorAll('[data-auth-tab]').forEach(tab => {
    tab.classList.toggle('is-active',
      (view === 'login'    && tab.dataset.authTab === 'login') ||
      (view === 'register' && tab.dataset.authTab === 'register'));
  });

  document.querySelectorAll('[data-auth-view]').forEach(v => {
    v.classList.toggle('is-active', v.dataset.authView === view);
  });
}

/* ── banner / error ── */
function setAuthBanner(msg) {
  const el = document.getElementById('sonya-auth-banner');
  if (!el) return;
  if (msg) { el.textContent = msg; el.hidden = false; }
  else     { el.textContent = ''; el.hidden = true; }
}

function normalizeAuthError(value, fallback = 'Не удалось выполнить запрос') {
  if (typeof value === 'string' && value.trim()) {
    return value;
  }

  if (value && typeof value === 'object') {
    const code = value.error || value.code || '';

    const messages = {
      unauthorized: 'Необходимо войти в аккаунт',
      invalid_email: 'Введите корректный email',
      email_send_failed: 'Не удалось отправить письмо. Попробуйте позже.',
      rate_limited: 'Слишком много попыток. Подождите и повторите.',
      invalid_code: 'Неверный код. Проверьте письмо и повторите.',
      code_expired: 'Срок действия кода истёк. Запросите новый код.'
    };

    return messages[code] || value.message || fallback;
  }

  return fallback;
}

function setAuthError(msg) {
  const el = document.getElementById('sonya-auth-error');
  if (!el) return;

  const text = normalizeAuthError(msg, 'Произошла ошибка. Попробуйте позже.');
  el.textContent = text;
  el.hidden = !text;
}
// NOT setAuthError('') -- an empty string fails normalizeAuthError's
// value.trim() truthiness check and falls through to its generic fallback
// text, so routing "clear" through setAuthError would paint the red
// fallback message instead of hiding it.
function clearAuthError() {
  const el = document.getElementById('sonya-auth-error');
  if (!el) return;
  el.textContent = '';
  el.hidden = true;
}

/* ── consent helper (registration) ── */
function getRegisterConsents() {
  const types = ['terms_of_service', 'privacy_policy', 'personal_data_processing'];
  const out = [];
  for (const t of types) {
    const el = document.querySelector(`#sonya-auth-overlay input[name="${t}"]`);
    if (!el || !el.checked) return null;
    out.push({ type: t, version: '1.0' });
  }
  return out;
}

/* ── Step 1: request code ── */
async function handleRequestCode(purpose) {
  clearAuthError();

  const inputId = purpose === 'registration' ? 'sonya-register-email' : 'sonya-login-email';
  const btnId   = purpose === 'registration' ? 'sonya-register-code-btn' : 'sonya-login-code-btn';

  const emailEl = document.getElementById(inputId);
  const email = (emailEl?.value || '').trim().toLowerCase();
  if (!email || !email.includes('@')) { setAuthError('Введите корректный email'); return; }

  // Validate consents on registration BEFORE network call
  let consents = null;
  if (purpose === 'registration') {
    consents = getRegisterConsents();
    if (!consents) { setAuthError('Необходимо принять все три соглашения'); return; }
  }

  const btn = document.getElementById(btnId);
  setCtaLoading(btn, true);

  const res  = await apiRequestCode(email, purpose);
  const data = await safeJson(res);

  setCtaLoading(btn, false);

  if (res._networkError) { setAuthError('Сервер недоступен. Проверьте соединение и повторите попытку.'); return; }
  if (res.status === 404) { setAuthError(data.detail || 'Аккаунт не найден. Зарегистрируйтесь.'); return; }
  if (res.status === 429) { setAuthError('Слишком много попыток. Подождите и повторите.'); return; }
  if (!res.ok)            { setAuthError(data.detail || 'Ошибка отправки. Попробуйте позже.'); return; }

  _authPurpose  = purpose;
  _authEmail    = email;
  _authConsents = consents;

  setAuthView('code');

  // Update code subtitle with email
  const sub = document.getElementById('sonya-code-subtitle');
  if (sub) sub.innerHTML = `Мы отправили код подтверждения на <b>${escHtml(email)}</b>. Проверьте «Входящие» и «Спам».`;

  startResendCooldown();
  setTimeout(() => {
    const codeEl = document.getElementById('sonya-code-input');
    if (codeEl) { codeEl.value = ''; codeEl.focus(); }
  }, 80);
}

/* ── Resend code (from code view) ── */
async function handleResendCode() {
  if (Date.now() < _resendDeadline) return;
  if (!_authEmail) { setAuthView(_authPurpose === 'registration' ? 'register' : 'login'); return; }
  clearAuthError();

  const btn = document.getElementById('sonya-auth-resend');
  if (btn) { btn.disabled = true; btn.textContent = 'Отправка…'; }

  const res  = await apiRequestCode(_authEmail, _authPurpose);
  const data = await safeJson(res);

  if (btn) { btn.disabled = false; btn.textContent = 'Отправить код повторно'; }

  if (res._networkError) { setAuthError('Сервер недоступен. Проверьте соединение.'); return; }
  if (res.status === 429) { setAuthError('Превышен лимит запросов. Попробуйте позже.'); return; }
  if (!res.ok)            { setAuthError(data.detail || 'Не удалось отправить код. Попробуйте позже.'); return; }

  showToast('Код отправлен повторно', 'success');
  startResendCooldown();
}

/* ── Back from code view ── */
function goBackFromCodeView() {
  setAuthView(_authPurpose === 'registration' ? 'register' : 'login');
  clearAuthError();
  stopResendCooldown();
  setTimeout(() => {
    const id = _authPurpose === 'registration' ? 'sonya-register-email' : 'sonya-login-email';
    document.getElementById(id)?.focus();
  }, 60);
}

/* ── Step 2: verify code ── */
async function handleVerifyCode() {
  clearAuthError();
  const codeEl = document.getElementById('sonya-code-input');
  const code = (codeEl?.value || '').trim();
  if (code.length !== 6 || !/^\d{6}$/.test(code)) { setAuthError('Введите 6-значный код из письма'); return; }
  if (!_authEmail) { setAuthError('Сессия устарела. Запросите код заново.'); return; }

  const submitBtn = document.getElementById('sonya-verify-code-btn');
  setCtaLoading(submitBtn, true);

  const payload = { email: _authEmail, code, purpose: _authPurpose };
  if (_authPurpose === 'registration') {
    payload.consents = _authConsents || getRegisterConsents();
    if (!payload.consents) {
      setCtaLoading(submitBtn, false);
      setAuthError('Необходимо принять все три соглашения');
      return;
    }
  }

  const res  = await apiVerifyCode(payload);
  const data = await safeJson(res);

  setCtaLoading(submitBtn, false);

  if (res._networkError) { setAuthError('Сервер недоступен. Проверьте соединение.'); return; }
  if (res.status === 400) { setAuthError(data.detail || 'Неверный код. Попробуйте снова.'); return; }
  if (res.status === 410) { setAuthError('Срок действия кода истёк. Запросите новый код.'); return; }
  if (res.status === 429) { setAuthError('Превышено количество попыток. Запросите новый код.'); return; }
  if (!res.ok)            { setAuthError(data.detail || 'Ошибка. Попробуйте позже.'); return; }

  stopResendCooldown();
  closeAuthModal();
  await refreshAuthState();

  const user = authState.user;
  if (user) {
    showToast(data.is_new_user
      ? 'Добро пожаловать! Вам доступно 1 бесплатное видео.'
      : `С возвращением, ${user.email}!`,
      'success');
  }

  // Came from the bot («Войти и привязать», ?tg=connect): finish the Telegram link right after sign-in.
  if (user && _pendingTgConnect) { _pendingTgConnect = false; _tgConnectGo(); return; }

  // Guest picked a plan before signing in -- bring them straight back to
  // it (same mode, plan and consent), one click away from paying.
  const resume = _pendingCheckout;
  _pendingCheckout = null;
  if (user && resume) openPaywallModal(resume);
}

/* ─────────────────────────────────────────────
   PAYWALL MODAL
   Views: offer -> loading -> (redirect away) | error | already-pro
   Source of truth for "is Pro active" is always the server
   (authState.user, refreshed from /api/auth/me) -- this mirrors that
   check for UI purposes only, it never gates the actual purchase.
───────────────────────────────────────────── */
function _isProActive(user) {
  if (!user || user.plan_type !== 'pro' || user.plan_status !== 'active') return false;
  if (!user.plan_active_until) return false;
  return new Date(user.plan_active_until).getTime() > Date.now();
}

function _formatPlanUntil(iso) {
  if (!iso) return '';
  try {
    return new Date(iso).toLocaleDateString('ru-RU', { day: 'numeric', month: 'long', year: 'numeric' });
  } catch (_e) {
    return '';
  }
}

/* Public plan catalog -- the ONLY place plan names, prices and limits live
   on the frontend. Only user-facing facts belong here (no GPU budgets,
   margins, fees). `id` is what POST /billing/checkout receives; the
   backend PLAN_CATALOG (scripts/pricing.py) must know the same ids --
   tests/test_pricing_catalog.py fails if prices/limits drift apart.
   `recommended` marks the plan preselected when a mode is opened. */
const SONYA_PRICING = {
  period: 'месяц',
  modes: [
    {
      id: 'cut', label: 'Нарезка', quotaLabel: 'Обработки',
      plans: [
        { id: 'cut_start',  name: 'Start',  price: 1090, quota: 30, source: 'до 60 минут' },
        { id: 'cut_pro',    name: 'Pro',    price: 2690, quota: 50, source: 'до 2 часов', recommended: true },
        { id: 'cut_studio', name: 'Studio', price: 4990, quota: 80, source: 'до 3 часов' },
      ],
    },
    {
      id: 'trailer', label: 'Трейлер', quotaLabel: 'Генерации',
      plans: [
        { id: 'trailer_start',  name: 'Start',  price: 1190, quota: 8,  source: 'до 90 минут' },
        { id: 'trailer_pro',    name: 'Pro',    price: 2190, quota: 12, source: 'до 2 часов', recommended: true },
        { id: 'trailer_studio', name: 'Studio', price: 4990, quota: 24, source: 'до 3 часов' },
      ],
    },
    {
      id: 'streamer', label: 'Стример', quotaLabel: 'Стримы',
      plans: [
        { id: 'streamer_start', name: 'Start', price: 1990, quota: 10, source: 'до 4 часов' },
      ],
    },
    {
      // One-off pack (not a subscription): never expires, spent after the
      // 20 free images of the month (backend: prod_generation_api._gen_quota).
      id: 'gen', label: 'Gen', quotaLabel: 'Картинки', oneOff: true,
      note: 'Если у вас пайплайн — попросите администратора выдать вам доступ к SONYA API.',
      plans: [
        { id: 'gen_100', name: '100 картинок', price: 349, quota: 100, source: 'не сгорает' },
      ],
    },
    {
      // One-off pack: never expires, spent after the free quota (backend: prod_job_store.create_job_with_quota pack_mode).
      id: 'beep', label: 'BEEP', quotaLabel: 'Обработки', oneOff: true,
      note: 'Сначала тратятся бесплатные обработки, потом пакет. Безлимит Pro пакет не тратит.',
      plans: [
        { id: 'beep_10', name: '10 обработок', price: 249, quota: 10, source: 'не сгорает' },
      ],
    },
  ],
};

// checkoutAvailable: whether the server has a payment provider configured
// (GET /billing/checkout-availability). Starts false and stays false on any
// error -- the pay CTA is then "Оплата скоро будет доступна", never a live
// payment and never an error after the click.
const _paywall = { modeId: 'cut', planId: null, checkoutAvailable: false };

async function loadCheckoutAvailability() {
  try {
    const res = await apiFetch('/billing/checkout-availability');
    const data = res.ok ? await safeJson(res) : null;
    _paywall.checkoutAvailable = Boolean(data && data.available === true);
  } catch (_e) {
    _paywall.checkoutAvailable = false;
  }
  _syncPaywallPayButton();
}

// { mode, planId, consent } a guest chose before being sent to sign in;
// in-memory on purpose -- auth happens in-page, no reload in between.
let _pendingCheckout = null;

// Pricing modes that sit behind a feature flag: mode id -> flag name. They
// are listed in the plan picker only while the flag is on. The mode entry
// itself stays in SONYA_PRICING: the account menu names existing
// subscriptions from it, and tests/test_pricing_catalog.py pins its prices.
const SONYA_FLAGGED_PRICING_MODES = { trailer: 'trailer' };

// Modes a user may BUY right now.
function _visiblePricingModes() {
  return SONYA_PRICING.modes.filter(m => !SONYA_FLAGGED_PRICING_MODES[m.id] || _featureOn(SONYA_FLAGGED_PRICING_MODES[m.id]));
}

function _pricingMode(modeId) {
  const visible = _visiblePricingModes();
  return visible.find(m => m.id === modeId) || visible[0];
}

function _defaultPlanId(mode) {
  return (mode.plans.find(p => p.recommended) || mode.plans[0]).id;
}

function _selectedPlan() {
  return _pricingMode(_paywall.modeId).plans.find(p => p.id === _paywall.planId) || null;
}

function _formatRub(amount) {
  return `${Number(amount).toLocaleString('ru-RU')} ₽`;
}

// Generation clipType -> pricing mode, so a 402 from the trailer or
// streamer flow opens the picker on the plans that actually apply.
function _pricingModeForClipType(clipType) {
  const t = String(clipType || '').toLowerCase();
  if (t.includes('trailer')) return 'trailer';
  if (t.includes('streamer')) return 'streamer';
  return 'cut';
}

function renderPaywallModes() {
  const wrap = document.getElementById('paywall-plan-modes');
  if (!wrap) return;
  wrap.innerHTML = _visiblePricingModes().map(m => `
    <button class="sonya-auth-tab${m.id === _paywall.modeId ? ' is-active' : ''}" type="button"
            role="tab" aria-selected="${m.id === _paywall.modeId}" aria-controls="paywall-plans"
            tabindex="${m.id === _paywall.modeId ? '0' : '-1'}" data-plan-mode="${m.id}">${m.label}</button>`
  ).join('');
}

function renderPaywallPlans({ animate = false } = {}) {
  const grid = document.getElementById('paywall-plans');
  if (!grid) return;
  const mode = _pricingMode(_paywall.modeId);
  grid.classList.toggle('is-solo', mode.plans.length === 1);
  grid.setAttribute('aria-label', `Тарифы · ${mode.label}`);
  grid.innerHTML = mode.plans.map((p, i) => `
    <label class="plan-card${animate ? ' is-entering' : ''}" style="--i:${i}">
      <input class="plan-card-input" type="radio" name="paywall-plan" value="${p.id}"${p.id === _paywall.planId ? ' checked' : ''}>
      <span class="plan-card-head">
        <span class="plan-card-radio" aria-hidden="true"></span>
        <span class="plan-card-name">${p.name}</span>
        ${p.recommended && mode.plans.length > 1 ? '<span class="acct-drop-plan plan-card-tag">Рекомендуем</span>' : ''}
      </span>
      <span class="plan-card-price">
        <span class="plan-card-amount">${_formatRub(p.price)}</span>
        <span class="plan-card-period">${mode.oneOff ? 'разово' : `/ ${SONYA_PRICING.period}`}</span>
      </span>
      <span class="plan-card-specs">
        <span class="plan-card-spec"><span class="plan-card-key">${mode.quotaLabel}</span><span class="plan-card-val">${p.quota}</span></span>
        <span class="plan-card-spec"><span class="plan-card-key">${mode.oneOff ? 'Срок' : 'Исходник'}</span><span class="plan-card-val">${p.source}</span></span>
      </span>
    </label>`
  ).join('') + (mode.note ? `<p class="plan-card-note">${mode.note}</p>` : '');
  _syncPaywallPayButton();
}

function setPaywallMode(modeId, { animate = true } = {}) {
  const mode = _pricingMode(modeId);
  if (mode.id === _paywall.modeId && _paywall.planId) return;
  _paywall.modeId = mode.id;
  _paywall.planId = _defaultPlanId(mode);
  renderPaywallModes();
  renderPaywallPlans({ animate });
}

// Pay button reflects both the consent checkbox and the selected plan.
function _syncPaywallPayButton() {
  const consent = document.getElementById('paywall-consent-checkbox');
  const payBtn = document.getElementById('paywall-pay-btn');
  if (!payBtn) return;
  const plan = _selectedPlan();
  const label = payBtn.querySelector('span');
  if (label) {
    label.textContent = !_paywall.checkoutAvailable ? 'Оплата скоро будет доступна'
      : plan ? `Оплатить ${_formatRub(plan.price)}` : 'Выберите тариф';
  }
  const enabled = Boolean(_paywall.checkoutAvailable && plan && consent && consent.checked);
  payBtn.disabled = !enabled;
  payBtn.setAttribute('aria-disabled', String(!enabled));
}

function showPaywallView(viewName) {
  document.querySelectorAll('#sonya-paywall-modal [data-paywall-view]').forEach(el => {
    el.classList.toggle('is-active', el.getAttribute('data-paywall-view') === viewName);
  });
  // Only the plan picker needs the wide sheet; loading/error/already-pro
  // keep the compact single-column modal.
  document.querySelector('#sonya-paywall-modal .paywall-modal')
    ?.classList.toggle('is-plans', viewName === 'offer');
}

let _checkoutInFlight = false;

function openPaywallModal(opts = {}) {
  document.getElementById('sonya-paywall-modal').classList.add('is-open');
  // «Продлить или повысить тариф» when the period's videos are spent (quota-glass.js), «Тарифы» otherwise
  const plansTitle = document.getElementById('paywall-plans-title');
  if (plansTitle) plansTitle.textContent = opts.title || 'Тарифы';

  if (_isProActive(authState.user) && opts.mode !== 'gen' && opts.mode !== 'beep') {
    const untilText = document.getElementById('paywall-active-until-text');
    if (untilText) {
      const until = _formatPlanUntil(authState.user.plan_active_until);
      untilText.textContent = until ? `Действует до ${until}` : 'Подписка активна.';
    }
    showPaywallView('already-pro');
    return;
  }

  // Reset the offer form every time it's (re)opened; keep the last mode
  // unless the caller knows which flow the user came from.
  const consent = document.getElementById('paywall-consent-checkbox');
  if (consent) consent.checked = Boolean(opts.consent);
  _paywall.planId = null;
  setPaywallMode(opts.mode || _paywall.modeId, { animate: false });
  if (opts.planId && _pricingMode(_paywall.modeId).plans.some(p => p.id === opts.planId)) {
    _paywall.planId = opts.planId;
    renderPaywallPlans();
  }
  showPaywallView('offer');
  loadCheckoutAvailability();
}

function closePaywallModal() {
  document.getElementById('sonya-paywall-modal').classList.remove('is-open');
}

/* ─────────────────────────────────────────────
   CHECKOUT  (POST /api/billing/checkout)
   One Idempotency-Key per purchase intent (plan), kept across a network
   retry, a re-click after an error, a Back from the payment page and a
   reload of this tab (sessionStorage) -- the backend answers all of them
   with the SAME payment. Cleared only by an answer that ends the intent
   (paid, plan refused, signed out).
   Selfwork smart payment (docs.selfwork.ru): the browser itself POSTs the
   server-signed form to Selfwork /init, straight from this page.
───────────────────────────────────────────── */
const _CHECKOUT_ATTEMPT_STORAGE = 'sonya_checkout_attempt';
const _PROVIDER_FORM_ORIGINS = ['https://pro.selfwork.ru'];
const _CHECKOUT_TIMEOUT_MS = 45000;
const _NAVIGATION_WATCHDOG_MS = 30000;
// Server error (detail.error) -> diagnostic code in the console log.
const _CHECKOUT_ERROR_CODES = {
  unauthorized: 'auth_session',
  payment_unavailable: 'checkout_disabled',
  selfwork_rejected: 'selfwork_rejected',
  internal_error: 'sonya_internal',
  wrong_origin: 'wrong_origin',
  unknown_plan: 'unknown_plan',
  feature_disabled: 'feature_disabled',
  subscription_active: 'subscription_active',
  invalid_idempotency_key: 'sonya_internal',
};
const _CONTINUE_SAME = 'Нажмите «Попробовать снова» — продолжим тот же платёж, новый не создаётся.';
const _CHECKOUT_ERROR_TEXT = {
  unauthorized: 'Сессия истекла. Войдите снова, чтобы продолжить оплату.',
  unknown_plan: 'Этот тариф недоступен. Обновите страницу и попробуйте снова.',
  feature_disabled: 'Этот тариф временно недоступен. Обновите страницу и выберите другой.',
  subscription_active: 'У вас уже есть тариф для этого режима, и в нём ещё остались операции. Новый можно оформить, когда они закончатся.',
  payment_unavailable: 'Оплата временно отключена. Попробуйте позже.',
  selfwork_rejected: 'Платёжная система не подтвердила этот платёж. Напишите в поддержку — повторно деньги не спишутся.',
  internal_error: `Не удалось подготовить платёж. ${_CONTINUE_SAME}`,
};
let _checkoutAttempt = null; // { planId, key }
let _navigationWatchdog = null;

// crypto.randomUUID is missing before iOS Safari 15.4 -- never let that
// make checkout impossible.
function _newCheckoutKey() {
  if (typeof crypto.randomUUID === 'function') return crypto.randomUUID();
  const b = crypto.getRandomValues(new Uint8Array(16));
  return Array.from(b, (x) => x.toString(16).padStart(2, '0')).join('');
}

function _checkoutKeyFor(planId) {
  if (!_checkoutAttempt) {
    try { _checkoutAttempt = JSON.parse(sessionStorage.getItem(_CHECKOUT_ATTEMPT_STORAGE) || 'null'); } catch (_e) { _checkoutAttempt = null; }
  }
  if (!_checkoutAttempt || _checkoutAttempt.planId !== planId || !_checkoutAttempt.key) {
    _checkoutAttempt = { planId, key: _newCheckoutKey() };
    try { sessionStorage.setItem(_CHECKOUT_ATTEMPT_STORAGE, JSON.stringify(_checkoutAttempt)); } catch (_e) { /* memory only */ }
  }
  return _checkoutAttempt.key;
}

function _clearCheckoutAttempt() {
  _checkoutAttempt = null;
  try { sessionStorage.removeItem(_CHECKOUT_ATTEMPT_STORAGE); } catch (_e) { /* nothing stored */ }
}

function _logCheckout(code, extra) {
  // Diagnostic only: never the form fields / signature / cookies.
  console.warn(`[SONYA checkout] ${code}` + (extra ? ` ${extra}` : ''));
}

// Top-level POST of the provider form from this page (Origin/Referer =
// sonya.group, as Selfwork requires). false = not a form we recognise.
function _postProviderForm(form) {
  let action;
  try { action = new URL(form.action); } catch (_e) { return false; }
  if (!_PROVIDER_FORM_ORIGINS.includes(action.origin) || String(form.method).toUpperCase() !== 'POST'
      || !Array.isArray(form.fields)) return false;
  const el = document.createElement('form');
  el.method = 'POST';
  el.action = action.href;
  el.hidden = true;
  form.fields.forEach(([name, value]) => {
    const input = document.createElement('input');
    input.type = 'hidden';
    input.name = name;
    input.value = value;
    el.appendChild(input);
  });
  document.body.appendChild(el);
  el.submit();
  return true;
}

// The page normally unloads right after the form post / location change.
// Still here long after that -> the navigation was blocked or stalled;
// the retry button continues the same payment (same Idempotency-Key).
function _watchNavigation(invoiceId) {
  clearTimeout(_navigationWatchdog);
  _navigationWatchdog = setTimeout(() => {
    // A hidden page is leaving (or backgrounded) -- not a failed navigation.
    if (document.visibilityState === 'hidden') return;
    _logCheckout('navigation_failed', `invoice_id=${invoiceId}`);
    _showPaywallError(`Страница оплаты не открылась. ${_CONTINUE_SAME}`);
  }, _NAVIGATION_WATCHDOG_MS);
}

function _onCheckoutPageHide() {
  clearTimeout(_navigationWatchdog);
}

// Back from the payment page restores this page from the bfcache with the
// paywall still on "loading" -- reset it so the user can continue (the
// next click reuses the same payment).
function _onCheckoutPageShow(e) {
  if (!e.persisted) return;
  clearTimeout(_navigationWatchdog);
  if (!_checkoutInFlight) return;
  _checkoutInFlight = false;
  if (document.getElementById('sonya-paywall-modal')?.classList.contains('is-open')) {
    showPaywallView('offer');
    _syncPaywallPayButton();
  }
}

async function _postCheckout(planId) {
  const send = () => apiFetch('/billing/checkout', {
    method: 'POST',
    headers: { 'Idempotency-Key': _checkoutKeyFor(planId) },
    body: JSON.stringify({ plan_id: planId }),
    timeoutMs: _CHECKOUT_TIMEOUT_MS,
  });
  let res = await send();
  if (res._networkError && res._networkErrorKind === 'network') {
    // The request may have reached the server -- same key, so a retry
    // can only return the payment it already created.
    _logCheckout('network_error_retry');
    await new Promise(r => setTimeout(r, 1500));
    res = await send();
  }
  return res;
}

async function startCheckout() {
  if (_checkoutInFlight) return; // double-click / double-submit guard
  const plan = _selectedPlan();
  if (!plan || !_paywall.checkoutAvailable) return;

  // Guest: auth is required only now, at the pay CTA. Remember the exact
  // choice so handleVerifyCode() can reopen the picker on it.
  if (!authState.user) {
    const mode = _pricingMode(_paywall.modeId);
    closePaywallModal();
    openAuthModal('login',
      `Войдите или зарегистрируйтесь, чтобы оформить «${mode.label} · ${plan.name}» — ${_formatRub(plan.price)} / ${SONYA_PRICING.period}`,
      { resumeCheckout: { mode: mode.id, planId: plan.id, consent: true } });
    return;
  }

  _checkoutInFlight = true;

  const payBtn = document.getElementById('paywall-pay-btn');
  if (payBtn) { payBtn.disabled = true; payBtn.setAttribute('aria-disabled', 'true'); }
  showPaywallView('loading');

  try {
    const res = await _postCheckout(plan.id);
    if (res._networkError) {
      _logCheckout(`${res._networkErrorKind}_error`);
      _showPaywallError(res._networkErrorKind === 'timeout'
        ? `Сервер слишком долго отвечает. ${_CONTINUE_SAME}`
        : `Нет связи с сервером. Проверьте интернет. ${_CONTINUE_SAME}`);
      return;
    }

    const data = await safeJson(res);
    if (!res.ok) {
      const err = data?.detail?.error || (res.status === 401 ? 'unauthorized' : 'unknown');
      _logCheckout(_CHECKOUT_ERROR_CODES[err] || 'sonya_internal', `http=${res.status} error=${err} trace_id=${data?.detail?.trace_id || '-'}`);
      if (err === 'wrong_origin' && data.detail.canonical_url) {
        // Payments only start on the canonical site (the provider checks
        // the domain) -- continue there, nothing was created here.
        window.location.href = `${data.detail.canonical_url}/?paywall=1`;
        return;
      }
      if (['unauthorized', 'unknown_plan', 'feature_disabled', 'subscription_active'].includes(err)) _clearCheckoutAttempt();
      _showPaywallError(_CHECKOUT_ERROR_TEXT[err] || _CHECKOUT_ERROR_TEXT.internal_error);
      return;
    }

    if (data?.status === 'paid' && data.redirect_url) {
      _clearCheckoutAttempt();
      window.location.href = data.redirect_url;
      return;
    }
    if (data?.form && _postProviderForm(data.form)) {
      _watchNavigation(data.invoice_id);
      return;
    }
    if (data?.redirect_url) {
      window.location.href = data.redirect_url;
      _watchNavigation(data.invoice_id);
      return;
    }
    _logCheckout('unknown_provider_response', `http=${res.status} invoice_id=${data?.invoice_id || '-'}`);
    _showPaywallError(_CHECKOUT_ERROR_TEXT.internal_error);
  } catch (_e) {
    _logCheckout('sonya_internal', 'exception');
    _showPaywallError(_CHECKOUT_ERROR_TEXT.internal_error);
  }
}

function _showPaywallError(message) {
  _checkoutInFlight = false;
  const el = document.getElementById('paywall-error-message');
  if (el) el.textContent = message;
  showPaywallView('error');
}

/* ─────────────────────────────────────────────
   GENERATE FLOW  (called from app.js)
   Returns: 'ok' | 'auth' | 'paywall' | 'error'
───────────────────────────────────────────── */
async function checkAndCreateVideoJob(formData) {
  try { return await _checkAndCreateVideoJob(formData); }
  finally { window.SONYA_TRIM = null; }
}

async function _checkAndCreateVideoJob(formData) {
  // A. Check auth
  const meRes = await apiGetMe();
  if (meRes.status === 401) {
    // Expected guest state -- prompt to sign in, not an error.
    openAuthModal('login', 'Создайте аккаунт, чтобы получить 1 бесплатное видео');
    return 'auth';
  }
  if (!meRes.ok) {
    // Real failure (5xx / network) -- must NOT be presented as "please log
    // in", that hides an actual outage/bug behind a normal-looking prompt.
    reportAuthError('checkAndCreateVideoJob:auth/me', meRes);
    const meData = await safeJson(meRes);
    showToast(meData?.detail || 'Не удалось проверить авторизацию. Попробуйте позже.', 'error');
    return 'error';
  }

  // B. Try to create video job — URL mode goes through the ingest+poll
  // flow above; upload mode is the original direct-multipart path. Both
  // resolve to the same Response-shaped result, so everything below is
  // shared.
  const isUrlMode = formData?.source?.mode === 'url' && String(formData?.source?.url || '').trim();

  // A0. 2026-11-22 «Мой лимит»: nothing left for this run -> «Продлить / повысить» right away, before any upload.
  if (window.SonyaQuota) {
    const pre = await window.SonyaQuota.precheck(formData);
    if (pre === 'paywall') return 'paywall';
    if (pre === 'short') return 'cancel';
  }

  // B0. A file's length is read from its metadata BEFORE uploading it (nothing is sent for a too-long video).
  if (!isUrlMode && !window.SONYA_TRIM) {
    const f = formData?.file || formData?.source?.file ||
      (typeof appState !== 'undefined' ? appState.uploadedFile : null);
    const maxSec = f ? await apiGetSourceLimit(normalizeMode(formData?.clipType || formData?.mode)) : null;
    const durSec = maxSec ? await probeVideoDurationSec(f) : null;
    if (maxSec && durSec && durSec > maxSec + 1) {
      const seg = await askTrimDialog(durSec, maxSec);
      if (!seg) return 'cancel';
      window.SONYA_TRIM = seg;
    }
  }
  const jobRes = isUrlMode ? await apiCreateVideoJobFromUrl(formData) : await apiCreateVideoJob(formData);

  if (jobRes._networkError) {
    // Client doesn't know whether the server received the request --
    // keep the same Idempotency-Key so a retry of this same attempt
    // can't create a duplicate job.
    const data = await safeJson(jobRes);
    showToast(data?.detail || 'Ошибка создания задания', 'error');
    return 'error';
  }

  // Any real HTTP response means this attempt is settled one way or
  // another -- the next click is always a new logical run.
  clearJobIdempotencyKey();
  const jobData = await safeJson(jobRes);

  if ([200, 201, 202].includes(jobRes.status)) {
    // The response just arrived -- this is the one point where "creating
    // the task" is a real, observable transition, unlike the wait leading
    // up to it (see docs/patches/generation-upload-timeout-plan.md §3).
    if (typeof window.sonyaSetGenerateButtonText === 'function') window.sonyaSetGenerateButtonText('Создаём задачу…');
    // Refresh account state to update free_video_used counter
    window.SONYA_LAST_JOB = jobData;
    console.log('[SONYA] generation job created', jobData);
    await refreshAuthState();
    // «Мой лимит» drops at once (refreshAuthState just reset the cache; afterLaunch re-reads the server)
    if (window.SonyaQuota) window.SonyaQuota.afterLaunch(formData);
    return 'ok';
  }

  if (jobRes.status === 402) {
    const code = jobData?.detail?.code;
    if (PAYWALL_CODES.includes(code)) {
      if (code !== 'FREE_PLAN_USED' && jobData?.detail?.message) showToast(jobData.detail.message, 'error');
      openPaywallModal({ mode: jobData?.detail?.plan_mode || _pricingModeForClipType(formData?.clipType),
        title: code === 'PLAN_LIMIT_REACHED' || code === 'SUBSCRIPTION_EXPIRED' ? 'Продлить или повысить тариф' : undefined });
      if (window.SonyaQuota) window.SonyaQuota.refreshLaunch(true);
      return 'paywall';
    }
  }

  if (jobRes.status === 401) {
    openAuthModal('login', 'Создайте аккаунт, чтобы получить 1 бесплатное видео');
    return 'auth';
  }

  // Generic error
  const msg = jobData?.detail?.message || jobData?.detail || 'Ошибка создания задания';
  showToast(msg, 'error');
  return 'error';
}

/* ─────────────────────────────────────────────
   TELEGRAM CONNECT (from the bot: /?tg=connect)
───────────────────────────────────────────── */
let _pendingTgConnect = false;
async function _tgConnectGo() {
  try {
    const st = await apiFetch('/telegram/status');
    if (st.ok) {
      const d = await safeJson(st);
      if (d && d.linked) {
        showToast('Telegram уже подключён — вернитесь в бота и нажмите «Я привязал — продолжить».', 'success');
        return;
      }
    }
    const res = await apiFetch('/telegram/link-token', { method: 'POST', body: '{}' });
    const data = await safeJson(res);
    if (res.ok && data && data.deep_link) {
      showToast('Открываем Telegram…', 'info');
      window.location.href = data.deep_link;
      return;
    }
  } catch (e) { /* fall through */ }
  showToast('Не удалось подключить Telegram. Попробуйте ещё раз.', 'error');
}

/* ─────────────────────────────────────────────
   TOAST
───────────────────────────────────────────── */
function showToast(message, type = 'info') {
  let toast = document.getElementById('sonya-toast');
  if (!toast) {
    toast = document.createElement('div');
    toast.id = 'sonya-toast';
    document.body.appendChild(toast);
  }
  const icons = { success: '\u2713', error: '\u26A0', info: '' };
  const ic = icons[type] || '';
  toast.innerHTML = ic
    ? `<span class="sonya-toast-ic" aria-hidden="true">${ic}</span><span>${escHtml(message)}</span>`
    : escHtml(message);
  toast.className = `sonya-toast sonya-toast--${type} is-visible`;
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => toast.classList.remove('is-visible'), 4200);
}

/* ─────────────────────────────────────────────
   HELPERS
───────────────────────────────────────────── */
function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

async function safeJson(res) {
  try { return await res.json(); } catch { return {}; }
}

/* ─────────────────────────────────────────────
   INIT
───────────────────────────────────────────── */
function initAuth() {
  // Account button
  const accountBtn = document.getElementById('sonya-account-btn');
  if (accountBtn) {
    accountBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      const drop = document.getElementById('sonya-account-drop');
      if (drop && drop.classList.contains('is-open')) closeAccountDropdown();
      else openAccountDropdown();
    });
  }

  /* ── Auth modal ── */
  // Close
  document.getElementById('sonya-auth-close')?.addEventListener('click', closeAuthModal);
  document.getElementById('sonya-auth-overlay')?.addEventListener('click', (event) => {
    if (event.target.id === 'sonya-auth-overlay') closeAuthModal();
  });

  // Tabs (login / register)
  document.querySelectorAll('[data-auth-tab]').forEach(tab => {
    tab.addEventListener('click', () => {
      setAuthView(tab.dataset.authTab);
      clearAuthError();
    });
  });

  // Inline switch buttons inside views
  document.querySelectorAll('[data-switch-auth]').forEach(btn => {
    btn.addEventListener('click', () => {
      setAuthView(btn.dataset.switchAuth);
      clearAuthError();
    });
  });

  // Login flow
  const loginEmail = document.getElementById('sonya-login-email');
  if (loginEmail) {
    loginEmail.addEventListener('input', clearAuthError);
    loginEmail.addEventListener('keydown', e => { if (e.key === 'Enter') handleRequestCode('login'); });
  }
  document.getElementById('sonya-login-code-btn')?.addEventListener('click', () => handleRequestCode('login'));

  // Register flow
  const regEmail = document.getElementById('sonya-register-email');
  if (regEmail) {
    regEmail.addEventListener('input', clearAuthError);
    regEmail.addEventListener('keydown', e => { if (e.key === 'Enter') handleRequestCode('registration'); });
  }
  document.getElementById('sonya-register-code-btn')?.addEventListener('click', () => handleRequestCode('registration'));

  // Code flow
  const codeInput = document.getElementById('sonya-code-input');
  if (codeInput) {
    codeInput.addEventListener('input', () => {
      codeInput.value = codeInput.value.replace(/\D/g, '').slice(0, 6);
      clearAuthError();
    });
    codeInput.addEventListener('keydown', e => { if (e.key === 'Enter') handleVerifyCode(); });
  }
  document.getElementById('sonya-verify-code-btn')?.addEventListener('click', handleVerifyCode);
  document.getElementById('sonya-auth-resend')?.addEventListener('click', handleResendCode);
  document.getElementById('sonya-auth-back')?.addEventListener('click', goBackFromCodeView);

  // Esc closes any open modal
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    const auth = document.getElementById('sonya-auth-overlay');
    if (auth && auth.classList.contains('is-open')) { closeAuthModal(); return; }
    const open = document.querySelector('.sonya-modal-overlay.is-open');
    if (!open) return;
    if (open.id === 'sonya-paywall-modal') closePaywallModal();
  });

  // Paywall modal
  document.getElementById('paywall-modal-close')?.addEventListener('click', closePaywallModal);
  document.getElementById('sonya-paywall-modal')?.addEventListener('click', e => {
    if (e.target === e.currentTarget) closePaywallModal();
  });
  document.getElementById('paywall-later-btn')?.addEventListener('click', closePaywallModal);
  document.getElementById('paywall-error-later-btn')?.addEventListener('click', closePaywallModal);
  document.getElementById('paywall-already-pro-close-btn')?.addEventListener('click', closePaywallModal);

  // Plan picker: mode tabs (click + arrow keys) and plan radios.
  const planModes = document.getElementById('paywall-plan-modes');
  planModes?.addEventListener('click', e => {
    const tab = e.target.closest('[data-plan-mode]');
    if (tab) setPaywallMode(tab.dataset.planMode);
  });
  planModes?.addEventListener('keydown', e => {
    if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
    const ids = _visiblePricingModes().map(m => m.id);
    const step = e.key === 'ArrowRight' ? 1 : -1;
    const next = ids[(ids.indexOf(_paywall.modeId) + step + ids.length) % ids.length];
    setPaywallMode(next);
    planModes.querySelector(`[data-plan-mode="${next}"]`)?.focus();
    e.preventDefault();
  });
  document.getElementById('paywall-plans')?.addEventListener('change', e => {
    if (e.target.name !== 'paywall-plan') return;
    _paywall.planId = e.target.value;
    _syncPaywallPayButton();
  });

  document.getElementById('paywall-consent-checkbox')?.addEventListener('change', _syncPaywallPayButton);
  document.getElementById('paywall-pay-btn')?.addEventListener('click', startCheckout);
  window.addEventListener('pagehide', _onCheckoutPageHide);
  window.addEventListener('pageshow', _onCheckoutPageShow);
  document.getElementById('paywall-retry-btn')?.addEventListener('click', () => {
    showPaywallView('offer');
    _checkoutInFlight = false;
    // startCheckout() disables the pay button unconditionally
    // before the request; re-sync it with the (still-checked) consent
    // checkbox and selected plan instead of leaving it stuck disabled.
    _syncPaywallPayButton();
  });

  // Initial state fetch (non-blocking). If we were sent back here from
  // payment/fail.html's "Попробовать снова" link (?paywall=1), reopen the
  // paywall once we know the current auth/plan state, then drop the
  // query param so a page refresh doesn't reopen it again. The session
  // could have expired between the failed payment and this click (or the
  // link could be opened signed-out in another browser) -- purchase must
  // never start before auth, so a signed-out visitor gets the login modal
  // instead of the paywall.
  refreshAuthState().then(function () {
    var params = new URLSearchParams(window.location.search);
    if (params.get('tg') === 'connect') {
      // @sonya_group_ai «Войти и привязать» (2026-11-20): signed in -> the existing link-token -> back to the bot;
      // a guest signs in first (the bot keeps the video link and continues after /start TOKEN).
      params.delete('tg');
      var q = params.toString();
      history.replaceState(null, '', window.location.pathname + (q ? '?' + q : '') + window.location.hash);
      if (authState.user) _tgConnectGo();
      else { _pendingTgConnect = true; openAuthModal('login', 'Войдите, чтобы подключить Telegram к SONYA'); }
    }
    if (params.get('signup') === '1') {
      // sonya.group/invite «Попробовать» (2026-11-23): a guest gets the registration form, a signed-in user just the home page.
      params.delete('signup');
      var sq = params.toString();
      history.replaceState(null, '', window.location.pathname + (sq ? '?' + sq : '') + window.location.hash);
      if (!authState.user) openAuthModal('register');
    }
    if (params.get('paywall') === '1') {
      if (authState.user) openPaywallModal();
      else openAuthModal('login');
      params.delete('paywall');
      var qs = params.toString();
      history.replaceState(null, '', window.location.pathname + (qs ? '?' + qs : ''));
    }
  });
}

// Auto-init when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initAuth);
} else {
  initAuth();
}
