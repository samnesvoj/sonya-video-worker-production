"""crop-shots-v5: calm, face-safe 9:16 framing for one clip (owner 2026-11-13).

Owner, after clip 4 of job cd6916ff: «лицо не попадает целиком … режет экран на два, будто там два лица —
а их нет; и потом обратно растягивает». Root causes in v3:
  * a fast-moving (hand-held / selfie) face fell into two "people" clusters → a false split screen;
  * a dead-zone follower lagged a moving face → the face left the window;
  * the layout was re-decided every 0.5 s → single / split / close-up flicker.

v5 rules (also in docs CUT_PLAYBOOK, section «Кроп v5»):
  1. People are TRACKS; two tracks that are never on screen at the same time are one person (merged).
  2. A split needs a real second person: comparable size (≥ 0.6 of the main face), confident, fully inside
     the frame, alive (talks or moves, not a poster), on screen together with the main person for ≥ 2 s.
  3. A layout lasts ≥ MIN_LAYOUT_SEC; it changes only at a shot cut or when a second person really comes / goes.
  4. Large face (≥ CLOSEUP_FRAC of the 9:16 window) or a face moving faster than a 9:16 window can follow
     calmly → close-up layout: wide square window (full source height), the scene blurred above / below.
  5. The window path keeps the WHOLE face inside with a margin at every sample (lazy camera: moves only as
     much as needed, forward+backward average so it starts moving a little early) — never a lagging follow.
  6. Aspect is never changed: every window is scaled uniformly.
The returned dict has the v3 shape (keyframes / closeup / split / shots / cuts) so render + text_zones are unchanged.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

CROP_VERSION_V5 = "crop-shots-v5"
MIN_SCORE = 0.8              # YuNet confidence for any face we frame
SPLIT_SCORE = 0.85           # … and for the second person of a split
SPLIT_SIZE = 0.6             # second face width ≥ this × main face width
SPLIT_MIN_SEC = 2.0          # both on screen together at least this long
SPLIT_COVER = 0.7            # … in at least this share of the samples of that stretch
MIN_LAYOUT_SEC = 1.6         # no layout shorter than this (except a whole clip shorter than it)
CLOSEUP_FRAC = 0.45          # median face width ≥ this × 9:16 window width → close-up layout
MARGIN_FRAC = 0.22           # face keeps this × its width free on both sides inside the window
CAPTION_BAND = 0.70          # burned-in captions: text boxes whose top is below this × source height …
CAPTION_CENTER = 0.22        # … centred within this × width of the middle


# ── people ───────────────────────────────────────────────────────────────────────────────────────────

def faces_of(samples: List[Dict[str, Any]], H: int, min_score: float = MIN_SCORE) -> List[Tuple[float, List[float]]]:
    out = []
    for s in samples:
        for b in s.get("faces") or []:
            sc = b[4] if len(b) > 4 else 1.0
            if sc >= min_score and b[3] <= 0.8 * H and b[2] > 0:
                out.append((float(s["t"]), b))
    return out


def tracks(dets: List[Tuple[float, List[float]]], max_gap: float = 1.0) -> List[Dict[str, Any]]:
    """Faces → person tracks. A box joins the track whose last box is near (≤ 1 face width, similar size,
    seen ≤ max_gap ago); then tracks that never share a sample and have a similar face size are merged
    (a hand-held camera moves one face fast; a turned head drops a few detections)."""
    out: List[Dict[str, Any]] = []
    by_t: Dict[float, List[List[float]]] = {}
    for t, b in dets:
        by_t.setdefault(t, []).append(b)
    for t in sorted(by_t):
        free = [tr for tr in out if t - tr["boxes"][-1][0] <= max_gap]
        pairs = []
        for i, b in enumerate(by_t[t]):
            cx, w = b[0] + b[2] / 2.0, float(b[2])
            for tr in free:
                _, lb = tr["boxes"][-1]
                lcx, lw = lb[0] + lb[2] / 2.0, float(lb[2])
                d = abs(cx - lcx)
                if d <= 1.0 * max(w, lw) and 0.6 <= w / max(lw, 1.0) <= 1.67:
                    pairs.append((d, i, id(tr), tr))
        used_b, used_t = set(), set()
        for d, i, k, tr in sorted(pairs, key=lambda p: p[0]):
            if i in used_b or k in used_t:
                continue
            used_b.add(i)
            used_t.add(k)
            tr["boxes"].append((t, by_t[t][i]))
        for i, b in enumerate(by_t[t]):
            if i not in used_b:
                out.append({"boxes": [(t, b)]})
    # merge tracks that never co-occur and look like the same face size
    out.sort(key=lambda tr: -len(tr["boxes"]))
    merged: List[Dict[str, Any]] = []
    for tr in out:
        ts = {t for t, _ in tr["boxes"]}
        w = _med([b[2] for _, b in tr["boxes"]])
        for m in merged:
            mts = {t for t, _ in m["boxes"]}
            mw = _med([b[2] for _, b in m["boxes"]])
            if len(ts & mts) <= max(1, 0.05 * len(ts)) and 0.7 <= w / max(mw, 1.0) <= 1.43:
                m["boxes"] = sorted(m["boxes"] + [x for x in tr["boxes"] if x[0] not in mts], key=lambda x: x[0])
                break
        else:
            merged.append({"boxes": list(tr["boxes"])})
    for tr in merged:
        bs = [b for _, b in tr["boxes"]]
        tr["w"] = _med([b[2] for b in bs])
        tr["score"] = sum(b[4] if len(b) > 4 else 1.0 for b in bs) / len(bs)
        tr["weight"] = sum(b[2] * b[2] * (b[4] if len(b) > 4 else 1.0) for b in bs)
        mm = [float(b[5]) for b in bs if len(b) > 5 and b[5] is not None]
        tr["mouth"] = sum(mm) / len(mm) if len(mm) >= 5 else 0.0
        cxs = [b[0] + b[2] / 2.0 for b in bs]
        mu = sum(cxs) / len(cxs)
        tr["x_std"] = (sum((c - mu) ** 2 for c in cxs) / len(cxs)) ** 0.5
        tr["times"] = {t for t, _ in tr["boxes"]}
    merged.sort(key=lambda tr: -tr["weight"])
    return merged


def _med(v: List[float]) -> float:
    s = sorted(v)
    if not s:
        return 0.0
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def real_partner(main: Dict[str, Any], other: Dict[str, Any], W: int) -> bool:
    """Is `other` a second person worth half the screen? (size, confidence, fully in frame, alive, apart)."""
    if other["w"] < SPLIT_SIZE * main["w"] or other["score"] < SPLIT_SCORE:
        return False
    bs = [b for _, b in other["boxes"]]
    inside = sum(1 for b in bs if b[0] >= 0.005 * W and b[0] + b[2] <= 0.995 * W) / len(bs)
    if inside < 0.8:
        return False                       # a face cut by the frame edge: a passer-by / shoulder
    alive = other["mouth"] >= max(0.8, 0.35 * main["mouth"]) or other["x_std"] >= 0.004 * W
    if not alive:
        return False                       # a poster, a photo on a screen
    gap = abs(_cx(main) - _cx(other))
    return gap >= 0.8 * max(main["w"], other["w"])   # a reflection / the same face twice sits on top of it


def pair_fits(main: Dict[str, Any], other: Dict[str, Any], win: int) -> bool:
    """Both faces (with their margins) fit in one window: frame them together — a split would show the
    same pair twice (owner 2026-11-09 «почему в кадре два одинаковых фрагмента»)."""
    a, b = sorted((main, other), key=_cx)
    left = _cx(a) - a["w"] / 2.0 - MARGIN_FRAC * a["w"]
    right = _cx(b) + b["w"] / 2.0 + MARGIN_FRAC * b["w"]
    return right - left <= 0.92 * win or (_cx(b) + b["w"] / 2.0) - (_cx(a) - a["w"] / 2.0) <= 0.95 * win


def pair_boxes(main: Dict[str, Any], other: Dict[str, Any]) -> List[Tuple[float, List[float]]]:
    """Per sample: the union box of both faces (one alone where the other is not seen)."""
    ob = {t: b for t, b in other["boxes"]}
    out = []
    for t, b in main["boxes"]:
        o = ob.get(t)
        if o is None:
            out.append((t, b))
            continue
        x0, x1 = min(b[0], o[0]), max(b[0] + b[2], o[0] + o[2])
        out.append((t, [x0, min(b[1], o[1]), x1 - x0, max(b[3], o[3]), b[4]]))
    return out


def _cx(tr: Dict[str, Any]) -> float:
    return _med([b[0] + b[2] / 2.0 for _, b in tr["boxes"]])


# ── window path ──────────────────────────────────────────────────────────────────────────────────────

def lazy_path(boxes: List[Tuple[float, List[float]]], s0: float, s1: float, win: int, max_x: int,
              smooth_sec: float = 0.4) -> List[List[float]]:
    """Window x keyframes over [s0, s1] that keep every face box inside [x + m, x + win - m].
    Forward lazy pass + backward lazy pass, averaged (stays feasible: both are inside each sample's
    interval), light moving average, re-clamped to the feasible interval, then simplified."""
    def even(x: float) -> int:
        return int(max(0, min(max_x, round(x))) // 2 * 2)
    pts = sorted(((t, b) for t, b in boxes if s0 - 1e-6 <= t <= s1 + 1e-6), key=lambda p: p[0])
    if not pts:
        return []
    iv = []
    for t, b in pts:
        m = MARGIN_FRAC * b[2]
        lo, hi = b[0] + b[2] + m - win, b[0] - m           # window x range that contains the padded face
        if lo > hi:                                          # face (+ margin) wider than the window: centre it
            lo = hi = b[0] + b[2] / 2.0 - win / 2.0
        iv.append((t, max(0.0, min(max_x, lo)), max(0.0, min(max_x, hi))))
    first = _med([(a + b) / 2.0 for _, a, b in iv[:5]])
    fwd, x = [], first
    for _, lo, hi in iv:
        x = min(max(x, lo), hi)
        fwd.append(x)
    bwd, x = [0.0] * len(iv), _med([(a + b) / 2.0 for _, a, b in iv[-5:]])
    for i in range(len(iv) - 1, -1, -1):
        x = min(max(x, iv[i][1]), iv[i][2])
        bwd[i] = x
    path = [(a + b) / 2.0 for a, b in zip(fwd, bwd)]
    ts = [t for t, _, _ in iv]
    sm = []
    for i, t in enumerate(ts):
        win_v = [path[j] for j in range(len(ts)) if abs(ts[j] - t) <= smooth_sec / 2.0]
        v = sum(win_v) / len(win_v)
        sm.append(min(max(v, iv[i][1]), iv[i][2]))
    keys = [[round(max(s0, ts[0]) if len(ts) > 1 else s0, 3), even(sm[0])]]
    keys[0][0] = round(s0, 3)
    pts2 = [[round(t, 3), even(v)] for t, v in zip(ts, sm)]
    keys += _simplify(pts2, tol=max(4.0, 0.012 * win))[1:] if len(pts2) > 1 else []
    if keys[-1][0] < s1:
        keys.append([round(s1, 3), keys[-1][1]])
    return [k for k in keys if s0 - 1e-6 <= k[0] <= s1 + 1e-6]


def _simplify(p: List[List[float]], tol: float) -> List[List[float]]:
    """Ramer–Douglas–Peucker on (t, x) with the error measured in x."""
    if len(p) < 3:
        return p
    (t0, x0), (t1, x1) = p[0], p[-1]
    best, idx = -1.0, 0
    for i in range(1, len(p) - 1):
        t, x = p[i]
        xl = x0 + (x1 - x0) * (t - t0) / (t1 - t0) if t1 > t0 else x0
        e = abs(x - xl)
        if e > best:
            best, idx = e, i
    if best <= tol:
        return [p[0], p[-1]]
    return _simplify(p[:idx + 1], tol)[:-1] + _simplify(p[idx:], tol)


def path_travel(keys: List[List[float]]) -> float:
    return sum(abs(b[1] - a[1]) for a, b in zip(keys, keys[1:]))


# ── layouts ──────────────────────────────────────────────────────────────────────────────────────────

def _runs(flags: List[Tuple[float, bool]], s0: float, s1: float, min_sec: float,
          bridge_sec: float = 1.0) -> List[List[Any]]:
    """Per-sample «both people on screen» → [[flag, a, b]] covering [s0, s1]. A gap < bridge_sec between two
    True stretches is a turned head (bridged); a True stretch shorter than min_sec is a flicker (dropped)."""
    if not flags:
        return [[False, s0, s1]]
    runs: List[List[Any]] = []
    for t, f in flags:
        if runs and runs[-1][0] == f:
            runs[-1][2] = t
        else:
            runs.append([f, t, t])
    edges = [s0] + [(a[2] + b[1]) / 2.0 for a, b in zip(runs, runs[1:])] + [s1]
    runs = [[r[0], a, b] for r, a, b in zip(runs, edges, edges[1:])]

    def join(rs: List[List[Any]]) -> List[List[Any]]:
        out: List[List[Any]] = []
        for r in rs:
            if out and out[-1][0] == r[0]:
                out[-1][2] = r[2]
            else:
                out.append(list(r))
        return out
    runs = join([[True if (not r[0] and 0 < k < len(runs) - 1 and r[2] - r[1] < bridge_sec) else r[0], r[1], r[2]]
                 for k, r in enumerate(runs)])
    return join([[r[0] and r[2] - r[1] >= min_sec, r[1], r[2]] for r in runs])


def crop_track_v5(analysis: Optional[Dict[str, Any]], duration: float, *, width: int, height: int) -> Dict[str, Any]:
    from .smart_crop_adapter import crop_window, split_window
    W, H = int(width), int(height)
    cw, ch = crop_window(W, H)
    max_x = W - cw
    center = max_x // 2 // 2 * 2
    sq = (min(W, H) // 2) * 2
    sq_max = W - sq
    meta = (analysis or {}).get("meta") or {}
    info: Dict[str, Any] = {"crop_version": CROP_VERSION_V5, "source_width": W, "source_height": H,
                            "output_aspect": "9:16", "detector": meta.get("detector"),
                            "analysis_version": meta.get("version"), "crop_w": cw, "crop_h": ch,
                            "y": (H - ch) // 2 // 2 * 2}
    if analysis is None:
        return dict(info, crop_status="center_fallback", fallback_reason="detector_unavailable",
                    keyframes=[[0.0, center], [round(duration, 3), center]], shots=[], cuts=[])
    samples = analysis.get("samples") or []
    cuts = [c for c in analysis.get("cuts") or [] if 0.0 < c < duration]
    bounds = [0.0] + cuts + [duration]
    shots: List[Dict[str, Any]] = []
    # 1. per shot: who is there, which layout
    for s0, s1 in zip(bounds, bounds[1:]):
        ins = [x for x in samples if s0 - 1e-6 <= x["t"] < s1 - 1e-6]
        trs = tracks(faces_of(ins, H))
        n = max(1, len(ins))
        trs = [tr for tr in trs if len(tr["times"]) >= max(3, 0.25 * n)]
        shot: Dict[str, Any] = {"start": round(s0, 3), "end": round(s1, 3), "faces": len(trs), "parts": []}
        if not trs:
            shot["kind"] = "none"
            shots.append(shot)
            continue
        main = trs[0]
        shot["main"] = main
        partner = next((tr for tr in trs[1:] if real_partner(main, tr, W)), None)
        crowd = [tr for tr in trs[1:] if tr["w"] >= SPLIT_SIZE * main["w"] and tr["score"] >= SPLIT_SCORE]
        if len(crowd) >= 2:          # a group shot: halves would pick 2 of many → show the group, wide
            partner = None
            shot["group"] = True
        big = main["w"] >= CLOSEUP_FRAC * cw
        if not big:   # a face the 9:16 window cannot follow calmly → the wide window
            k9 = lazy_path(main["boxes"], s0, s1, cw, max_x)
            big = path_travel(k9) > max(0.6 * cw, 0.35 * cw * (s1 - s0))
        shot["kind"] = "closeup" if big or shot.get("group") else "single"
        if partner is not None and pair_fits(main, partner, cw):
            shot["pair"] = pair_boxes(main, partner)
            shot["pair_w"] = max(main["w"], partner["w"])
            partner = None
        if partner is not None:
            ts = sorted({x["t"] for x in ins})
            both = [(t, t in main["times"] and t in partner["times"]) for t in ts]
            runs = _runs(both, s0, s1, SPLIT_MIN_SEC)
            for f, a, b in runs:
                cover = [x for x in ins if a <= x["t"] < b]
                together = sum(1 for x in cover if x["t"] in main["times"] and x["t"] in partner["times"])
                if f and b - a >= SPLIT_MIN_SEC and together >= SPLIT_COVER * max(1, len(cover)):
                    shot["parts"].append(["split", a, b])
                else:
                    shot["parts"].append([shot["kind"], a, b])
            shot["partner"] = partner
        shots.append(shot)
    # 2. faceless shots and very short shots take a neighbour's layout (no flicker)
    for i, sh in enumerate(shots):
        if not sh["parts"]:
            sh["parts"] = [[sh["kind"], sh["start"], sh["end"]]]
    for i, sh in enumerate(shots):
        if sh["kind"] == "none":
            prev = next((s for s in reversed(shots[:i]) if s["kind"] != "none"), None)
            nxt = next((s for s in shots[i + 1:] if s["kind"] != "none"), None)
            lay = prev["parts"][-1][0] if prev else nxt["parts"][0][0] if nxt else "closeup"
            sh["parts"] = [["closeup" if lay == "split" else lay, sh["start"], sh["end"]]]
    seq = [[p[0], p[1], p[2], i] for i, sh in enumerate(shots) for p in sh["parts"]]
    changed = True
    while changed and len(seq) > 1:
        changed = False
        for k, p in enumerate(seq):
            if p[2] - p[1] >= MIN_LAYOUT_SEC:
                continue
            prev = seq[k - 1] if k > 0 else None
            nxt = seq[k + 1] if k + 1 < len(seq) else None
            if (prev and prev[0] == p[0]) or (nxt and nxt[0] == p[0]):
                continue
            src = prev if prev and (not nxt or prev[2] - prev[1] >= nxt[2] - nxt[1]) else nxt
            lay = src[0]
            if lay == "split":       # a split needs both people in THIS stretch: take the other neighbour / wide
                alt = nxt if src is prev else prev
                lay = alt[0] if alt and alt[0] != "split" else "closeup"
            if lay != p[0]:
                p[0] = lay
                changed = True
        # same-layout neighbours inside one shot are joined
        out: List[List[Any]] = []
        for p in seq:
            if out and out[-1][0] == p[0] and out[-1][3] == p[3]:
                out[-1][2] = p[2]
            else:
                out.append(p)
        seq = out
    # 3. window paths
    keys: List[List[float]] = []
    cu_keys: List[List[float]] = []
    cu_iv: List[List[float]] = []
    splits: List[Dict[str, Any]] = []
    last_x, last_cx = center, (sq_max // 2) // 2 * 2
    for lay, a, b, si in seq:
        sh = shots[si]
        main = sh.get("main")
        if lay == "split":
            partner = sh["partner"]
            splits.append({"start": round(a, 3), "end": round(b, 3), "people": sorted((main, partner), key=_cx)})
            keys += [[round(a, 3), last_x], [round(b, 3), last_x]]
            continue
        win, mx = (sq, sq_max) if lay == "closeup" else (cw, max_x)
        k = lazy_path(sh.get("pair") or main["boxes"], a, b, win, mx) if main else []
        if not k:
            x0 = (last_cx if lay == "closeup" else last_x) if a > 0 and b - a < 1.0 else \
                ((sq_max // 2) // 2 * 2 if lay == "closeup" else center)
            k = [[round(a, 3), x0], [round(b, 3), x0]]
        if lay == "closeup":
            cu_keys += k
            cu_iv.append([round(a, 3), round(b, 3)])
            last_cx = k[-1][1]
            keys += [[round(a, 3), last_x], [round(b, 3), last_x]]
        else:
            keys += k
            last_x = k[-1][1]
    for sh in shots:
        sh["target"] = {"none": "none", "closeup": "closeup", "single": "pair" if sh.get("pair") else "person"}[sh["kind"]]
        if any(p[0] == "split" for p in sh["parts"]):
            sh["target"] = "split"
        sh["segments"] = [{"start": round(p[1], 3), "end": round(p[2], 3), "layout": p[0]}
                          for p in seq if p[3] == shots.index(sh)]
        for k in ("main", "partner", "parts", "kind", "pair", "pair_w", "group"):
            sh.pop(k, None)
    clean: List[List[float]] = []
    for k in keys:
        if not (clean and clean[-1] == k):
            clean.append(k)
    status = "tracked" if any(s["target"] != "none" for s in shots) else "center_fallback"
    out = dict(info, crop_status=status, fallback_reason=None if status == "tracked" else "no_subject_found",
               keyframes=clean, shots=shots, cuts=cuts, layouts=[[p[0], round(p[1], 3), round(p[2], 3)] for p in seq])
    if cu_iv:
        out["closeup"] = {"size": sq, "y": (H - sq) // 2 // 2 * 2, "intervals": _join(cu_iv), "x": cu_keys}
    if splits:
        gaps = [abs(_cx(x["people"][1]) - _cx(x["people"][0])) for x in splits]
        sw, sh_ = split_window(W, H, min(gaps))
        tr: Dict[str, Any] = {"w": sw, "h": sh_, "intervals": [[x["start"], x["end"]] for x in splits]}
        out["splits"] = [{"start": x["start"], "end": x["end"]} for x in splits]
        for idx, name in enumerate(("top", "bottom")):
            xk: List[List[float]] = []
            yk: List[List[float]] = []
            for x in splits:
                p = x["people"][idx]
                fy = _med([bb[1] + bb[3] / 2.0 for t, bb in p["boxes"] if x["start"] - 0.3 <= t <= x["end"] + 0.3]
                          or [H / 3.0])
                y = int(max(0, min(H - sh_, round(fy - 0.35 * sh_))) // 2 * 2)
                k = lazy_path(p["boxes"], x["start"], x["end"], sw, W - sw) or \
                    [[x["start"], max(0, min(W - sw, int(_cx(p) - sw / 2))) // 2 * 2], [x["end"], 0]]
                if len(k) == 2 and k[1][1] == 0 and k[0][1] != 0:
                    k[1][1] = k[0][1]
                xk += k
                yk += [[x["start"], y], [x["end"], y]]
            tr[name] = {"x": xk, "y": yk}
        out["split"] = tr
    return out


def _join(iv: List[List[float]]) -> List[List[float]]:
    out: List[List[float]] = []
    for a, b in iv:
        if out and abs(out[-1][1] - a) < 1e-3:
            out[-1][1] = b
        else:
            out.append([a, b])
    return out


# ── burned-in captions of the source ─────────────────────────────────────────────────────────────────

def caption_boxes(analysis: Optional[Dict[str, Any]], duration: float, hold: float = 0.6) -> List[List[float]]:
    """The source's own subtitles (white text low and centred, e.g. an interviewer's question translated in
    the original video): [[t0, t1, x, y, w, h]] in source pixels. They are blurred in the render — otherwise
    a second, different-looking line of subtitles shows up in the clip (owner 2026-11-13, clip 1)."""
    meta = (analysis or {}).get("meta") or {}
    W, H = int(meta.get("width") or 0), int(meta.get("height") or 0)
    if not analysis or not W or not H:
        return []
    spans: List[List[float]] = []
    for smp in analysis.get("text") or []:
        bs = list(smp.get("captions") or []) + [
            b for b in smp.get("boxes") or []
            if b[1] >= CAPTION_BAND * H and abs(b[0] + b[2] / 2.0 - W / 2.0) <= CAPTION_CENTER * W and b[2] >= 0.12 * W]
        if not bs:
            continue
        t = float(smp["t"])
        x0, y0 = min(b[0] for b in bs), min(b[1] for b in bs)
        x1, y1 = max(b[0] + b[2] for b in bs), max(b[1] + b[3] for b in bs)
        if spans and t - hold <= spans[-1][1] + 1e-6:
            s = spans[-1]
            s[1] = t + hold
            s[2], s[3], s[4], s[5] = min(s[2], x0), min(s[3], y0), max(s[4], x1), max(s[5], y1)
        else:
            spans.append([max(0.0, t - hold), t + hold, x0, y0, x1, y1])
    out = []
    for t0, t1, x0, y0, x1, y1 in spans:
        px, py = 0.03 * W, 0.025 * H
        x0, y0 = max(0, int(x0 - px)) // 2 * 2, max(0, int(y0 - py)) // 2 * 2
        x1, y1 = min(W, int(x1 + px)), min(H, int(y1 + py))
        out.append([round(t0, 3), round(min(duration, t1), 3), x0, y0, (x1 - x0) // 2 * 2, (y1 - y0) // 2 * 2])
    return out


def caption_blur_graph(boxes: List[List[float]]) -> str:
    """Filter-graph prefix (source → source) that blurs each caption box while it is on screen."""
    if not boxes:
        return ""
    n = len(boxes)
    parts = [f"split={n + 1}[cb0]" + "".join(f"[cbs{i}]" for i in range(n))]
    cur = "cb0"
    for i, (t0, t1, x, y, w, h) in enumerate(boxes):
        parts.append(f"[cbs{i}]crop={int(w)}:{int(h)}:{int(x)}:{int(y)},gblur=sigma=24[cbb{i}]")
        parts.append(f"[{cur}][cbb{i}]overlay={int(x)}:{int(y)}:enable='between(t\\,{t0:.3f}\\,{t1:.3f})'[cbo{i}]")
        cur = f"cbo{i}"
    parts[-1] = parts[-1].rsplit("[", 1)[0]
    return ";".join(parts)
