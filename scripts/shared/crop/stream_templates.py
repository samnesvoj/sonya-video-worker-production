"""
stream_templates.py
===================
Five «webcam + game» layouts for the streamer mode (parameter ``layout_template`` = 1..5, chosen on the site;
owner 2026-11-10). All are 1080x1920, no chat / overlays, the webcam is never upscaled more than MAX_UPSCALE (×2),
text stays inside the frame (subtitles are placed by ``seam_y``: the text sits just above that line).

  1 top_bottom  webcam on top, the whole game below (the v2 default, unchanged)
  2 bubble      game full-screen 9:16 (centre of the action), a round webcam bubble in the top-right corner
  3 cam_big     webcam large on top (60 %), game below (40 %, sides trimmed)
  4 cinema      blurred background, the whole game in the middle, a small webcam window above it
  5 framed      blurred background, the game on top, the webcam in a thin frame below with the nickname

Quiet look (Emil Kowalski): flat shapes, hairline light frames, no glass, no gold, no shadows.
``apply_template`` only changes the track dict; ``track_graph`` (smart_crop_adapter) renders it.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .stream_layout import MAX_UPSCALE, OUT_H, OUT_W, cam_fit, _even

TEMPLATES = {1: "top_bottom", 2: "bubble", 3: "cam_big", 4: "cinema", 5: "framed"}
TITLES_RU = {1: "Вебка сверху, игра снизу", 2: "Игра на весь экран + вебка-кружок",
             3: "Крупная вебка сверху 60 %, игра 40 %", 4: "Кинотеатр: игра по центру, вебка мини-окном",
             5: "Игра сверху, вебка в рамке с ником"}
DEFAULT_TEMPLATE = 1
RING = "0xE6E6E6"                      # hairline frame colour


def normalize_template(value: Any) -> int:
    """1..5 (int / numeric string / name) → template number; anything else → the default (1)."""
    if isinstance(value, str):
        v = value.strip().lower()
        for n, name in TEMPLATES.items():
            if v == name:
                return n
    try:
        n = int(value)
    except (TypeError, ValueError):
        return DEFAULT_TEMPLATE
    return n if n in TEMPLATES else DEFAULT_TEMPLATE


def clean_nick(nick: Any) -> Optional[str]:
    """Nickname for the caption: letters / digits / space . _ - @ only (it goes into a filtergraph), ≤ 24 chars."""
    s = re.sub(r"[^\w .@\-]", "", str(nick or ""), flags=re.UNICODE).strip()[:24].strip()
    return s or None


def _crop_box(game: List[int], masks: List[List[int]], aspect: float):
    """Trim the game box (centred) to width/height = aspect; masks (game-box px) follow."""
    x, y, w, h = game
    if w / float(h) > aspect:
        nw, nh = _even(h * aspect), h
    else:
        nw, nh = w, _even(w / aspect)
    dx, dy = (w - nw) // 2 // 2 * 2, (h - nh) // 2 // 2 * 2
    out = []
    for mx, my, mw, mh in masks:
        x0, y0, x1, y1 = max(0, mx - dx), max(0, my - dy), min(nw, mx + mw - dx), min(nh, my + mh - dy)
        if x1 - x0 >= 8 and y1 - y0 >= 8:
            out.append([x0 // 2 * 2, y0 // 2 * 2, (x1 - x0) // 2 * 2, (y1 - y0) // 2 * 2])
    return [x + dx, y + dy, nw, nh], out


def apply_template(track: Optional[Dict[str, Any]], template: Any, *, nick: Any = None) -> Optional[Dict[str, Any]]:
    """Track of the game layout → the same track in the chosen template (a copy). Other layouts: unchanged."""
    n = normalize_template(template)
    if not track or track.get("layout") != "game" or not track.get("cam_src") or n == 1:
        if track is not None:
            track = dict(track, template=1 if track.get("layout") == "game" else None)
        return track
    t = dict(track, template=n, template_name=TEMPLATES[n])
    cam, face = [float(v) for v in t["cam_src"]], t.get("face") or [cam[0] + cam[2] / 2, cam[1] + cam[3] / 2, 0, 0]
    game, masks = list(t["game"]), [list(m) for m in t.get("game_masks") or []]
    gh_full = _even(OUT_W * game[3] / float(game[2]))
    if n == 3:                                          # 60 % webcam / 40 % game: the same graph as 1, other split
        game_h = 768
        g, m = _crop_box(game, masks, OUT_W / float(game_h))
        top_h = OUT_H - game_h
        crop, size = cam_fit(cam, face[0], top_h)
        t.update(game=g, game_masks=m, game_h=game_h, top_h=top_h, cam=crop, cam_size=size,
                 cam_pos=[(OUT_W - size[0]) // 2 // 2 * 2, (top_h - size[1]) // 2 // 2 * 2],
                 cam_upscale=round(size[0] / float(crop[2]), 3), seam_y=top_h)
    elif n == 2:
        side = min(cam[2], cam[3])
        if face[2]:
            side = min(side, max(2.4 * face[2], 0.45 * min(cam[2], cam[3])))     # face-centred, not the whole room
        side = _even(side)
        d = min(320, _even(MAX_UPSCALE * side))
        sx = min(max(cam[0], face[0] - side / 2.0), cam[0] + cam[2] - side)
        sy = min(max(cam[1], face[1] - side / 2.0), cam[1] + cam[3] - side)
        t["tpl"] = {"kind": "bubble", "crop": [int(sx) // 2 * 2, int(sy) // 2 * 2, side, side], "d": d,
                    "pos": [OUT_W - d - 48, 250]}
        t.update(cam_upscale=round(d / float(side), 3), seam_y=1500)
    elif n == 4:
        gh = min(760, gh_full)
        crop, size = cam_fit(cam, face[0], 380, out_w=600)
        gy = 700
        t["tpl"] = {"kind": "cinema", "game_y": gy, "game_h": gh, "cam": crop, "cam_size": size,
                    "cam_pos": [(OUT_W - size[0] - 6) // 2 // 2 * 2, 190]}
        t.update(cam_upscale=round(size[0] / float(crop[2]), 3), seam_y=gy + gh + 190)   # subtitles under the game
    else:
        gh = min(640, gh_full)
        crop, size = cam_fit(cam, face[0], 440, out_w=760)
        card_y = 1120
        t["tpl"] = {"kind": "framed", "game_y": 250, "game_h": gh, "cam": crop, "cam_size": size,
                    "cam_pos": [(OUT_W - size[0] - 8) // 2 // 2 * 2, card_y], "nick": clean_nick(nick),
                    "nick_y": card_y + size[1] + 8 + 26}
        t.update(cam_upscale=round(size[0] / float(crop[2]), 3), seam_y=1090)
    return t


# ── graphs ──────────────────────────────────────────────────────────────────

def _polish(crop: List[int], size: List[int]) -> str:
    from .smart_crop_adapter import CAM_DENOISE, CAM_SHARPEN
    up = size[0] / float(crop[2])
    p = [CAM_DENOISE] if up >= 1.5 and CAM_DENOISE else []
    p.append(f"scale={size[0]}:{size[1]}:flags=lanczos")
    if up > 1.05 and CAM_SHARPEN:
        p.append(CAM_SHARPEN)
    return ",".join(p)


def _game_native(track: Dict[str, Any], src: str, out: str) -> List[str]:
    """The game box of the source with the webcam / chat boxes blurred over → [out] (native size)."""
    gx, gy, gw, gh = track["game"]
    masks = track.get("game_masks") or []
    g = f"[{src}]crop={gw}:{gh}:{gx}:{gy}"
    if not masks:
        return [f"{g}[{out}]"]
    parts = [f"{g},split={len(masks) + 1}[gm0]" + "".join(f"[gm{i + 1}]" for i in range(len(masks)))]
    cur = "gm0"
    for i, (mx, my, mw, mh) in enumerate(masks):
        parts.append(f"[gm{i + 1}]crop={mw}:{mh}:{mx}:{my},gblur=sigma=18,eq=brightness=-0.08[mb{i}]")
        parts.append(f"[{cur}][mb{i}]overlay={mx}:{my}[{'gdone' if i == len(masks) - 1 else f'go{i}'}]")
        cur = f"go{i}"
    return parts[:-1] + [parts[-1].replace("[gdone]", f"[{out}]")]


def _blur_bg(src: str, out: str) -> str:
    return (f"[{src}]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,gblur=sigma=6,"
            f"scale={OUT_W}:{OUT_H}:flags=bicubic,eq=brightness=-0.2:saturation=0.75,setsar=1[{out}]")


def _framed_cam(crop: List[int], size: List[int], frame: int) -> str:
    cx, cy, cw, ch = crop
    return (f"crop={cw}:{ch}:{cx}:{cy},{_polish(crop, size)},"
            f"pad={size[0] + 2 * frame}:{size[1] + 2 * frame}:{frame}:{frame}:color={RING},setsar=1")


def _font_arg() -> str:
    from scripts.shared.subtitle_styles import FONTS_DIR
    p = str(FONTS_DIR / "Montserrat-SemiBold.ttf").replace("\\", "/").replace(":", "\\:")
    return f"fontfile='{p}'"


def template_graph(track: Dict[str, Any]) -> str:
    """-vf graph (source → 1080x1920) of templates 2, 4, 5."""
    tp = track["tpl"]
    kind = tp["kind"]
    if kind == "bubble":
        sx, sy, side, _ = tp["crop"]
        d = tp["d"]
        r = f"hypot(X-W/2+0.5,Y-H/2+0.5)"
        ring = f"gt({r},W/2-3)"
        parts = ["split=2[fa][fc]"] + _game_native(track, "fc", "gn")
        parts.append(f"[gn]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase:flags=lanczos,"
                     f"crop={OUT_W}:{OUT_H},setsar=1[gbg]")
        parts.append(f"[fa]crop={side}:{side}:{sx}:{sy},{_polish(tp['crop'], [d, d])},setsar=1,format=yuva444p,"
                     f"geq=lum='if({ring},230,lum(X,Y))':cb='if({ring},128,cb(X,Y))':cr='if({ring},128,cr(X,Y))':"
                     f"a='clip(255*(W/2-{r}),0,255)'[bub]")
        parts.append(f"[gbg][bub]overlay={tp['pos'][0]}:{tp['pos'][1]}:format=auto")
        return ";".join(parts)
    parts = ["split=3[fa][fb][fc]", _blur_bg("fb", "bg")] + _game_native(track, "fc", "gn")
    parts.append(f"[gn]scale={OUT_W}:{tp['game_h']}:flags=lanczos,setsar=1[gm]")
    frame = 3 if kind == "cinema" else 4
    parts.append(f"[fa]{_framed_cam(tp['cam'], tp['cam_size'], frame)}[cm]")
    parts.append(f"[bg][gm]overlay=0:{tp['game_y']}[b1]")
    cx, cy = tp["cam_pos"]
    if kind == "framed" and tp.get("nick"):
        parts.append(f"[b1][cm]overlay={cx}:{cy}[b2]")
        parts.append(f"[b2]drawtext={_font_arg()}:text='{tp['nick']}':fontsize=38:fontcolor=white@0.9:"
                     f"x=(w-text_w)/2:y={tp['nick_y']}")
    else:
        parts.append(f"[b1][cm]overlay={cx}:{cy}")
    return ";".join(parts)
