#!/usr/bin/env /usr/bin/python3
"""Procedural bottom-filler videos for the split-screen option (bottom_filler=gameplay|asmr). 100% ours: every pixel is drawn
by this script (PIL + ffmpeg), no third-party footage, no logos. Output: silent 1080x920 H.264, 20 s, 30 fps, seamless loop
(all motion is periodic over the clip length). Needs Pillow (system python3) and ffmpeg.

  /usr/bin/python3 tools/gen_fillers.py OUT_DIR [name ...]
"""
from __future__ import annotations

import math
import random
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter

FPS, SECS = 30, 20
N = FPS * SECS
RW, RH = 540, 460            # drawn size, upscaled x2 by ffmpeg
TAU = 2 * math.pi


def ph(i: int, cycles: float = 1.0) -> float:
    return TAU * cycles * i / N


def lerp(a, b, k):
    return tuple(int(a[j] + (b[j] - a[j]) * k) for j in range(3))


def vgrad(top, bot, w=RW, h=RH):
    im = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(im)
    for y in range(h):
        d.line([(0, y), (w, y)], fill=lerp(top, bot, y / (h - 1)))
    return im


# ───────────────────────── gameplay ─────────────────────────
def runner_side(i, bg=[None]):
    """Side-scroller: a cube hops over spikes, parallax hills, coins."""
    if bg[0] is None:
        bg[0] = vgrad((70, 40, 140), (255, 140, 120))
    im = bg[0].copy()
    d = ImageDraw.Draw(im)
    scroll = (i / N) * 1080                      # px per loop, all layers wrap on 540 multiples
    for layer, (col, k, base, amp) in enumerate([((90, 60, 150), 1, 330, 70), ((60, 40, 120), 2, 370, 50)]):
        pts = [(0, RH)]
        for x in range(0, RW + 10, 10):
            xx = x + scroll * k / 2
            pts.append((x, base - amp * 0.5 * (1 + math.sin(TAU * xx / 540 * 1) * math.cos(TAU * xx / 270))))
        pts.append((RW, RH))
        d.polygon(pts, fill=col)
    ground = 400
    d.rectangle([0, ground, RW, RH], fill=(30, 20, 60))
    for x in range(-60, RW + 60, 60):
        xx = x - (scroll * 2) % 60
        d.rectangle([xx, ground, xx + 30, ground + 6], fill=(255, 90, 160))
    # obstacles (spikes) every 270 px, 4 per loop of 1080 px scrolled at 2x
    for k in range(-1, 4):
        ox = k * 270 - (scroll * 2) % 270 + 270
        for s in range(2):
            x0 = ox + s * 26
            d.polygon([(x0, ground), (x0 + 13, ground - 34), (x0 + 26, ground)], fill=(255, 235, 90))
    # player hops when a spike group passes x=120
    t = ((scroll * 2) % 270) / 270
    jump = max(0.0, math.sin(math.pi * min(1.0, max(0.0, (t - 0.0) / 0.45)))) if t < 0.45 else 0.0
    py = ground - 40 - 95 * jump
    rot = -90 * min(1.0, max(0.0, t / 0.45)) if t < 0.45 else 0
    cx, cy, r = 120, py + 20, 20
    a = math.radians(rot)
    pts = [(cx + r * math.cos(a + q) * 1.414, cy + r * math.sin(a + q) * 1.414) for q in (math.pi / 4, 3 * math.pi / 4, 5 * math.pi / 4, 7 * math.pi / 4)]
    d.polygon(pts, fill=(80, 230, 255), outline=(255, 255, 255))
    # coins
    for k in range(5):
        cxx = (k * 216 - scroll * 2) % 1080 - 100
        if -20 < cxx < RW + 20:
            d.ellipse([cxx - 9, 250 - 9 + 8 * math.sin(ph(i, 8) + k), cxx + 9, 250 + 9 + 8 * math.sin(ph(i, 8) + k)], fill=(255, 215, 0))
    return im


def runner_lanes(i):
    """Pseudo-3D three-lane runner: road in perspective, blocks and coins coming at the camera."""
    im = vgrad((120, 200, 255), (255, 240, 200))
    d = ImageDraw.Draw(im)
    hor, vx = 150, RW / 2
    d.polygon([(0, RH), (0, hor + 40), (RW, hor + 40), (RW, RH)], fill=(70, 150, 90))
    d.polygon([(vx - 30, hor), (vx + 30, hor), (RW - 20, RH), (20, RH)], fill=(70, 70, 85))
    z0 = (i / N) * 8                            # 8 stripes of road per loop
    for k in range(0, 18):
        z = (k - (z0 % 1)) / 18
        if z <= 0:
            continue
        y = hor + (RH - hor) * z ** 2
        w = 2 + 7 * z ** 2
        d.line([(vx - 30 - (vx - 50) * z ** 2 * 0 , y), (vx + 30, y)], fill=(70, 70, 85))
        for lane in (-1, 1):
            x = vx + lane * (10 + 150 * z ** 2) * 0.55
            d.line([(x, y), (x, y + w * 2)], fill=(255, 255, 255), width=int(1 + 3 * z ** 2))
    rng = random.Random(5)
    items = [(rng.choice((-1, 0, 1)), rng.random(), rng.random() < 0.55) for _ in range(10)]
    for lane, off, coin in sorted(items, key=lambda it: -((it[1] + i / N * 2) % 1)):
        z = (off + i / N * 2) % 1
        if z < 0.05:
            continue
        s = z ** 2
        y = hor + (RH - hor) * s
        x = vx + lane * (10 + 150 * s) * 0.9
        w = 8 + 70 * s
        if coin:
            d.ellipse([x - w / 3, y - w / 1.5, x + w / 3, y - w / 6], fill=(255, 210, 0), outline=(255, 250, 160))
        else:
            d.rectangle([x - w / 2, y - w, x + w / 2, y], fill=(240, 70, 90), outline=(255, 190, 200))
    # runner (bobbing, lane change sine)
    lane_x = vx + 80 * math.sin(ph(i, 4))
    bob = 6 * abs(math.sin(ph(i, 40)))
    d.ellipse([lane_x - 22, RH - 62 - bob, lane_x + 22, RH - 18 - bob], fill=(255, 160, 60), outline=(255, 255, 255))
    d.ellipse([lane_x - 10, RH - 94 - bob, lane_x + 10, RH - 70 - bob], fill=(255, 220, 170))
    return im


def hopper(i):
    """Vertical auto-jumper: a blob bounces up a column of platforms, camera scrolls up."""
    im = vgrad((20, 30, 70), (60, 20, 100))
    d = ImageDraw.Draw(im)
    rng = random.Random(11)
    for _ in range(60):
        d.point((rng.randrange(RW), rng.randrange(RH)), fill=(255, 255, 255))
    spacing, cnt = 92, 10                       # 10 platforms wrap per loop
    shift = (i / N) * spacing * cnt
    prng = random.Random(3)
    xs = [prng.randrange(70, RW - 140) for _ in range(cnt)]
    for k in range(cnt + 2):
        y = RH - 40 - k * spacing + (shift % (spacing * cnt)) % spacing + (shift // spacing % 1)
        y = (RH - 40 - k * spacing + shift) % (spacing * cnt) - 80
        x = xs[k % cnt] + 60 * math.sin(ph(i, 3) + k)
        d.rounded_rectangle([x, y, x + 90, y + 14], 7, fill=(90, 255, 160), outline=(220, 255, 230))
    hop = abs(math.sin(ph(i, 20)))
    bx = RW / 2 + 150 * math.sin(ph(i, 5))
    by = RH - 150 - 120 * hop
    sq = 1 + 0.15 * (1 - hop)
    d.ellipse([bx - 24 * sq, by - 24 / sq, bx + 24 * sq, by + 24 / sq], fill=(255, 110, 200), outline=(255, 255, 255))
    d.ellipse([bx - 9, by - 8, bx - 3, by - 2], fill=(255, 255, 255))
    d.ellipse([bx + 3, by - 8, bx + 9, by - 2], fill=(255, 255, 255))
    return im


# ───────────────────────── ASMR ─────────────────────────
def _goo(circles, color, w=270, h=230, blur=7, thr=120, hi=(255, 255, 255)):
    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    for (x, y, r) in circles:
        d.ellipse([x - r, y - r, x + r, y + r], fill=255)
    m = m.filter(ImageFilter.GaussianBlur(blur))
    solid = m.point(lambda v: 255 if v > thr else 0).filter(ImageFilter.GaussianBlur(0.8))
    height = solid.filter(ImageFilter.GaussianBlur(10))
    base = Image.new("RGB", (w, h), color)
    dark = Image.new("RGB", (w, h), tuple(int(c * 0.62) for c in color))
    shade = ImageChops.offset(height, -6, -8)
    lit = Image.composite(base, dark, shade.point(lambda v: min(255, int(v * 1.6))))
    spec = ImageChops.subtract(ImageChops.offset(height, -4, -5), ImageChops.offset(height, 4, 5)).point(lambda v: min(255, v * 6))
    spec = Image.composite(Image.new("RGB", (w, h), hi), lit, spec.point(lambda v: v // 2 + (v > 40) * 60))
    return solid, spec


def slime_blob(i, color=(255, 120, 190), bg_top=(255, 225, 240), bg_bot=(250, 190, 215), seed=1):
    rng = random.Random(seed)
    w, h = 270, 230
    circles = []
    for k in range(7):
        a0, rr, sp = rng.random() * TAU, 20 + rng.random() * 36, rng.choice((1, 2))
        x = w / 2 + 52 * math.cos(ph(i, sp) + a0) * (0.6 + 0.4 * math.sin(ph(i, 1) + k))
        y = h / 2 + 38 * math.sin(ph(i, sp) * 1 + a0 * 1.7) + 12 * math.sin(ph(i, 3) + k)
        circles.append((x, y, rr + 6 * math.sin(ph(i, 2) + k)))
    circles.append((w / 2, h / 2 + 30, 58 + 8 * math.sin(ph(i, 2))))
    solid, lit = _goo(circles, color)
    bg = vgrad(bg_top, bg_bot, w, h)
    sh = ImageChops.offset(solid.filter(ImageFilter.GaussianBlur(8)), 0, 10).point(lambda v: v // 3)
    bg = Image.composite(Image.new("RGB", (w, h), (120, 70, 100)), bg, sh)
    return Image.composite(lit, bg, solid).resize((RW, RH), Image.BICUBIC)


def slime_blue(i):
    return slime_blob(i, color=(90, 200, 255), bg_top=(225, 245, 255), bg_bot=(190, 225, 250), seed=7)


def soap_cut(i):
    """A pastel soap block is sliced by a thin blade, slices peel off to the right; periodic."""
    im = vgrad((245, 238, 230), (225, 215, 205))
    d = ImageDraw.Draw(im)
    cols = [(255, 170, 200), (170, 220, 255), (190, 255, 200), (255, 230, 150)]
    cycle = N // 5                                # five slices per loop
    k, t = divmod(i, cycle)
    u = t / cycle
    bw, bh, y0 = 270, 200, 170
    x0 = 60
    slice_w = 34
    # remaining block: starts wider, loses one slice per cycle; wraps (loop invariant: 5 cycles fully shift)
    for s in range(7):
        idx = k + s
        sx = x0 + (s - (0 if True else 0)) * slice_w - (u * slice_w if s > 0 else 0)
        if s == 0:
            continue
        c = cols[(idx // 2) % len(cols)]
        d.rounded_rectangle([sx, y0, sx + slice_w - 1, y0 + bh], 5, fill=c, outline=tuple(max(0, v - 40) for v in c))
    # blade drops in first half of cycle, lifts in second; falling slice at x0
    blade_y = y0 - 40 + 260 * min(1, u * 2) if u < 0.5 else y0 + 220 - 260 * (u - 0.5) * 2
    d.rectangle([x0 - 6, blade_y - 120, x0 - 3, blade_y], fill=(170, 175, 190))
    d.polygon([(x0 - 6, blade_y), (x0 - 3, blade_y), (x0 - 4, blade_y + 6)], fill=(210, 215, 225))
    d.rectangle([x0 - 6, blade_y - 160, x0 + 40, blade_y - 120], fill=(70, 70, 80))
    # cut slice peels away
    if u > 0.45:
        e = (u - 0.45) / 0.55
        c = cols[((k) // 2) % len(cols)]
        sx = x0 - 6 - e * 130
        sy = y0 + e * e * 260
        d.rounded_rectangle([sx, sy, sx + slice_w - 1, sy + bh], 5, fill=c, outline=tuple(max(0, v - 40) for v in c))
    d.rectangle([0, y0 + bh, RW, RH], fill=(200, 190, 180))
    return im


def bubbles(i):
    im = vgrad((250, 235, 255), (220, 235, 255))
    rng = random.Random(21)
    ov = Image.new("RGBA", (RW, RH), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    for k in range(26):
        r = 16 + rng.random() * 48
        spd = rng.choice((1, 2, 3))
        x = rng.random() * RW + 30 * math.sin(ph(i, 2) + k)
        y = (rng.random() * (RH + 2 * r) - (i / N) * spd * (RH + 2 * r)) % (RH + 2 * r) - r
        col = rng.choice([(255, 130, 200), (130, 200, 255), (160, 255, 200), (255, 220, 130), (200, 160, 255)])
        d.ellipse([x - r, y - r, x + r, y + r], fill=col + (70,), outline=col + (170,), width=2)
        d.ellipse([x - r * 0.55, y - r * 0.6, x - r * 0.15, y - r * 0.3], fill=(255, 255, 255, 150))
    return Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB")


SCENES = {                    # name -> (kind, frame fn, description)
    "gameplay_runner_side": ("gameplay", runner_side, "2D side-scroller cube hops over spikes, neon dusk"),
    "gameplay_runner_lanes": ("gameplay", runner_lanes, "3-lane pseudo-3D runner, blocks and coins"),
    "gameplay_hopper": ("gameplay", hopper, "vertical auto-jumper over green platforms"),
    "asmr_slime_pink": ("asmr", slime_blob, "glossy pink slime blob, soft squish"),
    "asmr_slime_blue": ("asmr", slime_blue, "glossy blue slime blob, soft squish"),
    "asmr_soap_cut": ("asmr", soap_cut, "pastel soap block cut into slices"),
    "asmr_bubbles": ("asmr", bubbles, "translucent pastel bubbles rising"),
}


def render(name: str, out: Path) -> None:
    fn = SCENES[name][1]
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{RW}x{RH}", "-r", str(FPS),
           "-i", "-", "-vf", "scale=1080:920:flags=lanczos,format=yuv420p", "-an", "-c:v", "libx264", "-crf", "24", "-preset", "medium",
           "-movflags", "+faststart", "-metadata", "comment=SONYA own procedural filler", str(out)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(N):
        fr = fn(i)
        if fr.size != (RW, RH):
            fr = fr.resize((RW, RH))
        p.stdin.write(fr.convert("RGB").tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise SystemExit(f"ffmpeg failed for {name}")


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for n in (sys.argv[2:] or SCENES):
        render(n, out / f"{n}.mp4")
        print("ok", n)
