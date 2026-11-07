"""
formula.py — TeX → transparent RGBA image. Input has already passed tex.check_tex.

Two backends behind one function:
  mathtext  Matplotlib's own TeX subset. Default: Matplotlib is already in requirements/worker.lock.txt, no browser,
            no Node, no TeX, so the worker image does not grow.
  mathjax   MathJax in Node → SVG paths → filled with OpenCV. Only if SONYA_MATHJAX_DIR points at a pre-installed
            mathjax-full. Better typography; needs a Node layer, so it stays an explicit opt-in.
A parse failure raises FormulaError: the card is dropped, never drawn half-right.
"""
from __future__ import annotations

import io
import os
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

RGB = Tuple[int, int, int]
_NODE_SCRIPT = Path(__file__).with_name("tex2svg.cjs")


class FormulaError(RuntimeError):
    pass


def backend_name(requested: Optional[str] = None) -> str:
    want = requested or os.environ.get("SONYA_EDU_FORMULA_BACKEND") or "mathtext"
    return "mathjax" if want == "mathjax" and os.environ.get("SONYA_MATHJAX_DIR") else "mathtext"


def render_mathtext(tex: str, height_px: int, color: RGB) -> Image.Image:
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.font_manager import FontProperties
    except ImportError as e:
        raise FormulaError("matplotlib_not_installed") from e
    import matplotlib
    dpi = 200
    # a size at which the formula's height comes out close to height_px; tall fractions are scaled by the caller
    fig = Figure(dpi=dpi)
    FigureCanvasAgg(fig)
    with matplotlib.rc_context({"mathtext.fontset": "stix"}):
        fig.text(0.02, 0.3, f"${tex}$", fontproperties=FontProperties(size=height_px * 72.0 / dpi * 1.15),
                 color=tuple(c / 255.0 for c in color))
        buf = io.BytesIO()
        try:
            fig.savefig(buf, format="png", dpi=dpi, transparent=True, bbox_inches="tight", pad_inches=0.02)
        except ValueError as e:         # matplotlib's mathtext ParseException derives from ValueError
            raise FormulaError("mathtext_parse_error") from e
    buf.seek(0)
    im = Image.open(buf).convert("RGBA")
    box = im.getbbox()
    if not box:
        raise FormulaError("empty_render")
    return im.crop(box)


def _tex_to_svg(tex: str) -> str:
    env = {"PATH": os.environ.get("PATH", ""), "SONYA_MATHJAX_DIR": os.environ["SONYA_MATHJAX_DIR"]}
    r = subprocess.run(["node", str(_NODE_SCRIPT)], input=tex, capture_output=True, text=True, timeout=30, env=env)
    if r.returncode != 0 or "<svg" not in r.stdout or "data-mjx-error" in r.stdout or "merror" in r.stdout:
        raise FormulaError("mathjax_failed")
    return r.stdout


_TRANSFORM = re.compile(r"(translate|scale|matrix)\(([^)]*)\)")


def _matrix(transform: Optional[str]) -> np.ndarray:
    m = np.eye(3)
    for name, args in _TRANSFORM.findall(transform or ""):
        v = [float(x) for x in re.split(r"[\s,]+", args.strip())]
        if name == "translate":
            t = np.array([[1, 0, v[0]], [0, 1, v[1] if len(v) > 1 else 0.0], [0, 0, 1]])
        elif name == "scale":
            t = np.array([[v[0], 0, 0], [0, v[1] if len(v) > 1 else v[0], 0], [0, 0, 1]])
        else:
            t = np.array([[v[0], v[2], v[4]], [v[1], v[3], v[5]], [0, 0, 1]])
        m = m @ t
    return m


def _flatten(d: str) -> List[np.ndarray]:
    """SVG path data → polylines (Beziers sampled)."""
    from fontTools.pens.recordingPen import RecordingPen
    from fontTools.svgLib.path import parse_path
    pen = RecordingPen()
    parse_path(d, pen)
    contours: List[List[Tuple[float, float]]] = []
    cur: List[Tuple[float, float]] = []
    for op, args in pen.value:
        if op == "moveTo":
            cur = [args[0]]
            contours.append(cur)
        elif op == "lineTo":
            cur.append(args[0])
        elif op == "qCurveTo":
            pts = list(args)
            p0 = cur[-1]
            for i in range(len(pts) - 1):
                c = pts[i]
                e = pts[i + 1] if i == len(pts) - 2 else ((pts[i][0] + pts[i + 1][0]) / 2, (pts[i][1] + pts[i + 1][1]) / 2)
                for s in np.linspace(0, 1, 9)[1:]:
                    cur.append(((1 - s) ** 2 * p0[0] + 2 * (1 - s) * s * c[0] + s ** 2 * e[0],
                                (1 - s) ** 2 * p0[1] + 2 * (1 - s) * s * c[1] + s ** 2 * e[1]))
                p0 = e
        elif op == "curveTo":
            p0, c1, c2, e = cur[-1], args[0], args[1], args[2]
            for s in np.linspace(0, 1, 13)[1:]:
                cur.append(((1 - s) ** 3 * p0[0] + 3 * (1 - s) ** 2 * s * c1[0] + 3 * (1 - s) * s ** 2 * c2[0] + s ** 3 * e[0],
                            (1 - s) ** 3 * p0[1] + 3 * (1 - s) ** 2 * s * c1[1] + 3 * (1 - s) * s ** 2 * c2[1] + s ** 3 * e[1]))
    return [np.array(c, dtype=np.float64) for c in contours if len(c) > 2]


def rasterize_mathjax_svg(svg: str, height_px: int, color: RGB) -> Image.Image:
    """Enough SVG for MathJax output (fontCache none): nested <g>, <path>, <rect>. Not a general SVG renderer."""
    import cv2
    root = ET.fromstring(svg)
    tag = lambda e: e.tag.split("}")[-1]        # noqa: E731
    x0, y0, vw, vh = (float(v) for v in root.attrib["viewBox"].split())
    ss = 4
    k = height_px * ss / vh
    W, H = int(np.ceil(vw * k)), int(np.ceil(vh * k))
    mask = np.zeros((H, W), np.uint8)

    def walk(node, m):
        m = m @ _matrix(node.attrib.get("transform"))
        for ch in node:
            t = tag(ch)
            if t == "g":
                walk(ch, m)
            elif t == "path" and ch.attrib.get("d"):
                polys = []
                mm = m @ _matrix(ch.attrib.get("transform"))
                for c in _flatten(ch.attrib["d"]):
                    xy = (mm @ np.c_[c, np.ones(len(c))].T).T[:, :2]
                    polys.append(np.round((xy - [x0, y0]) * k).astype(np.int32))
                if polys:
                    cv2.fillPoly(mask, polys, 255, cv2.LINE_AA)
            elif t == "rect":
                mm = m @ _matrix(ch.attrib.get("transform"))
                x, y, w, h = (float(ch.attrib.get(a, 0)) for a in ("x", "y", "width", "height"))
                pts = np.array([[x, y, 1], [x + w, y, 1], [x + w, y + h, 1], [x, y + h, 1]]).T
                xy = (mm @ pts).T[:, :2]
                cv2.fillPoly(mask, [np.round((xy - [x0, y0]) * k).astype(np.int32)], 255, cv2.LINE_AA)
            elif t == "svg":
                walk(ch, m)

    walk(root, np.eye(3))
    small = cv2.resize(mask, (max(1, W // ss), max(1, H // ss)), interpolation=cv2.INTER_AREA)
    rgba = np.zeros(small.shape + (4,), np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = color
    rgba[..., 3] = small
    im = Image.fromarray(rgba, "RGBA")
    box = im.getbbox()
    if not box:
        raise FormulaError("empty_render")
    return im.crop(box)


def render_formula(tex: str, height_px: int, color: RGB, backend: Optional[str] = None) -> Image.Image:
    name = backend_name(backend)
    if name == "mathjax":
        return rasterize_mathjax_svg(_tex_to_svg(tex), height_px, color)
    return render_mathtext(tex, height_px, color)
