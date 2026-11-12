# Educational v2 — overlay cards (formula, fact, term, diagram, chart)

Status: development candidate, tested on CPU (Apple Silicon) only. **Not enabled in production.**
`educational` stays an opt-in mode in `scripts/mode_policy.py`; the overlay pass runs only when a job passes
`params["edu_overlays"]`, and that flag does not enable the mode. Trailer stays OFF.

## What it does

For each Educational clip, after the existing cut/compose step:

1. **Plan** (optional, `params["llm"] == "on"`): `planner.py` sends the clip transcript (sentence ids) to the
   configured provider (`scripts/shared/llm_provider.py`) and gets proposals classified by meaning:
   `formula | fact | term | diagram | chart | none`. Analysis only: nothing is generated. Hard limits in
   `budget.py`: 5 ₽ per video, 30 ₽ total, worst-case forecast **before** the request; over the limit → no request.
   No opt-in / no provider / bad answer → no cards, no error. An edited manifest (`options["manifest"]`) replaces the planner.
2. **Validate** (`manifest.py`, `evidence.py`, `tex.py`): every proposal is re-checked against the ASR words; the
   model's own times are ignored. Nothing is dropped silently: each item gets `status` `ok | review | rejected | skipped` and a `reason`.
3. **Render** (`cards.py`, `formula.py`, `themes.py`): cards as transparent PNG layers, cached by hash of content,
   style, size and renderer version.
4. **Place** (`placement.py`): reuses `scripts/shared/crop/clip_analysis` on the rendered clip (YuNet faces,
   burned-in text boxes, scene cuts) plus our subtitle band, mobile UI zones and busy areas; cards never straddle a cut. No safe place → the card is skipped.
5. **Composite** (`compose.py`): one FFmpeg pass (overlay + fade/slide), audio stream copied untouched.
   Any failure keeps the plain clip and adds a warning.

## Evidence rules (no invented formulas or facts)

| Type | Shown only if |
|---|---|
| formula | TeX passes the whitelist (no `\input`, `\def`, `\write`, URLs, files…), the quote is found in the speech, every symbol/operation of the formula is stated in that speech (or the formula is in the author's materials), no hedging around it, ASR confident |
| fact | quote found in speech; every number on the card is spoken; `kind`: `quote` (Цитата из видео), `author_data` (Данные автора), `verified` (Проверенный факт) — `verified` **requires** a source from `verified_sources` that contains the same numbers and text. No external fact-checking services are called |
| term | the term is actually said; the definition comes from the speech or the author's material, not from the model's knowledge |
| diagram | `steps` (3), `cause_effect` (2), `a_vs_b` (2), `timeline` (2–4); every node quoted from speech, in spoken order; nodes reveal when said |
| chart | `line`/`bar`/`share`; unit required; every value spoken in its quote; zero-based axis; shares ≤ 100 %; each point appears when its number is said; last point stays ≥ 3 s |

Hedges in the same breath ("кажется", "точно не помню", "говорят", …) reject the item. Uncertain ASR or a
proposal marked ambiguous → `review_required: true`, not drawn.

## Manifest (editable JSON, clip-local times)

```json
{"schema_version": 1, "time_base": "clip_ms", "preset": "deep", "aspect": "9:16",
 "overlays": [{
   "id": "ov01", "type": "formula", "start_ms": 16300, "end_ms": 21000,
   "content": {"tex": "P = \\frac{A}{t}", "caption": "Мощность – это работа, деленная на время"},
   "source_segment_id": "s0005",
   "evidence": {"quote": "Мощность – это работа, деленная на время.", "matched_speech": "…", "match_ratio": 1.0,
                "basis": "stated_in_speech"},
   "confidence": 1.0, "style": {"theme": "modern-tech", "intensity": "normal"},
   "position": "auto", "review_required": false, "status": "ok", "reason": null}]}
```

Content shapes:

- fact: `{"kind": "author_data", "headline": "120 тысяч рублей", "body": "…"}` (+ `source_label` for `verified`)
- term: `{"term": "Фотосинтез", "definition": "процесс, в котором …"}`
- diagram: `{"kind": "steps", "title": "…", "nodes": [{"label": "письмо принимают", "quote": "Первый шаг – …"}, …]}`
- chart: `{"kind": "line", "title": "…", "unit": "тысяч рублей", "points": [{"label": "2023", "value": 120, "quote": "…"}, …]}`

## Options (`params["edu_overlays"]`)

`preset` `minimal` (term, fact) · `balanced` (formula, diagram — default) · `deep` (all five);
`enabled_types`, `max_per_min` (default 2, never two cards at once), `theme`
(`academic-minimal` light, `modern-tech` dark, `clean-whiteboard` white), `intensity` (`soft|normal|strong`), `size`,
`position` (`auto` or a fixed corner), `aspect` (`9:16`, `16:9`, `1:1`), `burn_subtitles`, `manifest`,
`author_materials`, `verified_sources`, `formula_backend` (`mathtext` default | `mathjax`).

## Dependencies and image size

Default path adds **nothing** to the worker image: FFmpeg, Pillow, NumPy, OpenCV and Matplotlib (mathtext for TeX)
are already in `requirements/worker.lock.txt`. MathJax (`mathjax-full` 3.2.1, Apache-2.0) is an optional backend
through Node (`SONYA_MATHJAX_DIR`); it is not installed in the image and must be pre-baked if ever wanted
(~50 MB node_modules, never installed at Instance start).

## Engineering comparison (licenses read from each repository's LICENSE file, Nov 2026)

| Project | License | Verdict | Why |
|---|---|---|---|
| FFmpeg overlay (+ libass for subtitles) | LGPL/GPL build already shipped; libass ISC | **ADOPT** | already in the image; one pass; audio untouched |
| Matplotlib mathtext | Matplotlib (PSF-style) | **ADOPT** (default TeX) | already in lock; no browser/Node/TeX; covers school formulas |
| MathJax (mathjax-full 3.2.1) | Apache-2.0 | **ADOPT as option** | better typography, SVG without a browser; needs Node layer |
| KaTeX 0.19 | MIT | IDEAS ONLY | outputs HTML+CSS, needs a browser to rasterise; security advisories exist |
| SymPy | BSD-3 | IDEAS ONLY | syntax/algebra check for step-by-step (P1); it cannot check facts about the world |
| Manim Community 0.22 | MIT | IDEAS ONLY (pre-render experiment) | Cairo/Pango/LaTeX stack, hundreds of MB, seconds per scene; not for every Instance |
| Mermaid | MIT | REJECT for runtime | headless browser to render; open advisories; our 4 diagram templates cover P0 |
| Plotly.py | MIT | REJECT | static export needs Chrome/Kaleido |
| PySceneDetect | BSD-3 | IDEAS ONLY | `clip_analysis` already detects cuts; cards are snapped inside one shot |
| WhisperX | BSD-2 | REJECT | torch + pyannote; existing ASR word timestamps suffice |
| MoviePy | MIT | REJECT | slower than a single FFmpeg filter graph; adds nothing |
| OpenCV | Apache-2.0 | **ADOPT** (already present) | YuNet faces, text boxes, busy-area checks via existing `clip_analysis` |
| SAM 2 | Apache-2.0 | REJECT for P0 | GPU, large weights; only for a future Highlight Zone |

Two implementations compared (33 s clip, Apple M-series CPU, libx264): A "PNG layers + FFmpeg overlay" 1.68 s vs
B "pre-rendered overlay video then overlay" 1.92 s + 4.2 MB intermediate; plain re-encode 1.42 s. A is the default.
Card render 0.03 s (cache hit 0.003 s). **These are CPU numbers, not a Vast GPU measurement.**

## Not done / next

- P1 (step-by-step equation, Highlight Zone, Recap, whiteboard strokes) — not implemented.
- GPU smoke test on Vast with NVENC; real educational videos with consent for a broader test.
- Author materials / verified sources UI: the code accepts them, nothing supplies them yet.
