# streamer (v2, beta)

Highlights from a long stream VOD (1–3 h), vertical 1080×1920, rendered like SONYA Cut. Details and the
measurements behind every choice: `docs/STREAMER_V2.md`.

## Pipeline

1. **ASR** once (GigaAM v3 by default, `scripts/shared/asr`).
2. **Signals per second** (`scripts/shared/stream_signals.py`): voice loudness and pitch (shouting), speech rate,
   reaction words (mat, «ахаха», «что?!», «да ладно»), game events (hard cuts, flashes), chat density when the VOD
   has a chat replay (`params.chat_path` or `<source>.live_chat.json`).
3. **Moments** (`scripts/shared/stream_moments.py`): whole sentences only, 25–60 s, the peak inside with time for
   the reaction, a lively first 4 s (hook); promo / donation reads dropped; a pool of ≤ 30 goes to the Gemini editor
   (analysis only: ids, scores, hook; hard cap 10 ₽ per stream; any failure → local order); word-exact boundaries
   (`scripts/shared/boundaries.py`).
4. **Layout per clip** (`scripts/shared/crop/stream_layout.py`): the webcam rectangle by its persistent border
   lines (+ a whole-stream survey), chat / overlay columns by small text that keeps changing. Top = the largest
   1080:960 window **inside** the webcam (no chat, alerts or frames), bottom = the game centre moved off chat and
   the webcam. No webcam → the Cut person / object / centre crop.
5. **Render**: one ffmpeg pass from the original source (lanczos, burned subtitles on the seam, 60 fps kept,
   6 Mb/s: NVENC on Vast, libx264 with a VBV cap on CPU).

## Switches

| env / param | default | |
|---|---|---|
| `params.editor` / `SONYA_LLM_EDITOR` | on when a provider is configured | Gemini editor |
| `STREAMER_VISUAL_SIGNALS` | true | game-event pass (one low-res decode of the VOD) |
| `STREAMER_HWACCEL` | — | `cuda` on Vast, `videotoolbox` on a Mac |
| `params.webcam_detector` / `STREAMER_WEBCAM_DETECTOR` | false | YOLO webcam box as a seed |
| `params.chat_path` | — | Twitch (TwitchDownloader JSON) / YouTube live chat replay |
