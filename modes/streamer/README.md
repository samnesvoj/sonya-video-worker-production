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

## «Завозный коммент» (moment type `comment`, 2026-11-19)

The streamer reads a viewer's comment aloud and reacts → chat card (nick, colour, text) while it is read, the big
two-line title for the first ~3.5 s, the reaction, and — when the stream has it — a glitch cut to the moment the
comment is about. `scripts/shared/stream_comment.py` (reads: chat replay matched to the ASR words, else ASR cues
«пишет / пишут / спрашивают»; payoff options by shared stems; one Gemini call ≤ 1.5 ₽), pictures and the glitch in
`scripts/shared/comment_card.py`, chat replay in `scripts/shared/stream_chat.py` (the worker fetches Twitch VOD chat
over GQL / YouTube live_chat with yt-dlp next to the source while ASR runs). Chosen alone → only comment clips
(regular ones if none found); «Авто» → ≤ 2 comment clips with final ≥ 78.

**v2 «жесть» (2026-11-20, owner: «надо искать какую-то жесть, долбанину»):** one Gemini call (≤ 1.5 ₽) rates
every candidate's *spice* (provocation / absurd / roast / scandal / shock / lewd / dark humour / troll) and the
streamer's *reaction* to the comment itself; the local reaction (loudness, laughs, reply length, pitch, chat burst,
webcam motion inside the OBS webcam rect vs. the 8 s before the read) is blended in. Taken only when spice ≥ 78
(ASR-only reads ≥ 84), reaction ≥ 50, final = 0.6·spice + 0.4·reaction (0.3·editor + 0.7·signals) ≥ 76, ≤ 3 per job. Greetings / thanks /
donations, routine questions and requests never qualify; platform-unsafe ones (calls to violence, extremism,
hate by nation / race / religion, suicide) are dropped by a local filter and by the editor — мат / пошлость /
троллинг / чёрный юмор are fine (titles get мат masked). No editor verdict or nothing sharp → no comment clips
(the regular Streamer moments). `selection.comment.board` lists every candidate with its scores and decision.

## Switches

| env / param | default | |
|---|---|---|
| `params.editor` / `SONYA_LLM_EDITOR` | on when a provider is configured | Gemini editor |
| `STREAMER_VISUAL_SIGNALS` | true | game-event pass (one low-res decode of the VOD) |
| `STREAMER_HWACCEL` | — | `cuda` on Vast, `videotoolbox` on a Mac |
| `params.webcam_detector` / `STREAMER_WEBCAM_DETECTOR` | false | YOLO webcam box as a seed |
| `params.chat_path` | — | Twitch (TwitchDownloader JSON) / YouTube live chat replay |
| `STREAMER_COMMENT` / `params.comment_moments` | true | «Завозный коммент» pass (off = no comment clips at all) |
| `STREAMER_COMMENT_CAM` | true | webcam motion as a reaction signal (off = audio / speech / chat only) |
| `STREAMER_CHAT_FETCH` | on | worker fetches the VOD chat for streamer jobs by link |
| `STREAMER_CHAT_WAIT_SEC` | 180 | how long the runner waits for a chat fetch still running |
