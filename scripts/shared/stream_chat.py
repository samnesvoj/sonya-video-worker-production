"""
stream_chat.py
==============
Chat replay of a stream VOD with the message TEXT (stream_signals.chat_times only needs the times):

  load(path)            → [{"t", "author", "text", "color"}] from any supported replay file:
                          * yt-dlp YouTube live chat (``*.live_chat.json``, JSON lines, replayChatItemAction);
                          * Twitch VOD chat: TwitchDownloader / rechat JSON ({"comments": [...]}) and our own
                            fetch_twitch() output (the same shape);
                          * a plain JSON list of {"t", "author", "text"}.
  fetch_twitch(id, …)   → Twitch VOD chat over the public GQL API (the web player's client id, no login), several
                          offset ranges in parallel; best effort, time-capped.
  fetch_youtube(url, …) → yt-dlp ``--write-subs --sub-langs live_chat --skip-download``; only streams that were
                          live with a chat replay have one. Best effort.
  fetch_for_source(…)   → whichever of the two fits the link; writes ``<source>.live_chat.json`` (the runner's
                          _chat_path picks it up), never raises.

Used by the «Завозный коммент» moments (scripts/shared/stream_comment.py) and the chat-density signal.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

TWITCH_GQL = "https://gql.twitch.tv/gql"
TWITCH_CLIENT_ID = "kimne78kx3ncx6brgo4mv6wki5h1ko"          # the public web player id (TwitchDownloader uses it too)
_TW_QUERY = ("query($id:ID!,$off:Int){video(id:$id){comments(contentOffsetSeconds:$off)"
             "{edges{cursor node{contentOffsetSeconds commenter{displayName login} "
             "message{fragments{text} userColor}}} pageInfo{hasNextPage}}}}")


# ── loading ─────────────────────────────────────────────────────────────────

def _msg(t: Any, author: Any, text: Any, color: Any = None) -> Optional[Dict[str, Any]]:
    try:
        t = float(t)
    except (TypeError, ValueError):
        return None
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return None
    col = str(color) if isinstance(color, str) and re.fullmatch(r"#[0-9A-Fa-f]{6}", color) else None
    return {"t": round(t, 3), "author": str(author or "").strip()[:40], "text": text[:500], "color": col}


def _yt_runs(runs: Any) -> str:
    out = []
    for r in runs or []:
        if not isinstance(r, dict):
            continue
        if "text" in r:
            out.append(str(r["text"]))
        elif r.get("emoji"):                            # custom / unicode emoji: its shortcut or the char itself
            e = r["emoji"]
            sc = (e.get("shortcuts") or [None])[0]
            out.append(e.get("emojiId") if e.get("emojiId") and len(str(e.get("emojiId"))) <= 2 else (sc or ""))
    return "".join(out)


def load(path: str) -> List[Dict[str, Any]]:
    """Messages sorted by time; [] for an unreadable / empty file."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    out: List[Dict[str, Any]] = []
    try:
        data = json.loads(text)
    except ValueError:
        data = None
    if isinstance(data, dict) and isinstance(data.get("comments"), list):
        for c in data["comments"]:
            if not isinstance(c, dict):
                continue
            who = c.get("commenter") or {}
            msg = c.get("message") or {}
            body = msg.get("body")
            if body is None:
                body = "".join(str(f.get("text") or "") for f in msg.get("fragments") or [] if isinstance(f, dict))
            m = _msg(c.get("content_offset_seconds", c.get("t")), who.get("display_name") or who.get("name")
                     or c.get("author"), body, msg.get("user_color") or c.get("color"))
            if m:
                out.append(m)
    elif isinstance(data, list):
        for c in data:
            if isinstance(c, dict):
                m = _msg(c.get("t"), c.get("author"), c.get("text"), c.get("color"))
                if m:
                    out.append(m)
    else:
        for line in text.splitlines():                  # yt-dlp live_chat: JSON lines
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            act = obj.get("replayChatItemAction") if isinstance(obj, dict) else None
            if not act:
                continue
            off = act.get("videoOffsetTimeMsec")
            for a in act.get("actions") or []:
                it = (a.get("addChatItemAction") or {}).get("item") or {}
                r = it.get("liveChatTextMessageRenderer") or it.get("liveChatPaidMessageRenderer")
                if not r or off is None:
                    continue
                m = _msg(float(off) / 1000.0, (r.get("authorName") or {}).get("simpleText"),
                         _yt_runs((r.get("message") or {}).get("runs")))
                if m:
                    out.append(m)
    out.sort(key=lambda m: m["t"])
    return out


# ── fetching ────────────────────────────────────────────────────────────────

def twitch_video_id(url: str) -> Optional[str]:
    m = re.search(r"twitch\.tv/(?:[\w-]+/)?(?:videos?|v)/(\d+)", url or "")
    return m.group(1) if m else None


def _gql(body: Dict[str, Any], timeout: float = 15.0) -> Dict[str, Any]:
    req = urllib.request.Request(TWITCH_GQL, data=json.dumps(body).encode("utf-8"), method="POST",
                                 headers={"Client-ID": TWITCH_CLIENT_ID, "Content-Type": "application/json",
                                          "User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:      # noqa: S310 - fixed https host
        return json.loads(r.read().decode("utf-8"))


def _twitch_range(vid: str, a: int, b: int, deadline: float, gql=_gql) -> List[Dict[str, Any]]:
    """Comments with offsets in [a, b). Paged by OFFSET (cursor pages need a Twitch integrity token since 2024):
    the next page starts at the last offset seen (duplicates dropped by the caller), +1 s when a page did not
    move forward (a very busy second may lose a few messages — fine for finding the read-out ones)."""
    out: List[Dict[str, Any]] = []
    off = a
    while off < b and time.monotonic() < deadline:
        var = {"id": vid, "off": int(off)}
        try:
            d = gql({"query": _TW_QUERY, "variables": var})
        except Exception as exc:                       # one retry after a pause, then give the range up
            logger.info("[stream_chat] twitch page failed at %s: %s", off, exc)
            time.sleep(1.0)
            try:
                d = gql({"query": _TW_QUERY, "variables": var})
            except Exception:
                break
        com = (((d.get("data") or {}).get("video") or {}).get("comments") or {})
        edges = com.get("edges") or []
        if not edges:
            break
        last_t = off
        for e in edges:
            n = e.get("node") or {}
            t = n.get("contentOffsetSeconds")
            if t is None:
                continue
            last_t = max(last_t, t)
            if t < a or t >= b:
                continue
            who = n.get("commenter") or {}
            msg = n.get("message") or {}
            out.append({"content_offset_seconds": t,
                        "commenter": {"display_name": who.get("displayName") or who.get("login") or ""},
                        "message": {"body": "".join(str(f.get("text") or "") for f in msg.get("fragments") or []),
                                    "user_color": msg.get("userColor")}})
        if not (com.get("pageInfo") or {}).get("hasNextPage"):
            break
        off = last_t if last_t > off else off + 1
    return out


def fetch_twitch(vid: str, out_path: str, start: float = 0.0, end: Optional[float] = None, *,
                 parallel: int = 8, budget_sec: float = 240.0, gql=_gql) -> int:
    """Twitch VOD chat for [start, end) of the VOD, offsets rebased to the downloaded piece (t - start).
    Writes the rechat shape {"comments": [...]}; returns the number of messages (0 → no file written)."""
    if end is None:
        try:
            d = gql({"query": "query($id:ID!){video(id:$id){lengthSeconds}}", "variables": {"id": vid}})
            end = float(((d.get("data") or {}).get("video") or {}).get("lengthSeconds") or 0)
        except Exception as exc:
            logger.info("[stream_chat] twitch length failed: %s", exc)
            return 0
    a0, b0 = int(max(0.0, start)), int(end or 0)
    if b0 <= a0:
        return 0
    n = max(1, min(parallel, (b0 - a0) // 300 or 1))
    step = (b0 - a0 + n - 1) // n
    ranges = [(a0 + i * step, min(b0, a0 + (i + 1) * step)) for i in range(n)]
    deadline = time.monotonic() + budget_sec
    with ThreadPoolExecutor(max_workers=n) as pool:
        parts = list(pool.map(lambda r: _twitch_range(vid, r[0], r[1], deadline, gql), ranges))
    seen, comments = set(), []
    for p in parts:
        for c in p:
            key = (c["content_offset_seconds"], c["commenter"]["display_name"], c["message"]["body"])
            if key in seen:
                continue
            seen.add(key)
            c = dict(c)
            c["content_offset_seconds"] = round(float(c["content_offset_seconds"]) - a0, 3)
            comments.append(c)
    if not comments:
        return 0
    comments.sort(key=lambda c: c["content_offset_seconds"])
    Path(out_path).write_text(json.dumps({"video": {"id": vid, "start": a0, "end": b0}, "comments": comments},
                                         ensure_ascii=False), encoding="utf-8")
    return len(comments)


def fetch_youtube(url: str, out_path: str, start: float = 0.0, *, timeout: float = 300.0,
                  extra_opts: Optional[Dict[str, Any]] = None) -> int:
    """yt-dlp live_chat replay → out_path (JSON lines, offsets rebased by ``start``). 0 when the video has none."""
    import tempfile
    try:
        import yt_dlp
    except ImportError:
        return 0
    with tempfile.TemporaryDirectory(prefix="sonya-chat-") as tmp:
        opts = dict(extra_opts or {})
        opts.update({"skip_download": True, "writesubtitles": True, "subtitleslangs": ["live_chat"],
                     "outtmpl": os.path.join(tmp, "chat.%(ext)s"), "quiet": True, "no_warnings": True,
                     "socket_timeout": 30})
        t0 = time.monotonic()
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
        except Exception as exc:
            logger.info("[stream_chat] youtube live_chat failed: %s", str(exc)[:200])
            return 0
        got = next((p for p in Path(tmp).glob("*.live_chat.json")), None)
        if got is None:
            return 0
        n = 0
        with open(got, encoding="utf-8", errors="replace") as src, open(out_path, "w", encoding="utf-8") as dst:
            for line in src:
                try:
                    obj = json.loads(line)
                    act = obj["replayChatItemAction"]
                    off = float(act["videoOffsetTimeMsec"]) / 1000.0 - start
                except (ValueError, KeyError, TypeError):
                    continue
                if off < 0:
                    continue
                act["videoOffsetTimeMsec"] = str(int(off * 1000))
                dst.write(json.dumps(obj, ensure_ascii=False) + "\n")
                n += 1
        logger.info("[stream_chat] youtube live_chat: %d lines in %.0fs", n, time.monotonic() - t0)
        return n


def fetch_for_source(url: str, platform: str, video_path: str, start: float = 0.0,
                     end: Optional[float] = None, yt_opts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Chat replay next to the downloaded source (``<video>.live_chat.json``). Never raises.
    -> {"status": "ok"|"none"|"error"|"off", "messages": n, "platform"}"""
    if os.environ.get("STREAMER_CHAT_FETCH", "on").strip().lower() in ("0", "off", "false", "no"):
        return {"status": "off"}
    dest = str(Path(video_path).with_suffix(".live_chat.json"))
    t0 = time.monotonic()
    try:
        if platform == "twitch" or "twitch.tv" in (url or ""):
            vid = twitch_video_id(url)
            n = fetch_twitch(vid, dest, start, end) if vid else 0
        elif platform == "youtube" or "youtu" in (url or ""):
            n = fetch_youtube(url, dest, start, extra_opts=yt_opts)
        else:
            return {"status": "none", "platform": platform}
    except Exception as exc:                           # a chat is a bonus, never a failure
        return {"status": "error", "error": str(exc)[:200], "platform": platform}
    return {"status": "ok" if n else "none", "messages": n, "platform": platform, "sec": round(time.monotonic() - t0, 1),
            **({"path": dest} if n else {})}


PENDING_SUFFIX = ".live_chat.pending"


def start_background(url: str, platform: str, video_path: str, sf: Optional[Dict[str, Any]] = None):
    """Worker side (owner 2026-11-19, «Завозный коммент»): fetch the chat next to the downloaded source in a thread
    while models load and ASR runs. A ``<video>.live_chat.pending`` marker tells the runner to wait for it
    (wait_pending). Returns the thread (or None when off / not a chat platform)."""
    import threading
    plat = platform or ("twitch" if "twitch.tv" in (url or "") else "youtube" if "youtu" in (url or "") else "")
    if plat not in ("twitch", "youtube") or os.environ.get("STREAMER_CHAT_FETCH", "on").strip().lower() in (
            "0", "off", "false", "no"):
        return None
    start, length = 0, 0
    try:
        from scripts.url_ingest import _youtube_opts, trim_request
        start, length = trim_request(sf or {}, 10 ** 7)
        yt_opts = _youtube_opts("youtube") if plat == "youtube" else None
    except Exception:                                  # noqa: BLE001
        yt_opts = None
    marker = Path(video_path).with_suffix(PENDING_SUFFIX)
    try:
        marker.write_text(str(time.time()), encoding="utf-8")
    except OSError:
        return None

    def _run():
        try:
            r = fetch_for_source(url, plat, video_path, float(start), float(start + length) if length else None,
                                 yt_opts=yt_opts)
            logger.info("[stream_chat] %s", {k: v for k, v in r.items() if k != "path"})
        finally:
            marker.unlink(missing_ok=True)
    th = threading.Thread(target=_run, name="stream-chat-fetch", daemon=True)
    th.start()
    return th


def wait_pending(video_path: str, timeout: float = 180.0) -> None:
    """Runner side: block (≤ timeout) while the worker's chat fetch for this source is still running."""
    marker = Path(video_path).with_suffix(PENDING_SUFFIX)
    t_end = time.monotonic() + timeout
    while marker.exists() and time.monotonic() < t_end:
        time.sleep(1.0)
