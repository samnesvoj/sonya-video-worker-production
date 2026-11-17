"""YouTube account cookies pool on the owner's Mac mini (owner's consent 2026-11-17: a SEPARATE, non-personal account).

The owner drops cookies.txt (browser extension "Get cookies.txt LOCALLY", signed in to the separate account) into
  /Users/agent/Business/SONYA/_incoming/yt-cookies/
`tick` (launchd group.sonya.ytcookies, every 2 min):
  1. ingest: each *.txt there is checked (Netscape format, youtube.com auth cookies), stored in the login Keychain
     (service group.sonya.yt-cookies, account pool-N.<chunk>; base64 in <=3000-char chunks via `security -i` stdin, so
     the value is never on a command line), and the file is overwritten with zeros and deleted;
  2. activate: the newest pool entry not marked bad is written to the relays' cookie files (0600):
       Mac  ~/SONYA/ytrelay/secret/cookies.txt
       bot  root@147.45.235.79:/etc/sonya-ytrelay/cookies.txt (over ssh stdin)
  3. rotate: when a relay heartbeat reports cookies "expired" for the active entry, it is marked bad and the next one
     is activated; none left -> the relays' copies are removed (the accountant alerts "cookies протухли").
The relays use cookies ONLY on a "not a bot" retry (scripts/url_ingest.py). Nothing here prints a cookie value.
State without secrets: ~/SONYA/ytrelay/secret/pool.json.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

INCOMING = Path(os.environ.get("SONYA_YT_COOKIES_INCOMING", "/Users/agent/Business/SONYA/_incoming/yt-cookies"))
SECRET = Path(os.environ.get("SONYA_YT_COOKIES_SECRET_DIR", str(Path.home() / "SONYA" / "ytrelay" / "secret")))
STATE = SECRET / "pool.json"
SERVICE = os.environ.get("SONYA_YT_COOKIES_SERVICE", "group.sonya.yt-cookies")
BOT = os.environ.get("SONYA_YT_COOKIES_BOT", "root@147.45.235.79")
CHUNK = 3000
AUTH_NAMES = ("SAPISID", "__Secure-3PSID", "__Secure-1PSID", "SID", "LOGIN_INFO", "__Secure-3PAPISID")


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def _state() -> dict:
    try:
        return json.loads(STATE.read_text())
    except Exception:  # noqa: BLE001
        return {"entries": [], "active": None}


def _save(st: dict) -> None:
    SECRET.mkdir(parents=True, exist_ok=True)
    os.chmod(SECRET, 0o700)
    STATE.write_text(json.dumps(st, indent=1))


def validate(data: bytes) -> str:
    """'' if usable, else the reason (no cookie value in it)."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return "not text"
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    rows = [ln.split("\t") for ln in lines]
    yt = [r for r in rows if len(r) >= 7 and r[0].lstrip(".").endswith(("youtube.com", "google.com"))]
    if not yt:
        return "no youtube.com cookies (Netscape cookies.txt expected)"
    if not any(r[5] in AUTH_NAMES for r in yt):
        return "no sign-in cookies: export while signed in to the separate account"
    return ""


def _security(cmds: str) -> bool:
    r = subprocess.run(["/usr/bin/security", "-i"], input=cmds, capture_output=True, text=True)   # output discarded
    return r.returncode == 0


def store(entry: str, data: bytes) -> bool:
    b64 = base64.b64encode(data).decode()
    parts = [b64[i:i + CHUNK] for i in range(0, len(b64), CHUNK)]
    cmds = "".join(f"add-generic-password -U -s {SERVICE} -a {entry}.{i} -w {p}\n" for i, p in enumerate(parts))
    cmds += f"add-generic-password -U -s {SERVICE} -a {entry}.n -w {len(parts)}\n"
    return _security(cmds) and load(entry) == data


def _get(account: str) -> str:
    r = subprocess.run(["/usr/bin/security", "find-generic-password", "-s", SERVICE, "-a", account, "-w"],
                       capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def load(entry: str) -> bytes:
    n = _get(f"{entry}.n")
    if not n.isdigit():
        return b""
    return base64.b64decode("".join(_get(f"{entry}.{i}") for i in range(int(n))))


def _shred(p: Path) -> None:
    try:
        size = p.stat().st_size
        with open(p, "r+b") as f:
            f.write(b"\0" * size)
            f.flush()
            os.fsync(f.fileno())
    except OSError:
        pass
    p.unlink(missing_ok=True)


def ingest(st: dict) -> bool:
    changed = False
    if not INCOMING.is_dir():
        return False
    for p in sorted(INCOMING.glob("*.txt")):
        if p.name.upper().startswith("README") or p.name.endswith(".rejected.txt"):
            continue
        data = p.read_bytes()
        why = validate(data)
        if why:
            log(f"rejected {p.name}: {why}")
            (INCOMING / (p.name + ".rejected.txt")).write_text(
                f"Файл не принят: {why}. Сам файл удалён. Выгрузите cookies.txt заново.\n")
            _shred(p)
            continue
        n = max([int(e["name"].split("-")[1]) for e in st["entries"]] + [0]) + 1
        name = f"pool-{n}"
        if store(name, data):
            st["entries"].append({"name": name, "added": time.time(), "bad": False})
            st["active"] = None                      # the newest good one becomes active below
            log(f"stored {p.name} as {name} (Keychain)")
            changed = True
        else:
            log(f"Keychain store failed for {p.name}; file left in place")
            continue
        _shred(p)
    return changed


def _write_targets(data: bytes) -> bool:
    mac = SECRET / "cookies.txt"
    SECRET.mkdir(parents=True, exist_ok=True)
    os.chmod(SECRET, 0o700)
    if data:
        fd = os.open(mac, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        cmd = "umask 077; mkdir -p /etc/sonya-ytrelay; cat > /etc/sonya-ytrelay/cookies.txt.new && mv /etc/sonya-ytrelay/cookies.txt.new /etc/sonya-ytrelay/cookies.txt"
    else:
        mac.unlink(missing_ok=True)
        cmd = "rm -f /etc/sonya-ytrelay/cookies.txt"
    if not BOT:
        return True
    r = subprocess.run(["/usr/bin/ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", BOT, cmd], input=data,
                       capture_output=True, timeout=60)
    return r.returncode == 0


def activate(st: dict) -> None:
    good = [e for e in st["entries"] if not e.get("bad")]
    want = good[-1]["name"] if good else None
    if want == st.get("active") and st.get("pushed"):
        return
    data = load(want) if want else b""
    if _write_targets(data):
        st["active"], st["pushed"] = want, True
        log(f"active cookies: {want or 'none'}")
    else:
        st["pushed"] = False
        log("push to the bot server failed; retry next tick")


def rotate(st: dict) -> bool:
    """Relay heartbeats say 'expired' for the active entry -> mark it bad."""
    if not st.get("active"):
        return False
    try:
        sys.path.insert(0, str(Path.home() / "SONYA" / "ytrelay" / "current"))
        from scripts import yt_relay
        hbs = yt_relay.heartbeats()
    except Exception as exc:  # noqa: BLE001 - offline: try later
        log(f"heartbeats unavailable: {type(exc).__name__}")
        return False
    since = next((e["added"] for e in st["entries"] if e["name"] == st["active"]), 0)
    expired = [n for n, h in hbs.items() if h.get("cookies") == "expired" and float(h.get("cookies_checked") or 0) > since + 60]
    if not expired:
        return False
    for e in st["entries"]:
        if e["name"] == st["active"]:
            e["bad"], e["bad_at"] = True, time.time()
    log(f"{st['active']} expired (relays: {', '.join(expired)}); rotating")
    st["pushed"] = False
    return True


def tick() -> int:
    st = _state()
    ingest(st)
    rotate(st)
    activate(st)
    _save(st)
    return 0


def status() -> int:
    st = _state()
    print(json.dumps({"active": st.get("active"), "entries": [{k: e[k] for k in ("name", "bad") if k in e} for e in st["entries"]]}))
    return 0


if __name__ == "__main__":
    sys.exit({"tick": tick, "status": status}.get(sys.argv[1] if len(sys.argv) > 1 else "tick", tick)())
