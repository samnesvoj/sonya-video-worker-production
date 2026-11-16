"""End to end: the real bot-side module (integration/accounting_bot/vast_reports.py) against the real server side, in process,
with a fake Telegram. Also proves the Telegram token never reaches the server side."""
import importlib.util
import json
import sys
import urllib.error
from datetime import date
from pathlib import Path

import pytest

from accounting.internal_api import AccountingApi
from tests.acct.helpers import msk
from tests.acct.test_scheduler_and_delivery import SECRET, service

spec = importlib.util.spec_from_file_location("vast_reports", Path(__file__).resolve().parents[2] / "integration/accounting_bot/vast_reports.py")
vr = importlib.util.module_from_spec(spec)
sys.modules["vast_reports"] = vr
spec.loader.exec_module(vr)

TG_TOKEN = "123456789:TELEGRAM_TOKEN_MUST_NEVER_REACH_SERVER"


class FakeTelegram:
    def __init__(self):
        self.sent, self.fail = [], False
        self.token = TG_TOKEN

    def send(self, chat_id, text, reply_to=None, parse_mode=None):
        if self.fail:
            raise RuntimeError("telegram down")
        self.sent.append((chat_id, text))
        self.modes = getattr(self, "modes", []) + [parse_mode]
        return 1000 + len(self.sent)


class InProcess:
    """urlopen replacement routing the bot's HTTP call into AccountingApi.handle; records every byte sent."""

    def __init__(self, api):
        self.api, self.wire, self.down = api, [], False

    def __call__(self, req, timeout=0):
        if self.down:
            raise urllib.error.URLError("net down")
        body = req.data or b""
        self.wire.append(json.dumps(dict(req.header_items())) + body.decode())
        status, data = self.api.handle(req.get_method(), req.selector, dict(req.header_items()), body)

        class R:
            def __init__(s): s.status = status
            def read(s): return json.dumps(data).encode()
            def __enter__(s): return s
            def __exit__(s, *a): pass
        return R()


def setup(tmp_path):
    svc, led = service()
    now = [msk(2026, 11, 7, 0, 6)]
    svc.run_due(now[0])
    api = AccountingApi(svc, SECRET, clock=lambda: now[0])
    net = InProcess(api)
    tg = FakeTelegram()
    puller = vr.VastReportPuller("https://sonya.example", SECRET, tg, [111, 222], tmp_path / "state.json", clock=lambda: now[0], opener=net, interval=0)
    return svc, led, api, net, tg, puller, now


def test_happy_path_sends_to_all_chats_and_acks(tmp_path):
    svc, led, api, net, tg, puller, now = setup(tmp_path)
    assert puller.poll() == 1
    assert [c for c, _ in tg.sent] == [111, 222] and tg.sent[0][1].startswith("<b>SONYA · Vast.ai — расходы за 06.11.2026</b>") and tg.modes == ["HTML", "HTML"]
    assert led.c.execute("SELECT status FROM report_outbox").fetchone()["status"] == "sent"
    assert puller.poll() == 0


def test_telegram_token_and_secret_never_on_the_wire(tmp_path):
    svc, led, api, net, tg, puller, now = setup(tmp_path)
    puller.poll()
    wire = "\n".join(net.wire)
    assert TG_TOKEN not in wire and SECRET not in wire and "TELEGRAM" not in wire.upper().replace("TELEGRAM SEND FAILED", "")


def test_telegram_failure_then_retry_sends_exactly_once(tmp_path):
    svc, led, api, net, tg, puller, now = setup(tmp_path)
    tg.fail = True
    assert puller.poll() == 0 and tg.sent == []
    assert led.c.execute("SELECT status, attempts FROM report_outbox").fetchone()["status"] == "pending"
    tg.fail = False
    now[0] += 3600
    assert puller.poll() == 1 and len(tg.sent) == 2
    now[0] += 3600
    assert puller.poll() == 0 and len(tg.sent) == 2


def test_lost_ack_does_not_resend_to_telegram(tmp_path):
    svc, led, api, net, tg, puller, now = setup(tmp_path)
    real = puller._post
    def lose_sent_ack(path, body):
        if path.endswith("/sent"):
            raise vr.VastReportError("network down")
        return real(path, body)
    puller._post = lose_sent_ack
    with pytest.raises(vr.VastReportError):
        puller.poll()
    assert len(tg.sent) == 2                                    # delivered once
    puller._post = real
    now[0] += 400                                               # lease expired -> server offers it again
    assert puller.poll() == 1
    assert len(tg.sent) == 2                                    # state file prevented a duplicate Telegram message
    assert led.c.execute("SELECT status FROM report_outbox").fetchone()["status"] == "sent"


def test_partial_delivery_resumes_only_missing_chats(tmp_path):
    svc, led, api, net, tg, puller, now = setup(tmp_path)
    orig = tg.send
    def flaky(chat, text, reply_to=None, parse_mode=None):
        if chat == 222 and not getattr(flaky, "ok", False):
            raise RuntimeError("blocked")
        return orig(chat, text, reply_to, parse_mode)
    tg.send = flaky
    assert puller.poll() == 0 and [c for c, _ in tg.sent] == [111]
    flaky.ok = True
    now[0] += 3600
    assert puller.poll() == 1 and [c for c, _ in tg.sent] == [111, 222]


def test_poll_safe_never_raises_and_is_rate_limited(tmp_path):
    svc, led, api, net, tg, puller, now = setup(tmp_path)
    puller.interval = 60
    net.down = True
    puller.poll_safe(); puller.poll_safe()                      # swallowed
    net.down = False
    puller.poll_safe()                                          # still inside the 60 s window
    assert tg.sent == []
    now[0] += 61
    puller.poll_safe()
    assert len(tg.sent) == 2


def test_state_file_permissions(tmp_path):
    svc, led, api, net, tg, puller, now = setup(tmp_path)
    puller.poll()
    assert (tmp_path / "state.json").stat().st_mode & 0o777 == 0o600


def test_vast_commands_send_html_and_unknown_text_is_not_ours(tmp_path):
    from accounting.summary import build_summary
    svc, led, api, net, tg, puller, now = setup(tmp_path)
    api._summary = lambda n: build_summary(svc.store, svc.rates, n, None, led)
    assert puller.command(111, "/vast") is True
    assert tg.sent[-1][1].startswith("<b>SONYA · Vast.ai — сводка</b>") and tg.modes[-1] == "HTML"
    assert puller.command(111, "/vast_day 06.11") is True and "расходы за 06.11.2026" in tg.sent[-1][1]
    assert puller.command(111, "/vast_day 01.11") is True and tg.sent[-1][1].startswith("Отчёта за этот день нет") and tg.modes[-1] is None
    assert puller.command(111, "/vast_day завтра") is True and tg.sent[-1][1].startswith("Не понял дату")
    n = len(tg.sent)
    assert puller.command(111, "/list") is False and len(tg.sent) == n


def test_markup_rejected_by_telegram_falls_back_to_plain_once(tmp_path):
    svc, led, api, net, tg, puller, now = setup(tmp_path)
    orig = tg.send
    def picky(chat, text, reply_to=None, parse_mode=None):
        if parse_mode:
            raise RuntimeError("telegram sendMessage: Bad Request: can't parse entities: unexpected end tag")
        return orig(chat, text, reply_to)
    tg.send = picky
    assert puller.poll() == 1
    assert all("<b>" not in t and "<pre>" not in t for _, t in tg.sent) and tg.sent[0][1].startswith("SONYA · Vast.ai — расходы за 06.11.2026")


def test_network_error_is_not_mistaken_for_markup_error(tmp_path):
    svc, led, api, net, tg, puller, now = setup(tmp_path)
    def down(chat, text, reply_to=None, parse_mode=None):
        raise RuntimeError("telegram unreachable (URLError)")
    tg.send = down
    assert puller.poll() == 0 and led.c.execute("SELECT status FROM report_outbox").fetchone()["status"] == "pending"


def test_unavailable_server_gives_a_short_plain_answer(tmp_path):
    svc, led, api, net, tg, puller, now = setup(tmp_path)
    net.down = True
    assert puller.command(111, "/vast") is True and tg.sent[-1][1].startswith("Не получилось получить данные Vast")
