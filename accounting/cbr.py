"""
Official CBR RF exchange rate (USD->RUB). Source: https://www.cbr.ru/scripts/XML_daily.asp?date_req=DD/MM/YYYY
The rate is stored per MSK day together with the CBR effective date; if CBR cannot be reached the last stored rate
at or before the day is used and the rate is flagged fallback=True (the report says so). Never invents a rate.
"""
from __future__ import annotations

import re
import urllib.request
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Callable, Optional

CBR_URL = "https://www.cbr.ru/scripts/XML_daily.asp?date_req={d:%d/%m/%Y}"


class RateUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class Rate:
    day: date
    value: Decimal            # RUB per 1 USD
    cbr_date: str             # date the CBR published the rate for (dd.mm.yyyy)
    fallback: bool = False


def parse_xml(xml: bytes | str) -> tuple[Decimal, str]:
    text = xml.decode("cp1251", "ignore") if isinstance(xml, bytes) else xml
    d = re.search(r'<ValCurs[^>]*Date="([0-9.]+)"', text)
    m = re.search(r'<Valute[^>]*>(?:(?!</Valute>).)*?<CharCode>USD</CharCode>(?:(?!</Valute>).)*?<Nominal>(\d+)</Nominal>(?:(?!</Valute>).)*?<Value>([\d,]+)</Value>', text, re.S)
    if not (d and m):
        raise RateUnavailable("USD not found in CBR answer")
    value = Decimal(m.group(2).replace(",", ".")) / Decimal(m.group(1))
    if value <= 0:
        raise RateUnavailable("non-positive rate")
    return value, d.group(1)


def http_fetch(day: date) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(CBR_URL.format(d=day), headers={"User-Agent": "sonya-accounting"}), timeout=20) as r:
        return r.read()


class CbrRates:
    def __init__(self, ledger, fetch: Callable[[date], bytes] = http_fetch):
        self.ledger, self._fetch = ledger, fetch

    def get(self, day: date) -> Rate:
        stored = self.ledger.get_rate(day)
        if stored and not stored["fallback"]:
            return Rate(day, stored["value"], stored["cbr_date"], False)
        try:
            value, cbr_date = parse_xml(self._fetch(day))
            self.ledger.put_rate(day, value, cbr_date, False)
            return Rate(day, value, cbr_date, False)
        except Exception as exc:  # noqa: BLE001 - network, parse, anything: fall back honestly
            prev = self.ledger.last_rate_before(day)
            if not prev:
                raise RateUnavailable("no CBR rate and no earlier stored rate") from exc
            self.ledger.put_rate(day, prev["value"], prev["cbr_date"], True)
            return Rate(day, prev["value"], prev["cbr_date"], True)
