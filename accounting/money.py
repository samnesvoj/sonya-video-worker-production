"""Money helpers. Every amount is a Decimal; floats never enter a calculation (they are converted via str())."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

try:
    from zoneinfo import ZoneInfo
    MSK = ZoneInfo("Europe/Moscow")
except Exception:  # pragma: no cover
    MSK = timezone(timedelta(hours=3))
UTC = timezone.utc
HOURS_PER_MONTH = Decimal("720")        # Vast prices storage per GB-month as /720 h (matches storage_total_cost in offers)


def D(v: Any) -> Decimal:
    if isinstance(v, Decimal):
        return v
    return Decimal(str(v)) if v not in (None, "") else Decimal("0")


def usd(v: Decimal) -> str:
    return f"${D(v).quantize(Decimal('0.01'), ROUND_HALF_UP):,.2f}"


def usd4(v: Decimal) -> str:
    return f"${D(v).quantize(Decimal('0.0001'), ROUND_HALF_UP):,.4f}"


def rub(v: Decimal) -> str:
    return f"{D(v).quantize(Decimal('0.01'), ROUND_HALF_UP):,.2f}".replace(",", " ") + " ₽"


def both(usd_v: Decimal, rate: Decimal) -> str:
    return f"{usd(usd_v)} / {rub(D(usd_v) * D(rate))}"


def msk_day_bounds(day: date) -> tuple[float, float]:
    s = datetime(day.year, day.month, day.day, tzinfo=MSK)
    return s.timestamp(), (s + timedelta(days=1)).timestamp()


def msk_date_of(ts: float) -> date:
    return datetime.fromtimestamp(ts, MSK).date()


def month_bounds_msk(year: int, month: int) -> tuple[date, date]:
    first = date(year, month, 1)
    nxt = date(year + (month == 12), 1 if month == 12 else month + 1, 1)
    return first, nxt - timedelta(days=1)
