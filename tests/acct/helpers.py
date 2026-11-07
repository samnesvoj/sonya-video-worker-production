import sqlite3
from datetime import date, datetime
from decimal import Decimal

from accounting.cbr import Rate
from accounting.ledger import Ledger
from accounting.money import MSK
from dispatcher.store import FleetStore


def msk(y, m, d, hh=0, mm=0, ss=0):
    return datetime(y, m, d, hh, mm, ss, tzinfo=MSK).timestamp()


class FixedRates:
    def __init__(self, value="80", fallback=False):
        self.value, self.fallback = Decimal(value), fallback

    def get(self, day):
        return Rate(day, self.value, day.strftime("%d.%m.%Y"), self.fallback)


class Jobs:
    def __init__(self, per_day=None):
        self.per_day = per_day or {}

    def completed_count(self, lo, hi):
        d = datetime.fromtimestamp(lo, MSK).date()
        return self.per_day.get(d, 0)


def fixture_store():
    """One RTX 3090: created 10:00, running 10:00-11:30, stopped 11:30-12:30, destroyed 12:30 on 2026-11-06 (MSK)."""
    st = FleetStore()
    t0 = msk(2026, 11, 6, 10)
    st.add(t0, id=1, label="sonya-pool-aa", state="destroyed", gpu_name="RTX 3090", dph_base=Decimal("0.20"),
           storage_gb_month=Decimal("0.36"), disk_gb=Decimal("80"), release_sha="a" * 40)
    st.event(t0, 1, "created")
    st.event(t0, 1, "running", via="creating")
    st.event(msk(2026, 11, 6, 11, 30), 1, "stopped")
    st.event(msk(2026, 11, 6, 12, 30), 1, "destroy_requested")
    return st
