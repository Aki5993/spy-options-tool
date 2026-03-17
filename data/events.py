from __future__ import annotations
"""FOMC dates, OPEX, VIX expiry computation."""
import os
import sys
import datetime
from typing import Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import FRED_API_KEY
from data.cache import cache_get, cache_set


def _third_friday(year: int, month: int) -> datetime.date:
    """Return 3rd Friday of given month."""
    first_day = datetime.date(year, month, 1)
    # Find first Friday
    days_until_friday = (4 - first_day.weekday()) % 7
    first_friday = first_day + datetime.timedelta(days=days_until_friday)
    return first_friday + datetime.timedelta(weeks=2)


def compute_opex_dates(start_year: int = 2000, end_year: int | None = None) -> list[datetime.date]:
    """Monthly OPEX = 3rd Friday of each month."""
    if end_year is None:
        end_year = datetime.date.today().year + 2
    dates = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            dates.append(_third_friday(year, month))
    return sorted(dates)


def compute_vix_expiry_dates(start_year: int = 2000, end_year: int | None = None) -> list[datetime.date]:
    """VIX expiry = Wednesday before the 3rd Friday of each month."""
    if end_year is None:
        end_year = datetime.date.today().year + 2
    dates = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            third_fri = _third_friday(year, month)
            # Wednesday before 3rd Friday = 3rd Friday - 2 days
            vix_exp = third_fri - datetime.timedelta(days=2)
            dates.append(vix_exp)
    return sorted(dates)


def fetch_fomc_dates() -> list[datetime.date]:
    """
    Fetch FOMC meeting dates from FRED or use fallback hardcoded 2025-2026 dates.
    Returns sorted list of datetime.date objects.
    """
    key = "fomc_dates"
    cached = cache_get(key, "fomc_dates")
    if cached is not None and not cached.empty:
        return [d.date() if hasattr(d, "date") else d for d in cached.index.tolist()]

    dates = []
    if FRED_API_KEY:
        try:
            from fredapi import Fred
            fred = Fred(api_key=FRED_API_KEY)
            # FOMC_MTGDATE series
            series = fred.get_series("FOMC_MTGDATE")
            dates = [d.date() if hasattr(d, "date") else d for d in series.index.tolist()]
        except Exception:
            dates = []

    if not dates:
        # Hardcoded approximate FOMC dates 2024-2026
        dates = [
            datetime.date(2024, 1, 31),
            datetime.date(2024, 3, 20),
            datetime.date(2024, 5, 1),
            datetime.date(2024, 6, 12),
            datetime.date(2024, 7, 31),
            datetime.date(2024, 9, 18),
            datetime.date(2024, 11, 7),
            datetime.date(2024, 12, 18),
            datetime.date(2025, 1, 29),
            datetime.date(2025, 3, 19),
            datetime.date(2025, 5, 7),
            datetime.date(2025, 6, 18),
            datetime.date(2025, 7, 30),
            datetime.date(2025, 9, 17),
            datetime.date(2025, 11, 5),
            datetime.date(2025, 12, 17),
            datetime.date(2026, 1, 28),
            datetime.date(2026, 3, 18),
            datetime.date(2026, 5, 6),
            datetime.date(2026, 6, 17),
            datetime.date(2026, 7, 29),
            datetime.date(2026, 9, 16),
            datetime.date(2026, 11, 4),
            datetime.date(2026, 12, 16),
        ]

    df = pd.DataFrame({"fomc": True}, index=pd.DatetimeIndex(dates))
    cache_set(key, df)
    return sorted(dates)


def days_to_next_fomc(as_of: Optional[datetime.date] = None) -> int:
    """Days until next FOMC meeting from `as_of` date."""
    if as_of is None:
        as_of = datetime.date.today()
    dates = fetch_fomc_dates()
    future = [d for d in dates if d >= as_of]
    if not future:
        return 999
    return (future[0] - as_of).days


def next_opex_date(as_of: Optional[datetime.date] = None) -> datetime.date:
    """Next monthly OPEX date."""
    if as_of is None:
        as_of = datetime.date.today()
    dates = compute_opex_dates()
    future = [d for d in dates if d >= as_of]
    return future[0] if future else datetime.date.today()


def days_to_next_opex(as_of: Optional[datetime.date] = None) -> int:
    """Days until next monthly OPEX."""
    if as_of is None:
        as_of = datetime.date.today()
    return (next_opex_date(as_of) - as_of).days


def is_opex_week(as_of: Optional[datetime.date] = None) -> bool:
    """True if `as_of` falls in the same calendar week as a monthly OPEX."""
    if as_of is None:
        as_of = datetime.date.today()
    opex = next_opex_date(as_of)
    delta = (opex - as_of).days
    return 0 <= delta <= 4
