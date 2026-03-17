from __future__ import annotations
"""Macro indicators: DXY, Oil, 10-yr yield, consumer sentiment, ISM PMI, economic calendar."""
import os
import sys
import datetime
import calendar as cal_mod
from typing import Optional

import pandas as pd
import numpy as np
import yfinance as yf
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import FRED_API_KEY, ALPHA_VANTAGE_KEY
from data.cache import cache_get, cache_set

# ── Tickers ────────────────────────────────────────────────────────────────────
MACRO_TICKERS = {
    "DXY":      "DX=F",        # US Dollar Index futures (DX-Y.NYB no longer works)
    "Oil":      "CL=F",        # WTI Crude Oil futures
    "Yield10Y": "^TNX",        # 10-year Treasury yield
}


def fetch_macro_history(years: int = 5) -> pd.DataFrame:
    """
    Download daily DXY, Oil, and 10-yr yield via yfinance, return as merged DataFrame.
    Columns: DXY, Oil, Yield10Y + 5-day ROC for each + yield level flags.
    """
    key = f"macro_history_{years}yr"
    cached = cache_get(key, "market_daily")
    if cached is not None and not cached.empty:
        return cached

    end   = datetime.date.today()
    start = end - datetime.timedelta(days=years * 365 + 60)
    frames = {}
    for name, ticker in MACRO_TICKERS.items():
        try:
            raw = yf.download(ticker, start=start.isoformat(), end=end.isoformat(),
                              auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw.index = pd.to_datetime(raw.index)
            frames[name] = raw["Close"].rename(name)
        except Exception:
            pass

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames.values(), axis=1).sort_index()

    # 5-day rate of change
    for col in list(MACRO_TICKERS.keys()):
        if col in df.columns:
            df[f"{col}_roc5"]  = df[col].pct_change(5)
            df[f"{col}_roc20"] = df[col].pct_change(20)

    df = df.ffill().dropna(how="all")
    cache_set(key, df)
    return df


def fetch_consumer_sentiment() -> pd.DataFrame:
    """
    University of Michigan Consumer Sentiment (monthly) via FRED.
    Falls back to empty DataFrame if FRED key unavailable.
    """
    key = "umcsent"
    cached = cache_get(key, "fomc_dates")   # weekly TTL is fine for monthly data
    if cached is not None and not cached.empty:
        return cached

    if FRED_API_KEY:
        try:
            from fredapi import Fred
            fred = Fred(api_key=FRED_API_KEY)
            series = fred.get_series("UMCSENT")
            df = series.rename("consumer_sentiment").to_frame()
            df.index = pd.to_datetime(df.index)
            cache_set(key, df)
            return df
        except Exception:
            pass

    return pd.DataFrame(columns=["consumer_sentiment"])


def fetch_ism_pmi() -> dict:
    """
    Fetch latest ISM Manufacturing and Services PMI.
    Tries FRED series NAPMPI (Mfg) and NMFCI (Services); falls back to Alpha Vantage DURABLES.
    Returns dict: {mfg_pmi, services_pmi, mfg_label, services_label}
    """
    result = {"mfg_pmi": None, "services_pmi": None,
              "mfg_label": "N/A", "services_label": "N/A"}

    if FRED_API_KEY:
        try:
            from fredapi import Fred
            fred = Fred(api_key=FRED_API_KEY)
            for series_id, key in [("NAPMPI", "mfg_pmi"), ("NMFCI", "services_pmi")]:
                try:
                    s = fred.get_series(series_id)
                    if len(s) > 0:
                        val = float(s.dropna().iloc[-1])
                        result[key] = val
                except Exception:
                    pass
        except Exception:
            pass

    # Label
    for k, label_k in [("mfg_pmi", "mfg_label"), ("services_pmi", "services_label")]:
        v = result[k]
        if v is not None:
            result[label_k] = "Expanding" if v > 50 else "Contracting"

    return result


# ── Economic Calendar ──────────────────────────────────────────────────────────

def _first_weekday_of_month(year: int, month: int, weekday: int) -> datetime.date:
    """Return the first occurrence of weekday (0=Mon…6=Sun) in given month."""
    d = datetime.date(year, month, 1)
    delta = (weekday - d.weekday()) % 7
    return d + datetime.timedelta(days=delta)


def _nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> datetime.date:
    """Return the Nth occurrence (1-indexed) of weekday in given month."""
    first = _first_weekday_of_month(year, month, weekday)
    return first + datetime.timedelta(weeks=n - 1)


def _first_business_day(year: int, month: int) -> datetime.date:
    d = datetime.date(year, month, 1)
    while d.weekday() >= 5:
        d += datetime.timedelta(days=1)
    return d


def _nth_business_day(year: int, month: int, n: int) -> datetime.date:
    d = datetime.date(year, month, 1)
    count = 0
    while True:
        if d.weekday() < 5:
            count += 1
            if count == n:
                return d
        d += datetime.timedelta(days=1)


def get_economic_calendar(days_ahead: int = 45) -> list[dict]:
    """
    Generate upcoming major economic events.

    Events included:
      NFP            — first Friday of month
      ISM Mfg PMI    — first business day of month
      ISM Svc PMI    — 3rd business day of month
      CPI            — ~2nd Tuesday of month (approximate)
      PPI            — ~2nd Wednesday of month (approximate)
      Retail Sales   — ~2nd Wednesday of month (approximate, offset)
      UMich Prelim   — 2nd Friday of month
      UMich Final    — last Friday of month
      GDP (advance)  — end of Jan/Apr/Jul/Oct (quarterly)
      FOMC           — from events module
      OPEX           — 3rd Friday of month
    """
    from data.events import fetch_fomc_dates, compute_opex_dates
    today = datetime.date.today()
    end   = today + datetime.timedelta(days=days_ahead)

    events: list[dict] = []

    # Iterate over relevant months
    months_to_check = set()
    d = datetime.date(today.year, today.month, 1)
    while d <= end + datetime.timedelta(days=31):
        months_to_check.add((d.year, d.month))
        if d.month == 12:
            d = datetime.date(d.year + 1, 1, 1)
        else:
            d = datetime.date(d.year, d.month + 1, 1)

    for year, month in sorted(months_to_check):
        def _add(date, name, importance, note=""):
            if today <= date <= end:
                events.append({
                    "date": date, "event": name,
                    "importance": importance, "note": note,
                    "days_away": (date - today).days,
                })

        # NFP: first Friday
        _add(_nth_weekday_of_month(year, month, 4, 1), "Non-Farm Payrolls", "HIGH",
             "Labor market — major market mover")

        # ISM Mfg PMI: first business day
        _add(_first_business_day(year, month), "ISM Manufacturing PMI", "HIGH",
             ">50 = expansion")

        # ISM Services PMI: 3rd business day
        _add(_nth_business_day(year, month, 3), "ISM Services PMI", "MEDIUM",
             "Services sector health")

        # CPI: approximate 2nd Tuesday
        _add(_nth_weekday_of_month(year, month, 1, 2), "CPI Inflation (est.)", "HIGH",
             "Drives Fed rate expectations")

        # PPI: approximate 2nd Wednesday
        _add(_nth_weekday_of_month(year, month, 2, 2), "PPI (est.)", "MEDIUM", "")

        # Retail Sales: approximate 2nd Wednesday + 2 days (Fri usually)
        rs_date = _nth_weekday_of_month(year, month, 4, 2)
        _add(rs_date, "Retail Sales (est.)", "MEDIUM", "Consumer spending proxy")

        # UMich Consumer Sentiment: 2nd Friday (preliminary)
        _add(_nth_weekday_of_month(year, month, 4, 2), "UMich Sentiment (Prelim)", "MEDIUM",
             "Consumer confidence")

        # UMich Final: last Friday of month
        last_fri = _nth_weekday_of_month(year, month, 4, 4)
        if last_fri.month != month:
            last_fri = _nth_weekday_of_month(year, month, 4, 3)
        _add(last_fri, "UMich Sentiment (Final)", "LOW", "")

        # GDP advance: end of Jan/Apr/Jul/Oct
        if month in (1, 4, 7, 10):
            gdp_date = datetime.date(year, month, 30)
            if gdp_date.weekday() >= 5:
                gdp_date -= datetime.timedelta(days=gdp_date.weekday() - 4)
            _add(gdp_date, "GDP Advance Estimate", "HIGH", "Quarterly economic output")

    # FOMC meetings
    for d in fetch_fomc_dates():
        if today <= d <= end:
            events.append({
                "date": d, "event": "FOMC Meeting",
                "importance": "HIGH", "note": "Fed rate decision",
                "days_away": (d - today).days,
            })

    # Monthly OPEX
    for d in compute_opex_dates():
        if today <= d <= end:
            events.append({
                "date": d, "event": "Monthly OPEX",
                "importance": "MEDIUM", "note": "Options expiration",
                "days_away": (d - today).days,
            })

    # Sort and deduplicate by (date, event)
    seen = set()
    unique = []
    for ev in sorted(events, key=lambda x: x["date"]):
        k = (ev["date"], ev["event"])
        if k not in seen:
            seen.add(k)
            unique.append(ev)
    return unique


def get_current_macro_snapshot() -> dict:
    """Return latest values for top-bar display."""
    snap = {"DXY": None, "Oil": None, "Yield10Y": None,
            "DXY_roc5": None, "Oil_roc5": None, "Yield10Y_roc5": None}
    try:
        df = fetch_macro_history(1)
        if not df.empty:
            last = df.iloc[-1]
            for k in snap:
                if k in df.columns:
                    snap[k] = float(last[k]) if not np.isnan(last[k]) else None
    except Exception:
        pass
    return snap
