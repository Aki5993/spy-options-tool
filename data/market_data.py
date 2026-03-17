from __future__ import annotations
"""SPY + VIX historical and real-time market data."""
import datetime
import os
import sys

import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import SPY_TICKER, VIX_TICKER, HISTORY_YEARS, POLYGON_API_KEY
from data.cache import cache_get, cache_set

import requests


def fetch_spy_history(years: int = HISTORY_YEARS) -> pd.DataFrame:
    """Fetch daily OHLCV for SPY going back `years` years."""
    key = f"spy_daily_{years}yr"
    cached = cache_get(key, "market_daily")
    if cached is not None:
        return cached

    end = datetime.date.today()
    start = end - datetime.timedelta(days=years * 365 + 30)
    df = yf.download(SPY_TICKER, start=start.isoformat(), end=end.isoformat(),
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    cache_set(key, df)
    return df


def fetch_vix_history(years: int = HISTORY_YEARS) -> pd.DataFrame:
    """Fetch daily VIX close history."""
    key = f"vix_daily_{years}yr"
    cached = cache_get(key, "market_daily")
    if cached is not None:
        return cached

    end = datetime.date.today()
    start = end - datetime.timedelta(days=years * 365 + 30)
    df = yf.download(VIX_TICKER, start=start.isoformat(), end=end.isoformat(),
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    df = df[["Close"]].rename(columns={"Close": "VIX"}).dropna()
    cache_set(key, df)
    return df


def fetch_realtime_spy() -> dict:
    """
    Fetch latest SPY quote.
    Uses Polygon if key is set, otherwise falls back to yfinance fast_info.
    Returns dict with keys: price, change_pct, volume.
    """
    if POLYGON_API_KEY:
        try:
            url = f"https://api.polygon.io/v2/last/trade/{SPY_TICKER}?apiKey={POLYGON_API_KEY}"
            r = requests.get(url, timeout=5)
            data = r.json()
            price = data["results"]["p"]
            return {"price": price, "change_pct": None, "volume": None}
        except Exception:
            pass

    ticker = yf.Ticker(SPY_TICKER)
    info = ticker.fast_info
    price = getattr(info, "last_price", None)
    prev_close = getattr(info, "previous_close", None)
    change_pct = ((price - prev_close) / prev_close * 100) if price and prev_close else None
    volume = getattr(info, "last_volume", None)
    return {"price": price, "change_pct": change_pct, "volume": volume}


def fetch_current_vix() -> float | None:
    """Return latest VIX level."""
    try:
        ticker = yf.Ticker(VIX_TICKER)
        info = ticker.fast_info
        return getattr(info, "last_price", None)
    except Exception:
        return None


def fetch_spy_hourly(days: int = 60) -> pd.DataFrame:
    """
    Fetch hourly SPY OHLCV data for the last `days` days.
    yfinance provides up to 60 days of 1h data for free.
    Returns DataFrame with DatetimeIndex (UTC-aware).
    """
    key = f"spy_hourly_{days}d"
    cached = cache_get(key, "market_intraday")
    if cached is not None and not cached.empty:
        return cached

    df = yf.download(
        SPY_TICKER,
        period=f"{min(days, 59)}d",
        interval="1h",
        auto_adjust=True,
        progress=False,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    cache_set(key, df)
    return df
