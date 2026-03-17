from __future__ import annotations
"""
Market breadth indicators: Zweig Breadth Thrust and Hindenburg Omen.

Uses 14 diversified ETFs as a proxy for NYSE breadth since individual
issue-level data is not freely available. Results approximate the spirit
of these indicators; treat as supporting evidence, not exact replications.
"""
import os
import sys

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.cache import cache_get, cache_set

# 11 SPDR Sector ETFs + Russell 2000 + Nasdaq 100 + Dow Jones = 14 instruments
BREADTH_ETFS = [
    "XLK", "XLV", "XLF", "XLE", "XLY", "XLP",
    "XLI", "XLB", "XLU", "XLRE", "XLC",
    "IWM", "QQQ", "DIA",
]

# Zweig Breadth Thrust thresholds
ZBT_LOW  = 0.40   # 10-day EMA must dip below this
ZBT_HIGH = 0.615  # then rise above this within 10 trading days → THRUST


def fetch_sector_breadth(years: int = 5) -> pd.DataFrame:
    """
    Download sector ETF daily closes and compute breadth metrics.

    Returns daily DataFrame with:
      breadth_ratio      — fraction of ETFs closing up vs prior day
      breadth_ema10      — 10-day EMA of breadth_ratio
      pct_above_sma20    — fraction of ETFs above their 20-day SMA
      pct_above_sma50    — fraction of ETFs above their 50-day SMA
      n_new_highs_52wk   — ETFs within 2% of 52-week high
      n_new_lows_52wk    — ETFs within 2% of 52-week low
    """
    key = f"sector_breadth_{years}yr"
    cached = cache_get(key, "market_daily")
    if cached is not None and not cached.empty:
        return cached

    import datetime
    end   = datetime.date.today()
    start = end - datetime.timedelta(days=years * 365 + 60)

    try:
        raw = yf.download(BREADTH_ETFS, start=start.isoformat(), end=end.isoformat(),
                          auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            closes = raw["Close"]
        else:
            closes = raw[["Close"]]
    except Exception:
        return pd.DataFrame()

    closes = closes.sort_index().ffill()

    # Daily up/down ratio
    daily_ret = closes.pct_change()
    breadth_ratio = (daily_ret > 0).sum(axis=1) / daily_ret.notna().sum(axis=1)
    breadth_ema10 = breadth_ratio.ewm(span=10, adjust=False).mean()

    # % above SMA20 and SMA50
    sma20 = closes.rolling(20).mean()
    sma50 = closes.rolling(50).mean()
    pct_above_sma20 = (closes > sma20).sum(axis=1) / closes.notna().sum(axis=1)
    pct_above_sma50 = (closes > sma50).sum(axis=1) / closes.notna().sum(axis=1)

    # 52-week highs / lows (within 2%)
    high_52wk = closes.rolling(252, min_periods=100).max()
    low_52wk  = closes.rolling(252, min_periods=100).min()
    n_new_highs = ((closes / high_52wk) >= 0.98).sum(axis=1)
    n_new_lows  = ((closes / low_52wk)  <= 1.02).sum(axis=1)

    df = pd.DataFrame({
        "breadth_ratio":   breadth_ratio,
        "breadth_ema10":   breadth_ema10,
        "pct_above_sma20": pct_above_sma20,
        "pct_above_sma50": pct_above_sma50,
        "n_new_highs_52wk": n_new_highs.astype(float),
        "n_new_lows_52wk":  n_new_lows.astype(float),
    }).dropna(how="all")

    cache_set(key, df)
    return df


def compute_zweig_breadth_thrust(breadth_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add `zweig_thrust` column to breadth_df.

    Signal fires on day T when:
      - breadth_ema10 was below ZBT_LOW within the last 10 trading days
      - breadth_ema10 is now above ZBT_HIGH
    This captures a rapid expansion from fearful to broad participation.
    """
    df = breadth_df.copy()
    ema = df["breadth_ema10"].values
    n   = len(ema)
    thrust = np.zeros(n, dtype=int)

    for i in range(10, n):
        window = ema[max(0, i - 10): i + 1]
        if ema[i] > ZBT_HIGH and window.min() < ZBT_LOW:
            thrust[i] = 1

    df["zweig_thrust"] = thrust
    return df


def compute_hindenburg_omen(breadth_df: pd.DataFrame,
                             spy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add `hindenburg_omen` column to breadth_df.

    Conditions (sector-ETF approximation):
      1. n_new_highs_52wk >= 2  (simultaneous new highs in multiple sectors)
      2. n_new_lows_52wk  >= 2  (simultaneous new lows  in multiple sectors)
      3. SPY is above its 50-day SMA                (market not in free-fall)
      4. breadth_ema10 < 0.50                       (McClellan proxy: breadth deteriorating)

    Note: original uses 2.2% of ~3,000 NYSE issues. With 14 ETFs, "≥ 2" is
    ~14% — a looser threshold. Treat signals as elevated-risk flags, not absolutes.
    """
    df = breadth_df.copy()

    # Align SPY SMA50
    spy_close = spy_df["Close"].reindex(df.index, method="ffill")
    spy_sma50 = spy_close.rolling(50).mean()
    spy_above_sma50 = spy_close > spy_sma50

    c1 = df["n_new_highs_52wk"] >= 2
    c2 = df["n_new_lows_52wk"]  >= 2
    c3 = spy_above_sma50
    c4 = df["breadth_ema10"] < 0.50

    df["hindenburg_omen"] = (c1 & c2 & c3 & c4).astype(int)
    return df


def build_full_breadth_df(spy_df: pd.DataFrame, years: int = 5) -> pd.DataFrame:
    """Convenience: fetch + compute Zweig + Hindenburg in one call."""
    raw = fetch_sector_breadth(years)
    if raw.empty:
        return pd.DataFrame()
    df = compute_zweig_breadth_thrust(raw)
    df = compute_hindenburg_omen(df, spy_df)
    return df


def get_breadth_snapshot(breadth_df: pd.DataFrame) -> dict:
    """Return latest breadth indicator values for dashboard display."""
    if breadth_df.empty:
        return {}
    last = breadth_df.iloc[-1]

    # Last Zweig thrust date
    thrust_dates = breadth_df.index[breadth_df.get("zweig_thrust", pd.Series(0)) == 1]
    last_thrust  = thrust_dates[-1].date() if len(thrust_dates) > 0 else None

    # Last Hindenburg date
    ho_dates   = breadth_df.index[breadth_df.get("hindenburg_omen", pd.Series(0)) == 1]
    last_ho    = ho_dates[-1].date() if len(ho_dates) > 0 else None

    import datetime
    today = datetime.date.today()

    return {
        "breadth_ratio":    float(last.get("breadth_ratio",   0.5)),
        "breadth_ema10":    float(last.get("breadth_ema10",   0.5)),
        "pct_above_sma50":  float(last.get("pct_above_sma50", 0.5)),
        "n_new_highs":      int(last.get("n_new_highs_52wk",  0)),
        "n_new_lows":       int(last.get("n_new_lows_52wk",   0)),
        "zweig_active":     bool(last.get("zweig_thrust", 0)),
        "hindenburg_active":bool(last.get("hindenburg_omen", 0)),
        "last_zweig_date":  last_thrust,
        "last_ho_date":     last_ho,
        "days_since_zweig": (today - last_thrust).days if last_thrust else None,
        "days_since_ho":    (today - last_ho).days     if last_ho     else None,
    }
