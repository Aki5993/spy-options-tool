from __future__ import annotations
"""CBOE Put/Call ratio computed from SPY options chain (yfinance, no key required)."""
import os
import sys
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.cache import cache_get, cache_set


def fetch_pcr_current() -> dict:
    """
    Compute current Put/Call ratio from SPY options open interest
    across the nearest 4 expirations.

    Returns dict: {pcr, put_oi, call_oi, label}
      pcr > 1.2  → elevated fear  → contrarian bullish
      pcr < 0.7  → complacency    → contrarian bearish
    """
    key = "pcr_current"
    cached = cache_get(key, "stocktwits")   # reuse 15-min TTL
    if cached is not None and not cached.empty:
        row = cached.iloc[-1]
        return {"pcr": float(row["pcr"]), "label": str(row["label"]),
                "put_oi": 0, "call_oi": 0}

    try:
        ticker = yf.Ticker("SPY")
        expirations = ticker.options[:4]
        total_put_oi = 0
        total_call_oi = 0
        for exp in expirations:
            chain = ticker.option_chain(exp)
            total_put_oi += int(chain.puts["openInterest"].fillna(0).sum())
            total_call_oi += int(chain.calls["openInterest"].fillna(0).sum())

        pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 1.0
        label = "Fearful" if pcr > 1.2 else ("Complacent" if pcr < 0.7 else "Neutral")

        df = pd.DataFrame(
            [{"pcr": pcr, "label": label}],
            index=pd.DatetimeIndex([pd.Timestamp.now()])
        )
        cache_set(key, df)
        return {"pcr": pcr, "put_oi": total_put_oi, "call_oi": total_call_oi, "label": label}

    except Exception as e:
        return {"pcr": 1.0, "put_oi": 0, "call_oi": 0, "label": "Neutral", "error": str(e)}
