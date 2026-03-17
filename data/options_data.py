from __future__ import annotations
"""SPY options chain fetching via yfinance."""
import os
import sys
import datetime

import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import SPY_TICKER


def get_options_chain(expiry: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (calls_df, puts_df) for nearest expiry >= 14 days out.
    expiry: 'YYYY-MM-DD' string or None to auto-select.
    """
    ticker = yf.Ticker(SPY_TICKER)
    expirations = ticker.options

    if expiry is None:
        today = datetime.date.today()
        min_date = today + datetime.timedelta(days=14)
        candidates = [e for e in expirations if datetime.date.fromisoformat(e) >= min_date]
        expiry = candidates[0] if candidates else expirations[0]

    chain = ticker.option_chain(expiry)
    return chain.calls, chain.puts


def get_atm_iv(expiry: str | None = None) -> float | None:
    """
    Return approximate at-the-money implied volatility (%) from calls chain.
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker(SPY_TICKER)
        price_info = ticker.fast_info
        spot = getattr(price_info, "last_price", None)
        if spot is None:
            return None

        calls, _ = get_options_chain(expiry)
        calls = calls.dropna(subset=["impliedVolatility", "strike"])
        calls["dist"] = (calls["strike"] - spot).abs()
        atm = calls.nsmallest(1, "dist")
        if atm.empty:
            return None
        iv = float(atm["impliedVolatility"].iloc[0]) * 100
        return iv
    except Exception:
        return None
