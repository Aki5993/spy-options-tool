from __future__ import annotations
"""Event-driven features: FOMC proximity, OPEX week, VIX regime."""
import datetime
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import VIX_LOW, VIX_HIGH
from data.events import (
    fetch_fomc_dates, compute_opex_dates, compute_vix_expiry_dates,
    days_to_next_fomc, is_opex_week, days_to_next_opex
)


def vix_regime(vix_level: float) -> int:
    """0=low, 1=mid, 2=high"""
    if vix_level < VIX_LOW:
        return 0
    elif vix_level < VIX_HIGH:
        return 1
    return 2


def get_event_features_today(vix_level: float | None = None) -> dict:
    """Return event feature dict for today."""
    today = datetime.date.today()
    dtf = days_to_next_fomc(today)
    dtop = days_to_next_opex(today)
    opex_wk = is_opex_week(today)
    regime = vix_regime(vix_level) if vix_level is not None else 1

    return {
        "days_to_fomc": dtf,
        "fomc_proximity": int(dtf <= 2),
        "is_opex_week": int(opex_wk),
        "days_to_opex": dtop,
        "vix_regime": regime,
    }


def build_event_feature_df(spy_df: pd.DataFrame, vix_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build event features aligned to spy_df index.
    spy_df: DatetimeIndex with at least Close column.
    vix_df: DatetimeIndex with VIX column.
    """
    fomc_dates = set(fetch_fomc_dates())
    opex_dates = set(compute_opex_dates())
    dates = pd.to_datetime(spy_df.index).normalize()

    rows = []
    for dt in dates:
        d = dt.date()
        # Days to next FOMC
        future_fomc = sorted([f for f in fomc_dates if f >= d])
        dtf = (future_fomc[0] - d).days if future_fomc else 999

        # Days to next OPEX
        future_opex = sorted([o for o in opex_dates if o >= d])
        dtop = (future_opex[0] - d).days if future_opex else 999
        opex_wk = int(dtop <= 4)

        rows.append({
            "days_to_fomc": dtf,
            "fomc_proximity": int(dtf <= 2),
            "is_opex_week": opex_wk,
            "days_to_opex": dtop,
        })

    event_df = pd.DataFrame(rows, index=spy_df.index)

    # VIX regime
    vix_aligned = vix_df.reindex(spy_df.index, method="ffill")
    vix_vals = vix_aligned["VIX"] if "VIX" in vix_aligned.columns else pd.Series(20.0, index=spy_df.index)
    event_df["vix_level"] = vix_vals.values
    event_df["vix_regime"] = event_df["vix_level"].apply(vix_regime)

    return event_df
