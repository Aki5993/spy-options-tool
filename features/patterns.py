from __future__ import annotations
"""Bull and bear flag chart pattern detection (vectorized)."""
import numpy as np
import pandas as pd


def detect_flag_patterns(df: pd.DataFrame,
                          pole_atr_mult: float = 1.6,
                          flag_max_range_pct: float = 0.04) -> pd.DataFrame:
    """
    Vectorized bull/bear flag detection across multiple (pole, flag) window sizes.

    Bull flag:
      1. Pole   — price rises > pole_atr_mult * ATR over pole_days (fast up-move)
      2. Flag   — consolidation: price range < flag_max_range_pct of price AND
                  < 50% of pole height over flag_days bars
      3. Break  — today's close exceeds the flag's rolling high (continuation)

    Bear flag is the mirror image.

    Adds columns: bull_flag (int), bear_flag (int)
    """
    if "ATR" not in df.columns:
        df = df.copy()
        df["bull_flag"] = 0
        df["bear_flag"] = 0
        return df

    df = df.copy()
    df["bull_flag"] = 0
    df["bear_flag"] = 0

    for pole_days in [5, 7, 10]:
        # Pole: net price change over pole_days
        pole_move = df["Close"].pct_change(pole_days)
        pole_threshold = pole_atr_mult * df["ATR"] / df["Close"]
        bull_pole = pole_move > pole_threshold
        bear_pole = pole_move < -pole_threshold

        for flag_days in [5, 8, 12]:
            # Shift poles forward so the flag follows the pole
            had_bull_pole = bull_pole.shift(flag_days).astype("boolean").fillna(False).astype(bool)
            had_bear_pole = bear_pole.shift(flag_days).astype("boolean").fillna(False).astype(bool)

            # Flag consolidation metrics over the flag window
            flag_high = df["High"].rolling(flag_days).max()
            flag_low  = df["Low"].rolling(flag_days).min()
            flag_range_pct = (flag_high - flag_low) / df["Close"].replace(0, np.nan)
            tight_flag = flag_range_pct < flag_max_range_pct

            # Also require flag range < 50% of the pole's absolute move
            pole_abs_move = (df["Close"].pct_change(pole_days).abs() * df["Close"]).shift(flag_days)
            flag_vs_pole = (flag_high - flag_low) < 0.5 * pole_abs_move.fillna(np.inf)

            # Break: today's close exits the flag in the pole direction
            bull_break = df["Close"] > flag_high.shift(1)
            bear_break = df["Close"] < flag_low.shift(1)

            df["bull_flag"] = (df["bull_flag"].astype(bool) |
                               (had_bull_pole & tight_flag & flag_vs_pole & bull_break)).astype(int)
            df["bear_flag"] = (df["bear_flag"].astype(bool) |
                               (had_bear_pole & tight_flag & flag_vs_pole & bear_break)).astype(int)

    # Never fire both on the same bar
    both = (df["bull_flag"] == 1) & (df["bear_flag"] == 1)
    df.loc[both, ["bull_flag", "bear_flag"]] = 0

    return df
