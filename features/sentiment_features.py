"""Normalize sentiment data into ML-ready features."""
import pandas as pd
import numpy as np


def normalize_fear_greed(value: float) -> float:
    """Normalize 0-100 Fear & Greed to 0-1."""
    return float(np.clip(value, 0, 100)) / 100.0


def get_sentiment_features(fear_greed_val: float,
                            bull_ratio: float,
                            news_score: float) -> dict:
    """
    Build sentiment feature dict suitable for ML prediction.

    Parameters
    ----------
    fear_greed_val : 0-100
    bull_ratio     : StockTwits bull proportion 0-1
    news_score     : Alpha Vantage score typically -1 to +1
    """
    fg_norm = normalize_fear_greed(fear_greed_val)
    # Contrarian flag: extreme fear/greed
    contrarian_bull = float(fear_greed_val < 20)
    contrarian_bear = float(fear_greed_val > 80)
    # News normalized to 0-1
    news_norm = float(np.clip((news_score + 1) / 2, 0, 1))

    return {
        "fg_normalized": fg_norm,
        "bull_ratio": float(bull_ratio),
        "news_norm": news_norm,
        "contrarian_bull": contrarian_bull,
        "contrarian_bear": contrarian_bear,
        "sentiment_composite": float((fg_norm + bull_ratio + news_norm) / 3),
    }


def merge_sentiment_into_df(df: pd.DataFrame,
                             fg_history: pd.DataFrame) -> pd.DataFrame:
    """
    Merge daily Fear & Greed history into main OHLCV+indicators DataFrame.
    fg_history: index=DatetimeIndex, columns=[value, classification]
    """
    if fg_history.empty:
        df["fg_normalized"] = 0.5
        df["contrarian_bull"] = 0
        df["contrarian_bear"] = 0
        return df

    fg = fg_history[["value"]].copy()
    fg["fg_normalized"] = fg["value"] / 100.0
    fg["contrarian_bull"] = (fg["value"] < 20).astype(int)
    fg["contrarian_bear"] = (fg["value"] > 80).astype(int)
    fg = fg.drop(columns=["value"])

    # Align on date only (strip time from index if present)
    fg.index = pd.to_datetime(fg.index).normalize()
    df.index = pd.to_datetime(df.index).normalize()
    df = df.join(fg, how="left")
    df[["fg_normalized", "contrarian_bull", "contrarian_bear"]] = (
        df[["fg_normalized", "contrarian_bull", "contrarian_bear"]].ffill().fillna(0.5)
    )
    return df
