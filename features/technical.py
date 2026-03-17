"""Technical indicators computed on OHLCV DataFrame."""
from __future__ import annotations
import pandas as pd
import numpy as np

try:
    import ta
    HAS_TA = True
except ImportError:
    HAS_TA = False


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
    return df


def add_bollinger_bands(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.DataFrame:
    sma = df["Close"].rolling(period).mean()
    rolling_std = df["Close"].rolling(period).std()
    df["BB_upper"] = sma + std * rolling_std
    df["BB_mid"] = sma
    df["BB_lower"] = sma - std * rolling_std
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["BB_mid"]
    df["BB_pct"] = (df["Close"] - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"])
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(com=period - 1, min_periods=period).mean()
    return df


def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    direction = np.sign(df["Close"].diff())
    direction.iloc[0] = 0
    df["OBV"] = (direction * df["Volume"]).cumsum()
    return df


def add_sma(df: pd.DataFrame, periods: list[int] = [20, 50, 200]) -> pd.DataFrame:
    for p in periods:
        df[f"SMA_{p}"] = df["Close"].rolling(p).mean()
    return df


def _find_pivot_highs(series: pd.Series, window: int = 5) -> pd.Series:
    """Return series of pivot high values (NaN elsewhere)."""
    roll_max = series.rolling(2 * window + 1, center=True, min_periods=window + 1).max()
    mask = series == roll_max
    return series.where(mask)


def _find_pivot_lows(series: pd.Series, window: int = 5) -> pd.Series:
    """Return series of pivot low values (NaN elsewhere)."""
    roll_min = series.rolling(2 * window + 1, center=True, min_periods=window + 1).min()
    mask = series == roll_min
    return series.where(mask)


def _project_line(x1: float, y1: float, x2: float, y2: float, x_now: float) -> float:
    """Project a line defined by two points to x_now."""
    if x2 == x1:
        return y2
    slope = (y2 - y1) / (x2 - x1)
    return y1 + slope * (x_now - x1)


def detect_trendline_breaks(df: pd.DataFrame, pivot_window: int = 5) -> pd.DataFrame:
    """
    Detect trendline breaks on any OHLCV DataFrame (hourly or daily).

    Algorithm:
      - Finds pivot highs (resistance) and pivot lows (support) via rolling window.
      - Projects a line through the last two pivots to the current bar.
      - Flags a break when price crosses the projected trendline.

    Adds columns:
      resistance      : projected resistance trendline value
      support         : projected support trendline value
      tl_break_up     : 1 when close crosses above resistance
      tl_break_down   : 1 when close crosses below support
    """
    df = df.copy()
    ts = df.index.astype(np.int64).values  # nanoseconds as numeric x-axis

    ph = _find_pivot_highs(df["High"], pivot_window).dropna()
    pl = _find_pivot_lows(df["Low"], pivot_window).dropna()

    resistances = np.full(len(df), np.nan)
    supports = np.full(len(df), np.nan)
    tl_break_up = np.zeros(len(df), dtype=int)
    tl_break_down = np.zeros(len(df), dtype=int)

    for i in range(1, len(df)):
        idx = df.index[i]
        x_now = ts[i]

        # Resistance: project line through last 2 pivot highs before idx
        prev_ph = ph[ph.index < idx].tail(2)
        if len(prev_ph) >= 2:
            x1 = ts[df.index.get_loc(prev_ph.index[0])]
            x2 = ts[df.index.get_loc(prev_ph.index[1])]
            res = _project_line(x1, prev_ph.iloc[0], x2, prev_ph.iloc[1], x_now)
        elif len(prev_ph) == 1:
            res = float(prev_ph.iloc[0])
        else:
            res = np.nan
        resistances[i] = res

        # Support: project line through last 2 pivot lows before idx
        prev_pl = pl[pl.index < idx].tail(2)
        if len(prev_pl) >= 2:
            x1 = ts[df.index.get_loc(prev_pl.index[0])]
            x2 = ts[df.index.get_loc(prev_pl.index[1])]
            sup = _project_line(x1, prev_pl.iloc[0], x2, prev_pl.iloc[1], x_now)
        elif len(prev_pl) == 1:
            sup = float(prev_pl.iloc[0])
        else:
            sup = np.nan
        supports[i] = sup

        curr_close = df["Close"].iloc[i]
        prev_close = df["Close"].iloc[i - 1]
        prev_res = resistances[i - 1]
        prev_sup = supports[i - 1]

        # Break up: was below or at resistance, now above
        if not np.isnan(res) and not np.isnan(prev_res):
            if prev_close <= prev_res and curr_close > res:
                tl_break_up[i] = 1

        # Break down: was above or at support, now below
        if not np.isnan(sup) and not np.isnan(prev_sup):
            if prev_close >= prev_sup and curr_close < sup:
                tl_break_down[i] = 1

    df["resistance"] = resistances
    df["support"] = supports
    df["tl_break_up"] = tl_break_up
    df["tl_break_down"] = tl_break_down
    return df


def get_hourly_trendline_signal(hourly_df: pd.DataFrame, pivot_window: int = 5) -> dict:
    """
    Run trendline break detection on hourly data and return a signal dict
    for the most recent session.

    Returns:
      tl_break_up   : bool — bullish trendline break in last 24 bars
      tl_break_down : bool — bearish trendline break in last 24 bars
      resistance    : float — current projected resistance
      support       : float — current projected support
      hourly_df     : DataFrame with trendline columns added (for charting)
    """
    if hourly_df.empty:
        return {"tl_break_up": False, "tl_break_down": False,
                "resistance": None, "support": None, "hourly_df": hourly_df}

    df_tl = detect_trendline_breaks(hourly_df, pivot_window=pivot_window)
    recent = df_tl.tail(24)  # last ~1 trading day of hourly bars
    break_up = bool(recent["tl_break_up"].any())
    break_down = bool(recent["tl_break_down"].any())
    latest = df_tl.iloc[-1]

    return {
        "tl_break_up": break_up,
        "tl_break_down": break_down,
        "resistance": float(latest["resistance"]) if not np.isnan(latest["resistance"]) else None,
        "support": float(latest["support"]) if not np.isnan(latest["support"]) else None,
        "hourly_df": df_tl,
    }


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all technical indicators to a copy of df."""
    df = df.copy()
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_atr(df)
    df = add_obv(df)
    df = add_sma(df)
    # MACD crossover signals
    df["MACD_cross_up"] = ((df["MACD"] > df["MACD_signal"]) &
                            (df["MACD"].shift(1) <= df["MACD_signal"].shift(1))).astype(int)
    df["MACD_cross_down"] = ((df["MACD"] < df["MACD_signal"]) &
                              (df["MACD"].shift(1) >= df["MACD_signal"].shift(1))).astype(int)
    # Price vs SMA flags
    df["above_sma20"] = (df["Close"] > df["SMA_20"]).astype(int)
    df["above_sma50"] = (df["Close"] > df["SMA_50"]).astype(int)
    df["above_sma200"] = (df["Close"] > df["SMA_200"]).astype(int)
    # Volatility squeeze: BB width below 20th percentile
    df["bb_squeeze"] = (df["BB_width"] < df["BB_width"].rolling(252).quantile(0.20)).astype(int)
    # Forward return (target for ML)
    df["fwd_5d_return"] = df["Close"].shift(-5) / df["Close"] - 1
    df["fwd_5d_sign"] = (df["fwd_5d_return"] > 0).astype(int)
    # Bull / Bear flag patterns
    from features.patterns import detect_flag_patterns
    df = detect_flag_patterns(df)
    return df
