"""Hybrid signal engine: rule-based layer + XGBoost ML layer."""
from __future__ import annotations
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    RULE_WEIGHT, ML_WEIGHT,
    SIGNAL_BULL_THRESHOLD, SIGNAL_BEAR_THRESHOLD,
    FEAR_GREED_EXTREME_FEAR, FEAR_GREED_EXTREME_GREED,
    FOMC_PROXIMITY_DAYS, VIX_HIGH,
)
from models.trainer import load_model, get_feature_cols

# PCR thresholds for contrarian signal
PCR_FEAR_THRESHOLD = 1.2     # PCR above this → fear → contrarian bullish
PCR_COMPLACENCY_THRESHOLD = 0.7  # PCR below this → greed → contrarian bearish


def _rule_score(row: pd.Series) -> float:
    """
    Compute rule-based score in [0, 1].
    0.5 = neutral, >0.5 bullish, <0.5 bearish.

    Rules (each appends a score to the list; final = mean):
      - RSI extremes + MACD confirmation (high conviction)
      - RSI mild signals
      - MACD crossovers (standalone)
      - Fear & Greed contrarian extremes
      - PCR contrarian extremes
      - Hourly trendline breaks (high conviction)
      - BB squeeze + VIX spike (risk flag → neutral)
    """
    signals = []

    rsi = row.get("RSI", 50)
    macd_cross_up = row.get("MACD_cross_up", 0)
    macd_cross_down = row.get("MACD_cross_down", 0)
    bb_squeeze = row.get("bb_squeeze", 0)
    vix = row.get("vix_level", 20)
    fg = row.get("fg_normalized", 0.5) * 100
    pcr = row.get("pcr", 1.0)
    above_sma50 = int(row.get("above_sma50", 1))
    tl_break_up = row.get("tl_break_up", 0)
    tl_break_down = row.get("tl_break_down", 0)

    # ── RSI ───────────────────────────────────────────────────────────────────
    if rsi < 30 and macd_cross_up:
        signals.append(0.84)   # oversold + MACD confirmation → high conviction bull
    elif rsi > 70 and macd_cross_down:
        signals.append(0.16)   # overbought + MACD confirmation → high conviction bear
    elif rsi < 35:
        signals.append(0.64)
    elif rsi > 65:
        signals.append(0.36)

    # ── MACD crossovers ───────────────────────────────────────────────────────
    if macd_cross_up:
        signals.append(0.68)
    if macd_cross_down:
        signals.append(0.32)

    # ── Fear & Greed contrarian ───────────────────────────────────────────────
    if fg < FEAR_GREED_EXTREME_FEAR:
        signals.append(0.74)
    elif fg > FEAR_GREED_EXTREME_GREED:
        signals.append(0.26)

    # ── Put/Call Ratio contrarian ─────────────────────────────────────────────
    if pcr > PCR_FEAR_THRESHOLD:
        signals.append(0.70)
    elif pcr < PCR_COMPLACENCY_THRESHOLD:
        signals.append(0.30)

    # ── Hourly trendline breaks (high conviction — add twice for emphasis) ────
    if tl_break_up:
        signals.extend([0.76, 0.76])
    if tl_break_down:
        signals.extend([0.24, 0.24])

    # ── BB squeeze + VIX spike → uncertainty → pull toward neutral ────────────
    if bb_squeeze and vix > VIX_HIGH:
        signals.append(0.5)

    if not signals:
        return 0.5

    base = float(np.mean(signals))

    # ── Trend filter (applied AFTER averaging, not as a vote) ─────────────────
    # Dampen counter-trend signals by 35%; trend-aligned signals pass through.
    if above_sma50 == 0 and base > 0.5:
        base = 0.5 + (base - 0.5) * 0.65   # bullish signal below SMA50 → reduce
    elif above_sma50 == 1 and base < 0.5:
        base = 0.5 - (0.5 - base) * 0.65   # bearish signal above SMA50 → reduce

    return base


def compute_signal(latest_row: pd.Series,
                   ml_model=None) -> dict:
    """
    Compute combined signal for a single row.

    Returns dict:
      score       : combined 0-1 score
      direction   : 'bullish' | 'bearish' | 'neutral'
      rule_score  : 0-1
      ml_prob     : 0-1 (or None if no model)
      confidence  : int 0-99
      reduce_conf : bool (FOMC proximity flag)
      trend_aligned : bool (signal agrees with SMA50 trend)
    """
    rule = _rule_score(latest_row)
    ml_prob = None

    if ml_model is not None:
        feat_cols = get_feature_cols()
        available = [c for c in feat_cols if c in latest_row.index]
        x = latest_row[available].values.reshape(1, -1).astype(float)
        try:
            ml_prob = float(ml_model.predict_proba(x)[0][1])
        except Exception:
            ml_prob = None

    combined = RULE_WEIGHT * rule + ML_WEIGHT * ml_prob if ml_prob is not None else rule

    fomc_proximity = bool(latest_row.get("fomc_proximity", 0))
    if fomc_proximity:
        combined = 0.5 + (combined - 0.5) * 0.6

    if combined >= SIGNAL_BULL_THRESHOLD:
        direction = "bullish"
    elif combined <= SIGNAL_BEAR_THRESHOLD:
        direction = "bearish"
    else:
        direction = "neutral"

    raw_conf = abs(combined - 0.5) * 2 * 100
    confidence = min(99, max(1, round(raw_conf)))

    above_sma50 = int(latest_row.get("above_sma50", 1))
    trend_aligned = (direction == "bullish" and above_sma50 == 1) or \
                    (direction == "bearish" and above_sma50 == 0) or \
                    (direction == "neutral")

    return {
        "score": combined,
        "direction": direction,
        "rule_score": rule,
        "ml_prob": ml_prob,
        "confidence": confidence,
        "reduce_conf": fomc_proximity,
        "trend_aligned": trend_aligned,
    }


def generate_signals_series(df: pd.DataFrame,
                             ml_model=None,
                             min_confidence: int = 0) -> pd.DataFrame:
    """
    Generate signal for each row in df.
    Rows below min_confidence are forced to 'neutral'.
    Returns df with added columns: signal_score, signal_direction.
    """
    scores = []
    directions = []
    for _, row in df.iterrows():
        sig = compute_signal(row, ml_model)
        if sig["confidence"] < min_confidence:
            directions.append("neutral")
        else:
            directions.append(sig["direction"])
        scores.append(sig["score"])
    df = df.copy()
    df["signal_score"] = scores
    df["signal_direction"] = directions
    return df
