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

# PCR thresholds
PCR_FEAR_THRESHOLD        = 1.2
PCR_COMPLACENCY_THRESHOLD = 0.7

# Macro thresholds
DXY_STRONG_THRESHOLD   = 0.010   # +1%  5-day ROC → dollar strengthening
DXY_WEAK_THRESHOLD     = -0.010  # -1%  5-day ROC → dollar weakening
OIL_SPIKE_THRESHOLD    = 0.050   # +5%  5-day ROC → oil spike
OIL_CRASH_THRESHOLD    = -0.050  # -5%  5-day ROC → oil crash
YIELD_HIGH_LEVEL       = 4.5     # 10yr yield above this = headwind
YIELD_RISING_THRESHOLD = 0.05    # +0.05 in 5 days = rising
YIELD_FALLING_THRESHOLD= -0.10   # -0.10 in 5 days = falling
SENTIMENT_LOW          = 65.0    # UMich < 65 → contrarian bull
SENTIMENT_HIGH         = 95.0    # UMich > 95 → contrarian bear


def _rule_score(row: pd.Series) -> float:
    """
    Rule-based signal score in [0, 1].  0.5 = neutral.

    Signal groups (each fires a score value; final = mean of all fired):
      Technical  : RSI + MACD combos, MACD standalone, BB squeeze risk
      Sentiment  : Fear & Greed extremes, PCR contrarian
      Macro      : DXY direction, Oil, 10yr yield level/direction,
                   Consumer sentiment contrarian
      Patterns   : Bull/bear flag breaks
      Breadth    : Zweig Breadth Thrust (strong bull), Hindenburg Omen (risk)
      Trendline  : Hourly resistance/support breaks

    Trend filter applied AFTER averaging — dampens counter-SMA50 signals by 35%.
    """
    signals = []

    # ── Pull features ────────────────────────────────────────────────────────
    rsi            = float(row.get("RSI",               50))
    macd_up        = int(row.get("MACD_cross_up",        0))
    macd_down      = int(row.get("MACD_cross_down",      0))
    bb_squeeze     = int(row.get("bb_squeeze",           0))
    vix            = float(row.get("vix_level",          20))
    fg             = float(row.get("fg_normalized",      0.5)) * 100
    pcr            = float(row.get("pcr",                1.0))
    above_sma50    = int(row.get("above_sma50",          1))
    tl_break_up    = int(row.get("tl_break_up",          0))
    tl_break_down  = int(row.get("tl_break_down",        0))
    bull_flag      = int(row.get("bull_flag",            0))
    bear_flag      = int(row.get("bear_flag",            0))
    zweig          = int(row.get("zweig_thrust",         0))
    hindenburg     = int(row.get("hindenburg_omen",      0))
    dxy_roc        = float(row.get("DXY_roc5",           0))
    oil_roc        = float(row.get("Oil_roc5",           0))
    yield_level    = float(row.get("Yield10Y",           4.0))
    yield_roc      = float(row.get("Yield10Y_roc5",      0))
    cons_sent      = float(row.get("consumer_sentiment", 80))

    # ── Technical ────────────────────────────────────────────────────────────
    if rsi < 30 and macd_up:
        signals.append(0.84)
    elif rsi > 70 and macd_down:
        signals.append(0.16)
    elif rsi < 35:
        signals.append(0.64)
    elif rsi > 65:
        signals.append(0.36)

    if macd_up:
        signals.append(0.68)
    if macd_down:
        signals.append(0.32)

    if bb_squeeze and vix > VIX_HIGH:
        signals.append(0.50)   # risk → neutral

    # ── Sentiment ────────────────────────────────────────────────────────────
    if fg < FEAR_GREED_EXTREME_FEAR:
        signals.append(0.74)
    elif fg > FEAR_GREED_EXTREME_GREED:
        signals.append(0.26)

    if pcr > PCR_FEAR_THRESHOLD:
        signals.append(0.70)
    elif pcr < PCR_COMPLACENCY_THRESHOLD:
        signals.append(0.30)

    # ── Macro ─────────────────────────────────────────────────────────────────
    # DXY: strong dollar = headwind for SPY; weak dollar = tailwind
    if dxy_roc > DXY_STRONG_THRESHOLD:
        signals.append(0.36)
    elif dxy_roc < DXY_WEAK_THRESHOLD:
        signals.append(0.62)

    # Oil: spike = inflationary pressure; crash = risk-off caution
    if oil_roc > OIL_SPIKE_THRESHOLD:
        signals.append(0.40)
    elif oil_roc < OIL_CRASH_THRESHOLD:
        signals.append(0.44)   # crash is also risky, mild bearish

    # 10yr yield: high + rising = equity headwind; falling = tailwind
    if yield_level > YIELD_HIGH_LEVEL and yield_roc > YIELD_RISING_THRESHOLD:
        signals.append(0.34)
    elif yield_roc < YIELD_FALLING_THRESHOLD:
        signals.append(0.63)

    # Consumer sentiment contrarian (UMich)
    if cons_sent < SENTIMENT_LOW:
        signals.append(0.70)   # depressed sentiment → contrarian bull
    elif cons_sent > SENTIMENT_HIGH:
        signals.append(0.30)   # euphoric sentiment → contrarian bear

    # ── Chart Patterns ───────────────────────────────────────────────────────
    if bull_flag:
        signals.extend([0.76, 0.76])   # continuation pattern — add twice for weight
    if bear_flag:
        signals.extend([0.24, 0.24])

    # ── Breadth ───────────────────────────────────────────────────────────────
    if zweig:
        signals.extend([0.86, 0.86])   # Zweig Breadth Thrust — rare, very bullish
    if hindenburg:
        signals.extend([0.20, 0.20])   # Hindenburg Omen — elevated risk warning

    # ── Hourly trendline breaks ───────────────────────────────────────────────
    if tl_break_up:
        signals.extend([0.76, 0.76])
    if tl_break_down:
        signals.extend([0.24, 0.24])

    if not signals:
        return 0.5

    base = float(np.mean(signals))

    # ── Trend filter (post-average dampener) ──────────────────────────────────
    if above_sma50 == 0 and base > 0.5:
        base = 0.5 + (base - 0.5) * 0.65
    elif above_sma50 == 1 and base < 0.5:
        base = 0.5 - (0.5 - base) * 0.65

    return base


def compute_signal(latest_row: pd.Series, ml_model=None) -> dict:
    """
    Compute combined signal for a single row.

    Returns dict: score, direction, rule_score, ml_prob, confidence,
                  reduce_conf, trend_aligned.
    """
    rule    = _rule_score(latest_row)
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

    raw_conf   = abs(combined - 0.5) * 2 * 100
    confidence = min(99, max(1, round(raw_conf)))

    above_sma50   = int(latest_row.get("above_sma50", 1))
    trend_aligned = (direction == "bullish" and above_sma50 == 1) or \
                    (direction == "bearish" and above_sma50 == 0) or \
                    (direction == "neutral")

    return {
        "score":         combined,
        "direction":     direction,
        "rule_score":    rule,
        "ml_prob":       ml_prob,
        "confidence":    confidence,
        "reduce_conf":   fomc_proximity,
        "trend_aligned": trend_aligned,
    }


def generate_signals_series(df: pd.DataFrame,
                             ml_model=None,
                             min_confidence: int = 0) -> pd.DataFrame:
    """
    Annotate every row in df with signal_score and signal_direction.
    Rows below min_confidence are set to 'neutral'.
    """
    scores     = []
    directions = []
    for _, row in df.iterrows():
        sig = compute_signal(row, ml_model)
        scores.append(sig["score"])
        directions.append("neutral" if sig["confidence"] < min_confidence else sig["direction"])
    df = df.copy()
    df["signal_score"]     = scores
    df["signal_direction"] = directions
    return df
