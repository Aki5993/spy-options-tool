from __future__ import annotations
"""Walk-forward backtest engine with stop-loss, profit-target, and min hold."""
import sys, os
import datetime
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    BACKTEST_MAX_HOLD_DAYS, BACKTEST_DEFAULT_CAPITAL,
    SIGNAL_BULL_THRESHOLD, SIGNAL_BEAR_THRESHOLD,
    RISK_FREE_RATE,
)
from backtest.options_sim import black_scholes_price
from backtest.metrics import compute_metrics, equity_curve_df


def _estimate_iv(vix_level: float) -> float:
    """SPY IV ≈ VIX / 100 with a small upward adjustment."""
    return max(0.05, min(1.0, vix_level / 100 * 1.1))


def run_backtest(df: pd.DataFrame,
                 start_date: str | None = None,
                 end_date: str | None = None,
                 initial_capital: float = BACKTEST_DEFAULT_CAPITAL,
                 max_hold_days: int = BACKTEST_MAX_HOLD_DAYS,
                 option_expiry_days: int = 45,
                 min_hold_days: int = 3,
                 stop_loss_pct: float = 0.55,
                 profit_target_pct: float = 0.50,
                 min_confidence: int = 0,
                 contract_size: int = 1) -> dict:
    """
    Walk-forward backtest over df[start_date:end_date].

    Key design: options are bought with `option_expiry_days` to expiry
    (default 45d) and held for at most `max_hold_days` (default 20d).
    This preserves time value at exit, avoiding the theta cliff of holding
    a 20d option to near-expiry.

    - min_hold_days: ignore signal reversals before this threshold
    - stop_loss_pct: close after min_hold if option down >X% (default 55%)
    - profit_target_pct: close immediately if option up >X% (default 50%)
    - min_confidence: skip entry when signal confidence is below this level

    df must have: Close, signal_direction, signal_score, vix_level.
    Returns dict: trades (list), metrics (dict), equity_curve (DataFrame).
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    if start_date:
        df = df[df.index >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df.index <= pd.Timestamp(end_date)]

    df = df.dropna(subset=["Close", "signal_direction"])

    trades = []
    i = 0
    dates = df.index.tolist()
    n = len(dates)

    while i < n - 1:
        row = df.loc[dates[i]]
        direction = row.get("signal_direction", "neutral")
        confidence = abs(float(row.get("signal_score", 0.5)) - 0.5) * 2 * 100

        if direction not in ("bullish", "bearish") or confidence < min_confidence:
            i += 1
            continue

        option_type = "call" if direction == "bullish" else "put"
        entry_spot = float(row["Close"])
        vix_entry = float(row.get("vix_level", 20.0))
        entry_iv = _estimate_iv(vix_entry)
        entry_date = dates[i]

        # ATM strike; buy with option_expiry_days time value
        K = entry_spot
        T_entry = option_expiry_days / 252

        entry_option_price = black_scholes_price(
            entry_spot, K, T_entry, RISK_FREE_RATE, entry_iv, option_type
        )

        # ── Day-by-day simulation ─────────────────────────────────────────
        exit_idx = min(i + max_hold_days, n - 1)
        exit_reason = "max_hold"
        exit_option_price = None

        for j in range(i + 1, min(i + max_hold_days + 1, n)):
            current_date = dates[j]
            current_row = df.loc[current_date]
            current_spot = float(current_row["Close"])
            current_vix = float(current_row.get("vix_level", vix_entry))
            current_iv = _estimate_iv(current_vix)

            days_elapsed = max((current_date - entry_date).days, 1)
            T_remaining = max((option_expiry_days - days_elapsed) / 252, 1 / 252)

            current_option_price = black_scholes_price(
                current_spot, K, T_remaining, RISK_FREE_RATE, current_iv, option_type
            )

            pnl_pct = (current_option_price - entry_option_price) / entry_option_price \
                if entry_option_price > 0 else 0.0

            # Profit-target check: applies from day 1 (locking in gains is always ok)
            if pnl_pct >= profit_target_pct:
                exit_idx = j
                exit_option_price = current_option_price
                exit_reason = "profit_target"
                break

            # Stop-loss and signal-reversal: only after min_hold_days
            # (theta alone can cause 40%+ intraday loss on short-dated options;
            #  don't stop out before the trade has had time to develop)
            if days_elapsed >= min_hold_days:
                if pnl_pct <= -stop_loss_pct:
                    exit_idx = j
                    exit_option_price = current_option_price
                    exit_reason = "stop_loss"
                    break

                future_dir = current_row.get("signal_direction", "neutral")
                if future_dir not in ("neutral", direction):
                    exit_idx = j
                    exit_option_price = current_option_price
                    exit_reason = "signal_reversal"
                    break

        # If no early exit, compute final price at exit_idx
        if exit_option_price is None:
            exit_date = dates[exit_idx]
            exit_row = df.loc[exit_date]
            exit_spot = float(exit_row["Close"])
            exit_vix = float(exit_row.get("vix_level", vix_entry))
            exit_iv = _estimate_iv(exit_vix)
            days_elapsed = max((exit_date - entry_date).days, 1)
            T_remaining = max((option_expiry_days - days_elapsed) / 252, 1 / 252)
            exit_option_price = black_scholes_price(
                exit_spot, K, T_remaining, RISK_FREE_RATE, exit_iv, option_type
            )

        exit_date = dates[exit_idx]
        days_held = max((exit_date - entry_date).days, 1)
        pnl_per_contract = (exit_option_price - entry_option_price) * 100
        pnl_pct_final = (exit_option_price - entry_option_price) / entry_option_price \
            if entry_option_price > 0 else 0.0

        trades.append({
            "entry_date": entry_date,
            "exit_date": exit_date,
            "direction": direction,
            "entry_spot": entry_spot,
            "exit_spot": float(df.loc[exit_date, "Close"]),
            "option_type": option_type,
            "strike": K,
            "days_held": days_held,
            "entry_price": entry_option_price,
            "exit_price": exit_option_price,
            "pnl": pnl_per_contract * contract_size,
            "pnl_pct": pnl_pct_final,
            "signal_score": float(row.get("signal_score", 0.5)),
            "exit_reason": exit_reason,
        })

        i = exit_idx + 1

    metrics = compute_metrics(trades, initial_capital)
    eq_curve = equity_curve_df(trades, initial_capital)

    return {"trades": trades, "metrics": metrics, "equity_curve": eq_curve}
