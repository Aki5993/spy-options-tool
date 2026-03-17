"""Backtest performance metrics."""
import numpy as np
import pandas as pd
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import RISK_FREE_RATE


def compute_metrics(trades: list[dict], initial_capital: float = 100_000) -> dict:
    """
    Compute performance metrics from a list of trade dicts.

    Each trade dict must have: pnl (dollar P&L), entry_date, exit_date.
    """
    if not trades:
        return {
            "win_rate": 0, "avg_pnl": 0, "total_pnl": 0,
            "sharpe": 0, "sortino": 0, "max_drawdown": 0,
            "n_trades": 0,
        }

    pnls = np.array([t["pnl"] for t in trades])
    wins = (pnls > 0).sum()
    win_rate = wins / len(pnls)
    avg_pnl = pnls.mean()
    total_pnl = pnls.sum()

    # Equity curve
    equity = initial_capital + np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_drawdown = drawdown.min()

    # Sharpe / Sortino (per-trade returns)
    returns = pnls / initial_capital
    rf_per_trade = RISK_FREE_RATE / 252 * np.mean([
        (pd.Timestamp(t["exit_date"]) - pd.Timestamp(t["entry_date"])).days
        for t in trades
    ])
    excess = returns - rf_per_trade
    sharpe = (excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0

    neg = excess[excess < 0]
    sortino = (excess.mean() / neg.std() * np.sqrt(252)) if len(neg) > 0 and neg.std() > 0 else 0

    return {
        "win_rate": win_rate,
        "avg_pnl": avg_pnl,
        "total_pnl": total_pnl,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "n_trades": len(trades),
        "equity_curve": equity.tolist(),
    }


def equity_curve_df(trades: list[dict], initial_capital: float = 100_000) -> pd.DataFrame:
    """Build equity curve DataFrame from trades."""
    if not trades:
        return pd.DataFrame(columns=["date", "equity"])
    rows = []
    equity = initial_capital
    for t in sorted(trades, key=lambda x: x["entry_date"]):
        equity += t["pnl"]
        rows.append({"date": t["exit_date"], "equity": equity})
    return pd.DataFrame(rows)
