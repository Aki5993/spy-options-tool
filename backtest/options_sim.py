"""Black-Scholes option pricing for backtest P&L simulation."""
import numpy as np
from scipy.stats import norm


def black_scholes_price(S: float, K: float, T: float, r: float,
                         sigma: float, option_type: str = "call") -> float:
    """
    Compute Black-Scholes theoretical price.

    Parameters
    ----------
    S : spot price
    K : strike price
    T : time to expiry in years
    r : risk-free rate
    sigma : implied volatility (e.g. 0.20 for 20%)
    option_type : 'call' or 'put'
    """
    if T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
        return intrinsic

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def simulate_option_trade(entry_spot: float,
                           exit_spot: float,
                           entry_iv: float,
                           exit_iv: float,
                           option_type: str,
                           days_held: int,
                           total_days: int,
                           r: float = 0.045,
                           strike_pct: float = 1.0) -> dict:
    """
    Simulate open/close of a single option trade using Black-Scholes.

    Parameters
    ----------
    entry_spot   : SPY price at entry
    exit_spot    : SPY price at close
    entry_iv     : IV at entry (0-1, e.g. 0.20)
    exit_iv      : IV at close
    option_type  : 'call' or 'put'
    days_held    : number of days held
    total_days   : total days in the contract (expiry horizon)
    strike_pct   : 1.0 = ATM, 1.02 = 2% OTM call, etc.

    Returns
    -------
    dict with: entry_price, exit_price, pnl, pnl_pct
    """
    K = entry_spot * strike_pct
    T_entry = total_days / 252
    T_exit = max((total_days - days_held) / 252, 1 / 252)

    entry_price = black_scholes_price(entry_spot, K, T_entry, r, entry_iv, option_type)
    exit_price = black_scholes_price(exit_spot, K, T_exit, r, exit_iv, option_type)

    # Each SPY option = 100 shares multiplier
    pnl = (exit_price - entry_price) * 100
    pnl_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0

    return {
        "entry_price": entry_price,
        "exit_price": exit_price,
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "strike": K,
        "option_type": option_type,
    }
