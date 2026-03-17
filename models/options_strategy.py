from __future__ import annotations
"""Map signal → specific options strategy recommendation."""
import datetime
import sys, os

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    SIGNAL_BULL_THRESHOLD, SIGNAL_BEAR_THRESHOLD,
    IV_LOW_THRESHOLD, IV_HIGH_THRESHOLD,
    DEFAULT_EXPIRY_DAYS_MIN, DEFAULT_EXPIRY_DAYS_MAX,
)
from data.events import days_to_next_fomc, is_opex_week


def recommend_strategy(signal: dict,
                        current_iv: float | None = None,
                        spot_price: float | None = None) -> dict:
    """
    Given a signal dict (from signal_generator.compute_signal), return a
    strategy recommendation dict.

    Returns dict with keys:
      strategy       : str description
      action         : 'buy' | 'sell' | 'spread' | 'none'
      option_type    : 'call' | 'put' | 'iron_condor' | 'vertical_spread' | None
      expiry_days    : int suggested days to expiry
      strike_offset  : 'ATM' | 'OTM_1' etc.
      rationale      : list of str
    """
    score = signal["score"]
    direction = signal["direction"]
    reduce_conf = signal["reduce_conf"]
    iv = current_iv if current_iv is not None else 20.0
    near_fomc = reduce_conf or days_to_next_fomc() <= 2
    opex_wk = is_opex_week()

    rationale = []
    expiry_days = DEFAULT_EXPIRY_DAYS_MAX  # default 4 weeks

    # Near-event override: prefer vertical spreads
    if near_fomc or opex_wk:
        rationale.append("Near FOMC/OPEX → capped-risk vertical spread preferred")
        if direction == "bullish":
            return {
                "strategy": "Bull Call Spread",
                "action": "buy",
                "option_type": "vertical_spread",
                "expiry_days": expiry_days,
                "strike_offset": "ATM / ATM+5",
                "rationale": rationale,
            }
        elif direction == "bearish":
            return {
                "strategy": "Bear Put Spread",
                "action": "buy",
                "option_type": "vertical_spread",
                "expiry_days": expiry_days,
                "strike_offset": "ATM / ATM-5",
                "rationale": rationale,
            }

    # High IV → sell premium
    if iv >= IV_HIGH_THRESHOLD and direction == "neutral":
        rationale.append(f"IV={iv:.1f}% elevated + neutral signal → sell iron condor")
        return {
            "strategy": "Iron Condor",
            "action": "sell",
            "option_type": "iron_condor",
            "expiry_days": DEFAULT_EXPIRY_DAYS_MAX,
            "strike_offset": "OTM_1SD",
            "rationale": rationale,
        }

    if score >= SIGNAL_BULL_THRESHOLD:
        rationale.append(f"Score={score:.2f} → bullish")
        if iv < IV_LOW_THRESHOLD:
            rationale.append(f"IV={iv:.1f}% cheap → buy ATM call")
            return {
                "strategy": "Long Call",
                "action": "buy",
                "option_type": "call",
                "expiry_days": expiry_days,
                "strike_offset": "ATM",
                "rationale": rationale,
            }
        else:
            rationale.append(f"IV={iv:.1f}% elevated → bull call spread instead")
            return {
                "strategy": "Bull Call Spread",
                "action": "buy",
                "option_type": "vertical_spread",
                "expiry_days": expiry_days,
                "strike_offset": "ATM / ATM+5",
                "rationale": rationale,
            }

    if score <= SIGNAL_BEAR_THRESHOLD:
        rationale.append(f"Score={score:.2f} → bearish")
        if iv < IV_LOW_THRESHOLD:
            rationale.append(f"IV={iv:.1f}% cheap → buy ATM put")
            return {
                "strategy": "Long Put",
                "action": "buy",
                "option_type": "put",
                "expiry_days": expiry_days,
                "strike_offset": "ATM",
                "rationale": rationale,
            }
        else:
            rationale.append(f"IV={iv:.1f}% elevated → bear put spread instead")
            return {
                "strategy": "Bear Put Spread",
                "action": "buy",
                "option_type": "vertical_spread",
                "expiry_days": expiry_days,
                "strike_offset": "ATM / ATM-5",
                "rationale": rationale,
            }

    # Neutral
    if iv >= IV_HIGH_THRESHOLD:
        rationale.append(f"Neutral + IV={iv:.1f}% high → iron condor or strangle")
        return {
            "strategy": "Iron Condor / Strangle",
            "action": "sell",
            "option_type": "iron_condor",
            "expiry_days": DEFAULT_EXPIRY_DAYS_MAX,
            "strike_offset": "OTM_1SD",
            "rationale": rationale,
        }

    rationale.append("Neutral signal, no strong setup → hold/wait")
    return {
        "strategy": "No trade",
        "action": "none",
        "option_type": None,
        "expiry_days": None,
        "strike_offset": None,
        "rationale": rationale,
    }
