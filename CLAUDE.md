# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git Workflow

After completing any meaningful unit of work (new feature, bug fix, refactor, new file), commit and push immediately so progress is never lost:

```bash
git add <specific files>
git commit -m "concise description of what and why"
git push
```

Commit messages should be lowercase, imperative, and specific (e.g. `add VIX regime feature to event_features`, `fix Black-Scholes IV estimation`, `wire backtest results to sidebar date range`). Never use `git add -A` or `git add .` without reviewing what's staged first.

## Running the App

```bash
# Launch dashboard (headless, skips Streamlit email prompt)
streamlit run app.py --server.headless true

# Or suppress the email prompt interactively
echo "" | streamlit run app.py
```

The app runs on **http://localhost:8501** by default.

## Environment Setup

Copy `.env.example` to `.env` and fill in API keys (all optional — free tiers or fallbacks exist for each):

```
POLYGON_API_KEY    # real-time SPY quotes (fallback: yfinance)
ALPHA_VANTAGE_KEY  # news sentiment (fallback: score=0)
FRED_API_KEY       # FOMC dates (fallback: hardcoded 2024-2026 dates in data/events.py)
```

Python 3.9 is in use — all type hints must use `from __future__ import annotations` at the top of files, not bare `X | Y` union syntax.

## Architecture

The pipeline flows in one direction: **Data → Features → Signal → Strategy → UI**.

### Data layer (`data/`)
- `market_data.py` — yfinance fetches for SPY OHLCV + VIX; Polygon.io for real-time quotes
- `sentiment.py` — Alternative.me Fear & Greed (no key), StockTwits bull/bear ratio, Alpha Vantage news score
- `events.py` — FOMC dates via FRED or hardcoded fallback; OPEX (3rd Friday monthly); VIX expiry (Wednesday before 3rd Friday), all computed algorithmically
- `cache.py` — Parquet + `.meta` file TTL cache stored in `.cache/` dir; keyed by MD5 hash of the cache key string

### Feature layer (`features/`)
Three modules each produce columns that are joined into one wide DataFrame before signal generation:
- `technical.py` — RSI, MACD (+crossover flags), Bollinger Bands (+squeeze flag), ATR, OBV, SMA 20/50/200, 5-day forward return/sign (ML target)
- `sentiment_features.py` — normalizes Fear & Greed to 0–1, adds contrarian flags; `merge_sentiment_into_df()` left-joins FG history onto the main df by date
- `event_features.py` — per-row FOMC proximity (days + binary flag), OPEX proximity, VIX regime (0/1/2)

### Signal layer (`models/`)
- `signal_generator.py` — hybrid engine: `_rule_score()` produces a 0–1 score from threshold rules (RSI, MACD crossovers, Fear & Greed extremes); combined with XGBoost `predict_proba` at 50/50 weight. Near-FOMC dampens score toward 0.5.
- `options_strategy.py` — maps signal score + IV + event flags to a specific strategy (long call/put, bull/bear spread, iron condor, no trade)
- `trainer.py` — XGBoost training; `FEATURE_COLS` list defines exactly which columns are used; model saved to `models/xgb_model.json` + `models/feature_names.txt`

### Backtest (`backtest/`)
- `engine.py` — walk-forward: opens position on each new directional signal, closes on reversal or `max_hold_days`; no overlapping trades
- `options_sim.py` — Black-Scholes pricing for entry/exit; IV estimated as `VIX / 100 * 1.1`; each contract = 100 shares multiplier
- `metrics.py` — win rate, Sharpe/Sortino (annualized), max drawdown, equity curve

### UI (`ui/` + `app.py`)
- `app.py` — orchestrates all `@st.cache_data` / `@st.cache_resource` loaders; passes serialized JSON between cached functions to avoid hashing large DataFrames
- `chart.py` — 4-row Plotly subplot: candlestick+BB+signals / volume / VIX / RSI; dark theme
- `sidebar.py` — returns a `cfg` dict consumed entirely by `app.py`
- `backtest_view.py` — renders equity curve + trade log from the dict returned by `backtest/engine.py`

### Key data flow in `app.py`
1. Load SPY, VIX, Fear & Greed history (cached)
2. Serialize to JSON → pass to `build_feature_df()` (cached) → returns wide feature DataFrame
3. Slice to sidebar date range
4. `generate_signals_series()` annotates every row with `signal_direction` + `signal_score`
5. Latest row → `compute_signal()` → `recommend_strategy()` → signal card
6. Backtest runs on demand via button; result stored in `st.session_state["bt_result"]`

### Adding a new signal rule
Edit `models/signal_generator.py` → `_rule_score()`. Append a score value (0–1) to the `signals` list; the rule score is the mean of all appended values. To add a new feature column, add it to `features/technical.py` or the relevant features module, then add it to `FEATURE_COLS` in `models/trainer.py` and retrain.
