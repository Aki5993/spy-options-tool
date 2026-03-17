import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

# Data settings
SPY_TICKER = "SPY"
VIX_TICKER = "^VIX"
HISTORY_YEARS = 20
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")

# Cache TTL in seconds
CACHE_TTL = {
    "market_daily": 3600 * 4,       # 4 hours
    "market_intraday": 60,           # 1 minute
    "fear_greed": 3600 * 6,          # 6 hours
    "stocktwits": 900,               # 15 minutes
    "news_sentiment": 3600 * 2,      # 2 hours
    "fomc_dates": 3600 * 24 * 7,     # 1 week
}

# Technical indicator parameters
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2
ATR_PERIOD = 14
SMA_PERIODS = [20, 50, 200]

# Signal thresholds
SIGNAL_BULL_THRESHOLD = 0.65
SIGNAL_BEAR_THRESHOLD = 0.35
IV_LOW_THRESHOLD = 20.0
IV_HIGH_THRESHOLD = 25.0
FEAR_GREED_EXTREME_FEAR = 20
FEAR_GREED_EXTREME_GREED = 80
FOMC_PROXIMITY_DAYS = 2

# Rule weight vs ML weight
RULE_WEIGHT = 0.5
ML_WEIGHT = 0.5

# Options defaults
DEFAULT_EXPIRY_DAYS_MIN = 14
DEFAULT_EXPIRY_DAYS_MAX = 28

# Backtest settings
BACKTEST_MAX_HOLD_DAYS = 20
BACKTEST_DEFAULT_CAPITAL = 100_000
RISK_FREE_RATE = 0.045  # 4.5% annualized

# VIX regime thresholds
VIX_LOW = 15.0
VIX_HIGH = 25.0

# Model persistence path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "xgb_model.json")
FEATURE_NAMES_PATH = os.path.join(os.path.dirname(__file__), "models", "feature_names.txt")
