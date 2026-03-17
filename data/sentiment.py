"""Sentiment data: Fear & Greed, StockTwits, Alpha Vantage news."""
import os
import sys
import requests
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import ALPHA_VANTAGE_KEY
from data.cache import cache_get, cache_set


def fetch_fear_greed() -> dict:
    """
    Fetch current Fear & Greed index from Alternative.me (free, no key needed).
    Returns dict: {value, classification, timestamp}
    """
    key = "fear_greed_current"
    cached = cache_get(key, "fear_greed")
    if cached is not None and not cached.empty:
        row = cached.iloc[-1]
        return {"value": row["value"], "classification": row["classification"], "timestamp": row.name}

    try:
        url = "https://api.alternative.me/fng/?limit=30&format=json"
        r = requests.get(url, timeout=10)
        data = r.json()["data"]
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
        df["value"] = df["value"].astype(int)
        df = df.set_index("timestamp").sort_index()
        df = df[["value", "value_classification"]].rename(columns={"value_classification": "classification"})
        cache_set(key, df)
        row = df.iloc[-1]
        return {"value": int(row["value"]), "classification": row["classification"], "timestamp": df.index[-1]}
    except Exception as e:
        return {"value": 50, "classification": "Neutral", "timestamp": None, "error": str(e)}


def fetch_fear_greed_history(days: int = 365) -> pd.DataFrame:
    """Return historical Fear & Greed as DataFrame with columns [value, classification]."""
    key = f"fear_greed_hist_{days}"
    cached = cache_get(key, "fear_greed")
    if cached is not None:
        return cached

    try:
        url = f"https://api.alternative.me/fng/?limit={days}&format=json"
        r = requests.get(url, timeout=15)
        data = r.json()["data"]
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
        df["value"] = df["value"].astype(int)
        df = df.set_index("timestamp").sort_index()
        df = df[["value", "value_classification"]].rename(columns={"value_classification": "classification"})
        cache_set(key, df)
        return df
    except Exception:
        return pd.DataFrame(columns=["value", "classification"])


def fetch_stocktwits_sentiment() -> dict:
    """
    Fetch StockTwits SPY stream and compute bull/bear ratio.
    Returns dict: {bull_ratio, bear_ratio, total_messages}
    """
    key = "stocktwits_spy"
    cached = cache_get(key, "stocktwits")
    if cached is not None and not cached.empty:
        row = cached.iloc[-1]
        return {"bull_ratio": row.get("bull_ratio", 0.5), "bear_ratio": row.get("bear_ratio", 0.5), "total": 0}

    try:
        url = "https://api.stocktwits.com/api/2/streams/symbol/SPY.json"
        r = requests.get(url, timeout=10)
        messages = r.json().get("messages", [])
        bulls = sum(1 for m in messages if m.get("entities", {}).get("sentiment", {}) and
                    m["entities"]["sentiment"].get("basic") == "Bullish")
        bears = sum(1 for m in messages if m.get("entities", {}).get("sentiment", {}) and
                    m["entities"]["sentiment"].get("basic") == "Bearish")
        total = bulls + bears
        bull_ratio = bulls / total if total > 0 else 0.5
        bear_ratio = bears / total if total > 0 else 0.5
        result = {"bull_ratio": bull_ratio, "bear_ratio": bear_ratio, "total": total}
        df = pd.DataFrame([{"bull_ratio": bull_ratio, "bear_ratio": bear_ratio}],
                          index=pd.DatetimeIndex([pd.Timestamp.now()]))
        cache_set(key, df)
        return result
    except Exception as e:
        return {"bull_ratio": 0.5, "bear_ratio": 0.5, "total": 0, "error": str(e)}


def fetch_news_sentiment() -> dict:
    """
    Fetch news sentiment from Alpha Vantage (requires API key).
    Returns dict: {score, label, articles_count}
    """
    if not ALPHA_VANTAGE_KEY:
        return {"score": 0.0, "label": "Neutral", "articles_count": 0}

    key = "news_sentiment_spy"
    cached = cache_get(key, "news_sentiment")
    if cached is not None and not cached.empty:
        row = cached.iloc[-1]
        return {"score": row.get("score", 0.0), "label": row.get("label", "Neutral"), "articles_count": 0}

    try:
        url = (
            f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT"
            f"&tickers=SPY&apikey={ALPHA_VANTAGE_KEY}&limit=50"
        )
        r = requests.get(url, timeout=15)
        data = r.json()
        feed = data.get("feed", [])
        scores = []
        for article in feed:
            for ts in article.get("ticker_sentiment", []):
                if ts.get("ticker") == "SPY":
                    try:
                        scores.append(float(ts.get("ticker_sentiment_score", 0)))
                    except ValueError:
                        pass
        avg_score = sum(scores) / len(scores) if scores else 0.0
        label = "Bullish" if avg_score > 0.15 else ("Bearish" if avg_score < -0.15 else "Neutral")
        result = {"score": avg_score, "label": label, "articles_count": len(feed)}
        df = pd.DataFrame([{"score": avg_score, "label": label}],
                          index=pd.DatetimeIndex([pd.Timestamp.now()]))
        cache_set(key, df)
        return result
    except Exception as e:
        return {"score": 0.0, "label": "Neutral", "articles_count": 0, "error": str(e)}
