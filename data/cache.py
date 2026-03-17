"""Disk-based Parquet caching with per-source TTL."""
from __future__ import annotations
import os
import time
import hashlib
import pandas as pd
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import CACHE_DIR, CACHE_TTL


def _cache_path(key: str) -> str:
    safe = hashlib.md5(key.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{safe}.parquet")


def _meta_path(key: str) -> str:
    safe = hashlib.md5(key.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{safe}.meta")


def cache_get(key: str, ttl_key: str) -> Optional[pd.DataFrame]:
    """Return cached DataFrame if fresh, else None."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _cache_path(key)
    meta = _meta_path(key)
    if not os.path.exists(path) or not os.path.exists(meta):
        return None
    with open(meta) as f:
        saved_at = float(f.read().strip())
    ttl = CACHE_TTL.get(ttl_key, 3600)
    if time.time() - saved_at > ttl:
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def cache_set(key: str, df: pd.DataFrame) -> None:
    """Persist DataFrame to cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _cache_path(key)
    meta = _meta_path(key)
    df.to_parquet(path, index=True)
    with open(meta, "w") as f:
        f.write(str(time.time()))


def cache_clear(key: str) -> None:
    for p in [_cache_path(key), _meta_path(key)]:
        if os.path.exists(p):
            os.remove(p)
