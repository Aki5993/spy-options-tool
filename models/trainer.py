"""XGBoost model training with walk-forward cross-validation."""
import os
import sys
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import MODEL_PATH, FEATURE_NAMES_PATH

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


FEATURE_COLS = [
    "RSI", "MACD", "MACD_signal", "MACD_hist", "MACD_cross_up", "MACD_cross_down",
    "BB_width", "BB_pct", "ATR",
    "above_sma20", "above_sma50", "above_sma200", "bb_squeeze",
    "fg_normalized", "contrarian_bull", "contrarian_bear",
    "days_to_fomc", "fomc_proximity", "is_opex_week", "days_to_opex",
    "vix_level", "vix_regime",
]


def get_feature_cols() -> list[str]:
    return FEATURE_COLS


def prepare_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extract features and target from the fully-merged DataFrame."""
    available = [c for c in FEATURE_COLS if c in df.columns]
    target_col = "fwd_5d_sign"
    mask = df[target_col].notna()
    for col in available:
        mask &= df[col].notna()
    X = df.loc[mask, available].astype(float)
    y = df.loc[mask, target_col].astype(int)
    return X, y


def train_model(df: pd.DataFrame,
                train_ratio: float = 0.9,
                n_estimators: int = 300,
                max_depth: int = 4,
                learning_rate: float = 0.05) -> "xgb.XGBClassifier":
    """Train XGBoost on first `train_ratio` of data, validate on the rest."""
    if not HAS_XGB:
        raise ImportError("xgboost is required for training")

    X, y = prepare_dataset(df)
    split = int(len(X) * train_ratio)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=False)

    val_acc = (model.predict(X_val) == y_val).mean()
    print(f"Validation accuracy: {val_acc:.3f} on {len(y_val)} samples")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save_model(MODEL_PATH)
    with open(FEATURE_NAMES_PATH, "w") as f:
        f.write("\n".join(X_train.columns.tolist()))

    return model


def load_model() -> "xgb.XGBClassifier | None":
    """Load trained model from disk, or return None."""
    if not HAS_XGB or not os.path.exists(MODEL_PATH):
        return None
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    return model


def get_feature_importance(model: "xgb.XGBClassifier") -> pd.DataFrame:
    """Return feature importance as sorted DataFrame."""
    if model is None:
        return pd.DataFrame()
    names_path = FEATURE_NAMES_PATH
    if os.path.exists(names_path):
        with open(names_path) as f:
            names = [l.strip() for l in f.readlines()]
    else:
        names = [f"f{i}" for i in range(len(model.feature_importances_))]
    imp = pd.DataFrame({"feature": names, "importance": model.feature_importances_})
    return imp.sort_values("importance", ascending=False).reset_index(drop=True)


def walk_forward_backtest_accuracy(df: pd.DataFrame,
                                    n_folds: int = 5) -> list[float]:
    """
    Walk-forward validation: returns list of accuracy per fold.
    """
    if not HAS_XGB:
        return []
    X, y = prepare_dataset(df)
    fold_size = len(X) // (n_folds + 1)
    accuracies = []
    for i in range(1, n_folds + 1):
        train_end = i * fold_size
        val_end = train_end + fold_size
        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
        if len(X_train) < 100 or len(X_val) < 10:
            continue
        m = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, n_jobs=-1,
        )
        m.fit(X_train, y_train, verbose=False)
        acc = (m.predict(X_val) == y_val).mean()
        accuracies.append(acc)
    return accuracies
