"""Shared model utilities: rolling splits, metrics, preprocessing."""
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler


def make_rolling_splits(
    df: pd.DataFrame,
    train_years: int = 5,
    test_years: int = 1,
    min_start_year: int = 2000,
    date_col: str = "date",
) -> List[Tuple[pd.Index, pd.Index]]:
    """Splits (train_index, test_index) with train strictly before test. No overlap."""
    df = df.sort_values(date_col).copy()
    df[date_col] = pd.to_datetime(df[date_col])
    years = df[date_col].dt.year.unique()
    years = sorted([y for y in years if y >= min_start_year])
    splits = []
    for i in range(len(years) - train_years - test_years + 1):
        train_y = years[i : i + train_years]
        test_y = years[i + train_years : i + train_years + test_years]
        train_idx = df[df[date_col].dt.year.isin(train_y)].index
        test_idx = df[df[date_col].dt.year.isin(test_y)].index
        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((train_idx, test_idx))
    return splits


def train_and_evaluate(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    scale: bool = False,
) -> Dict[str, float]:
    """Fit model and return ROC-AUC, precision, recall, F1. Optionally scale X."""
    if scale:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
        X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_test)
    pred = (proba >= 0.5).astype(int)
    res = {}
    if y_test.nunique() >= 2:
        res["roc_auc"] = roc_auc_score(y_test, proba)
    else:
        res["roc_auc"] = np.nan
    res["precision"] = precision_score(y_test, pred, zero_division=0)
    res["recall"] = recall_score(y_test, pred, zero_division=0)
    res["f1"] = f1_score(y_test, pred, zero_division=0)
    return res


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 100) -> float:
    """Precision@k: share of positives in top-k by score."""
    if len(y_true) < k or y_true.sum() == 0:
        return 0.0
    top_k_idx = np.argsort(y_score)[-k:]
    return float(y_true[top_k_idx].sum() / min(k, y_true.sum()))


def get_feature_columns(df: pd.DataFrame, exclude: List[str] | None = None) -> List[str]:
    """Numeric columns suitable as features (exclude ids, dates, labels)."""
    exclude = exclude or ["date", "permno", "ticker", "label_join", "label_leave"]
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
