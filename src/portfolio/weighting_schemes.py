"""Portfolio weighting: equal, probability-weighted, risk-parity."""
from typing import Optional

import pandas as pd
import numpy as np


def equal_weight(
    scores: pd.Series,
    top_decile: float = 0.10,
    *,
    gross_target: float = 1.0,
) -> pd.Series:
    """Equal weight within top decile; normalize so sum(abs(w)) = gross_target."""
    q = scores.quantile(1 - top_decile)
    mask = scores >= q
    w = mask.astype(float)
    w = w / w.sum() * gross_target if w.sum() > 0 else w
    return w


def probability_weight(
    scores: pd.Series,
    top_decile: float = 0.10,
    *,
    gross_target: float = 1.0,
) -> pd.Series:
    """Weights proportional to scores within top decile; normalize to gross_target."""
    q = scores.quantile(1 - top_decile)
    mask = scores >= q
    w = (scores * mask).fillna(0)
    if w.sum() > 0:
        w = w / w.sum() * gross_target
    return w


def risk_parity_weight(
    scores: pd.Series,
    volatility: pd.Series,
    top_decile: float = 0.10,
    *,
    gross_target: float = 1.0,
) -> pd.Series:
    """Inverse-volatility weights within top decile. volatility is per-name (e.g. rolling std)."""
    q = scores.quantile(1 - top_decile)
    mask = scores >= q
    inv_vol = (1.0 / volatility.replace(0, np.nan)).fillna(0)
    w = (mask * inv_vol).fillna(0)
    if w.sum() > 0:
        w = w / w.sum() * gross_target
    return w
