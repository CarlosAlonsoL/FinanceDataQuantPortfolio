"""Rolling-window features: momentum, volatility, liquidity (no lookahead)."""
from typing import List

import pandas as pd
import numpy as np


def add_momentum_features(
    df: pd.DataFrame,
    windows: List[int],
    *,
    ret_col: str = "ret",
    date_col: str = "date",
    permno_col: str = "permno",
) -> pd.DataFrame:
    """Add cumulative return over rolling windows (1m, 3m, 6m, 12m). Past data only."""
    df = df.sort_values([permno_col, date_col]).copy()
    for w in windows:
        df[f"ret_{w}d"] = df.groupby(permno_col)[ret_col].transform(
            lambda x: x.rolling(w, min_periods=1).apply(lambda y: (1 + y).prod() - 1 if len(y) >= 1 else np.nan)
        )
        # Shift so we use only past returns (no same-day lookahead)
        df[f"ret_{w}d"] = df.groupby(permno_col)[f"ret_{w}d"].shift(1)
    return df


def add_volatility_features(
    df: pd.DataFrame,
    windows: List[int],
    *,
    ret_col: str = "ret",
    permno_col: str = "permno",
) -> pd.DataFrame:
    """Rolling standard deviation of returns."""
    df = df.sort_values([permno_col, "date"]).copy()
    for w in windows:
        df[f"vol_{w}d"] = df.groupby(permno_col)[ret_col].transform(
            lambda x: x.rolling(w, min_periods=min(5, w)).std()
        ).shift(1)
    return df


def add_liquidity_features(
    df: pd.DataFrame,
    *,
    volume_col: str = "volume",
    cap_col: str = "market_cap",
    permno_col: str = "permno",
    windows: List[int] = [21],
) -> pd.DataFrame:
    """Rolling average volume; turnover = volume / (market_cap/1e6) as liquidity proxy if cap available."""
    df = df.sort_values([permno_col, "date"]).copy()
    if cap_col in df.columns and volume_col in df.columns:
        df["turnover"] = (df[volume_col] / (df[cap_col].replace(0, np.nan) / 1e6)).replace(np.inf, np.nan)
    else:
        df["turnover"] = np.nan
    for w in windows:
        df[f"turnover_avg_{w}d"] = df.groupby(permno_col)["turnover"].transform(
            lambda x: x.rolling(w, min_periods=1).mean()
        ).shift(1)
        df[f"volume_avg_{w}d"] = df.groupby(permno_col)[volume_col].transform(
            lambda x: x.rolling(w, min_periods=1).mean()
        ).shift(1)
    return df


def add_abnormal_performance(
    df: pd.DataFrame,
    windows: List[int],
    *,
    ret_col: str = "ret",
    market_ret_col: str = "market_ret",
    permno_col: str = "permno",
) -> pd.DataFrame:
    """Excess return vs market over rolling windows (sector/market relative)."""
    if market_ret_col not in df.columns:
        return df
    df = df.sort_values([permno_col, "date"]).copy()
    df["excess_ret"] = df[ret_col] - df[market_ret_col]
    for w in windows:
        df[f"excess_ret_{w}d"] = df.groupby(permno_col)["excess_ret"].transform(
            lambda x: x.rolling(w, min_periods=1).sum()
        ).shift(1)
    return df
