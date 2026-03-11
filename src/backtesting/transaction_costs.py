"""Transaction cost model (linear in turnover)."""
from typing import Optional

import pandas as pd
import numpy as np


def estimate_costs(
    turnover: pd.Series,
    config: dict | None = None,
    *,
    cost_bps: Optional[float] = None,
) -> pd.Series:
    """Return time series of transaction cost in return space (decimal). turnover = sum|delta w|."""
    if config:
        cost_bps = cost_bps or config.get("backtest", {}).get("transaction_cost_bps", 10)
    else:
        cost_bps = cost_bps or 10
    return (turnover * cost_bps / 10_000).fillna(0)


def apply_costs_to_returns(
    returns: pd.Series,
    turnover: pd.Series,
    cost_bps: float = 10,
) -> pd.Series:
    """Net return = gross return - cost (in decimal)."""
    cost = turnover * cost_bps / 10_000
    return returns - cost.fillna(0)
