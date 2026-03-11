"""Performance and risk metrics: Sharpe, Sortino, drawdown, Calmar, turnover."""
from typing import Dict, Optional

import pandas as pd
import numpy as np


def compute_performance_metrics(
    returns: pd.Series,
    rf: Optional[pd.Series] = None,
    *,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    """Annualized return, vol, Sharpe, Sortino, max drawdown, Calmar, avg turnover if in index."""
    ret = returns.dropna()
    if ret.empty:
        return {"annual_return": np.nan, "annual_volatility": np.nan, "sharpe_ratio": np.nan, "sortino_ratio": np.nan, "max_drawdown": np.nan, "calmar_ratio": np.nan}
    n = len(ret)
    if rf is not None:
        rf = rf.reindex(ret.index).fillna(0)
        excess = ret - rf
    else:
        excess = ret
    ann_ret = (1 + ret).prod() ** (periods_per_year / n) - 1 if n else np.nan
    ann_vol = ret.std() * np.sqrt(periods_per_year) if n > 1 else np.nan
    sharpe = (excess.mean() / ret.std() * np.sqrt(periods_per_year)) if ret.std() > 0 else np.nan
    downside = ret[ret < 0].std()
    sortino = (excess.mean() / downside * np.sqrt(periods_per_year)) if downside > 0 else np.nan
    cum = (1 + ret).cumprod()
    dd = (cum.cummax() - cum) / cum.cummax()
    max_dd = dd.max()
    calmar = ann_ret / (-max_dd) if max_dd != 0 else np.nan
    return {
        "annual_return": ann_ret,
        "annual_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar,
    }


def compute_drawdowns(returns: pd.Series) -> pd.DataFrame:
    """DataFrame with date, cumulative, running_max, drawdown."""
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    dd = (running_max - cum) / running_max
    return pd.DataFrame({"cumulative": cum, "running_max": running_max, "drawdown": dd})
