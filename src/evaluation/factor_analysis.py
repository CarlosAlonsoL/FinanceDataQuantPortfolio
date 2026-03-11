"""Fama-French factor regression: alpha and betas."""
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import numpy as np
import statsmodels.api as sm

from src.utils.config_loader import load_config


def run_factor_regression(
    returns: pd.Series,
    factors: pd.DataFrame,
    *,
    risk_free_col: str = "RF",
    factor_cols: Optional[list] = None,
) -> Dict:
    """Time-series regression: r - rf = alpha + beta_M*MKT_RF + beta_S*SMB + beta_H*HML + beta_U*MOM.

    factors: DataFrame with date index and columns MKT_RF, SMB, HML, MOM, RF (or similar).
    Returns dict with alpha, t_alpha, betas, r_squared, etc.
    """
    factor_cols = factor_cols or ["MKT_RF", "SMB", "HML", "MOM"]
    common = returns.index.intersection(factors.index).drop_duplicates()
    if len(common) < 30:
        return {"alpha": np.nan, "t_alpha": np.nan, "r_squared": np.nan, "betas": {}}
    y = returns.reindex(common).dropna()
    X = factors.reindex(common)[[c for c in factor_cols if c in factors.columns]].dropna(how="all")
    common = y.index.intersection(X.dropna(how="any").index)
    y = y.loc[common]
    X = sm.add_constant(X.loc[common])
    model = sm.OLS(y, X).fit()
    return {
        "alpha": model.params.get("const", np.nan),
        "t_alpha": model.tvalues.get("const", np.nan),
        "r_squared": model.rsquared,
        "betas": {c: model.params.get(c, np.nan) for c in factor_cols if c in model.params},
    }


def load_factors(path: str | Path | None = None, config: dict | None = None) -> Optional[pd.DataFrame]:
    """Load factor CSV if path exists. Columns: date, MKT_RF, SMB, HML, MOM, RF."""
    cfg = config or load_config()
    p = path or cfg.get("paths", {}).get("raw_factors")
    if not p:
        return None
    from pathlib import Path
    base = Path(__file__).resolve().parent.parent.parent
    full = base / p if isinstance(p, str) else Path(p)
    if not full.exists():
        return None
    df = pd.read_csv(full)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    return df
