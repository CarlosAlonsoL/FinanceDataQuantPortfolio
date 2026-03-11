"""Build event windows [-pre, +post] trading days around index add/delete events."""
from pathlib import Path
from typing import Optional

import pandas as pd


def generate_event_windows(
    events: pd.DataFrame,
    panel: pd.DataFrame,
    pre: int = 60,
    post: int = 60,
    *,
    date_col: str = "date",
    permno_col: str = "permno",
    event_date_col: str = "effective_date",
    event_type_col: str = "event_type",
) -> pd.DataFrame:
    """For each event, align [-pre, +post] trading days relative to event_date.

    Args:
        events: DataFrame with columns event_date_col, permno_col (or ticker mapped to permno), event_type_col.
        panel: Daily panel with date_col, permno_col; must contain trading calendar.
        pre: Number of trading days before event.
        post: Number of trading days after event.

    Returns:
        DataFrame with one row per (event_id, rel_day): event_id, permno, date, rel_day, event_type,
        announcement_flag, effective_flag. Events with insufficient data are dropped.
    """
    events = events.copy()
    events[event_date_col] = pd.to_datetime(events[event_date_col]).dt.normalize()
    if "permno" not in events.columns and "ticker" in events.columns:
        raise ValueError("events must have permno or ticker; map ticker->permno before calling.")

    # Trading calendar from panel
    trading_dates = pd.Series(panel[date_col].unique()).sort_values().reset_index(drop=True)
    trading_dates = pd.to_datetime(trading_dates).dt.normalize()
    date_to_idx = {d: i for i, d in enumerate(trading_dates)}

    rows = []
    for idx, row in events.iterrows():
        ed = row[event_date_col]
        if ed not in date_to_idx:
            # Find nearest trading date
            try:
                idx_ed = trading_dates.searchsorted(ed)[0]
                if idx_ed >= len(trading_dates):
                    continue
                ed = trading_dates.iloc[idx_ed]
            except Exception:
                continue
        center = date_to_idx.get(ed)
        if center is None:
            continue
        permno = row[permno_col]
        event_type = row.get(event_type_col, "ADD")
        for rel in range(-pre, post + 1):
            i = center + rel
            if i < 0 or i >= len(trading_dates):
                continue
            dt = trading_dates.iloc[i]
            rows.append({
                "event_id": idx,
                "permno": permno,
                "date": dt,
                "rel_day": rel,
                "event_type": event_type,
                "announcement_flag": 1 if rel == 0 else 0,
                "effective_flag": 1 if rel == 0 else 0,
            })

    return pd.DataFrame(rows)
