"""Compute abnormal returns, CAR, volume and volatility effects; produce plots and tables."""
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.config_loader import load_config, get_section
from src.utils.plotting import plot_car, set_plot_style


def compute_abnormal_returns(
    event_windows: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    benchmark: str = "market",
    ret_col: str = "ret",
    date_col: str = "date",
    permno_col: str = "permno",
) -> pd.DataFrame:
    """Compute abnormal return per (event_id, rel_day): stock return minus benchmark return."""
    # Market return: equal-weighted average return per day from panel
    daily_ret = panel.groupby(date_col)[ret_col].mean().reset_index()
    daily_ret = daily_ret.rename(columns={ret_col: "market_ret"})
    panel_sub = panel[[date_col, permno_col, ret_col]].merge(daily_ret, on=date_col, how="left")

    ew = event_windows.merge(
        panel_sub,
        on=[date_col, permno_col],
        how="left",
    )
    ew["abnormal_ret"] = ew[ret_col] - ew["market_ret"]
    return ew


def compute_car(abnormal: pd.DataFrame) -> pd.DataFrame:
    """Cumulative abnormal return per event over rel_day."""
    abnormal = abnormal.sort_values(["event_id", "rel_day"])
    abnormal["car"] = abnormal.groupby("event_id")["abnormal_ret"].cumsum()
    return abnormal


def aggregate_event_stats(
    car_df: pd.DataFrame,
    *,
    event_type_col: str = "event_type",
    rel_day_col: str = "rel_day",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average CAR by rel_day for each event_type; return (car_by_rel, volume_ratio if available)."""
    car_by_rel = car_df.groupby([rel_day_col, event_type_col])["car"].mean().unstack(fill_value=0)
    volume_ratio = None
    if "volume_ratio" in car_df.columns:
        volume_ratio = car_df.groupby([rel_day_col, event_type_col])["volume_ratio"].mean().unstack(fill_value=1)
    return car_by_rel, volume_ratio


def run_event_study(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    config: dict | None = None,
    *,
    output_dir_figures: str | Path | None = None,
    output_dir_tables: str | Path | None = None,
) -> dict:
    """Full event study: windows, AR, CAR, aggregate, plots and tables."""
    cfg = config or load_config()
    base = Path(__file__).resolve().parent.parent.parent
    paths = cfg.get("paths", {})
    out_fig = Path(output_dir_figures or base / paths.get("results_figures", "results/figures"))
    out_tab = Path(output_dir_tables or base / paths.get("results_tables", "results/tables"))
    out_fig.mkdir(parents=True, exist_ok=True)
    out_tab.mkdir(parents=True, exist_ok=True)

    pre = get_section(cfg, "event_study", "pre_window", default=60)
    post = get_section(cfg, "event_study", "post_window", default=60)
    benchmark = get_section(cfg, "event_study", "benchmark", default="market")

    from src.events.event_windows import generate_event_windows
    windows = generate_event_windows(events, panel, pre=pre, post=post)
    if windows.empty:
        return {"car_by_rel": pd.DataFrame(), "windows": windows}

    ar_df = compute_abnormal_returns(windows, panel, benchmark=benchmark)
    car_df = compute_car(ar_df)
    car_by_rel, _ = aggregate_event_stats(car_df)

    # Plots
    plot_car(
        car_by_rel,
        title="Average CAR: Joiners vs Leavers",
        save_path=out_fig / "event_study_car_joiners_leavers.png",
    )
    # Pre-announcement zoom
    pre_car = car_by_rel.loc[car_by_rel.index >= -pre]
    pre_car = pre_car.loc[pre_car.index <= -1]
    if not pre_car.empty:
        plot_car(
            pre_car,
            title="Pre-announcement drift (avg CAR)",
            save_path=out_fig / "event_study_car_pre_announcement.png",
        )
    # Post-inclusion zoom
    post_car = car_by_rel.loc[car_by_rel.index >= 0]
    post_car = post_car.loc[post_car.index <= post]
    if not post_car.empty:
        plot_car(
            post_car,
            title="Post-inclusion reversal (avg CAR)",
            save_path=out_fig / "event_study_car_post_inclusion.png",
        )

    # Tables
    car_by_rel.to_csv(out_tab / "event_study_car_by_rel_day.csv")
    windows.to_csv(out_tab / "event_study_windows_sample.csv", index=False)

    return {"car_by_rel": car_by_rel, "windows": windows, "car_df": car_df}
