"""Run event study: load data, build panel if needed, compute CAR and save figures/tables."""
import sys
from pathlib import Path

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config_loader import load_config
from src.data.load_data import load_events, load_config_paths
from src.data.preprocess_data import build_daily_panel
from src.events.event_study import run_event_study


def main() -> None:
    cfg = load_config()
    paths = load_config_paths(cfg)
    base = Path(__file__).resolve().parent.parent
    interim = base / paths.get("interim", "data/interim")
    panel_path = interim / "daily_panel.parquet"

    import pandas as pd
    panel_csv = interim / "daily_panel.csv"
    if panel_path.exists():
        panel = pd.read_parquet(panel_path)
    elif panel_csv.exists():
        panel = pd.read_csv(panel_csv, parse_dates=["date"])
    else:
        print("Building daily panel (this may take a while for full daily.csv)...")
        panel = build_daily_panel(config=cfg, max_chunks=20)  # use max_chunks for testing; set None for full
        print(f"Panel shape: {panel.shape}")

    events = load_events(config=cfg)
    # Map ticker to permno for event_windows
    from src.data.load_data import build_ticker_permno_bridge
    bridge = build_ticker_permno_bridge(config=cfg)
    ticker_to_permno = bridge.groupby("ticker")["permno"].first().to_dict()
    events["permno"] = events["ticker"].map(ticker_to_permno)
    events = events.dropna(subset=["permno"]).copy()
    events["permno"] = events["permno"].astype(int)

    print("Running event study...")
    result = run_event_study(panel, events, config=cfg)
    print("Event study done. Figures and tables in results/figures and results/tables.")


if __name__ == "__main__":
    main()
