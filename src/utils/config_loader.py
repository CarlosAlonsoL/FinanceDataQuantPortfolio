"""Load and expose configuration from config.yaml."""
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | None = None) -> dict[str, Any]:
    """Load YAML config from the project config file.

    Args:
        path: Path to config file. If None, uses config/config.yaml
            relative to project root (current working directory or parent of src).

    Returns:
        Nested dict of configuration.
    """
    if path is None:
        base = Path(__file__).resolve().parent.parent.parent
        path = str(base / "config" / "config.yaml")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def get_section(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Get a nested value from config by key path.

    Example:
        get_section(cfg, "event_study", "pre_window") -> 60
    """
    cur = config
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur
