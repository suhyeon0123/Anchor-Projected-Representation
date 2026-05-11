"""Shared config loader (yaml-based)."""
from __future__ import annotations

from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "models.yaml"


def load_config(path: Path | str | None = None) -> dict:
    """Load and return the YAML config as a plain dict."""
    p = Path(path) if path else DEFAULT_CONFIG
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
