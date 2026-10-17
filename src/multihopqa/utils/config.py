"""Small YAML configuration reader used by the entrypoints."""

from pathlib import Path

import yaml


def load_config(config_path):
    """Read a YAML file and check that it contains a mapping."""
    path = Path(config_path)
    with open(path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")

    return config


def check_data_config(config):
    """Report missing settings early with a readable message."""
    required = [
        "dataset_id",
        "dataset_config",
        "splits",
        "cache_dir",
        "output_dir",
    ]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing required data settings: {', '.join(missing)}")

    if not config["splits"]:
        raise ValueError("At least one dataset split must be listed.")
