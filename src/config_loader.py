"""
Utility module for loading project configuration parameters.

This module provides a single function — `load_config` — that reads
YAML-based configuration files and returns them as a Python dictionary.
Centralizing configuration management helps maintain clean code,
ensures reproducibility, and allows easy modification of project
parameters without touching the source code.
"""
from pathlib import Path
import yaml

def load_config() -> dict:
    """
    Load project configuration from YAML file.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing all configuration parameters.
    """
    # Шлях до кореневої папки проєкту
    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config
