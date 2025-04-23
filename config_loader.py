# config_loader.py
# Utility to load training and environment configuration from a YAML file

import yaml
from pathlib import Path

def load_config():
    """
    Loads the hyperparameter configuration from config/hyperparams.yml.

    Returns:
        dict: Parsed configuration dictionary.
    """
    # Determine the path to the configuration file
    config_path = Path(__file__).parent / "config" / "hyperparams.yml"
    
    # Load and return the YAML config
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
