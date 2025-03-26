import yaml
from pathlib import Path

def load_config():
    config_path = Path(__file__).parent / "config" / "hyperparams.yml"
    with open(config_path) as f:
        return yaml.safe_load(f)