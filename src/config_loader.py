from types import SimpleNamespace
from pathlib import Path
import yaml

def dict_to_namespace(data):
    if isinstance(data, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in data.items()})
    elif isinstance(data, list):
        return [dict_to_namespace(i) for i in data]
    else:
        return data

def load_config(config_name="parameters.yaml"):
    base_path = Path(__file__).parent.parent
    config_path = base_path / "config" / config_name

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return dict_to_namespace(config_dict)