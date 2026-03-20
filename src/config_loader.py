from pathlib import Path
import yaml


def load_config(config_name="parameters.yaml"):
    base_path = Path(__file__).parent.parent
    config_path = base_path / "config" / config_name

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return _to_namespace(config_dict)


def _to_namespace(data):
    if isinstance(data, dict):
        return type("Config", (), {k: _to_namespace(v) for k, v in data.items()})()
    return data