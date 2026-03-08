from pathlib import Path
from typing import Any, Optional, TextIO

import mergedeep
import yaml

CONFIG_DIR = Path(__file__).resolve().parent / "config"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default.yaml"


def _load_yaml_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    if not isinstance(config, dict):
        raise TypeError(f"Expected a mapping in config file '{path}'")
    return config


def _discover_type_configs() -> dict[str, Path]:
    type_configs: dict[str, Path] = {}
    for config_path in sorted(CONFIG_DIR.glob("default_*.yaml")):
        type_name = config_path.stem.removeprefix("default_")
        type_configs[type_name] = config_path
    return type_configs


DEFAULT_TYPE_CONFIGS = _discover_type_configs()


class ConfigHandler(dict):

    def __init__(
        self, config_file_object: Optional[TextIO] = None, type: Optional[str] = None, override: Optional[dict] = None
    ):
        mergedeep.merge(self, _load_yaml_config(DEFAULT_CONFIG_PATH))
        if type and type in DEFAULT_TYPE_CONFIGS:
            mergedeep.merge(self, _load_yaml_config(DEFAULT_TYPE_CONFIGS[type]))
        if config_file_object:
            config = yaml.safe_load(config_file_object) or {}
            if not isinstance(config, dict):
                raise TypeError("Expected a mapping in config file object")
            mergedeep.merge(self, config)
        if override:
            # Only override values that are already in the config that have been set to a non-None value
            mergedeep.merge(self, {k: v for k, v in override.items() if k in self and v is not None})

    def dump(self) -> str:
        dump: dict = dict()
        mergedeep.merge(dump, self)
        return yaml.dump(dump, default_flow_style=False)
