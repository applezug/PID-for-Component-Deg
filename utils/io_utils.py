"""Config and file IO."""

import os
import yaml
import json


def load_yaml_config(path: str) -> dict:
    if not os.path.isabs(path):
        path = os.path.join(os.getcwd(), path)
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_config_to_yaml(config: dict, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True)


def save_dict_to_json(d: dict, path: str, indent=2):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(d, f, indent=indent, ensure_ascii=False)


def load_dict_from_json(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
