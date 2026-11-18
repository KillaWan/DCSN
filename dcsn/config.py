"""Frozen main-result settings, without historical experiment switches."""
import copy
import json
from pathlib import Path


def load_config(backbone, dataset, horizon, path=None):
    path = Path(path) if path else Path(__file__).with_name('configs.json')
    configs = json.loads(path.read_text())
    key = f'{backbone}/{dataset}/{int(horizon)}'
    if key not in configs:
        raise ValueError(f'Unknown configuration: {key}')
    return copy.deepcopy(configs[key])
