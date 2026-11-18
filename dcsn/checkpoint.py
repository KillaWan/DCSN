"""Portable checkpoints contain their configuration and random Fourier modes."""
from pathlib import Path

import torch

from .model import ForecastModel
from .training import seed_everything


def save_checkpoint(path, model, metrics=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        torch.save(dict(format='dcsn-v1', config=model.config,
            state_dict={key: value.detach().cpu() for key, value in model.state_dict().items()},
            random_features=model.random_features(), metrics=metrics or {}), stream)


def load_checkpoint(path, device='cpu', config=None):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('format') == 'dcsn-v1':
        if config is not None and config != payload['config']:
            raise ValueError('Requested configuration differs from the checkpoint configuration')
        config = payload['config']
        seed_everything(config['training']['seed'])
        model = ForecastModel(config)
        model.load_state_dict(payload['state_dict'], strict=True)
        model.restore_random_features(payload['random_features'])
    else:
        if config is None:
            raise ValueError('Legacy checkpoints require their exact archived configuration')
        seed_everything(config['training']['seed'])
        model = ForecastModel(config)
        model.load_legacy_state(payload)
    return model.to(device).eval()
