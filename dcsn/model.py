"""A single normalizer, block-mean predictor and interchangeable backbone."""
from copy import deepcopy

import torch
from torch import nn

from .backbones import build_backbone
from .normalization import ShapeNormalizer


class ForecastModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = deepcopy(config)
        head = config['head']
        # Construction order preserves the historical seed stream. Registration
        # order preserves optimizer parameter order (backbone before reference).
        normalizer = ShapeNormalizer(config['seq_len'], config['pred_len'],
            config['enc_in'], config['block_len'], head['width'], head['ff'],
            head['layers'], head['dropout'], dataset=config['dataset'])
        self.backbone = build_backbone(config['backbone'], config['backbone_config'])
        self.normalizer = normalizer
        self.label_len = config['label_len']
        self.pred_len = config['pred_len']
        self.is_former = config['backbone'] in ('FEDformer', 'Autoformer')

    def reference_only(self, x):
        self.normalizer(x)
        return self.normalizer.reference_prediction.transpose(1, 2)

    def forward(self, x, x_mark=None, y_mark=None):
        residual = self.normalizer(x)
        if self.is_former:
            if x_mark is None or y_mark is None:
                raise ValueError('Former backbones require lookback and forecast time marks')
            decoder_marks = torch.cat([x_mark[:, -self.label_len:, :], y_mark], dim=1)
            decoder = torch.cat([residual[:, -self.label_len:, :],
                torch.zeros(residual.size(0), self.pred_len, residual.size(-1),
                            device=residual.device, dtype=residual.dtype)], dim=1)
            output = self.backbone(residual, x_mark, decoder, decoder_marks)
        else:
            output = self.backbone(residual)
        return self.normalizer.restore(output)

    def load_legacy_state(self, state):
        state = {key.removeprefix('module.'): value for key, value in state.items()}
        unexpected = [key for key in state if not key.startswith(('fm.', 'nm.apn_module.'))]
        if unexpected:
            raise ValueError(f'Not a current DCSN checkpoint: {unexpected}')
        self.backbone.load_state_dict({key[3:]: value for key, value in state.items()
                                      if key.startswith('fm.')}, strict=True)
        self.normalizer.load_legacy_state(state)

    def random_features(self):
        """Persist Fourier mode selections which upstream stored as Python lists."""
        return {name: {key: [int(index) for index in getattr(module, key)]
                      for key in ('index', 'index_q', 'index_kv') if hasattr(module, key)}
                for name, module in self.backbone.named_modules()
                if any(hasattr(module, key) for key in ('index', 'index_q', 'index_kv'))}

    def restore_random_features(self, features):
        modules = dict(self.backbone.named_modules())
        expected = self.random_features()
        if set(features) != set(expected):
            raise ValueError('Checkpoint Fourier module names do not match the model')
        for name, fields in features.items():
            if set(fields) != set(expected[name]):
                raise ValueError(f'Checkpoint Fourier fields do not match {name}')
            for key, value in fields.items():
                if len(value) != len(expected[name][key]):
                    raise ValueError(f'Checkpoint Fourier mode count does not match {name}.{key}')
                setattr(modules[name], key, deepcopy(value))
