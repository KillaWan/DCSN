"""Small standalone checks; no experiment archives or GPU are required."""
import json
from pathlib import Path
import tempfile
import unittest

import torch

from dcsn.checkpoint import load_checkpoint, save_checkpoint
from dcsn.config import load_config
from dcsn.model import ForecastModel
from dcsn.normalization import ShapeNormalizer
from dcsn.training import seed_everything


class StandaloneTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        seed_everything(2021)

    def test_presets(self):
        root = Path(__file__).resolve().parents[1]
        presets = json.loads((root / 'dcsn/configs.json').read_text())
        manifest = {row['dataset']: row for row in
                    json.loads((root / 'scripts/dataset_manifest.json').read_text())}
        self.assertEqual(len(presets), 96)
        for backbone in ('DLinear', 'FEDformer', 'Autoformer'):
            for dataset in ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather',
                            'Electricity', 'Traffic', 'ExchangeRate'):
                for horizon in (96, 192, 336, 720):
                    cfg = load_config(backbone, dataset, horizon)
                    self.assertEqual(cfg['pred_len'], horizon)
                    self.assertEqual(cfg['data']['path'], manifest[dataset]['path'])
                    self.assertEqual(cfg['enc_in'], manifest[dataset]['channels'])

    def test_map_inverse_and_mean_preservation(self):
        norm = ShapeNormalizer(13, 7, 3, 3, 8, 8, 1, 0.1)
        x = torch.randn(2, 13, 3, requires_grad=True)
        residual = norm(x)
        bct = x.transpose(1, 2)
        transformed = norm.coordinates(bct)
        torch.testing.assert_close(norm.coordinates(transformed, inverse=True), bct,
                                   atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(norm.project(transformed), norm.project(bct),
                                   atol=1e-6, rtol=1e-6)
        self.assertFalse(norm._analysis.requires_grad)
        torch.testing.assert_close(transformed[..., -1], bct[..., -1], atol=0, rtol=0)
        residual.square().mean().backward()
        self.assertIsNotNone(x.grad)

    def test_portable_checkpoint(self):
        cfg = load_config('DLinear', 'Weather', 336)
        model = ForecastModel(cfg).eval()
        x = torch.randn(2, cfg['seq_len'], cfg['enc_in'])
        with torch.no_grad():
            prediction = model(x)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'selected.pt'
            save_checkpoint(path, model)
            restored = load_checkpoint(path)
            with self.assertRaises(FileExistsError):
                save_checkpoint(path, model)
            wrong_config = load_config('DLinear', 'Weather', 96)
            with self.assertRaisesRegex(ValueError, 'differs'):
                load_checkpoint(path, config=wrong_config)
            with torch.no_grad():
                actual = restored(x)
        torch.testing.assert_close(actual, prediction, atol=0, rtol=0)
        self.assertEqual(restored.config, cfg)

    def test_block_shapes_are_centered_and_map_is_bounded(self):
        norm = ShapeNormalizer(24, 12, 3, 3, 8, 8, 1, 0.1)
        x = torch.randn(2, 24, 3)
        residual = norm(x).transpose(1, 2)
        torch.testing.assert_close(norm.project(residual), torch.zeros_like(residual),
                                   atol=1e-6, rtol=0)
        factors = norm._analysis
        contrasts, dual = factors[..., :3], factors[..., 3:]
        matrix = contrasts.transpose(-1, -2) @ dual
        self.assertLessEqual(float(torch.linalg.matrix_norm(matrix, ord=2).max()), 1.000001)
        forecast = torch.randn(2, 3, 12)
        transformed = norm.coordinates(forecast)
        torch.testing.assert_close(norm.coordinates(transformed, inverse=True), forecast,
                                   atol=1e-6, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
