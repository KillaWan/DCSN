"""Train, evaluate or convert a checkpoint using the frozen paper presets."""
import argparse
import json
from pathlib import Path

import torch

from .checkpoint import load_checkpoint, save_checkpoint
from .config import load_config
from .data import build_data
from .model import ForecastModel
from .training import evaluate, log_epoch, seed_everything, train


def main():
    parser = argparse.ArgumentParser(description='Difference-Conditioned Shape Normalization')
    parser.add_argument('action', choices=('train', 'evaluate', 'convert'))
    parser.add_argument('--backbone', choices=('DLinear', 'FEDformer', 'Autoformer'))
    parser.add_argument('--dataset', choices=('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2',
                                             'Weather', 'Electricity', 'Traffic', 'ExchangeRate'))
    parser.add_argument('--horizon', type=int, choices=(96, 192, 336, 720))
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    supplied = (args.backbone, args.dataset, args.horizon)
    if any(value is not None for value in supplied) and not all(value is not None for value in supplied):
        parser.error('--backbone, --dataset and --horizon must be given together')
    config = load_config(*supplied) if all(value is not None for value in supplied) else None
    if args.action == 'train':
        if config is None or args.output is None:
            parser.error('train requires --backbone, --dataset, --horizon and --output')
        if args.checkpoint is not None:
            parser.error('train starts from the archived initialization; use evaluate for checkpoints')
        if args.output.exists():
            parser.error('--output already exists; refusing to overwrite an experiment')
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        seed_everything(config['training']['seed'])
        data = build_data(config, args.data_root)
        model = ForecastModel(config).to(args.device)
        result = train(model, data, config, device=args.device, on_epoch=log_epoch)
        save_checkpoint(args.output / 'selected.pt', model, result['metrics'])
        report = {key: value for key, value in result.items() if key != 'state_dict'}
        (args.output / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        metrics = result['metrics']['test']
        print(json.dumps(dict(best_epoch=result['best_epoch'], mse=metrics['mse'], mae=metrics['mae']), indent=2))
    else:
        if args.checkpoint is None:
            parser.error('--checkpoint is required')
        model = load_checkpoint(args.checkpoint, args.device, config)
        if args.action == 'convert':
            if args.output is None or args.output.exists():
                parser.error('convert requires a new --output checkpoint path')
            save_checkpoint(args.output, model)
            print(args.output)
        else:
            data = build_data(model.config, args.data_root)
            metrics = evaluate(model, data.test_loader, args.device)
            print(json.dumps({key: metrics[key] for key in ('mse', 'mae')}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
