"""Standalone data contract for the three paper backbones.

Parsing, reductions and windows follow the frozen crossguard50_20260927
FAN loaders and ttn_norm/experiments/train.py, with timeF marks corrected by
backbone_contract_fix_20261020/contract.py. No downloads or archive imports.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, default_collate

from .timefeatures import time_features


DATA_PATHS = {
    **{name: f"{name}/{name}.csv" for name in ("ETTh1", "ETTh2", "ETTm1", "ETTm2")},
    "Electricity": "electricity/electricity.csv",
    "ExchangeRate": "exchange_rate/exchange_rate.csv",
    "Traffic": "traffic/traffic.txt",
    "Weather": "weather/weather/weather.csv",
}
SPLITS = ("train", "val", "test")


@dataclass
class StandardScaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, values):
        return (values - self.mean) / self.std

    def inverse_transform(self, values):
        if isinstance(values, torch.Tensor):
            return (values * torch.as_tensor(self.std, device=values.device)
                    + torch.as_tensor(self.mean, device=values.device))
        return values * self.std + self.mean


class WindowDataset(Dataset):
    def __init__(self, data, dates, bounds, scaler, seq_len, pred_len, horizon,
                 time_enc, freq):
        self.bounds, self.scaler = bounds, scaler
        self.window, self.steps, self.horizon = seq_len, pred_len, horizon
        self.time_enc, self.freq = time_enc, freq
        # Advanced indexing is intentional: FAN subsets are row-major copies.
        indices = range(*bounds)
        self.raw = data[indices]
        self.scaled_data = scaler.transform(self.raw)
        self.date_enc_data = time_features(dates.iloc[indices], time_enc, freq)
        if self.__len__() <= 0:
            raise ValueError(f"Split {bounds} has no complete forecasting windows")

    def __len__(self):
        return len(self.raw) - self.window - self.horizon - self.steps + 2

    def __getitem__(self, index):
        if not isinstance(index, (int, np.integer)):
            raise TypeError("Window indices must be integers")
        if not 0 <= index < len(self):
            raise IndexError(index)
        end = index + self.window
        start = end + self.horizon - 1
        stop = start + self.steps
        return (self.scaled_data[index:end], self.scaled_data[start:stop],
                self.raw[start:stop], self.date_enc_data[index:end],
                self.date_enc_data[start:stop])


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    datasets: dict
    bounds: dict
    scaler: StandardScaler
    info: dict

    @property
    def mean(self):
        return self.scaler.mean

    @property
    def std(self):
        return self.scaler.std

    @property
    def metadata(self):
        return self.info


def _read_data(path, dataset):
    if dataset == "Traffic":
        frame = pd.read_csv(path, header=None)
        if len(frame) != 17544:
            raise ValueError("Traffic must contain exactly 17544 hourly rows")
        frame["date"] = pd.date_range("2015-01-01", periods=17544, freq="h")
        data = frame.drop("date", axis=1).values
    elif dataset in ("Electricity", "ExchangeRate"):
        frame = pd.read_csv(path, parse_dates=["date"])
        data = frame.drop("date", axis=1).values
    else:
        frame = pd.read_csv(path, parse_dates=[0])
        # Weather's -9999 sentinel is a retained numeric value, not an NA token.
        data = frame.iloc[:, 1:].to_numpy()
    return data, pd.DataFrame({"date": frame.date})


def _collate(batch):
    return tuple(default_collate(batch))


def build_data(config, data_root=None, workers=None):
    """Build loaders from a config dict, preserving float64 source arithmetic.

    ``data.path`` is a raw file path relative to ``data_root`` (absolute paths
    also work). The default root is DCSN/datasets, independent of the cwd.
    ETT defaults to the popular 12/4/4 split; other datasets use the official
    ratio split. Scaling is always train-only, without epsilon.
    ``workers`` overrides ``training.workers``; RNG state is left to the caller.
    Batches are (x, scaled_y, raw_y, x_mark, y_mark), with forecast-only targets.
    """
    name = config["dataset"]
    if name not in DATA_PATHS:
        raise ValueError(f"Unsupported dataset: {name!r}")
    options, training = config.get("data", {}), config.get("training", {})
    root = Path(data_root).expanduser() if data_root is not None else Path(__file__).resolve().parents[1] / "datasets"
    path = (root / Path(options.get("path", DATA_PATHS[name])).expanduser()).resolve()
    window, steps = int(config["seq_len"]), int(config["pred_len"])
    horizon = int(options.get("horizon", 1))
    batch_size = int(training.get("batch_size", 32))
    workers = int(training.get("workers", 0) if workers is None else workers)
    if min(window, steps, horizon, batch_size) < 1 or workers < 0:
        raise ValueError("Lengths and batch_size must be positive; workers must be nonnegative")
    split = options.get("split", "popular" if name.startswith("ETT") else "ratio")
    if split not in ("popular", "ratio") or (split == "popular" and not name.startswith("ETT")):
        raise ValueError("split must be 'ratio', or 'popular' for ETT datasets")
    backbone = config["backbone"]
    if backbone not in ("DLinear", "FEDformer", "Autoformer"):
        raise ValueError(f"Unsupported backbone: {backbone!r}")
    time_enc = options.get("time_enc", int(backbone in ("FEDformer", "Autoformer")))
    if time_enc not in (0, 1):
        raise ValueError("data.time_enc must be 0 or 1")
    freq = options.get("freq", "d" if name == "ExchangeRate" else "t" if name.startswith("ETTm") else "h")
    data, dates = _read_data(path, name)
    if data.shape[1] != int(config["enc_in"]) or not np.isfinite(data).all():
        raise ValueError("Dataset channels differ from enc_in or contain non-finite values")
    total = len(data)
    if split == "popular":
        factor = 4 if name.startswith("ETTm") else 1
        ends = [months * 30 * 24 * factor for months in (12, 16, 20)]
        if total < ends[-1]:
            raise ValueError("ETT is shorter than the popular 12/4/4-month split")
        # These are the actual FAN borders, including its historical +horizon.
        starts = [0, ends[0] - window + horizon - 1, ends[1] - window + horizon - 1]
    else:
        train_ratio, val_ratio = float(options.get("train_ratio", 0.7)), float(options.get("val_ratio", 0.1))
        if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and train_ratio + val_ratio < 1):
            raise ValueError("Train/validation ratios must be positive and sum to less than one")
        train, val = int(train_ratio * total), int(val_ratio * total)
        ends = [train, train + val, total]
        starts = [0, train - window, train + val - window]
    if any(start < 0 or start >= end for start, end in zip(starts, ends)):
        raise ValueError("Split bounds cannot supply the requested context")
    # The official ratio fit uses the parser's original view (usually F order).
    # Popular fits use an indexed C-order copy; changing this changes bits.
    fit = data[:ends[0]] if split == "ratio" else data[list(range(ends[0]))]
    scaler = StandardScaler(np.mean(fit, axis=0), np.std(fit, axis=0))
    if not np.isfinite(scaler.mean).all() or not np.isfinite(scaler.std).all() or np.any(scaler.std == 0):
        raise ValueError("Training scaler has zero or non-finite variance")
    bounds = dict(zip(SPLITS, zip(starts, ends)))
    datasets = {key: WindowDataset(data, dates, bound, scaler, window, steps,
                                   horizon, time_enc, freq)
                for key, bound in bounds.items()}
    loaders = [DataLoader(datasets[key], batch_size=batch_size, shuffle=key == "train",
                          drop_last=key != "test", num_workers=workers, collate_fn=_collate)
               for key in SPLITS]
    if any(len(loader) == 0 for loader in loaders):
        raise ValueError("A loader is empty after the retained drop_last rule")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    info = {"file": str(path), "sha256": digest.hexdigest(), "dataset": name,
            "rows": total, "channels": data.shape[1], "split": split, "bounds": bounds,
            "windows": {key: len(ds) for key, ds in datasets.items()},
            "batches": dict(zip(SPLITS, map(len, loaders))),
            "drop_last": {"train": True, "val": True, "test": False},
            "scale_fit": "train only", "scaler_mean": scaler.mean.tolist(),
            "scaler_std": scaler.std.tolist(), "time_enc": time_enc, "freq": freq,
            "horizon": horizon, "workers": workers}
    return DataBundle(*loaders, datasets, bounds, scaler, info)
