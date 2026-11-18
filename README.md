# DCSN

**English** | [Nederlands](README.nl.md) | [中文（简体）](README.zh-CN.md)

**Difference-Conditioned Shape Normalization** is a normalization method for time series forecasting. Subtracting the mean of a block removes its level, but the shape left within the block can still depend on whether the level rose or fell. DCSN fits that relationship on each lookback window, removes the shape predicted by the map, and uses the same map to restore the forecast.

This repository reproduces the main results of `norm_backbone50_20261023`. It includes three forecasting backbones (**DLinear, FEDformer and Autoformer**), eight datasets to download, and four forecast horizons (96, 192, 336 and 720 steps). The [96 presets](dcsn/configs.json) use the original settings and seed 2021. Data, checkpoints and baseline normalization methods are not bundled.

## Download the data

From the repository root, run one of the commands below. The first downloads all eight datasets (about 253 MiB after extraction); the second downloads only the two named datasets. The downloader uses the Python 3.11 standard library and does not require PyTorch.

```bash
python scripts/download_datasets.py
python scripts/download_datasets.py Electricity Traffic
```

Files go into `datasets/`, which Git ignores. The script checks the size and SHA-256 of each file against the [dataset manifest](scripts/dataset_manifest.json). It skips an existing file if it matches and reports an error without replacing it if it does not.

| Dataset | Source | Path under the data root |
| --- | --- | --- |
| ETTh1, ETTh2, ETTm1, ETTm2 | [ETDataset / ETT-small](https://github.com/zhouhaoyi/ETDataset/tree/1d16c8f4f943005d613b5bc962e9eeb06058cf07/ETT-small) | `<name>/<name>.csv` |
| Electricity, ExchangeRate | [Multivariate benchmark mirror](https://github.com/wwy155/multivariate_timeseries_datasets/tree/89e003fc15d20a310c337f73363e1a07a4beb532) | `electricity/electricity.csv`, `exchange_rate/exchange_rate.csv` |
| Weather | [weather.zip](https://github.com/wwy155/multivariate_timeseries_datasets/blob/89e003fc15d20a310c337f73363e1a07a4beb532/weather.zip) | `weather/weather/weather.csv` |
| Traffic | [Multivariate time series data / traffic](https://github.com/laiguokun/multivariate-time-series-data/tree/7f402f185cc2435b5e66aed13a3b560ed142e023/traffic) | `traffic/traffic.txt` (no header or date column) |

For a compute node without internet access, download on another machine and copy the data directory. To use shared storage, point the downloader and the model at the same data root:

```bash
python scripts/download_datasets.py Weather --data-root /path/to/data
python -m dcsn train --backbone DLinear --dataset Weather --horizon 336 \
  --data-root /path/to/data --device cuda:0 --output runs/dlinear_weather_336
```

## Train and evaluate

Use Python 3.11 and the versions in [requirements.txt](requirements.txt). For GPU runs, choose a PyTorch build compatible with your CUDA environment. Run the following from the repository root. For an editable installation, use `pip install --no-deps --no-build-isolation -e .` after installing the dependencies.

```bash
python -m unittest discover -s tests -v

python -m dcsn train --backbone DLinear --dataset Weather --horizon 336 \
  --device cuda:0 --output runs/dlinear_weather_336

python -m dcsn evaluate --checkpoint runs/dlinear_weather_336/selected.pt \
  --device cuda:0
```

Training requires a new output directory. It writes the resolved configuration to `config.json`, the weights selected by validation MSE to `selected.pt`, and the training curves and selected epoch to `results.json`. When using a non-default data directory, pass `--data-root` to evaluation as well as training.

### Load a historical checkpoint

For a checkpoint from the original main results, specify its backbone, dataset and horizon explicitly. The presets do not describe ablations or runs with other seeds.

```bash
python -m dcsn evaluate --checkpoint /path/to/selected_s2021.pt \
  --backbone FEDformer --dataset Electricity --horizon 336 --device cuda:0

python -m dcsn convert --checkpoint /path/to/selected_s2021.pt \
  --backbone FEDformer --dataset Electricity --horizon 336 \
  --output checkpoints/fedformer_electricity_336.pt --device cpu
```

A converted checkpoint contains the configuration, weights and Fourier mode indices. It can be evaluated without the original project.

## How the method works

1. On each lookback window, DCSN fits a ridge regression from differences in block means to within-block shape. It caps the spectral norm of the map at one and caches the map for that window. The fit uses no forecast observations and is not differentiated.
2. DCSN transforms the history as `(I − B_X)X`. The backbone receives the block-centered transformed history and produces `F`. The block mean predictor reads the transformed history before block centering and forecasts block means `M`.
3. The output is `(I + B_X)(M + F)`, using the cached map. Gradients pass through the coordinate transforms to the predictors. The backbone output is not constrained to have zero block means.

Start with `ForecastModel.forward` in [model.py](dcsn/model.py). The forward and inverse coordinate transforms are in [normalization.py](dcsn/normalization.py). The [block mean predictor](dcsn/reference.py) uses a shared MLP, variable embeddings and two linear skip connections. Public inputs and outputs use `[batch, time, variables]`; the normalizer works internally in `[batch, variables, time]`.

## Reproducibility

The [training code](dcsn/training.py) pretrains two candidate block mean predictors from the same initialization (up to ten epochs in total). It selects one by validation error, warms up the backbone for two epochs, then trains both predictors jointly. Block length, head width, learning rates, batch size and epoch limits come from the corresponding preset.

The release preserves the original initialization draws, random-state restoration, data splits, time features, Weather sentinel values and handling of the final batch. MSE and MAE weight every evaluated element equally. [Validation results](VALIDATION.md) distinguish same-environment parity checks from retraining on different hardware, for which bitwise equality is not guaranteed.

See the [third-party notices](LICENSES/README.txt) for attribution and the [source manifest](LICENSES/SOURCE_REPORT.json) for pinned source files and checksums.
