# DCSN

**Difference-Conditioned Shape Normalization**，中文名：**均值差条件形状归一化**。

这是 `norm_backbone50_20261023` 主结果的独立实现，不是新的方法或调参版本。每个回看窗口拟合均值差到块内形状的岭回归映射，限制映射的谱范数，通过可逆变换交给块均值预测器和主干处理，再用缓存映射恢复输出。

## 内容

- `dcsn/normalization.py`：逐窗口岭回归映射、谱范数约束、变换和逆变换。
- `dcsn/reference.py`：块均值预测器，包含共享 MLP、变量嵌入和两条线性跳跃连接。
- `dcsn/model.py`、`dcsn/backbones/`：统一模型接口和论文主表使用的三个主干：DLinear、FEDformer、Autoformer。
- `dcsn/training.py`：两候选参考头预训练、两轮骨干预热、联合训练及验证集选模。
- `dcsn/data.py`、`timefeatures.py`：原有切分、训练期标准化、窗口和时间特征。
- `dcsn/configs.json`：三个主干、八个数据集、四个预测长度，共 96 项原配置，随 Python 包一起安装。
- `scripts/download_datasets.py`、`scripts/dataset_manifest.json`：数据下载脚本及复现用的 SHA-256 清单。仓库不包含原始数据。
- `tests/`：可独立运行的数值检查；`LICENSES/` 保留必要的主干来源和许可说明。

不含 SAN、DDN、RevIN 等对照归一化实现，不含历史探索分支、旧实验报告或检查点集合。旧版对照程序及详细验收输出放在项目外的 `../refactor_validation_20261113/`，正常训练和评估不读取该目录。

## 下载数据集

在仓库根目录运行以下命令（Python 3.11，只用标准库，不需要先安装 PyTorch）。不带数据集名称时下载全部八项；只跑某个实验时可以只下载需要的数据：

```bash
python scripts/download_datasets.py                    # 全部，解压后约 253 MiB
python scripts/download_datasets.py Electricity Traffic  # 只准备这两个数据集
```

脚本默认写入本地 `datasets/`（Git 已忽略），自动解压并逐个核对训练时文件的大小和 SHA-256。已有文件正确则跳过，校验不符则报错，不替换本地文件。八个文件及各自的固定版本下载链接见 [数据清单](scripts/dataset_manifest.json)。

| 数据集 | 下载来源 | 相对于数据根目录的文件 |
| --- | --- | --- |
| ETTh1、ETTh2、ETTm1、ETTm2 | [ETDataset / ETT-small](https://github.com/zhouhaoyi/ETDataset/tree/1d16c8f4f943005d613b5bc962e9eeb06058cf07/ETT-small) | `<名称>/<名称>.csv` |
| Electricity、ExchangeRate | [多变量基准数据镜像](https://github.com/wwy155/multivariate_timeseries_datasets/tree/89e003fc15d20a310c337f73363e1a07a4beb532) | `electricity/electricity.csv`、`exchange_rate/exchange_rate.csv` |
| Weather | [同一镜像的 weather.zip](https://github.com/wwy155/multivariate_timeseries_datasets/blob/89e003fc15d20a310c337f73363e1a07a4beb532/weather.zip) | `weather/weather/weather.csv` |
| Traffic | [multivariate-time-series-data / traffic](https://github.com/laiguokun/multivariate-time-series-data/tree/7f402f185cc2435b5e66aed13a3b560ed142e023/traffic) | `traffic/traffic.txt`（无表头，不是带日期列的 traffic.csv） |

计算节点不能联网时，在可联网机器上运行脚本，把生成的数据目录复制到计算节点。使用共享存储时，下载和训练使用同一个 `--data-root`，例如：

```bash
python scripts/download_datasets.py Weather --data-root /path/to/data
python -m dcsn train --backbone DLinear --dataset Weather --horizon 336 \
  --data-root /path/to/data --device cuda:0 --output runs/dlinear_weather_336
```

## 使用

在此目录中运行，使用 Python 3.11 和 `requirements.txt` 中的数值库版本。PyTorch 的 CUDA 构建需与运行环境匹配。已有 `wnj_venv` 环境无需重装。需要从其他目录导入时可使用 `pip install --no-deps --no-build-isolation -e .`。普通安装也包含全部配置，但不把数据集装进 Python 环境；此时用 `--data-root /path/to/DCSN/datasets` 指定数据位置。

```bash
python -m unittest discover -s tests -v

python -m dcsn train --backbone DLinear --dataset Weather --horizon 336 \
  --device cuda:0 --output runs/dlinear_weather_336

python -m dcsn evaluate --checkpoint runs/dlinear_weather_336/selected.pt \
  --device cuda:0
```

`--output` 必须是新路径，避免覆盖已有实验。训练仅打印阶段、轮次、损失、验证和测试 MSE/MAE；按验证集选中的权重保存在 `selected.pt`，训练曲线及选中轮次保存在 `results.json`。

数据集名称：`ETTh1`、`ETTh2`、`ETTm1`、`ETTm2`、`Weather`、`Electricity`、`Traffic`、`ExchangeRate`。预测长度为 `96`、`192`、`336`、`720`。默认数据位于本目录；也可通过 `--data-root` 指定具有相同文件结构的外部目录。

上述 96 项主结果的旧检查点可直接加载，但须明确指定其原配置。内置配置使用主结果的 seed 2021，不用于替代其他种子或消融变体的配置：

```bash
python -m dcsn evaluate --checkpoint /path/to/selected_s2021.pt \
  --backbone FEDformer --dataset Electricity --horizon 336 --device cuda:0

python -m dcsn convert --checkpoint /path/to/selected_s2021.pt \
  --backbone FEDformer --dataset Electricity --horizon 336 \
  --output checkpoints/fedformer_electricity_336.pt --device cpu
```

新检查点包含配置、权重和 Fourier 模式索引。转换后评估不需要原项目。全部历史检查点没有重复复制，避免增加约 6.8 GB 的占用。

## 阅读顺序

先读 `model.py` 的 `forward`，再读 `normalization.py` 的 `forward`、`_estimate_map`、`coordinates` 和 `restore`。块均值预测器在 `reference.py` 中独立定义；训练细节集中在 `training.py`，不需要跟踪实验脚本或动态替换函数。

```text
回看窗口 X  [batch, time, variables]
  |
  +-- 历史均值差与块内形状 --> 每个样本拟合并缓存 A
  |
  +-- (I - B_X) X --> 去块均值 --> 主干 --> 主干输出 F
  |          |
  |          +--> 块均值预测器 --> 未来块均值 M
  |
  +----------------------------------> (I + B_X)(M + F) --> 预测
```

映射只从当前回看窗口估计，不使用未来数据，拟合过程不求梯度。拟合好的映射用于正反两次坐标变换，梯度仍可穿过坐标变换传到预测器。主干输出不额外强制为零块均值；这里保持主结果的实际代码行为。

公共输入输出使用 `[B, T, C]`，归一化内部使用 `[B, C, T]`。`reference.py` 的 `raw` 参数指变换后的历史，不是未经标准化的原始数据。`normalizer.mlp` 保留这个属性名以兼容旧权重，其实际类名为 `BlockMeanPredictor`。

训练顺序固定为：参考头预训练最多 10 轮，包含两组从同一初始化出发的候选；按验证参考误差选候选；主干预热两轮；然后联合训练并按验证 MSE 选择最终模型。学习率、块长、头宽度等逐项读取原配置。

## 一致性约定

保留了原算法的浮点运算次序、初始化随机数消耗、两次预训练之间的随机状态恢复、冻结骨干时的 dropout 行为，以及逐元素加权的 MSE/MAE。`_initialization.py` 仅在构造时消耗随机数，以复现已有种子的初始化；它不参与预测，没有需要训练的参数。

每个配置继续使用自己的块长、参考头尺寸、学习率、批量和训练上限，没有统一替换为一套默认超参数。Weather 的标记值、ETT 的 popular 切分、实际时间特征和末批处理保持原样。

数值等价的验收结果见 `VALIDATION.md`。跨 GPU 型号、数值库版本或非确定性 CUDA 内核的重复训练不承诺逐位相同；重构与旧实现的对照在同一环境下进行。

第三方代码出处、保留的许可证以及修改范围见 [来源说明](LICENSES/README.txt)；各文件的固定版本链接和 SHA-256 见 [源码清单](LICENSES/SOURCE_REPORT.json)。
