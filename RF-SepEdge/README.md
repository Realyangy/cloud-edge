# RF-SepEdge：射频信号分析与恢复实验项目

> 当前状态（2026-10-28）：信号分离已覆盖四类现有干扰实验；频谱分析已完成扩大 TorchSig 数据训练、目标函数对比和独立模型选择。YOLO→IQ 提取→XCiT 联合流程默认使用 `xcit_joint_v4.ckpt`。

项目包含两个**相互独立**的实验任务。目前并未实现由频谱识别任务自动触发 WaveNet 分离的统一端到端系统。

| 子项目 | 目标 | 主要模型与依赖 | 状态 |
|---|---|---|---|
| [`signal-separation/`](signal-separation/README.md) | 从混合 IQ 中恢复目标 QPSK / OFDM-QPSK 信号并解调 | MIT RF Challenge 0.2.0 官方数据生成器、Sionna 0.10.0、WaveNet | 四类现有干扰实验已完成 |
| [`spectrum-analysis/`](spectrum-analysis/README.md) | 宽带频谱目标检测、IQ 提取、调制方式分类 | TorchSig 2.1.1、YOLO11n、XCiT | 联合 V4 已完成，独立测试家族准确率 82.94%、57 类 Top-1 38.06% |

## 代码与文件位置

```text
RF-SepEdge/
├── README.md
├── AUDIT.md                         # 代码整理前的历史源码审查记录
├── requirements-rf-sionna.txt      # Sionna 数据生成与解调环境
├── requirements-rf-spectrum.txt    # WaveNet、YOLO 和 XCiT 环境
├── demo-results/                    # 供网页直接读取的轻量 JSON/CSV 运行结果
├── signal-separation/
│   ├── mit_sionna_official_smoke.py
│   ├── mit_sionna_wavenet_infer.py
│   ├── mit_sionna_official_ber.py
│   ├── summarize_sinr.py
│   ├── rfchallenge/                 # MIT 官方源码
│   ├── dataset/                     # MIT 数据
│   ├── pretrained-models/           # 官方预训练权重
│   ├── output/                      # 本地生成结果，不纳入 Git
│   └── README.md
└── spectrum-analysis/
    ├── inference.py                 # YOLO 单图检测
    ├── benchmark_yolo.py            # YOLO 批量评测
    ├── xcit_inference_v211.py       # 官方窄带 XCiT 测试
    ├── combined_inference.py        # YOLO→IQ→XCiT 单图联合入口
    ├── benchmark_combined.py        # 联合流程批量评测
    ├── test_extract_iq_frequency_shift.py
    ├── finetune_xcit_joint.py
    ├── configs/
    ├── data/                         # 按需生成的训练数据，不纳入最终仓库
    ├── models/                       # YOLO、官方 XCiT、历史 V3 和正式 V4 权重
    ├── torchsig-models-1.0.0/
    ├── output/                      # 本地生成结果，不纳入 Git
    └── README.md
```

## 已完成的实验结果（本地重生成样本）

### 任务一：WaveNet 信号分离

- **测试设计**：QPSK、OFDM-QPSK 两种目标信号；CommSignal2 干扰；7 档标称 SINR（-24/-18/-15/-12/-9/-6/0 dB），每种目标信号每档 10 条，合计 140 条；Seed 42。
- **QPSK**：70 条的平均输入 BER 为 **17.291%**，恢复后平均 BER 为 **1.724%**；标称 -12 dB 及以上的 40 条样本未观测到比特错误。
- **OFDM-QPSK**：70 条的平均输入 BER 为 **29.890%**，恢复后平均 BER 为 **9.926%**；标称 0 dB 的 10 条样本未观测到比特错误。
- 按 SINR 分档的 BER、MSE 等指标可通过 `signal-separation/` 中的正式脚本重新生成。MSE(dB) 采用**先平均线性 MSE，再转换为 dB**的口径。

### 任务二：YOLO11n + XCiT

联合流程为：TorchSig 宽带复数 IQ → 频谱图 → YOLO11n 检测框 → 按框截取 IQ → 负号下变频 → FIR 低通 → 多相重采样 → 复数 RMS 归一化 → XCiT 分类。

- **YOLO 独立评测**：50 张 TorchSig 验证风格时频图、195 个真实信号；Precision **94.32%**、Recall **92.82%**、mAP@0.5 **94.94%**、mAP@0.5:0.95 **82.82%**。
- **主要问题与处理**：旧官方 XCiT 在官方窄带输入和联合流程截取输入之间存在明显域差异。当前联合入口使用按照正式 IQ 提取链和扩大后的 TorchSig 数据微调的 `models/finetuned/xcit_joint_v4.ckpt`。
- **V4 训练与验证**：训练数据来自 Seed 51042 的 10,000 个宽带混合样本，共 39,951 个截取目标；验证数据来自独立 Seed 61042 的 2,000 个混合样本，共 7,950 个目标。最终 V4 在验证集上的家族准确率为 **86.49%**，57 类细分类准确率为 **42.72%**。

Seed 161803 的 100 个宽带独立测试样本共有 407 个 GT，三组模型共享同一 YOLO 结果：TP/FP/FN=`381/32/26`。

| XCiT 权重 | 匹配目标家族准确率 | 匹配目标 57 类 Top-1 | 匹配目标 57 类 Top-5 | 全部 GT 57 类召回率 |
|---|---:|---:|---:|---:|
| 联合 V3 | 80.84% | 33.33% | 81.89% | 31.20% |
| **联合 V4（最终）** | 82.94% | **38.06%** | **83.46%** | **35.63%** |
| V4 家族联合目标（0.5） | **83.20%** | 37.27% | 83.20% | 34.89% |

这里的“家族准确率”表示对 ASK、PSK、QAM、FSK、OFDM 等调制家族的判断；“57 类 Top-1”要求识别到具体调制子类。最终按独立测试上的细分类 Top-1 选择纯 57 类目标 V4。以上均为本地生成样本上的结果，不是官方隐藏测试集成绩。

## 环境安装

两个任务需要两个独立环境，不能将所有依赖安装到同一个 Python 环境中：

| 环境 | Python | 用途 | 依赖文件 |
|---|---:|---|---|
| `rf-sionna` | 3.7.13 | MIT 数据生成、Sionna 解调和 BER 计算，当前按 CPU 运行 | `requirements-rf-sionna.txt` |
| `rf-spectrum` | 3.10.0 | WaveNet 推理、TorchSig 数据、YOLO 检测、XCiT 推理与微调 | `requirements-rf-spectrum.txt` |

### 1. 创建 Sionna 环境

```bash
conda create -n rf-sionna python=3.7.13 pip -y
conda activate rf-sionna
python -m pip install -r requirements-rf-sionna.txt
```

安装后可执行以下命令检查核心依赖：

```bash
python -c "import tensorflow, sionna, numpy, scipy, h5py; print(tensorflow.__version__, sionna.__version__)"
```

### 2. 创建 PyTorch/TorchSig 环境

```bash
conda create -n rf-spectrum python=3.10.0 pip -y
conda activate rf-spectrum
python -m pip install -r requirements-rf-spectrum.txt
```

`requirements-rf-spectrum.txt` 固定为 Lab-5090 当前验证过的 CUDA 12.8 PyTorch 版本，并从仓库中的 `spectrum-analysis/torchsig-models-1.0.0/` 安装本地 TorchSig Models 源码，因此安装命令应在 `RF-SepEdge` 根目录执行。目标机器必须具有兼容的 NVIDIA 驱动；如果只能使用其他 CUDA 版本，应先按照目标 CUDA 版本安装对应的 PyTorch，再安装文件中的其余依赖。

安装后可执行以下命令检查核心依赖和 CUDA：

```bash
python -c "import torch, torchsig, ultralytics; from torchsig_models.models import XCiTClassifier; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

服务器中的旧文件 `signal-separation/rfchallenge/rfsionna_env.yml` 和 `rftorch_env.yml` 来自 MIT 官方基线，包含原作者机器上的绝对路径及大量 Jupyter 依赖，仅作为历史参考；本项目复现应优先使用根目录的两份精简 requirements。

## 运行与复现

请进入对应子目录并参照各自 README：

- [WaveNet：数据生成 → 推理 → 官方解调 → SINR 汇总](signal-separation/README.md)
- [YOLO + XCiT：物理测试 → YOLO 评测 → 联合 V4 推理与评测 → 可选重新微调](spectrum-analysis/README.md)
- [网页展示数据：信号分离 BER/MSE 与频谱检测/分类结果](demo-results/README.md)

运行脚本前应按照上面的任务类型激活 `rf-sionna` 或 `rf-spectrum`。数据集和模型权重由 Git LFS 管理，克隆仓库前应先安装 Git LFS。

各脚本对部分输出目录设置了**非空目录拒绝覆盖**保护。重新运行时应选择新输出目录，或在确认备份后手动清空旧结果；不要直接将不同实验混入同一个目录。

## 技术边界与后续工作

- MIT WaveNet 使用已知的目标信号与干扰组合对应权重；测试的是受控生成条件下的信号分离与恢复，并非对任意未知 RF 信号的通用分离。
- 联合 V4 是“宽带检测框截取 IQ”输入域的专用权重，不是官方窄带模型的通用替代。`xcit_inference_v211.py` 继续使用官方窄带权重，联合入口默认使用 V4。
- 当前 V4 的家族识别已超过 82%，下一步优化重点仍是 57 类细粒度识别，而不是继续把早期域偏移基线当作当前结果。
- `AUDIT.md` 是后续优化前的历史静态审查记录；当前实现、性能和运行方式以本 README 及 `spectrum-analysis/README.md` 为准。
