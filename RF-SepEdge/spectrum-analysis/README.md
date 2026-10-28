# 频谱检测、IQ 提取与调制分类：YOLO11n + XCiT

状态（2026-10-28）：联合流程已完成频移修正、物理测试、扩大 TorchSig 数据微调和独立模型选择。默认联合权重为 `models/finetuned/xcit_joint_v4.ckpt`；官方窄带权重与联合 V3 历史基线均保留，可通过 `--xcit-weights` 显式选择。

## 处理链路

TorchSig 宽带复数 IQ → 频谱图 → YOLO11n 检测框 → 按时间/频率框截取 IQ → 负号下变频 → FIR 低通 → 多相重采样 → `2×4096` 复数 RMS 归一化 → XCiT 57 类预测及家族概率汇总。

## 文件与目录说明

| 文件或目录 | 作用 |
|---|---|
| `test_extract_iq_frequency_shift.py` | 不加载模型，使用已知正、负中心频率的复数信号验证下变频方向、零频峰值和滤波后能量。每次修改 `extract_iq()` 后应首先运行。 |
| `inference.py` | YOLO 单图检测入口，同时提供频谱图转换、框坐标、IoU 匹配等公共函数，供其他脚本导入。用于先确认宽带信号检测链路正常。 |
| `benchmark_yolo.py` | 批量生成宽带验证风格数据并独立评估 YOLO，输出 Precision、Recall 和 mAP。它不运行 XCiT。 |
| `xcit_inference_v211.py` | 在官方风格窄带数据上独立运行 XCiT，默认使用官方窄带 checkpoint。它用于验证原始窄带分类能力，不应默认换成联合 V4。 |
| `combined_inference.py` | 正式单图联合入口：YOLO 检测、IQ 截取、负号下变频、滤波、重采样、RMS 归一化和 XCiT 分类都在此串联；其中 `extract_iq()` 是唯一正式 IQ 提取实现。默认使用联合 V4。 |
| `benchmark_combined.py` | 在多张宽带样本上批量评估完整 YOLO→IQ→XCiT 流程，并报告检测、匹配目标分类和全部 GT 端到端召回率。默认使用联合 V4，也可通过参数切换为其他权重。 |
| `finetune_xcit_joint.py` | 包含 `prepare` 和 `train` 两个子命令：前者使用 GT 框生成联合输入数据，后者从指定 checkpoint 微调并另存副本。训练目标由 57 类细分类损失和调制家族损失组成，家族损失权重可配置；训练集和验证集必须使用不同 seed。 |
| `configs/wideband_clean_val_all.yaml` | 宽带 TorchSig 数据生成配置，供 YOLO、联合推理和联合训练数据准备使用。 |
| `configs/narrowband_clean_val_all.yaml` | 官方风格窄带数据配置，仅用于独立 XCiT 测试及窄带能力对照。 |
| `models/yolo11n_wideband_v1.0.0.pt` | 正式 YOLO11n 宽带检测权重。 |
| `models/xcit_narrowband_v1.0.0.ckpt` | 官方窄带 XCiT 权重，保留用于窄带推理和联合基线对照。 |
| `models/finetuned/xcit_joint_v4.ckpt` | 当前正式联合 XCiT 权重，由扩大后的 TorchSig 数据和纯 57 类目标微调得到，是联合入口默认模型。 |
| `models/finetuned/xcit_joint_v3.ckpt` | 上一版联合模型，仅保留作历史基线和复现实验起点。 |
| `data/` | 训练数据生成目录。大规模 `.npz` 可由固定 Seed 重新生成，不纳入最终仓库。 |
| `output/` | 本地评估与训练历史目录，不纳入 Git；服务器仅保留最终模型选择所需的少量结果。 |
| `torchsig-models-1.0.0/` | TorchSig Models 源码参考，用于核对 XCiT 结构和训练实现；正式命令依赖 `rf-spectrum` 环境中已安装的包。 |
| `README.md` | 记录处理链路、运行顺序、最终指标、模型适用范围和复现命令。 |

阶段性诊断、探针、消融脚本和旧代码备份已在结论确认后清理；相关结论与最终数值保留在本文档中。正式流程不依赖这些已删除文件。

## 当前结论

1. YOLO 不是联合分类低准确率的主要瓶颈。Seed 42 的 182 个相同匹配目标上，GT 框家族/细分类为 `52/182`、`25/182`，YOLO 框为 `55/182`、`27/182`。
2. 下变频必须使用 `exp(-2j*pi*center_hz*t)`；正、负频率物理测试均已通过。
3. 复数 RMS 归一化稳定有效。
4. 固定相对带宽、时间框扩展、FIR 长度/截止频率和三窗口平均均未同时在 Seed 42 与 Seed 2026 上稳定改善细分类，因此未合入正式 DSP。
5. 主要问题是官方窄带 XCiT 与宽带检测框截取 IQ 之间的输入分布差异。使用正式提取链路和更大规模 TorchSig 数据微调后，联合 V4 的细分类进一步提高。
6. 联合 V4 是联合截取输入的专用权重，不是官方窄带权重的通用替代；`xcit_inference_v211.py` 继续默认使用官方窄带权重，只有联合流程默认使用 V4。

## 最终模型选择：Seed 161803

V3、V4 纯 57 类目标和 V4 家族联合目标均使用相同的 100 个宽带样本、相同 YOLO 框、`conf=0.25`、匹配 `IoU=0.5` 和复数 RMS 归一化。该测试共有 407 个 GT，YOLO 结果固定为 TP/FP/FN=`381/32/26`。

| 权重 | 匹配目标家族准确率 | 匹配目标细分类 Top-1 | 匹配目标细分类 Top-5 | 全部 GT 细分类召回 |
|---|---:|---:|---:|---:|
| 联合 V3 | 308/381 = 80.84% | 127/381 = 33.33% | 312/381 = 81.89% | 127/407 = 31.20% |
| **联合 V4（最终）** | 316/381 = 82.94% | **145/381 = 38.06%** | **318/381 = 83.46%** | **145/407 = 35.63%** |
| V4 家族联合目标（0.5） | **317/381 = 83.20%** | 142/381 = 37.27% | 317/381 = 83.20% | 142/407 = 34.89% |

V4 训练数据来自 Seed 51042 的 10,000 个宽带混合样本，共 39,951 个 GT 框截取目标；验证数据来自独立 Seed 61042 的 2,000 个混合样本，共 7,950 个目标，二者均覆盖全部 57 类。纯 57 类目标模型的验证细分类/家族准确率为 `42.72%/86.49%`，家族联合目标模型为 `42.50%/86.69%`。最终按独立测试的细分类 Top-1 选择纯 57 类目标 V4。

仓库只保留正式 V4、历史 V3 和官方窄带权重。服务器本地只保留 Seed 161803 的三组模型选择结果与两组 V4 训练历史；大规模训练数据和其他中间输出可按下方固定 Seed 重新生成。

## 推荐运行顺序

下面分为“已有模型的正常运行流程”和“需要重新训练时的流程”。如果只是进行联合系统演示，完成步骤 1 后可以直接运行步骤 4；步骤 2、3 主要用于分别检查 YOLO 和官方窄带 XCiT。

### 0. 进入环境

```bash
cd ~/Sata2/lyy/cloud-edge/RF-SepEdge/spectrum-analysis
conda activate rf-spectrum
```

下面示例使用物理 GPU 0；如果机器上的空闲 GPU 编号不同，需要相应修改 `CUDA_VISIBLE_DEVICES`。

### 1. 先运行物理测试

```bash
PYTHONDONTWRITEBYTECODE=1 python test_extract_iq_frequency_shift.py
python -m py_compile \
  inference.py benchmark_yolo.py xcit_inference_v211.py \
  combined_inference.py benchmark_combined.py finetune_xcit_joint.py
```

这一步应最先执行。两个正负频率测试都通过且语法检查无报错后，再运行模型推理。

### 2. 单独检查 YOLO

先运行一张宽带样本：

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
  --seed 42 --device 0 \
  --out output/yolo_single_seed42_new
```

再运行批量检测评估：

```bash
CUDA_VISIBLE_DEVICES=0 python benchmark_yolo.py \
  --samples 50 --seed 42 --batch 8 --device 0 \
  --out output/yolo_batch50_seed42_new
```

`inference.py` 用于检查单张频谱图、预测框和匹配情况；`benchmark_yolo.py` 用于获得批量 Precision、Recall 和 mAP。二者都不执行调制分类。

### 3. 单独检查官方窄带 XCiT

```bash
CUDA_VISIBLE_DEVICES=0 python xcit_inference_v211.py \
  --samples 200 --seed 20261023 --device cuda \
  --out output/xcit_narrowband_official_new
```

该命令默认读取 `models/xcit_narrowband_v1.0.0.ckpt`，用于验证官方风格窄带分类能力。若只想查看某一类，可以额外使用例如 `--class-name qpsk`。

### 4. 运行单图联合流程

```bash
CUDA_VISIBLE_DEVICES=0 python combined_inference.py \
  --seed 42 --device 0 \
  --out output/combined_v4_seed42_new
```

这是一条完整演示命令，依次执行 YOLO 检测、IQ 提取和联合 V4 分类，并保存频谱图、框、分类候选及 JSON 结果。未指定 `--xcit-weights` 时默认使用 `models/finetuned/xcit_joint_v4.ckpt`。

### 5. 批量评估联合流程

```bash
CUDA_VISIBLE_DEVICES=0 python benchmark_combined.py \
  --samples 100 --seed 161803 --device 0 \
  --out output/combined_v4_seed161803_new
```

该命令用于正式统计联合系统指标，默认同时报告原始幅度输入和复数 RMS 归一化输入，正式结果以 `with_rms` 为主。输出包括 TP/FP/FN、匹配目标家族/细分类准确率、全部 GT 端到端召回率和逐框 JSON。

如需和官方窄带 checkpoint 做联合基线对照：

```bash
CUDA_VISIBLE_DEVICES=0 python benchmark_combined.py \
  --samples 100 --seed 161803 --device 0 \
  --xcit-weights models/xcit_narrowband_v1.0.0.ckpt \
  --out output/combined_official_seed161803_new
```

两次对照必须使用相同 `--samples`、`--seed`、`--conf` 和 `--iou`，并使用不同输出目录。

## 重新生成数据与微调

已有 V4 可以直接运行；只有在需要重新训练或继续优化细分类准确率时，才执行以下步骤。

本轮只使用已经接入的 TorchSig，没有引入外部数据集。训练集包含 10,000 个宽带混合样本和 39,951 个截取目标，验证集包含 2,000 个宽带混合样本和 7,950 个截取目标。

### 6. 分别准备训练集和验证集

```bash
python finetune_xcit_joint.py prepare \
  --mixtures 10000 --seed 51042 \
  --out data/joint_xcit_v4/train_seed51042_m10000.npz

python finetune_xcit_joint.py prepare \
  --mixtures 2000 --seed 61042 \
  --out data/joint_xcit_v4/val_seed61042_m2000.npz
```

`prepare` 使用 GT 框和正式 `extract_iq()` 生成 `2×4096` 输入。训练与验证必须采用不同 seed，且输出路径不能指向已有文件。

### 7. 微调并保存新的 checkpoint

```bash
CUDA_VISIBLE_DEVICES=0 python finetune_xcit_joint.py train \
  --train-data data/joint_xcit_v4/train_seed51042_m10000.npz \
  --val-data data/joint_xcit_v4/val_seed61042_m2000.npz \
  --base-checkpoint models/finetuned/xcit_joint_v3.ckpt \
  --out-checkpoint models/finetuned/xcit_joint_v4.ckpt \
  --history output/finetune_xcit_joint_v4/history.json \
  --epochs 10 --batch-size 32 \
  --learning-rate 5e-6 --family-loss-weight 0 --device cuda
```

总训练损失为 `57类交叉熵 + family_loss_weight × 家族交叉熵`。本轮比较了 `0` 与 `0.5`：后者家族准确率略高，但纯 57 类目标在独立测试上的细分类 Top-1 更高，因此正式 V4 使用 `--family-loss-weight 0`。训练从 V3 开始并按验证集细分类准确率保存最佳副本。

### 8. 用新的 checkpoint 做联合验证

```bash
CUDA_VISIBLE_DEVICES=0 python benchmark_combined.py \
  --samples 100 --seed 161803 --device 0 \
  --xcit-weights models/finetuned/xcit_joint_v4.ckpt \
  --out output/combined_v4_seed161803_new
```

Seed 161803 不参与训练或验证，用于统一比较 V3、V4 纯 57 类目标和 V4 家族联合目标。最终结果选择了细分类 Top-1 更高的纯 57 类目标 V4，并将其设为正式默认权重。

不同实验必须使用新的输出目录。不得覆盖官方模型、既有基线结果或其他随机种子的测试结果。
