# 频谱检测、IQ 提取与调制分类：YOLO11n + XCiT

状态（2026-10-23）：联合流程已完成频移修正、物理测试、输入适配消融和 XCiT 联合输入微调。默认联合权重为 `models/finetuned/xcit_joint_v3.ckpt`；官方窄带权重 `models/xcit_narrowband_v1.0.0.ckpt` 保留不变，可通过 `--xcit-weights` 显式选择。

## 处理链路

TorchSig 宽带复数 IQ → 频谱图 → YOLO11n 检测框 → 按时间/频率框截取 IQ → 负号下变频 → FIR 低通 → 多相重采样 → `2×4096` 复数 RMS 归一化 → XCiT 57 类预测及家族概率汇总。

## 文件与目录说明

| 文件或目录 | 作用 |
|---|---|
| `test_extract_iq_frequency_shift.py` | 不加载模型，使用已知正、负中心频率的复数信号验证下变频方向、零频峰值和滤波后能量。每次修改 `extract_iq()` 后应首先运行。 |
| `inference.py` | YOLO 单图检测入口，同时提供频谱图转换、框坐标、IoU 匹配等公共函数，供其他脚本导入。用于先确认宽带信号检测链路正常。 |
| `benchmark_yolo.py` | 批量生成宽带验证风格数据并独立评估 YOLO，输出 Precision、Recall 和 mAP。它不运行 XCiT。 |
| `xcit_inference_v211.py` | 在官方风格窄带数据上独立运行 XCiT，默认使用官方窄带 checkpoint。它用于验证原始窄带分类能力，不应默认换成联合 V3。 |
| `combined_inference.py` | 正式单图联合入口：YOLO 检测、IQ 截取、负号下变频、滤波、重采样、RMS 归一化和 XCiT 分类都在此串联；其中 `extract_iq()` 是唯一正式 IQ 提取实现。默认使用联合 V3。 |
| `benchmark_combined.py` | 在多张宽带样本上批量评估完整 YOLO→IQ→XCiT 流程，并报告检测、匹配目标分类和全部 GT 端到端召回率。默认使用联合 V3，也可通过参数切换为官方权重。 |
| `finetune_xcit_joint.py` | 包含 `prepare` 和 `train` 两个子命令：前者使用 GT 框生成联合输入数据，后者从指定 checkpoint 微调并另存副本。训练集和验证集必须使用不同 seed。 |
| `configs/wideband_clean_val_all.yaml` | 宽带 TorchSig 数据生成配置，供 YOLO、联合推理和联合训练数据准备使用。 |
| `configs/narrowband_clean_val_all.yaml` | 官方风格窄带数据配置，仅用于独立 XCiT 测试及窄带能力对照。 |
| `models/yolo11n_wideband_v1.0.0.pt` | 正式 YOLO11n 宽带检测权重。 |
| `models/xcit_narrowband_v1.0.0.ckpt` | 官方窄带 XCiT 权重，保留用于窄带推理和联合基线对照。 |
| `models/finetuned/xcit_joint_v3.ckpt` | 联合系统专用 XCiT 权重，是 `combined_inference.py` 和 `benchmark_combined.py` 的默认分类模型。 |
| `data/joint_xcit_v2/` | 当前保留的联合 V3 训练集和验证集，可直接用于续训或复现训练。 |
| `output/` | 保存推理图像、JSON、评估结果和训练历史；脚本通常拒绝覆盖非空结果目录，因此每次运行应使用新路径。 |
| `torchsig-models-1.0.0/` | TorchSig Models 源码参考，用于核对 XCiT 结构和训练实现；正式命令依赖 `rf-spectrum` 环境中已安装的包。 |
| `README.md` | 记录处理链路、运行顺序、最终指标、模型适用范围和复现命令。 |

阶段性诊断、探针、消融脚本和旧代码备份已在结论确认后清理；相关结论与最终数值保留在本文档中。正式流程不依赖这些已删除文件。

## 当前结论

1. YOLO 不是联合分类低准确率的主要瓶颈。Seed 42 的 182 个相同匹配目标上，GT 框家族/细分类为 `52/182`、`25/182`，YOLO 框为 `55/182`、`27/182`。
2. 下变频必须使用 `exp(-2j*pi*center_hz*t)`；正、负频率物理测试均已通过。
3. 复数 RMS 归一化稳定有效。
4. 固定相对带宽、时间框扩展、FIR 长度/截止频率和三窗口平均均未同时在 Seed 42 与 Seed 2026 上稳定改善细分类，因此未合入正式 DSP。
5. 主要问题是官方窄带 XCiT 与宽带检测框截取 IQ 之间的输入分布差异。使用最终提取链路微调权重副本后，两组独立联合测试均显著提高。
6. 联合 V3 是联合截取输入的专用权重，不是官方窄带权重的通用替代。在 Seed 20261023 的同一批 200 个官方风格窄带样本上，官方权重细分类为 `79/200 = 39.5%`，联合 V3 为 `24/200 = 12.0%`。因此 `xcit_inference_v211.py` 继续默认使用官方窄带权重，只有联合流程默认使用 V3。

## 最终独立联合测试

所有结果均使用相同 YOLO 框、`conf=0.25`、匹配 `IoU=0.5` 和复数 RMS 归一化；微调前后检测结果完全相同。

| 测试 | 权重 | TP/FP/FN | 匹配目标家族准确率 | 匹配目标细分类 Top-1 | 全部 GT 家族召回 | 全部 GT 细分类召回 |
|---|---|---:|---:|---:|---:|---:|
| Seed 314159，196 GT | 官方窄带 | 183/14/13 | 51/183 = 27.87% | 15/183 = 8.20% | 51/196 = 26.02% | 15/196 = 7.65% |
| Seed 314159，196 GT | 联合 V3 | 183/14/13 | **148/183 = 80.87%** | **59/183 = 32.24%** | **148/196 = 75.51%** | **59/196 = 30.10%** |
| Seed 271828，198 GT | 官方窄带 | 188/16/10 | 64/188 = 34.04% | 26/188 = 13.83% | 64/198 = 32.32% | 26/198 = 13.13% |
| Seed 271828，198 GT | 联合 V3 | 188/16/10 | **150/188 = 79.79%** | **64/188 = 34.04%** | **150/198 = 75.76%** | **64/198 = 32.32%** |

V3 训练数据来自 Seed 31042 的 2,000 个宽带混合样本，共 8,033 个 GT 框截取目标；验证数据来自独立 Seed 41042 的 400 个混合样本，共 1,610 个目标。两者均覆盖全部 57 类，且不与上述最终测试流重叠。V3 在 GT 框验证集上的细分类准确率为 `608/1610 = 37.76%`，家族准确率为 `1336/1610 = 82.98%`。

为控制目录规模，当前只保留两组最终 holdout 的官方/V3 对照、官方/V3 窄带适用边界对照、V2/V3 训练历史和一组默认 V3 单图冒烟结果，其余中间实验输出均已删除，必要时可按命令重新生成。

## 推荐运行顺序

下面分为“已有模型的正常运行流程”和“需要重新训练时的流程”。如果只是进行联合系统演示，完成步骤 1 后可以直接运行步骤 4；步骤 2、3 主要用于分别检查 YOLO 和官方窄带 XCiT。

### 0. 进入环境

```bash
cd ~/Sata2/lyy/cloud-edge/RF-SepEdge/spectrum-analysis
conda activate rf-spectrum
```

默认使用第二张可见 GPU；如果机器上的空闲 GPU 编号不同，需要相应修改 `CUDA_VISIBLE_DEVICES`。

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
CUDA_VISIBLE_DEVICES=1 python inference.py \
  --seed 42 --device 0 \
  --out output/yolo_single_seed42_new
```

再运行批量检测评估：

```bash
CUDA_VISIBLE_DEVICES=1 python benchmark_yolo.py \
  --samples 50 --seed 42 --batch 8 --device 0 \
  --out output/yolo_batch50_seed42_new
```

`inference.py` 用于检查单张频谱图、预测框和匹配情况；`benchmark_yolo.py` 用于获得批量 Precision、Recall 和 mAP。二者都不执行调制分类。

### 3. 单独检查官方窄带 XCiT

```bash
CUDA_VISIBLE_DEVICES=1 python xcit_inference_v211.py \
  --samples 200 --seed 20261023 --device cuda \
  --out output/xcit_narrowband_official_new
```

该命令默认读取 `models/xcit_narrowband_v1.0.0.ckpt`，用于验证官方风格窄带分类能力。若只想查看某一类，可以额外使用例如 `--class-name qpsk`。

### 4. 运行单图联合流程

```bash
CUDA_VISIBLE_DEVICES=1 python combined_inference.py \
  --seed 42 --device 0 \
  --out output/combined_v3_seed42_new
```

这是一条完整演示命令，依次执行 YOLO 检测、IQ 提取和联合 V3 分类，并保存频谱图、框、分类候选及 JSON 结果。未指定 `--xcit-weights` 时默认使用 `models/finetuned/xcit_joint_v3.ckpt`。

### 5. 批量评估联合流程

```bash
CUDA_VISIBLE_DEVICES=1 python benchmark_combined.py \
  --samples 50 --seed 271828 --device 0 \
  --out output/combined_v3_seed271828_new
```

该命令用于正式统计联合系统指标，默认同时报告原始幅度输入和复数 RMS 归一化输入，正式结果以 `with_rms` 为主。输出包括 TP/FP/FN、匹配目标家族/细分类准确率、全部 GT 端到端召回率和逐框 JSON。

如需和官方窄带 checkpoint 做联合基线对照：

```bash
CUDA_VISIBLE_DEVICES=1 python benchmark_combined.py \
  --samples 50 --seed 271828 --device 0 \
  --xcit-weights models/xcit_narrowband_v1.0.0.ckpt \
  --out output/combined_official_seed271828_new
```

两次对照必须使用相同 `--samples`、`--seed`、`--conf` 和 `--iou`，并使用不同输出目录。

## 重新生成数据与微调

已有 V3 可以直接运行；只有在需要重新训练或继续优化细分类准确率时，才执行以下步骤。

### 6. 分别准备训练集和验证集

```bash
python finetune_xcit_joint.py prepare \
  --mixtures 2000 --seed 50001 \
  --out data/joint_xcit_new/train_seed50001_m2000.npz

python finetune_xcit_joint.py prepare \
  --mixtures 400 --seed 50002 \
  --out data/joint_xcit_new/val_seed50002_m400.npz
```

`prepare` 使用 GT 框和正式 `extract_iq()` 生成 `2×4096` 输入。训练与验证必须采用不同 seed，且输出路径不能指向已有文件。

### 7. 微调并保存新的 checkpoint

```bash
CUDA_VISIBLE_DEVICES=1 python finetune_xcit_joint.py train \
  --train-data data/joint_xcit_new/train_seed50001_m2000.npz \
  --val-data data/joint_xcit_new/val_seed50002_m400.npz \
  --base-checkpoint models/xcit_narrowband_v1.0.0.ckpt \
  --out-checkpoint models/finetuned/xcit_joint_new.ckpt \
  --history output/finetune_xcit_joint_new/history.json \
  --epochs 10 --batch-size 32 \
  --learning-rate 1e-5 --device cuda
```

该命令不会覆盖基础权重，会按照验证集细分类准确率保存最佳副本。若从现有联合模型继续训练，可把 `--base-checkpoint` 改为 `models/finetuned/xcit_joint_v3.ckpt`，并适当降低学习率。

### 8. 用新的 checkpoint 做联合验证

```bash
CUDA_VISIBLE_DEVICES=1 python benchmark_combined.py \
  --samples 50 --seed 60001 --device 0 \
  --xcit-weights models/finetuned/xcit_joint_new.ckpt \
  --out output/combined_joint_new_seed60001
```

最终测试应使用未参与训练、验证和参数选择的新 seed。确认新权重在独立测试上优于 V3 后，才能考虑修改正式默认权重。

不同实验必须使用新的输出目录。不得覆盖官方模型、既有基线结果或其他随机种子的测试结果。
