# RF-SepEdge：射频信号分析与恢复实验项目

> 当前状态（2026-10-23）：信号分离基线已经完成；频谱分析联合流程已经完成频移修正、物理测试、联合输入域微调和两组独立测试。YOLO→IQ 提取→XCiT 联合流程默认使用 `xcit_joint_v3.ckpt`。

项目包含两个**相互独立**的实验任务。目前并未实现由频谱识别任务自动触发 WaveNet 分离的统一端到端系统。

| 子项目 | 目标 | 主要模型与依赖 | 状态 |
|---|---|---|---|
| [`signal-separation/`](signal-separation/README.md) | 从混合 IQ 中恢复目标 QPSK / OFDM-QPSK 信号并解调 | MIT RF Challenge 0.2.0 官方数据生成器、Sionna 0.10.0、WaveNet | Seed 42 基线已完成 |
| [`spectrum-analysis/`](spectrum-analysis/README.md) | 宽带频谱目标检测、IQ 提取、调制方式分类 | TorchSig 2.1.1、YOLO11n、XCiT | 联合 V3 已完成，两组独立测试的家族准确率约 80% |

## 代码与文件位置

```text
RF-SepEdge/
├── README.md
├── AUDIT.md                         # 代码整理前的历史源码审查记录
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
    ├── data/joint_xcit_v2/          # 联合 V3 训练集和验证集
    ├── models/                       # YOLO、官方 XCiT 和联合 V3 权重
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
- **主要问题与处理**：旧官方 XCiT 在官方窄带输入和联合流程截取输入之间存在明显域差异。当前联合入口使用按照正式 IQ 提取链微调的 `models/finetuned/xcit_joint_v3.ckpt`，不再使用早期 Seed 42 联合分类基线作为当前性能。
- **V3 训练与验证**：训练数据来自 Seed 31042 的 2,000 个宽带混合样本，共 8,033 个截取目标；验证数据来自独立 Seed 41042 的 400 个混合样本，共 1,610 个目标。V3 在验证集上的家族准确率为 **82.98%**，57 类细分类准确率为 **37.76%**。

两组未参与训练、验证和参数选择的最终联合测试如下。微调前后使用相同 YOLO 框，检测结果不因 XCiT 权重改变。

| 独立测试 | XCiT 权重 | TP/FP/FN | 匹配目标家族准确率 | 匹配目标 57 类 Top-1 | 全部 GT 家族召回率 | 全部 GT 57 类召回率 |
|---|---|---:|---:|---:|---:|---:|
| Seed 314159，196 GT | 官方窄带 | 183/14/13 | 27.87% | 8.20% | 26.02% | 7.65% |
| Seed 314159，196 GT | **联合 V3** | 183/14/13 | **80.87%** | **32.24%** | **75.51%** | **30.10%** |
| Seed 271828，198 GT | 官方窄带 | 188/16/10 | 34.04% | 13.83% | 32.32% | 13.13% |
| Seed 271828，198 GT | **联合 V3** | 188/16/10 | **79.79%** | **34.04%** | **75.76%** | **32.32%** |

这里的“家族准确率”表示对 ASK、PSK、QAM、FSK、OFDM 等调制家族的判断；“57 类 Top-1”要求识别到具体调制子类，难度更高。V3 已显著解决联合输入下的家族识别问题，但细粒度 57 类识别仍有继续优化空间。以上均为本地生成样本上的结果，不是官方隐藏测试集成绩。

## 运行与复现

请进入对应子目录并参照各自 README：

- [WaveNet：数据生成 → 推理 → 官方解调 → SINR 汇总](signal-separation/README.md)
- [YOLO + XCiT：物理测试 → YOLO 评测 → 联合 V3 推理与评测 → 可选重新微调](spectrum-analysis/README.md)

两个任务使用不同环境：MIT 的 Sionna 生成/解调阶段使用 `rf-sionna`（CPU）；MIT WaveNet 推理以及 TorchSig/YOLO/XCiT 使用 `rf-spectrum`（PyTorch GPU）。数据集和模型权重由 Git LFS 管理，克隆仓库前应先安装 Git LFS。

各脚本对部分输出目录设置了**非空目录拒绝覆盖**保护。重新运行时应选择新输出目录，或在确认备份后手动清空旧结果；不要直接将不同实验混入同一个目录。

## 技术边界与后续工作

- MIT WaveNet 使用已知的目标信号与干扰组合对应权重；测试的是受控生成条件下的信号分离与恢复，并非对任意未知 RF 信号的通用分离。
- 联合 V3 是“宽带检测框截取 IQ”输入域的专用权重，不是官方窄带模型的通用替代。`xcit_inference_v211.py` 继续使用官方窄带权重，联合入口默认使用 V3。
- 当前 V3 的家族识别已达到约 80%，下一步优化重点是 57 类细粒度识别，而不是继续把早期域偏移基线当作当前结果。
- `AUDIT.md` 是后续优化前的历史静态审查记录；当前实现、性能和运行方式以本 README 及 `spectrum-analysis/README.md` 为准。
