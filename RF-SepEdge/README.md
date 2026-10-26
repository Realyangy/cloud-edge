# RF-SepEdge：射频信号分析与恢复实验项目

> 基线状态：2026-10-23 已完成固定随机种子 42 的本地复现实验。本文档描述**当前项目**，不再是源码补丁包的安装说明。

项目包含两个**相互独立**的实验任务。目前并未实现由频谱识别任务自动触发 WaveNet 分离的统一端到端系统。

| 子项目 | 目标 | 主要模型与依赖 | 状态 |
|---|---|---|---|
| [`signal-separation/`](signal-separation/README.md) | 从混合 IQ 中恢复目标 QPSK / OFDM-QPSK 信号并解调 | MIT RF Challenge 0.2.0 官方数据生成器、Sionna 0.10.0、WaveNet | Seed 42 基线已完成 |
| [`spectrum-analysis/`](spectrum-analysis/README.md) | 宽带频谱目标检测、IQ 提取、调制方式分类 | TorchSig 2.1.1、YOLO11n、XCiT | Seed 42 基线已完成；XCiT 分类仍待优化 |

## 代码与文件位置

```text
RF-SepEdge/
├── README.md
├── AUDIT.md                         # 2026-10-23 的历史源码审查记录
├── signal-separation/
│   ├── mit_sionna_official_smoke.py
│   ├── mit_sionna_wavenet_infer.py
│   ├── mit_sionna_official_ber.py
│   ├── summarize_sinr.py
│   ├── rfchallenge/                 # MIT 官方源码
│   ├── dataset/                     # MIT 数据
│   ├── pretrained-models/           # 官方预训练权重
│   ├── output/
│   └── README.md
└── spectrum-analysis/
    ├── inference.py
    ├── benchmark_yolo.py
    ├── xcit_inference_v211.py
    ├── combined_inference.py
    ├── benchmark_combined.py
    ├── xcit_conditions_benchmark.py # 可选诊断工具，不在正式主流程内
    ├── configs/
    ├── models/
    ├── torchsig-models-1.0.0/
    ├── output/
    └── README.md
```

## 已完成的实验结果（本地重生成样本）

### 任务一：WaveNet 信号分离

- **测试设计**：QPSK、OFDM-QPSK 两种目标信号；CommSignal2 干扰；7 档标称 SINR（-24/-18/-15/-12/-9/-6/0 dB），每种目标信号每档 10 条，合计 140 条；Seed 42。
- **QPSK**：70 条的平均输入 BER 为 **17.291%**，恢复后平均 BER 为 **1.724%**；标称 -12 dB 及以上的 40 条样本未观测到比特错误。
- **OFDM-QPSK**：70 条的平均输入 BER 为 **29.890%**，恢复后平均 BER 为 **9.926%**；标称 0 dB 的 10 条样本未观测到比特错误。
- 按 SINR 分档的 BER、MSE 等指标由 `signal-separation/output/mit_sionna_seed42_report/by_sinr.csv` 记录。MSE(dB) 采用**先平均线性 MSE，再转换为 dB**的口径。

### 任务二：YOLO11n + XCiT

- **YOLO 独立评测**：新生成的 50 张 TorchSig 验证风格时频图、195 个真实信号；Precision **94.32%**、Recall **92.82%**、mAP@0.5 **94.94%**、mAP@0.5:0.95 **82.82%**（Ultralytics 验证口径）。
- **联合评测**：50 张图、GT=195、预测框=198、TP=182、FP=16、FN=13；固定 conf=0.25、匹配 IoU=0.5 下 Precision **91.92%**、Recall **93.33%**。
- **XCiT 联合分类（使用 RMS）**：成功匹配的 182 个目标中，调制家族正确 **25/182=13.74%**，细粒度类别正确 **7/182=3.85%**；未使用 RMS 时分别为 **15/182=8.24%** 和 **6/182=3.30%**。**分类性能明显不足，不能以 YOLO 的 mAP 代替端到端分类性能。**

上述数值来自本地生成的样本，不是官方隐藏测试集成绩；单样本分类置信度也不等同于分类准确率。

## 运行与复现

请进入对应子目录，参照：

- [WaveNet：生成 → 推理 → 官方解调 → SINR 汇总](signal-separation/README.md)
- [YOLO + XCiT：两个展示样本 → YOLO 批量评估 → 联合批量评估](spectrum-analysis/README.md)

两个任务使用不同环境：MIT 的 Sionna 生成/解调阶段使用 `rf-sionna`（CPU）；MIT WaveNet 推理以及 TorchSig/YOLO/XCiT 使用 `rf-spectrum`（PyTorch GPU）。使用官方/可信来源的 checkpoint；旧 checkpoint 的 `weights_only=False` 涉及 pickle 反序列化风险。

各脚本对部分输出目录设置了**非空目录拒绝覆盖**保护。重新运行时应选择新输出目录，或在确认备份后手动清空旧结果；不要直接将不同实验混入同一个目录。

## 技术边界与后续工作

- MIT WaveNet 使用已知的目标信号与干扰组合对应权重；测试的是受控生成条件下的信号分离与恢复，并非对任意未知 RF 信号的通用分离。
- YOLO 能定位宽带信号，但当前 XCiT 对从宽带混合 IQ 提取的窄带信号识别能力有限。后续考虑核查通道化、采样率/带宽比、多窗口分类与训练/推理分布偏移。
- `AUDIT.md` 记录的是**代码整理时的历史静态审查**；本文档记录的是**后续服务器复现实验的当前状态**。
