# RF-SepEdge 网页展示结果数据

本目录保存两个任务已经运行得到的轻量结果，供后续网页读取和展示。这里只保留 JSON/CSV 指标及逐目标预测，不保存可重新生成的大型训练集、原始 IQ、WaveNet 中间数组或缓存。

## 目录结构

```text
demo-results/
├── manifest.json
├── showcase/
│   ├── mit-wavenet/                 # 24 条真实记录的波形、时频图和指标
│   └── yolo-xcit-v4/                # 4 张真实时频图、检测框与局部特征
├── yolo-detection/
│   └── seed42-50/summary.json        # 50 图 YOLO 正式检测指标
├── signal-separation/
│   ├── commsignal2/
│   ├── commsignal3/
│   ├── commsignal5g1/
│   └── emisignal1/
│       ├── summary.json
│       ├── by_sinr.csv
│       └── per_sample.csv
└── spectrum-analysis/
    ├── model-selection/
    │   ├── v3/
    │   ├── v4-final/
    │   └── v4-hier-w05/
    │       ├── summary.json
    │       ├── predictions.json
    │       └── extraction_errors.json
    └── training-history/
        ├── v4-final/history.json
        └── v4-hier-w05/history.json
```

## 信号分离结果

- `summary.json`：该干扰类型下 QPSK、OFDM-QPSK 各 SINR 档位的汇总指标和计算口径。
- `by_sinr.csv`：网页绘制 BER、MSE 随 SINR 变化曲线时优先读取该文件。
- `per_sample.csv`：每个测试样本的输入及 WaveNet 恢复结果，可用于明细表或误差分布图。
- 所有结果均为 Seed 42 的本地生成实验，不是官方隐藏测试集结果。

## 频谱分析结果

- `summary.json`：YOLO 检测指标、XCiT 家族/57 类准确率和端到端召回率。
- `predictions.json`：100 个宽带样本的逐检测框置信度、框位置、XCiT Top-1/Top-5、家族概率、IQ 提取信息及对应真实标签。
- `extraction_errors.json`：IQ 提取异常；当前三组模型选择实验均为空数组。
- `history.json`：V4 两种训练目标的逐 epoch 训练和验证曲线。
- 三组模型选择结果使用同一 Seed 161803 和同一批 YOLO 检测框，可以直接进行公平对比。

`v4-final/summary.json` 中记录的权重名 `xcit_joint_v4_flat.ckpt` 是实验运行时名称；该权重选为正式版本后已重命名为 `models/finetuned/xcit_joint_v4.ckpt`，模型内容未改变。

## 网页使用建议

网页应首先读取根目录的 `manifest.json`，其中已经索引所有展示样本、指标 JSON 和 PNG 的相对路径。

- `showcase/mit-wavenet/`：QPSK、OFDM-QPSK × 四种干扰 × `-18/-9/0 dB`，共 24 条代表性记录。每条记录的混合输入、目标真值、WaveNet 恢复结果均来自相同 `sample_index`，并提供复数 I/Q 波形、时频图、BER、MSE、实际 SINR 和数组 SHA-256。
- `showcase/yolo-xcit-v4/`：Seed 161803 的 Sample 8、54、60、61。真实时频图由同一 Seed 数据流恢复，检测框和分类结果直接复用 `model-selection/v4-final/predictions.json`，没有重新执行 YOLO 或 XCiT 推理。
- `yolo-detection/seed42-50/summary.json`：Seed 42、50 图、195 个 GT 的 YOLO 指标。旧生成目录清理后只保留了正式文档中的四位小数，因此未保留的 `mAP75` 没有补写或猜测。

频谱任务的总体统计建议使用 `spectrum-analysis/model-selection/v4-final/`；V3 和 `v4-hier-w05` 仅用于模型性能对比。信号分离总体曲线仍可按干扰类型读取 `signal-separation/*/by_sinr.csv`。
