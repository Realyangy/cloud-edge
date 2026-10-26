# RF-SepEdge 源码审查记录（2026-10-23）

审查范围：用户上传的源码包，包含 `spectrum-analysis/*.py` 六个（其中 `xcit_conditions_benchmark.py` 属历史诊断）、`signal-separation/*.py` 三个，TorchSig 及 MIT 官方代码，配置文件。**仅完成静态审查和语法验证**，由于缺乏现场模型权重/完整软件环境，没有在本地进行 GPU 端到端复测。

## 核心结论

- 现有算法主链完整。YOLO→频谱框→原始 IQ 定位→频移/滤波/重采样→复数 RMS→XCiT；Sionna 数据生成→WaveNet 推理→Sionna 解调/BER。
- 已核对 `combined_inference.py` 的正号频移、复数 RMS，`benchmark_combined.py` 复用同一 `extract_iq()`。无需再次更换频移符号；请勿将这视为已解决 XCiT 分类域偏移。
- 全部 9 个项目级 Python 文件通过静态语法解析；本整理包 9 个脚本（含 1 个纯标准库报表工具）通过语法检查。模型加载与 GPU 执行仍以服务器上既有运行记录为准。

## 发现与处理

| 级别 | 发现 | 处理 |
|---|---|---|
| P1 | `mit_sionna_official_ber.py` 汇总 JSON 的描述恒为 `2-record MINI-set`，实际 140 条时失真 | 改为按 `len(all_rows)` 动态写入 |
| P1 | `mit_sionna_official_smoke.py`、`mit_sionna_wavenet_infer.py`、`mit_sionna_official_ber.py`、`combined_inference.py`、`inference.py`、`xcit_inference_v211.py` 可以覆写旧输出，可能导致不同 seed 的结果混杂 | 统一增加非空目录拒绝覆盖 |
| P2 | `benchmark_yolo.py` 控制台标题固定 `50-sample`，`--samples` 可变 | 改为动态 |
| P2 | YAML seed 仍为 `1234567891`；直接省略 `--seed` 会恢复旧种子 | 不改官方 YAML，README 强制每次 `--seed 42`/`43` |
| P2 | YOLO mAP 和联合固定阈值 P/R 口径不同 | 明确区分，分别记录数据集规模/置信度/IoU |
| P2 | `benchmark_combined.py` 是 RMS 前后对照；不能当独立 XCiT 官方 held-out 分类评测 | 保留用于基线审查，并记录依赖相同检测框 |
| P2 | MIT mini 生成器可创建伪官方 TestSet1Example，但使用 TestSet1 原始干扰生成；不是官方隐藏测试集 | README 标注为本地重生成验证 |
| 注意 | `xcit_conditions_benchmark.py` 并不是 `combined_inference` 的导入依赖 | 建议从最终入口归档，不必要打包 |

## 保留依赖

- `combined_inference.py` -> `inference.py` + `xcit_inference_v211.py`；`benchmark_combined.py` 依赖前两者及 `combined_inference.py`。
- MIT 3 脚本依赖 `rfchallenge/`、`pretrained-models/`、`dataset/testset1_frame/`；三个阶段使用两套 Conda 环境。
- 原始 MIT 官方 checkpoint 采用 Pickle/OmegaConf，`weights_only=False` **仅限确认来源可信的模型**；本整理不更改加载机制。

## 当前未修的研究问题

- XCiT 对从宽带时频框提取的窄带 IQ 的识别效果可能显著低于独立窄带测试；现在按用户要求保留模型、暂不优化算法。
- 目标/预测框 `evaluate()` 采用逐预测贪心 IoU 匹配（非全局最优二分匹配），如需严谨公布边缘案例指标，建议另做一致性测试；当前先不改原评估以保持可比性。
- `benchmark_combined` 和 `benchmark_yolo` 生成的 50 张图需核对真正使用相同生成序列才可以逐样本配对；即使 seed 一致，也不应默认不同程序的样本逐个一致。
- `mit_sionna_official_ber` 汇总没有按 SINR 保存分组表。可从 `per_sample.csv` 根据 Sionna 生成器顺序还原，本整理包已增加 `summarize_sinr.py` 独立报表工具，不改动原核心测评代码。
