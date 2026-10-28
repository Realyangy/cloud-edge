# 信号分离与恢复：MIT Sionna + WaveNet

**状态（2026-10-28）**：QPSK 与 OFDM-QPSK 两类目标信号、四种现有干扰的 Seed 42 本地实验均已完成。生成、推理和 BER 脚本支持按 `--interference` 选择干扰。该任务与 `spectrum-analysis` 独立运行。

## 数据、模型与环境

- 官方代码：`rfchallenge/`，MIT RF Challenge 0.2.0。
- 目标信号：`QPSK`（每条 5,120 个原始比特）、`OFDMQPSK`（每条 57,344 个原始比特）。
- 干扰：支持 `CommSignal2`、`CommSignal3`、`CommSignal5G1`、`EMISignal1`；对应 HDF5 位于 `dataset/testset1_frame/`。生成器使用官方 Sionna 代码生成 SOI 并与所选干扰混合，保存 SOI/IQ/比特真值。
- 输入：每条 **40,960 个 complex64 IQ 点**；两类目标和四种干扰的八组预训练权重位于 `pretrained-models/WaveNet/dataset_<soi>_<interference>_mixture_wavenet/weights.pt`。
- 环境一：`rf-sionna`（Python 3.7.13、TensorFlow 2.8.2、Sionna 0.10.0；**CPU 生成/解调**）。
- 环境二：`rf-spectrum`（Python 3.10、PyTorch 2.9.1+cu128；**GPU WaveNet 推理**）。

## 目录结构

```text
signal-separation/
├── dataset/                   # 四类原始干扰与 TestSet1 干扰帧
├── pretrained-models/         # 官方 U-Net、WaveNet 预训练权重
├── rfchallenge/               # MIT RF Challenge 0.2.0 官方代码
├── output/                    # 本地生成、恢复、BER 与汇总结果（Git 忽略）
├── mit_sionna_official_smoke.py   # 生成混合 IQ、目标 IQ 和比特真值
├── mit_sionna_wavenet_infer.py    # 调用对应 WaveNet 权重恢复目标 IQ
├── mit_sionna_official_ber.py     # 官方解调辅助函数与 BER/MSE 评估
├── summarize_sinr.py              # 按标称 SINR 汇总逐样本结果
└── README.md
```

正式流程只需要依次运行上述四个顶层脚本。`rfchallenge/` 保留官方实现供脚本调用和结果溯源，`dataset/` 与 `pretrained-models/` 不应在实验过程中改写。

## 最终复现命令

以下命令在 `signal-separation/` 目录依次执行。默认 GPU 示例是 `CUDA_VISIBLE_DEVICES=0`；如服务器 GPU 编号或占用情况变化，请自行调整。

```bash
cd ~/Sata2/lyy/cloud-edge/RF-SepEdge/signal-separation

# 1. 官方 Sionna 生成：2 种目标 x 7 档 SINR x 10 条 = 140 条
conda activate rf-sionna
CUDA_VISIBLE_DEVICES=-1 python mit_sionna_official_smoke.py \
  --soi-types QPSK,OFDMQPSK --interference CommSignal2 \
  --samples-per-sinr 10 --sinrs=-24,-18,-15,-12,-9,-6,0 \
  --seed 42 --out output/mit_sionna_seed42

# 2. WaveNet 恢复 IQ
conda activate rf-spectrum
CUDA_VISIBLE_DEVICES=0 python mit_sionna_wavenet_infer.py \
  --interference CommSignal2 \
  --source output/mit_sionna_seed42 \
  --out output/mit_sionna_seed42_infer --batch-size 1

# 3. Sionna 官方解调辅助函数、BER/MSE 评估
conda activate rf-sionna
CUDA_VISIBLE_DEVICES=-1 python mit_sionna_official_ber.py \
  --interference CommSignal2 \
  --source output/mit_sionna_seed42 \
  --predictions output/mit_sionna_seed42_infer \
  --out output/mit_sionna_seed42_ber

# 4. 按 SINR 汇总
python summarize_sinr.py \
  --generated output/mit_sionna_seed42 \
  --evaluated output/mit_sionna_seed42_ber \
  --out output/mit_sionna_seed42_report
```

`mit_sionna_wavenet_infer.py` 和 `mit_sionna_official_ber.py` 会同时读取生成阶段的 `summary.json`，并检查命令行干扰类型是否与数据一致，防止误用其他干扰对应的权重或文件。

## 四种干扰的完整批量复现

GPU 空闲后，可在 `signal-separation/` 目录运行下面的循环。每种干扰分别生成独立输出目录；如目录已有结果，请更换输出目录名，脚本不会覆盖非空目录。

```bash
for interference in CommSignal2 CommSignal3 CommSignal5G1 EMISignal1; do
  slug=$(printf '%s' "$interference" | tr '[:upper:]' '[:lower:]')

  conda run -n rf-sionna env CUDA_VISIBLE_DEVICES=-1 \
    python mit_sionna_official_smoke.py \
      --soi-types QPSK,OFDMQPSK --interference "$interference" \
      --samples-per-sinr 10 --sinrs=-24,-18,-15,-12,-9,-6,0 \
      --seed 42 --out "output/${slug}_seed42"

  conda run -n rf-spectrum env CUDA_VISIBLE_DEVICES=0 \
    python mit_sionna_wavenet_infer.py \
      --interference "$interference" \
      --source "output/${slug}_seed42" \
      --out "output/${slug}_seed42_infer" --batch-size 1

  conda run -n rf-sionna env CUDA_VISIBLE_DEVICES=-1 \
    python mit_sionna_official_ber.py \
      --interference "$interference" \
      --source "output/${slug}_seed42" \
      --predictions "output/${slug}_seed42_infer" \
      --out "output/${slug}_seed42_ber"

  conda run -n rf-sionna \
    python summarize_sinr.py \
      --generated "output/${slug}_seed42" \
      --evaluated "output/${slug}_seed42_ber" \
      --out "output/${slug}_seed42_report"
done
```

首次在新环境执行时，可以先将 `--samples-per-sinr 10` 临时改为 `1`，确认数据生成、权重选择、推理和解调均正常，再运行完整规模。

### 每类实验的四阶段输出

以下以 `<name>` 表示干扰的小写名称，例如 `commsignal3`。早期保留的 CommSignal2 基线使用兼容目录名 `mit_sionna_seed42*`。

| 路径（均在 `output/` 下） | 内容 |
|---|---|
| `<name>_seed42/` | 官方生成器的小规模混合 IQ、真值 SOI、比特真值与生成参数 |
| `<name>_seed42_infer/` | 两种信号的 WaveNet 恢复 IQ `.npy` 与波形误差汇总 |
| `<name>_seed42_ber/` | 逐样本 BER/MSE、解调比特与总览 |
| `<name>_seed42_report/` | 最终按 7 档 SINR 分组的 `by_sinr.csv`、`summary.json` |

## 四类干扰的总体结果：Seed 42 / 每档 10 条

每个干扰包含 QPSK 与 OFDM-QPSK，各使用 7 档 SINR、每档 10 条，即每类干扰共 140 条记录。

| 干扰 | 目标信号 | 输入 BER | WaveNet BER |
|---|---|---:|---:|
| CommSignal2 | QPSK | 17.291% | **1.724%** |
| CommSignal2 | OFDM-QPSK | 29.890% | **9.926%** |
| CommSignal3 | QPSK | 24.555% | **18.683%** |
| CommSignal3 | OFDM-QPSK | 32.576% | **28.965%** |
| CommSignal5G1 | QPSK | 27.194% | **1.224%** |
| CommSignal5G1 | OFDM-QPSK | 13.176% | **3.156%** |
| EMISignal1 | QPSK | 32.247% | **0.033%** |
| EMISignal1 | OFDM-QPSK | 13.855% | **1.568%** |

WaveNet 对四类干扰均降低了 BER，但 `CommSignal3` 的改善明显较弱。当前统计表明它具有宽带、持续且高度平稳的通信干扰特征，与目标波形特别是 OFDM-QPSK 更难分离。以上是本地 Seed 42 小规模实验，不是官方隐藏测试集成绩；后续如需形成稳定结论，应增加独立随机种子。

### CommSignal2 分档结果

| 标称 SINR | QPSK 输入 BER | QPSK WaveNet BER | OFDM-QPSK 输入 BER | OFDM-QPSK WaveNet BER |
|---:|---:|---:|---:|---:|
| -24 dB | 36.111% | 11.137% | 44.282% | 33.397% |
| -18 dB | 27.287% | 0.873% | 39.014% | 19.228% |
| -15 dB | 22.357% | 0.059% | 35.004% | 11.713% |
| -12 dB | 16.934% | **0.000%** | 30.716% | 3.853% |
| -9 dB | 11.465% | **0.000%** | 26.222% | 1.145% |
| -6 dB | 6.865% | **0.000%** | 21.513% | 0.148% |
| 0 dB | 0.016% | **0.000%** | 12.478% | **0.000%** |

跨全部 70 条样本：QPSK 输入 BER 17.291%，WaveNet BER **1.724%**；OFDM-QPSK 输入 BER 29.890%，WaveNet BER **9.926%**。这两个总体值混合了不同 SINR，仅供整体描述，正式分析应优先采用分档结果。

- **BER**：解调错误比特占比，越低越好。`0.000%` 指这次测试未观测到误码，而不是理论 BER 恒为 0。
- **MSE**：波形恢复误差；`summarize_sinr.py` 按档位先求线性 MSE 均值再转 dB。**NMSE** 为相对真实 SOI 功率归一化后的误差，MSE 与 NMSE 不可混称。
- **标称 SINR 与实际 SINR**：由于输入混合的功率定义/实测干扰帧，OFDM 与 QPSK 实际 SINR 略有不同。`by_sinr.csv` 按生成时的**标称档位**汇总。
- **口径**：自行调用官方 Sionna 生成器、已发布的官方 WaveNet 权重以及官方解调辅助函数，使用本地 TestSet1 干扰帧构造样本；**并非官方隐藏测试集/正式榜单得分**。

TensorFlow 在 `CUDA_VISIBLE_DEVICES=-1` 下报告 `CUDA_ERROR_NO_DEVICE`，但只要数据和解调正常生成，不影响 CPU 操作。

实验输出目录非空时部分脚本会拒绝覆盖。更新实验请使用新目录；旧版 PyTorch checkpoint 使用 `weights_only=False`，只对可信官方权重操作。
