# 信号分离与恢复：MIT Sionna + WaveNet

**状态（2026-10-23）**：QPSK 与 OFDM-QPSK 两类目标信号、CommSignal2 干扰的 Seed 42 本地基线已完成。该任务与 `spectrum-analysis` 独立运行。

## 数据、模型与环境

- 官方代码：`rfchallenge/`，MIT RF Challenge 0.2.0。
- 目标信号：`QPSK`（每条 5,120 个原始比特）、`OFDMQPSK`（每条 57,344 个原始比特）。
- 干扰：`dataset/testset1_frame/CommSignal2_test1_raw_data.h5`；生成器使用官方 Sionna 代码生成 SOI 并与干扰混合，保存 SOI/IQ/比特真值。
- 输入：每条 **40,960 个 complex64 IQ 点**；两类信号分别使用 `pretrained-models/WaveNet/dataset_qpsk_commsignal2_mixture_wavenet/weights.pt` 与 `pretrained-models/WaveNet/dataset_ofdmqpsk_commsignal2_mixture_wavenet/weights.pt`。
- 环境一：`rf-sionna`（Python 3.7.13、TensorFlow 2.8.2、Sionna 0.10.0；**CPU 生成/解调**）。
- 环境二：`rf-spectrum`（Python 3.10、PyTorch 2.9.1+cu128；**GPU WaveNet 推理**）。

## 最终复现命令

以下命令在 `signal-separation/` 目录依次执行。默认 GPU 示例是 `CUDA_VISIBLE_DEVICES=1`；如服务器 GPU 编号变化，请自行调整。

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
CUDA_VISIBLE_DEVICES=1 python mit_sionna_wavenet_infer.py \
  --source output/mit_sionna_seed42 \
  --out output/mit_sionna_seed42_infer --batch-size 1

# 3. Sionna 官方解调辅助函数、BER/MSE 评估
conda activate rf-sionna
CUDA_VISIBLE_DEVICES=-1 python mit_sionna_official_ber.py \
  --source output/mit_sionna_seed42 \
  --predictions output/mit_sionna_seed42_infer \
  --out output/mit_sionna_seed42_ber

# 4. 按 SINR 汇总
python summarize_sinr.py \
  --generated output/mit_sionna_seed42 \
  --evaluated output/mit_sionna_seed42_ber \
  --out output/mit_sionna_seed42_report
```

### 保留的四类输出

| 路径（均在 `output/` 下） | 内容 |
|---|---|
| `mit_sionna_seed42/` | 官方生成器的小规模混合 IQ、真值 SOI、比特真值与生成参数 |
| `mit_sionna_seed42_infer/` | 两种信号的 WaveNet 恢复 IQ `.npy` 与波形误差汇总 |
| `mit_sionna_seed42_ber/` | 逐样本 BER/MSE、解调比特与总览 |
| `mit_sionna_seed42_report/` | 最终按 7 档 SINR 分组的 `by_sinr.csv`、`summary.json` |

## 结果：Seed 42 / 每档 10 条

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
