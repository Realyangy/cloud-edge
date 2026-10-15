# AIOps 模型内网重新量化包

## 目标与边界

本包用于在内网云端重新生成 Qwen2.5-0.5B AIOps 模型的 Q6_K GGUF。最终制品供
GTX 1050 2GB 边端使用。云端有两张昇腾 910B，但本次 GGUF
量化使用云端 **CPU**，不调用 NPU、CUDA 或 vLLM-Ascend。把模型搬到 910B4
并不会自动让当前 Q6_K 进入 vLLM-Ascend；那是另一条模型格式和推理后端链路。

## 包内文件

| 路径 | 用途 |
|---|---|
| `../model/` | AIOps 2020 微调后、LoRA 已合并的 Hugging Face 完整模型 |
| `source/llama.cpp-091b6a7-source.tar.gz` | 固定提交的转换、量化、推理源码 |
| `quantize_offline.sh` | HF 模型转 F16 GGUF、构建 CPU 工具并执行 Q6_K 量化 |
| `verification/` | 六条测试数据、基线指标及回放脚本 |

源模型是 `cloude-edge/model/` 中的合并模型。运行时需要保持 `model/` 与
`offline-requantization/` 为同级目录；迁移到云侧时一并复制这两个目录。源码
提交为 `091b6a7fc561b78da90a0ed2d524ec995458c657`。

## 内网环境

推荐在与内网云服务器相同 CPU 架构和 Linux 发行版的环境执行。需预先离线安装
`bash`、`tar`、CMake、C++ 编译器及 Python 3。脚本从 Hugging Face 模型开始转换，
还需要源码中的 `requirements/requirements-convert_hf_to_gguf.txt` 依赖，包括
PyTorch 2.11.0、NumPy 2.2.6、Transformers 4.57.6、sentencepiece、gguf 和
protobuf。断网环境需提前准备与 Kylin ARM64、Python 3.10 匹配的安装包；当前包
没有附带这些 wheel。此流程不需要 CUDA Toolkit、CANN 或网络。

建议云端系统 RAM 至少 16GB，工作目录至少留 6GB 空间；这包含源码解包、编译
与输出文件的余量，不是 NPU 显存需求。目标边端另需 NVIDIA 驱动以及支持
GTX 1050 (sm_61) 的 CUDA 12.x 推理构建。边端 16GB RAM 与 GTX 1050 的
2GB 显存不能相加。

## 在内网重新量化

保持两个目录同级后，在 `offline-requantization/` 中运行：

```bash
bash quantize_offline.sh /data/aiops-quantized
```

脚本从 `../model/` 读取完整微调模型，先生成临时
F16 GGUF，再构建并调用 CPU 版 `llama-quantize` 生成
`aiops-qwen25-0.5b-Q6_K.gguf`。脚本不会覆盖已有输出，结束后自动删除临时
转换和构建目录。若云端是 ARM64，CMake 会在该机器上重新编译；不要从 x86_64
复制编译后的二进制。

`Q6_K` 是量化请求的类型，不代表全部张量均为 6 位。本模型有 144/290 个张量
回退到 Q8_0；量化日志记录的实际平均值约 8.09 bit/weight，文件大小约
482 MiB。容量和质量判断应以生成的文件及实测指标为准。

## 验收与部署

量化后把生成的 GGUF、`source/`、`verification/` 复制到 GTX 1050 终端。
云端 910B4 的算力和显存不能推出边端速度。终端需有 NVIDIA 驱动、支持
Pascal sm_61 的 CUDA 12.x Toolkit、CMake 和 C++ 编译器。这些系统依赖也须
提前离线准备；本包不含驱动和 Toolkit。在终端执行（以 `/data/aiops` 为包目录）：

```bash
cd /data/aiops
tar -xzf source/llama.cpp-091b6a7-source.tar.gz
cmake -S llama.cpp -B llama.cpp/build \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61 \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_USE_PREBUILT_UI=OFF
cmake --build llama.cpp/build --target llama-server llama-bench -j 4
CUDA_VISIBLE_DEVICES=0 llama.cpp/build/bin/llama-server \
  -m /data/aiops-quantized/aiops-qwen25-0.5b-Q6_K.gguf \
  -ngl 99 -c 3072 -b 128 -ub 64 -np 1 \
  --host 127.0.0.1 --port 8095 --no-ui
```

另开终端检查显存并回放：

```bash
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv
python3 verification/evaluate_gguf.py --chat-api \
  --url http://127.0.0.1:8095/v1/chat/completions \
  --model /data/aiops-quantized/aiops-qwen25-0.5b-Q6_K.gguf \
  --test-data verification/test.jsonl \
  --baseline-metrics verification/baseline-metrics.json \
  --output-dir /data/aiops-gtx1050-test
```

目标为单并发、3072 tokens context、128 batch、64 ubatch。如果显存不足，
逐步降低 `-ngl`，并重新测端侧时延。六条样本上的 F1 和准确率在原服务器上
与合并模型一致，但这不是完整数据集保证。应记录目标机空闲显存、峰值显存、
每条时延和 `quality_passes_all`。终端实测前，不能宣称 2GB 显存和性能目标
已达标。若量化文件要进入平台模型仓库，登记新路径及 SHA-256，不要沿用
本机绝对路径。

若云端还要用 910B4 做推理，应另行验证 llama.cpp CANN 支持的 GGUF 类型，
或按 vLLM-Ascend 支持的量化流程从 HF 合并模型生成专用制品；当前 Q6_K
交付目标是 GTX 1050 CUDA 边端。
