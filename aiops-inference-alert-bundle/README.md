# AIOps 故障检测与告警推理包

本包只用于推理和告警验证，不包含训练代码，也不需要原始 Qwen 基础模型或 LoRA 适配器。
`../model/` 已经是 Qwen2.5-0.5B 与 LoRA 合并后的独立模型，与本包目录同级。

## 目录

```text
../model/                      合并后的完整模型（与本目录同级）
test_inference/run_test.py     测试集批量推理入口
test_inference/alarm_engine.py 告警阈值、连续窗口和冷却策略
data/processed/test.jsonl      固定测试集
data/processed/test_incidents.json
reference_results/             A100 上已经完成的参考验证结果
requirements.txt               推理依赖
```

## 安装依赖

推荐 Python 3.11：

```bash
pip install -r requirements.txt
```

## GPU 推理

在包的根目录执行：

```bash
python test_inference/run_test.py \
  --model ../model \
  --test-data data/processed/test.jsonl \
  --gpu 0 \
  --device cuda \
  --dtype auto
```

程序默认会在本包的 `output/test-时间戳/` 下生成结果，
也可以通过 `--output-dir` 指定其他目录：

- `predictions.jsonl`：逐条模型输出与真实标签；
- `alarms.jsonl`：告警引擎最终产生的告警；
- `metrics.json`：检测、诊断、告警、时延和显存指标；
- `test.log`：本次运行所用模型和数据路径。

对不支持 BF16 的老显卡可明确指定 `--dtype float16`。CPU 测试可使用
`--device cpu --dtype float32`，但速度会明显更慢。

## 更换另一个完整模型

代码无需修改，只替换模型路径：

```bash
python test_inference/run_test.py \
  --model /absolute/path/to/another_model \
  --test-data data/processed/test.jsonl \
  --device cuda \
  --gpu 0
```

新模型需要能够被 Transformers 的 `AutoModelForCausalLM` 和 `AutoTokenizer` 加载，
并能按照测试提示输出约定的 JSON 字段。

