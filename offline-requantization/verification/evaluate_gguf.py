from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


def args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an AIOps GGUF model served by llama-server")
    parser.add_argument("--url", default="http://127.0.0.1:8095/completion")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default="", help="HF tokenizer directory for raw /completion mode")
    parser.add_argument("--chat-api", action="store_true", help="Use /v1/chat/completions without Python model dependencies")
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-metrics", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--confidence-threshold", type=float, default=0.75)
    parser.add_argument("--consecutive-windows", type=int, default=1)
    parser.add_argument("--cooldown-seconds", type=int, default=300)
    return parser.parse_args()


def extract_json(text: str) -> dict:
    text = text.strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("model output is not a JSON object")
    return value


class AlarmEngine:
    def __init__(self, threshold: float, windows: int, cooldown: int) -> None:
        self.threshold = threshold
        self.windows = windows
        self.cooldown = timedelta(seconds=cooldown)
        self.streaks: dict[tuple[str, str], int] = {}
        self.last_alarm: dict[tuple[str, str], datetime] = {}

    def process(self, diagnosis: dict, observed_at: datetime, sample_id: str) -> dict | None:
        component = str(diagnosis.get("root_cause_component") or "unknown")
        fault_type = str(diagnosis.get("fault_type") or "unknown")
        key = (component, fault_type)
        qualifies = (
            bool(diagnosis.get("is_anomaly"))
            and bool(diagnosis.get("should_alert"))
            and float(diagnosis.get("confidence", 0.0)) >= self.threshold
        )
        if not qualifies:
            self.streaks[key] = 0
            return None
        self.streaks[key] = self.streaks.get(key, 0) + 1
        if self.streaks[key] < self.windows:
            return None
        previous = self.last_alarm.get(key)
        if previous and observed_at - previous < self.cooldown:
            return None
        self.last_alarm[key] = observed_at
        return {
            "alarm_id": f"alarm-{sample_id}",
            "observed_at": observed_at.isoformat(),
            "component": component,
            "fault_category": diagnosis.get("fault_category", "unknown"),
            "fault_type": fault_type,
            "severity": diagnosis.get("severity", "major"),
            "confidence": float(diagnosis.get("confidence", 0.0)),
            "evidence": diagnosis.get("evidence", []),
            "recommendation": diagnosis.get("recommendation", ""),
            "source_sample_id": sample_id,
        }


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def metrics_for(rows: list[dict], alarms: list[dict], latencies: list[float], elapsed: float) -> dict:
    tp = fp = tn = fn = 0
    valid = fault_correct = component_correct = anomaly_count = 0
    for row in rows:
        truth = row["truth"]
        pred = row["prediction"]
        valid += int(row["json_valid"])
        actual = bool(truth.get("is_anomaly"))
        predicted = bool(pred.get("is_anomaly"))
        tp += int(actual and predicted)
        fp += int(not actual and predicted)
        tn += int(not actual and not predicted)
        fn += int(actual and not predicted)
        if actual:
            anomaly_count += 1
            fault_correct += int(pred.get("fault_type") == truth.get("fault_type"))
            component_correct += int(pred.get("root_cause_component") == truth.get("root_cause_component"))
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return {
        "sample_count": len(rows),
        "detection": {
            "precision": precision, "recall": recall,
            "f1": safe_div(2 * precision * recall, precision + recall),
            "accuracy": safe_div(tp + tn, len(rows)),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        },
        "diagnosis": {
            "fault_type_accuracy_on_anomalies": safe_div(fault_correct, anomaly_count),
            "component_accuracy_on_anomalies": safe_div(component_correct, anomaly_count),
            "json_valid_rate": safe_div(valid, len(rows)),
        },
        "alerting": {
            "alarm_count": len(alarms),
            "alerted_sample_recall": safe_div(sum(
                1 for row in rows if row["alarm"] and row["truth"].get("is_anomaly")
            ), anomaly_count),
            "false_alarm_count": sum(1 for row in rows if row["alarm"] and not row["truth"].get("is_anomaly")),
        },
        "performance": {
            "wall_seconds": elapsed,
            "samples_per_second": safe_div(len(rows), elapsed),
            "latency_mean_seconds": statistics.fmean(latencies) if latencies else 0.0,
            "latency_p95_seconds": sorted(latencies)[max(0, math.ceil(len(latencies) * 0.95) - 1)] if latencies else 0.0,
            "peak_allocated_gib": None,
        },
    }


def main() -> None:
    args = args_parser()
    model = Path(args.model)
    if not model.is_file():
        raise FileNotFoundError(f"GGUF model not found: {model}")
    parsed_url = urlsplit(args.url)
    props_url = urlunsplit((parsed_url.scheme, parsed_url.netloc, "/props", "", ""))
    with urlopen(props_url, timeout=10) as response:
        server_props = json.load(response)
    served_model = Path(server_props["model_path"])
    if served_model.resolve() != model.resolve():
        raise RuntimeError(f"server loaded {served_model}, expected {model}")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = None
    if not args.chat_api:
        if not args.tokenizer:
            raise ValueError("--tokenizer is required for raw /completion mode")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    source_rows = [json.loads(line) for line in Path(args.test_data).read_text(encoding="utf-8").splitlines() if line.strip()]
    engine = AlarmEngine(args.confidence_threshold, args.consecutive_windows, args.cooldown_seconds)
    rows, alarms, latencies = [], [], []
    wall_start = time.perf_counter()
    for index, record in enumerate(source_rows, start=1):
        messages = [m for m in record["messages"] if m.get("role") != "assistant"]
        if args.chat_api:
            payload = {
                "model": "local", "messages": messages,
                "max_tokens": args.max_new_tokens,
                "temperature": 0, "stream": False,
            }
        else:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            payload = {
                "prompt": prompt, "n_predict": args.max_new_tokens,
                "temperature": 0, "cache_prompt": False,
            }
        body = json.dumps(payload).encode("utf-8")
        started = time.perf_counter()
        with urlopen(Request(args.url, data=body, headers={"Content-Type": "application/json"}), timeout=300) as response:
            result = json.load(response)
        latency = time.perf_counter() - started
        latencies.append(latency)
        generated = result["choices"][0]["message"]["content"] if args.chat_api else result["content"]
        try:
            prediction = extract_json(generated)
            json_valid = True
        except (ValueError, json.JSONDecodeError):
            prediction = {}
            json_valid = False
        truth = json.loads(next(m["content"] for m in reversed(record["messages"]) if m.get("role") == "assistant"))
        meta = record.get("meta", {})
        alarm = engine.process(prediction, datetime.fromisoformat(meta["window_end"]), str(meta.get("window_id", index))) if json_valid else None
        row = {
            "sample_id": meta.get("window_id", str(index)), "meta": meta,
            "truth": truth, "prediction": prediction,
            "raw_output": generated, "json_valid": json_valid,
            "latency_seconds": latency, "alarm": alarm,
        }
        rows.append(row)
        if alarm:
            alarms.append(alarm)
        print(f"[{index}/{len(source_rows)}] valid={json_valid} alarm={bool(alarm)} latency={latency:.3f}s", flush=True)
    metrics = metrics_for(rows, alarms, latencies, time.perf_counter() - wall_start)
    baseline = json.loads(Path(args.baseline_metrics).read_text(encoding="utf-8"))
    comparisons = {}
    for section, key in [
        ("detection", "f1"), ("detection", "accuracy"),
        ("diagnosis", "fault_type_accuracy_on_anomalies"),
        ("diagnosis", "component_accuracy_on_anomalies"),
        ("diagnosis", "json_valid_rate"),
        ("alerting", "alerted_sample_recall"),
    ]:
        original = baseline[section][key]
        actual = metrics[section][key]
        comparisons[f"{section}.{key}"] = {
            "baseline": original, "actual": actual,
            "retention": safe_div(actual, original) if original else None,
            "passes_80_percent": actual + 1e-9 >= original * 0.8,
        }
    metrics["quality_comparison"] = comparisons
    metrics["quality_passes_all"] = all(x["passes_80_percent"] for x in comparisons.values())
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output / "alarms.jsonl").open("w", encoding="utf-8") as handle:
        for alarm in alarms:
            handle.write(json.dumps(alarm, ensure_ascii=False) + "\n")
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
