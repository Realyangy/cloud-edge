"""云侧 Linux / 端侧 Windows 的设备采集；只读系统指标，不调用模型。"""
import csv
import io
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def run_command(args):
    result = subprocess.run(args, capture_output=True, text=True, errors="replace", timeout=12)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"退出码 {result.returncode}")
    return result.stdout


def number(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def parse_nvidia(text):
    devices = []
    for row in csv.reader(io.StringIO(text)):
        if len(row) != 7:
            continue
        index, name, total, used, free, temperature, utilization = [v.strip() for v in row]
        devices.append({"id": index, "kind": "GPU", "name": name,
                        "memory_total_mb": number(total), "memory_used_mb": number(used),
                        "memory_free_mb": number(free), "temperature_c": number(temperature),
                        "utilization_pct": number(utilization)})
    return devices


def parse_npu_table(text):
    """解析现场 npu-smi info 的两行设备表；HBM 和主机内存不混用。"""
    devices, pending = [], None
    for line in text.splitlines():
        if "Process id" in line or "Process name" in line:
            break
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        if len(cells) < 3:
            continue
        head = re.fullmatch(r"(\d+)\s+(\S+)", cells[0])
        if head:
            values = cells[2].split()
            if len(values) < 2:
                continue
            pending = {"id": head[1], "kind": "NPU", "name": head[2], "health": cells[1],
                       "power_w": number(values[0]), "temperature_c": number(values[1]),
                       "utilization_pct": None, "memory_total_mb": None,
                       "memory_used_mb": None, "memory_free_mb": None}
            devices.append(pending)
        elif pending is not None and re.fullmatch(r"\d+", cells[0]):
            pending["chip_id"] = cells[0]
            values = cells[2].split()
            pending["utilization_pct"] = number(values[0]) if values else None
            pairs = re.findall(r"([\d.]+)\s*/\s*([\d.]+)", cells[2])
            if pairs:
                # 910B 的最后一组是 HBM-Usage；有些板卡只有 Memory-Usage。
                used, total = map(float, pairs[-1])
                if total > 0:
                    pending.update(memory_total_mb=total, memory_used_mb=used,
                                   memory_free_mb=max(0, total - used))
            pending = None
    return devices


def cpu_memory_linux():
    def cpu_ticks():
        values = list(map(int, Path("/proc/stat").read_text().splitlines()[0].split()[1:9]))
        return sum(values), values[3] + values[4]
    before, idle_before = cpu_ticks()
    time.sleep(0.15)
    after, idle_after = cpu_ticks()
    cpu = round(100 * (1 - (idle_after - idle_before) / (after - before)), 1) if after > before else None
    mem = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        mem[key] = int(value.strip().split()[0])
    return {"cpu_percent": cpu, "ram_total_mb": mem.get("MemTotal", 0) / 1024,
            "ram_available_mb": mem.get("MemAvailable", 0) / 1024}


def cpu_memory_windows():
    # 优先使用端侧已有的 psutil；无该包时，使用 Windows 自带 PowerShell。
    try:
        import psutil
    except ImportError:
        script = ('$o=Get-CimInstance Win32_OperatingSystem; '
                  '$c=Get-CimInstance Win32_Processor; '
                  '@{cpu_percent=($c | Measure-Object LoadPercentage -Average).Average; '
                  'ram_total_mb=$o.TotalVisibleMemorySize/1024; '
                  'ram_available_mb=$o.FreePhysicalMemory/1024} | ConvertTo-Json -Compress')
        return json.loads(run_command(["powershell", "-NoProfile", "-Command", script]))
    memory = psutil.virtual_memory()
    return {"cpu_percent": psutil.cpu_percent(interval=0.15),
            "ram_total_mb": memory.total / 1024**2,
            "ram_available_mb": memory.available / 1024**2}


def collect_metrics(kind):
    metrics = {"sampled_at": now_iso(), "hostname": socket.gethostname(),
               "os": platform.system(), "architecture": platform.machine(),
               "cpu_count": os.cpu_count(), "accelerators": [], "errors": []}
    try:
        metrics.update(cpu_memory_windows() if os.name == "nt" else cpu_memory_linux())
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        metrics["errors"].append(f"CPU/内存采集失败：{exc}")
    name = "npu-smi" if kind == "NPU" else "nvidia-smi"
    executable = shutil.which(name)
    if not executable:
        candidate = (Path("/usr/local/Ascend/driver/tools/npu-smi") if kind == "NPU" else
                     Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "NVIDIA Corporation/NVSMI/nvidia-smi.exe")
        executable = str(candidate) if candidate.exists() else None
    if not executable:
        metrics["errors"].append(f"找不到 {name}，加速卡指标暂不可用")
        return metrics
    try:
        if kind == "NPU":
            raw = run_command([executable, "info"])
            metrics["accelerators"] = parse_npu_table(raw)
        else:
            fields = "index,name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu"
            raw = run_command([executable, "--query-gpu=" + fields, "--format=csv,noheader,nounits"])
            metrics["accelerators"] = parse_nvidia(raw)
        if not metrics["accelerators"]:
            metrics["errors"].append(f"{name} 输出未解析到设备，请查看原始输出")
            metrics["accelerator_raw_output"] = raw
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        metrics["errors"].append(f"{name} 采集失败：{exc}")
    return metrics
