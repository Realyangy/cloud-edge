"""Windows 端侧：采集、交接报告、接收决策与模型文件。不会执行修复或自动启动模型。"""
import argparse
import importlib.metadata
import json
import os
import platform
import socket
import sys
import threading
import time
import zipfile
from pathlib import Path
from urllib.parse import quote
from urllib.request import ProxyHandler, Request, build_opener
from uuid import uuid4

if __package__:
    from .telemetry import collect_metrics, now_iso, run_command
else:
    from telemetry import collect_metrics, now_iso, run_command


def log(message):
    print(f"[{now_iso()}] {message}", flush=True)


def save_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def request_json(cloud, path, data=None):
    request = Request(cloud + path, data=None if data is None else json.dumps(data, ensure_ascii=False).encode("utf-8"),
                      headers={"Content-Type": "application/json"})
    with build_opener(ProxyHandler({})).open(request, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def runtime_info():
    packages = {}
    for name in ("torch", "transformers", "sentence-transformers", "llama-cpp-python", "onnxruntime"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python_version": platform.python_version(), "python_executable": sys.executable,
            "packages_in_agent_environment": packages,
            "note": "这里只列采集程序所在 Python 环境；其他环境是否安装推理组件尚未确认"}


def edge_context(process_names):
    """链路状态只代表网卡与IP配置，不把已连接误写为能访问互联网。"""
    if os.name != "nt":
        return {"network": {"status": "未实现此系统的链路采集"}, "critical_software": {"selected": process_names, "status": "未知"}}
    script = ("$n=@(Get-NetAdapter | ForEach-Object { "
              "$a=$_; $ips=@(Get-NetIPAddress -InterfaceIndex $a.ifIndex -AddressFamily IPv4 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty IPAddress); "
              "@{name=$a.Name;status=[string]$a.Status;link_speed=[string]$a.LinkSpeed;ipv4=$ips} }); "
              "$p=@(Get-Process | Select-Object -ExpandProperty ProcessName); "
              "@{interfaces=$n;running_processes=$p} | ConvertTo-Json -Depth 5 -Compress")
    # 使用 UTF-8 输出，保留中文网卡名。
    script = "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; " + script
    data = json.loads(run_command(["powershell", "-NoProfile", "-Command", script]))
    running = {str(name).lower() for name in data["running_processes"]}
    selected = [{"name": name, "running": Path(name).stem.lower() in running} for name in process_names]
    return {"network": {"interfaces": data["interfaces"], "note": "网卡连接与IP配置状态，不代表端到端网络可达"},
            "critical_software": {"selected": selected, "note": "未配置关键软件" if not selected else "按进程名称观测，进程存在不等于业务健康"}}


def upload_reports(args, work):
    manifest_file = work / "uploaded_reports.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8")) if manifest_file.exists() else {}
    for path in sorted(Path(args.report_dir).glob("*")):
        if not path.is_file() or path.suffix.lower() not in (".json", ".txt"):
            continue
        # 交接目录的文件视为完整报告；同名报告更新使用大小与修改时间识别，不扫描模型文件。
        stat = path.stat()
        revision = [stat.st_mtime_ns, stat.st_size]
        key = path.name
        item = manifest.get(key)
        if item and item.get("revision") == revision and item.get("uploaded"):
            continue
        if not item or item.get("revision") != revision:
            item = {"revision": revision, "report_id": args.node_id + ":" + uuid4().hex, "uploaded": False}
            manifest[key] = item
            save_json(manifest_file, manifest)
        try:
            text = path.read_text(encoding="utf-8-sig")
            content = json.loads(text) if path.suffix.lower() == ".json" else text
            if isinstance(content, dict) and "report" in content and "node_id" in content:
                if content["node_id"] != args.node_id:
                    raise ValueError("转交包的设备编号与本机配置不一致")
                payload = {"node_id": args.node_id, "report": content["report"],
                           "report_id": content.get("report_id") or item["report_id"],
                           "observed_at": content.get("observed_at")}
            else:
                payload = {"node_id": args.node_id, "report": content, "report_id": item["report_id"]}
            result = request_json(args.cloud, "/api/reports", payload)
            item["uploaded"] = True
            item["cloud_report_id"] = result["report_id"]
            save_json(manifest_file, manifest)
            log("报告已上传：" + path.name)
        except Exception as exc:
            log("报告暂未上传，已保留本地文件 " + path.name + "：" + str(exc))


def receive_decisions(args, work):
    base = "/api/nodes/" + quote(args.node_id, safe="")
    records = request_json(args.cloud, base + "/decisions/pending")
    for record in records:
        identifier = record["decision_id"]
        destination = work / "decisions" / (identifier + ".json")
        save_json(destination, record)
        request_json(args.cloud, base + "/decisions/" + identifier + "/ack", {})
        log("云侧决策已保存：" + str(destination))
        log(json.dumps(record["result"], ensure_ascii=False))


def model_receiver(args, work, stop):
    """大模型下载独立运行，避免传输期间心跳停止。"""
    base = "/api/nodes/" + quote(args.node_id, safe="")
    while not stop.is_set():
        identifier = None
        received = 0
        total = None
        try:
            jobs = request_json(args.cloud, base + "/models/pending")
            for job in jobs:
                if stop.is_set():
                    return
                identifier = job["selection_id"]
                destination = work / "models" / identifier
                receipt = destination / "received.json"
                if not receipt.exists():
                    package = work / "models" / (identifier + ".zip.partial")
                    log("开始接收模型 " + job["model_id"] + "；心跳继续上报")
                    url = args.cloud + "/api/selections/" + identifier + "/package"
                    progress_url = base + "/models/" + identifier + "/progress"
                    with build_opener(ProxyHandler({})).open(url, timeout=600) as response, package.open("wb") as output:
                        total = int(response.headers.get("Content-Length", "0")) or None
                        received = 0
                        last_update = 0
                        request_json(args.cloud, progress_url, {"phase": "downloading", "bytes_received": 0, "total_bytes": total})
                        while True:
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            output.write(chunk)
                            received += len(chunk)
                            if time.monotonic() - last_update >= 2:
                                try:
                                    request_json(args.cloud, progress_url, {"phase": "downloading", "bytes_received": received, "total_bytes": total})
                                    last_update = time.monotonic()
                                except Exception as exc:
                                    log("进度上报暂时失败，模型下载继续：" + str(exc))
                    request_json(args.cloud, progress_url, {"phase": "extracting", "bytes_received": received, "total_bytes": total})
                    destination.mkdir(parents=True, exist_ok=True)
                    with zipfile.ZipFile(package) as archive:
                        archive.extractall(destination)
                    save_json(receipt, {"selection_id": identifier, "model_id": job["model_id"], "received_at": now_iso()})
                    package.unlink()
                    log("模型文件保存至 " + str(destination) + "；尚未启动推理")
                request_json(args.cloud, base + "/models/" + identifier + "/ack", {})
        except Exception as exc:
            try:
                if identifier:
                    request_json(args.cloud, base + "/models/" + identifier + "/progress",
                                 {"phase": "error", "bytes_received": received,
                                  "total_bytes": total, "error": str(exc)})
            except Exception:
                pass
            log("模型接收等待重试：" + str(exc))
        stop.wait(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cloud", default="http://130.2.211.215:8000")
    parser.add_argument("--node-id", default=socket.gethostname())
    parser.add_argument("--name", default=None)
    parser.add_argument("--work-dir", default="edge-data")
    parser.add_argument("--report-dir", default="reports-outbox")
    parser.add_argument("--process", action="append", default=[], help="关键软件进程名称，可重复指定，例如 --process MyService")
    args = parser.parse_args()
    args.cloud = args.cloud.rstrip("/")
    work = Path(args.work_dir).resolve()
    for folder in (work, work / "decisions", work / "models", Path(args.report_dir)):
        folder.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    receiver = threading.Thread(target=model_receiver, args=(args, work, stop), daemon=True)
    receiver.start()
    log("端侧编号：" + args.node_id + "；云侧：" + args.cloud)
    log("将完整的 UTF-8 JSON/TXT 转交包放入 " + str(Path(args.report_dir).resolve()))
    try:
        while True:
            started = time.monotonic()
            metrics = collect_metrics("GPU")
            metrics["runtime"] = runtime_info()
            try:
                metrics.update(edge_context(args.process))
            except Exception as exc:
                metrics["errors"].append("链路/软件采集失败：" + str(exc))
            try:
                request_json(args.cloud, "/api/nodes/heartbeat", {"node_id": args.node_id, "name": args.name or args.node_id, "metrics": metrics})
                upload_reports(args, work)
                receive_decisions(args, work)
            except Exception as exc:
                log("云侧暂不可达；报告留在本地，等待下次重试：" + str(exc))
            time.sleep(max(0.1, 5 - (time.monotonic() - started)))
    except KeyboardInterrupt:
        stop.set()
        log("采集程序已退出，未发送的报告保留在本地")


if __name__ == "__main__":
    main()
