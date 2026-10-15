# 云边运维演示程序

离线可部署的 FastAPI + HTML 页面。云端用 MindIE 的 OpenAI 兼容接口生成模型选型和故障决策；Windows 端侧定时上报 CPU、内存、链路、指定软件进程和 NVIDIA GPU 指标，接收决策 JSON 与选中的模型文件。

## 页面布局和三个按钮

三个主按钮位于页面上方，下面只有一个内容区域。同一时间只展示设备信息、模型仓库或总体决策其中一项；打开页面默认显示设备信息。页面顶端的消息栏显示操作状态。设备卡片每 5 秒自动更新，也可以点“更新设备信息”立即刷新。端侧超过 30 秒没有心跳会显示离线，保留最后一份快照。新端侧运行 `edge_agent.py` 后自动登记。NPU/GPU 不支持的指标显示为 `—`，不会以 0 代替未知值。

中部三按钮：

1. **更新设备信息**：切换到设备视图，刷新云侧 NPU、端侧 CPU/内存和加速卡指标。Windows 端侧默认通过 `nvidia-smi` 读取 GPU。链路卡片列出接口状态与 IPv4；这是网卡本地状态，不代表云端网络可达。关键软件按 `--process` 名称检查进程是否存在。
2. **模型仓库**：切换到模型视图，读取 `model_catalog.json` 中两条模型目录，展示权重文件总大小和用途说明。点“选择合适模型”后，云侧 Qwen 收到三项输入：模型库元数据、固定检测任务、目标端侧最近的资源快照。结果包含选型理由、量化判断和部署前提，不展示原始提示词或调试输出。下发模型按钮会打包所选模型目录，端侧接收并保存文件，不会自动安装推理环境或启动模型；下载字节数与百分比会动态显示。
3. **生成总体决策**：切换到报告与决策视图，选择当前设备的一条报告，可选最多 5 条同设备较早报告，然后点“开始研判”。决策 JSON 在该视图中展示，并可分别保存到 SQLite 或排队回传到端侧。端侧收到后将 JSON 保存到本机 `edge-data/decisions/` 并回 ACK；页面会区分排队与端侧已接收。回传不代表执行了修复。

## 提示词约定

- 模型选型提示词只接收模型信息、固定检测任务、目标设备算力信息三部分。模型权重文件大小不等于推理所需显存；未知项按未知处理；可以选择“暂无合适模型”；不会强行选 Qwen 或强制量化。量化判断只讨论选中模型。
- 故障决策要求输出 `{node_id, fault_type, repair_suggestions}`。历史事件与当前状态分开，网络断开不能直接推断网卡硬件损坏；缺少依据时输出“未知”并给出需要补采的信息。云端保存完整 JSON、模型原始回答及实际输入，便于查看依据。

## 云侧部署

将 `app.py`、`cloud_ui.html`、`telemetry.py`、`model_catalog.json` 放在同一目录。模型目录路径在 `model_catalog.json` 配置。默认 MindIE 地址是 `http://127.0.0.1:1025/v1/chat/completions`，模型名 `qwen2.5-7b`。

使用已有的 Python 3.10 环境（下例为 `d1_env-3`）。如果 FastAPI/Uvicorn/Pydantic 已离线安装，略过 pip；否则需事先将 `requirements.txt` 对应 wheels 放到服务器，再执行离线安装：

```bash
conda activate d1_env-3
cd /root/cloud-dashboard
python -m pip install --no-index --find-links ./wheels -r requirements.txt
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

打开 `http://云侧服务器IP:8000`。防火墙/安全组需允许端侧到云侧 TCP 8000。 MindIE 保持监听本机 1025 端口。可通过环境变量指定 MindIE：

```bash
MINDIE_API=http://127.0.0.1:1025/v1/chat/completions MINDIE_MODEL=qwen2.5-7b python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

默认 SQLite 与草稿位于 `./data/`；可用 `CLOUD_DATA_DIR=/持久化目录` 指定数据目录。报告一旦导入就存入 SQLite；决策生成后先写入 JSON 草稿，点“保存”才进入保存决策表；回传队列与接收时间存于 SQLite。需要备份时复制该目录。

## Windows 端侧部署

将 `edge_agent.py` 与 `telemetry.py` 放在同一目录。在 PowerShell 中运行（把 IP 与端口替换成实际地址）：

```powershell
py edge_agent.py --cloud http://130.2.211.215:8000 --node-id DESKTOP-PP86E2M --process MyService.exe
```

每个关键软件重复指定 `--process`。无需 `psutil`：Windows CPU/内存和网卡通过系统 PowerShell 读取；GPU 指标需要 NVIDIA 驱动提供的 `nvidia-smi`。如果本机尚未安装 NVIDIA 驱动或 `nvidia-smi`，卡片会显示不可用原因。端侧运行该程序的 Python 环境会显示其 Python 版本和常见推理依赖是否安装，这不能代表机器上其他 Python/conda 环境的状态。

完整 `.txt` 或 `.json` 转交报告可放入当前目录下 `reports-outbox/`。程序上传成功后保留原文件并在 `edge-data/uploaded_reports.json` 记录状态；网络恢复后会重试。端侧也可不启动采集程序，在网页里手动粘贴或选择报告文件。用户给出的“智能远维平台·告警报告”文本按原样保存在报告记录中，提取报告编号、设备名和事件时间供列表显示。

## Windows 端侧可离线接收模型

选型后点击“下发所选模型文件”。端侧程序每 5 秒检查队列并下载完整模型目录压缩包，解压到 `edge-data/models/<选型编号>/`，再报告“端侧已保存”。所选目录里的权重文件会完整打包；模型越大，云端临时空间、链路传输量和等待时间越大。请预留模型权重总量约两倍的端侧/云侧空间（端侧 zip 下载完成后会删除 zip）。本功能不自动部署该模型的 Python 运行时、任务模板适配器或推理服务。

端侧每 5 秒轮询决策。收到后先写入本地 `edge-data/decisions/<决策编号>.json` 再向云侧 ACK；终端显示 JSON 摘要。PowerShell 窗口退出后程序停止，未上传的报告仍保留供下次运行。

## 报告 JSON 示例

```json
{
  "node_id": "DESKTOP-PP86E2M",
  "report_id": "DESKTOP-PP86E2M:200657bd07e4d3143f14c325",
  "observed_at": "2026-10-15T11:38:55+08:00",
  "report": "智能远维平台·告警报告\n报告编号:200657bd07e4d3143f14c325\n设备:DESKTOP-PP86E2M\n事件记录时间:2026-10-15T11:38:55+08:00\n记录当时的状态:故障告警\n告警内容:网络未连接\n级别:主要\n监测链路:以太网2"
}
```

可直接粘贴原报告文本，云端会保留全文并尝试提取可见字段。设备名不匹配时会阻止导入错设备。相同报告编号和内容重复上传会返回既有报告；相同编号内容不同会提示冲突。

## 当前边界

此版本适用于单台可信云服务器与授权接入的端侧演示，使用 HTTP 传输，没有账户登录或 TLS。部署到跨网络环境前应由部署方配置 TLS/网关。端侧链路显示本机网卡信息，不做云侧主动探测；关键软件只按显式提供的进程名判断是否运行，不做业务级健康检查。MindIE 模型选型与决策接口是实际调用；模型权重下发是完整文件传输，推理启动和修复执行仍需端侧程序另行实现。
