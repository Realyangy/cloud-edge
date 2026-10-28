# RF-SepEdge 双任务前端（真实实验回放版）

## 已实现

- **任务一：MIT WaveNet 混合信号分离**：任务简介；QPSK / OFDM-QPSK、四种干扰、三档 SINR 的 24 条展示样本；波形/时频图切换与放大；四阶段自动/手动回放；单样本输入/恢复 BER、MSE、SINR；七档 SINR 的 BER/MSE 曲线与四类干扰对比。
- **任务二：YOLO11n + XCiT V4**：任务简介；Seed 161803 的四张真实宽带时频图；可点击检测框与目标列表；YOLO 置信度、中心频率、带宽、时长；XCiT V4 家族、细类、Top-5；四阶段回放；YOLO mAP、V3/V4 比较、调制家族差异及 V4 验证曲线。
- 所有指标、图片和逐目标预测都从 `../demo-results/` 读取，不运行任何推理，也不改动权重及正式评测数据。
- **视觉规范**：与先前浅色蓝色工作台风格一致，仅使用一套系统中文无衬线字体、14/16/20/24/28px 五种字号和 400/500/600 三种字重。

## 部署（推荐）

将本压缩包中的 `frontend/` 目录放入已有仓库 `RF-SepEdge/` 根目录，使其与 `demo-results/` 同级：

```text
RF-SepEdge/
├── frontend/
│   ├── index.html
│   ├── styles.css
│   ├── app.js
│   └── README.md
├── demo-results/
│   ├── manifest.json
│   ├── showcase/
│   └── ...
├── signal-separation/
└── spectrum-analysis/
```

然后从 `RF-SepEdge/` 根目录启动静态网页服务（不需要安装前端依赖）：

```bash
cd ~/Sata2/lyy/cloud-edge/RF-SepEdge
python -m http.server 8080 --bind 127.0.0.1
```

在**服务器本机**浏览器打开：`http://127.0.0.1:8080/frontend/`。

若希望在自己的 Windows 浏览器查看远程服务器页面，可以在 Windows PowerShell 中建立 SSH 端口转发（把 `<服务器地址>` 替换为你实际使用的地址）：

```powershell
ssh -L 8080:127.0.0.1:8080 flyvideo@<服务器地址>
```

随后打开 `http://127.0.0.1:8080/frontend/`。不建议直接向公网暴露整个 `RF-SepEdge` 项目目录，因为它包含模型和数据文件。

## 单文件演示

同目录外另附 `RF-SepEdge_射频双任务前端_v2_单文件.html`，可以尝试直接在浏览器中打开。它会从 GitHub 的公开 `demo-results/` 读取数据；必须能够连接 GitHub，且远程资源加载取决于浏览器网络策略。**正式演示仍推荐使用上述本地目录方式**，即使断网也能回放已同步到服务器的图片与统计数据。

## 数据对齐与评价说明

- MIT 样本均来自 Seed 42；24 条可视化样本来自已有 560 条批量记录；图像中混合输入、目标 SOI 真值、WaveNet 估计来自相同样本。目标真值**不作为推理输入**。
- MIT 的“误码率相对下降”按 `(BER_input - BER_restored) / BER_input` 计算。曲线的 MSE 直接采用原报告的“线性 MSE 先求平均、再转 dB”的统计结果。0 误码只意味着当前样本未观测到错误。
- XCiT 的家族概率来自 **57 类概率按家族求和**，不等于细类 Top-1 简单映射家族；当前页面默认使用 RMS 归一化后的 V4 分类结果。
- 在频谱图中点击检测框，右侧显示同一 `sample.json` 中对应的预测内容。真实标签仅用于评测，模型并未将其作为输入。未匹配 GT 的预测显示为“未匹配真实目标”。
- YOLO 的 94.94% mAP@0.5 与 82.82% mAP@0.5:0.95 来自 Seed 42 的 50 图独立评测；XCiT V4 的 82.94% 家族准确率与 38.06% 57 类 Top-1 来自 Seed 161803 的 100 图模型选择用留出数据，不能混称同一批实验。
- 这些结果都是本地 TorchSig/Sionna 生成条件下的复现实验，不是现场实时接收电磁信号、真实部署实测或官方隐藏测试集成绩。
- 已上传 PNG 内部的标题及坐标轴字体由原导出脚本控制，网页只能统一页面中的字体。遇到图片小字可使用右上角“放大查看”。如果要彻底统一图片内的字体，则需从原实验数据按统一 Matplotlib 样式重新导出。

## 数据源

默认优先读取 `../demo-results/manifest.json`。如果本地不存在，会尝试 GitHub 原始数据链接：

`https://raw.githubusercontent.com/Realyangy/cloud-edge/main/RF-SepEdge/demo-results/manifest.json`

浏览器必须允许加载此 URL；正式离线部署应保持本地 `demo-results/` 文件齐全。没有素材时会显示加载错误，不用随机图表或伪造指标代替。

## 维护提示

页面代码是静态 `index.html + styles.css + app.js`，不依赖第三方前端包。需要变更样本时先更新 `demo-results/manifest.json` 和相应真实结果，不要直接硬编码新的实验数据在前端 JS 中。
