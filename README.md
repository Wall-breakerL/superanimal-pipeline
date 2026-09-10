# SuperAnimal Pipeline

**把视频整理成可追溯、可复用的单鼠姿态数据。**

SuperAnimal 推理 → 置信度过滤 → 缺失补齐 → 时序平滑 → 10/27 点 H5 → 标注视频与质量检查。

这是独立的批处理工具，不需要 AutoCBE，不包含 MoSeq、行为命名、LLM 或组间统计。当前范围是顶视、每段画面一只小鼠。已有双箱视频可在 manifest 指定两个裁剪区域。

> **方法声明**：历史兼容清洗使用 **5 帧居中滑动均值，不是中值滤波**。长缺失段和 likelihood 也会插值。技术验收不等于姿态精度已经得到人工真值验证。

当前验收：两平台各 19 项测试、34,938 帧历史结果精确回归、真实 200 帧清洗与视频导出通过。当前服务器无 GPU、内存限额 2 GiB，新推理未跑通；换机后仍需短视频完整 smoke。详见 [验收记录](docs/verification.md)。

## 快速开始

推荐 Linux、Python 3.11、NVIDIA GPU。CPU 支持清洗、导出、测试；推理性能与设备支持以 `doctor` 和实际 smoke 为准。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# 本次固定依赖版本；完整 GPU 安装/推理仍需在目标设备上验收。
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[inference]'
python -m pip check
sap doctor
```

不需要推理、仅处理已有 SuperAnimal JSON/H5 时：`python -m pip install -e .`，不要求 DLC 或 GPU。由于系统驱动、模型缓存与模型下载条件不同，换机仍必须做一次短视频 smoke。

1. 复制 `examples/config.yaml`，填写两个 checkpoint 路径和输出目录。
2. 在 `examples/trials.csv` 写入视频路径；trial ID 按字符串保留，`05` 不会变成 `5`。
3. 执行：

```bash
sap check --config examples/config.yaml
sap run --config examples/config.yaml
sap verify --config examples/config.yaml
```

`check` 检查配置、路径、权重存在性和输入哈希，不执行模型，也不代表质量验收。`run` 显示当前视频、步骤、缓存复用及叠加视频帧进度。只有 `COMPLETE` 且 `verify` 通过才算技术完成。

## 断线也能继续

```bash
# 已激活 .venv，且位于仓库根目录
tmux new -s superanimal
set -o pipefail
sap run --config examples/config.yaml 2>&1 | tee run.log
```

先按 Ctrl+B，再按 D，退出查看但保持运行；重连用 `tmux attach -t superanimal`。

```bash
sap status --config examples/config.yaml  # 最后事件，不是进程存活证明
tail -f run.log                          # Ctrl+C 仅退出日志查看
sap run --config examples/config.yaml    # 失败/中断后，同一命令按步骤续跑
```

未完成 attempt 保留用于排查；不会凭“文件存在”跳过。输入、参数、代码、依赖版本或权重改变，必须使用新输出目录。单个任务失败时 fail-fast，保留此前完成凭证，整批不会假装完成。

## 三种输入模式

| model.mode | 用途 | 必需项 |
|---|---|---|
| `checkpoint` | 固定适配权重批量推理 | pose + detector checkpoint |
| `zero-shot` | 使用上游预训练模型 | 不填自定义权重；可能首次联网下载模型 |
| `precomputed` | 只清洗已有原始预测 | manifest 中同源视频、raw_json、raw_h5 |

`run` 始终 `video_adapt=False`。需要新数据适配时，显式运行一次：

```bash
sap adapt --video /path/to/already-cropped-single-mouse.mp4 --output /path/to/new-adaptation-run
```

沿用历史 detector 4 + pose 4 epochs；输出 `adaptation.json` 中两个 checkpoint 路径，再填入批处理配置。新适配输出目录不能复用。失败适配不支持伪装为恢复训练；另建目录重新开始。

## 数据目录

```text
outputs/my-run/
├── data/05/                 # 指向已验收 export 的相对符号链接
│   ├── 01_original.mp4      # 分析用单鼠视频；如裁剪/格式转换过，不是原始整幅 AVI
│   ├── 02_cleaned.h5       # 配置选择的 10 或 27 点
│   ├── 02_cleaned_27kp.h5  # 选择 10 点时保留完整 27 点副本
│   ├── 03_labeled_cleaned.mp4
│   └── qc.json
├── meta.csv                # trial、animal、group 与相对数据路径
├── provenance.json         # 输入/权重哈希、参数、代码、依赖版本
├── receipts/               # 每一步完整文件哈希与依赖
├── work/                   # 原始预测、cleaned JSON、gap、mask、失败 attempt
├── status.json
└── verification.json
```

原 H5/视频不覆盖；保留原生 FPS 和实际可解码帧数，不补造帧。坐标单位为**像素**；没有自动毫米标定。`data/` 用相对链接，迁移整个输出根目录即可；只拷贝 workspace 数据请用 `rsync -aL output/data/ destination/data/`，同时携带 `meta.csv`。仅拷贝这些数据不包含原始预测与完整重现证据。

## 看清“异常处理”做了什么

| 部分 | 本工具的行为 |
|---|---|
| 低置信度 | bbox < 0.5、关键点 < 0.1 各自置空；阈值可配置 |
| 缺失补齐 | 线性、双向、无 gap 长度上限，包括 confidence；保存原始缺失 mask 和 gap |
| 平滑 | 仅 x/y，居中 rolling mean，默认 5 帧、min_periods=1 |
| 跳变 | `off` / 人工指定边界 / 同步双视频候选检测；仅阻止平滑跨边界 |
| 无法补齐 | 整列无有效值时失败，保留 gap 诊断，不伪造输出 |
| 高速度/出界 | 写入 QC，不自动删点；图像范围检查不是场地 ROI 检查 |
| 10 点 | 对清洗后的 27 点做精确列选择，不再次滤波 |

跳变检测只适用于**同一时间线同步拍摄**的两个视图，不等于生物学配对。阈值为各自 bbox 位移的 99.9% 分位数；候选不是真值。平滑分段不会修复源时间线，当前插值仍不分段。详见 [方法与边界](docs/methods.md)。

## 测试与验收

```bash
python -m unittest discover -s tests -v
```

测试不下载模型，涵盖历史数值兼容、gap、切点、ID/路径、帧数、10/27 点导出、视频、失败恢复、输入/输出篡改拒绝。真实数据与设备验证另见 [验收记录](docs/verification.md)，不要把合成测试当作 GPU 或科学效果证明。

## 代码导航

| 文件 | 职责 |
|---|---|
| `config.py` | 严格配置、manifest、路径与参数校验 |
| `runner.py` | 冻结、步骤凭证、进度、恢复、workspace 发布 |
| `processing.py` | 视频准备、推理适配、分段平滑、清洗、导出、叠加 |
| `legacy.py` | 原历史清洗脚本，逐字保留以便追溯 |
| `cli.py` | check/run/status/verify/doctor/adapt |

## 来源与公开发布

模型通过 DeepLabCut 调用，不复制上游模型实现，不把权重或实验视频放进 Git。官方参考：[SuperAnimal 文档](https://deeplabcut.github.io/DeepLabCut/examples/COLAB/COLAB_YOURDATA_SuperAnimal.html)、[安装说明](https://deeplabcut.github.io/DeepLabCut/docs/installation.html)。依赖以此项目的实际验证版本为准，不自动追随文档最新版。

本项目目前为本地交付，**尚未选择项目自身的开源许可证，也未公开发布**。发布前需要仓库所有者确认历史脚本的授权、许可证以及实验材料可公开范围。详见 [来源记录](docs/provenance.md)。

阶段范围与迁移前检查见 [交接说明](docs/handoff.md)。
