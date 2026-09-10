# 2026-09-10 验收记录

结论：独立仓库的配置、清洗、导出、续跑与技术验收功能已实现，并通过历史数据回归。**本次新视频的完整模型推理尚未通过运行验收，不能把整条 GPU 链路标为已验证。**

## 已通过

| 验收项 | 证据与范围 |
|---|---|
| 原清洗核心保留 | `legacy.py` SHA-256 与原服务器脚本一致，见 `provenance.md` |
| macOS 干净 CPU 环境 | 新建 Python 3.11.15 venv，安装项目核心依赖；未安装 DLC/Torch；依赖检查通过 |
| 自动测试 | macOS 与 Linux 各 19 项通过；包括模拟推理接口、10/27 点导出、帧数检查、失败续跑、篡改拒绝、分段平滑和历史序列化一致性 |
| 静态检查 | Ruff lint、format check 通过；历史核心不重新格式化 |
| 安装包 | wheel 和 sdist 构建成功；另建 venv 安装 wheel，确认从 site-packages 导入后 19 项测试通过；包内未包含私有数据 |
| 两条完整历史结果回归 | 各 17,469 帧，合计 34,938 帧；27 点 × 3 列 H5 的索引、列、dtype、数值精确一致；cleaned JSON 对象精确相等 |
| 同步跳变候选 | 从原始预测重新计算边界，得到 `[3511, 5116, 13114, 13115, 13116]`，与历史记录一致 |
| 真实短片清洗与导出 | macOS 和 Linux 均通过；原始视频和对应原始 JSON/H5 的前 200 帧，约 8 秒；完整走过 precomputed → clean → export → verify，不是复制旧 cleaned H5 |
| 两平台续跑 | 再次运行相同配置，五个阶段均 `CACHED`，最终仍为 `COMPLETE`；输入、代码或结果变动会拒绝复用 |

完整回归使用最终包代码在本地执行，而不是仅测试服务器早期开发快照。旧版函数本身仍原样保留；新 `serialize_predictions` 只优化写回取数方式，测试中直接与旧函数比较，包括空检测、缺失值、补齐后数据和额外字段。

两条完整回归中，清洗调用实测分别为 6.58 和 5.99 秒，包括验证、清洗和 H5/JSON/QC 写出，不含模型推理、视频叠加或文件下载。不同机器耗时不可直接横比，也不能据此估计 GPU 推理吞吐量。

200 帧演示的 QC：27 点清洗输入中 43 个低置信度点次，0 个长 gap，0 个出图像边界点次；10 点是从清洗后的 27 点精确选列。FPS 为 24.996，未重采样。检查了叠加预览，点与骨架覆盖鼠体；这只是有限视觉抽查，不是全数据定位精度验收。

## 运行环境限制与未通过项

服务器只在原有环境上建立独立 venv（共享已有 site-packages）并安装此包，**不是从零重建 GPU 环境**。包元数据为 DLC 3.0.0rc14、Torch 2.8.0+cu128、torchvision 0.23.0+cu128、Python 3.11.16。DLC 启动文字曾显示 rc13，与安装包元数据不一致，故版本记录以安装元数据为准，同时保留日志。

本次服务器现场检查：`nvidia-smi` 无设备，`torch.cuda.is_available()` 为 false；容器 cgroup 内存上限 2,147,483,648 字节（2 GiB），不能采用宿主机 `free` 显示的总内存作本容器配额。

使用历史 pose/detector 权重，尝试 12 帧真实视频的新推理：

- batch size 2 两次在 detector 中途终止；第二次通过 tmux 重试，准备阶段正确复用，推理阶段新建 attempt。
- 单独设置 batch size 1、单 CPU 线程、新输出目录后仍在 detector 阶段退出，记录退出码 137。
- 退出码说明进程被强制终止；结合 2 GiB 配额，资源不足是可能原因，但 cgroup 的 `oom_kill` 计数为 0，**不能将具体杀进程原因宣称已确诊**。
- 未产生完成的 inference 凭证，也未伪报整体成功。以上不是 GPU 推理通过证据。不能用模拟接口测试替代实际推理验收。

不继续在该受限状态反复重试。恢复 GPU 和充足内存后，先用 **新的输出目录**做短视频完整 smoke：`check → run → verify`，确认设备、帧数、27/10 点、叠加视频后再整批运行。

`sap adapt` 的 4+4 epoch 参数有接口测试；本次没有重新训练，也没有对新适配效果作验证。zero-shot 模型下载、陌生服务器从零安装和 GitHub Actions 在线运行均未完成现场验证。当前 CI 文件仅为已配置。

## 证据保存与公开边界

以下相对路径是交付机器上的私有验收资料，均被 `.gitignore` 排除，不会进入代码发布包：

- `.local/unit-tests-final.log`：本地 19 项测试日志。
- `.local/wheel-tests.log`：独立 wheel 安装后的 19 项测试日志。
- `.local/full-regression.log`、`.local/full-regression-output/regression.json`：完整历史回归和旧结果哈希。
- `.local/demo-final/verification.json`、`.local/demo-final.log`、`.local/demo-final-resume.log`：最终演示与续跑凭证。
- `.local/demo-final/data/demo05/03_labeled_cleaned.mp4`：10 点演示视频。
- 服务器 `.local/unit-tests-final.log`、`.local/inference*.log`：Linux 测试与未完成推理记录。

原始实验文件与师兄 AutoCBE tracked 源码未修改。仓库没有公开上传；未选择项目许可证。后续边界见 `handoff.md`。
