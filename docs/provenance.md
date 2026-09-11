# 来源与授权边界

此项目从 2026-09-05 实际运行的独立 SuperAnimal 处理脚本整理，不是 AutoCBE 源码分支。

| 文件/模块 | 来源 |
|---|---|
| `src/superanimal_pipeline/legacy.py` | 服务器 `depression_open_field_20260905/code/clean_v4_core.py` 逐字复制 |
| `processing.segmented_smooth` | 按同次 `run_cleaning_v4_cutaware_batch.py` 的分段 rolling mean 逻辑整理 |
| `processing.paired_cuts` | 按同次 `detect_pair_discontinuities.py` 的同步 bbox 位移分位数逻辑整理 |
| 固定权重 inference 参数 | 同次 `run_superanimal_batch.py` |
| 显式 adaptation 参数 | 同次 `adapt_checkpoint.py`，4+4 epoch |
| 通用 config/runner/导出/叠加/测试 | 本次新增外层，不复制 AutoCBE 引擎 |
| `processing.serialize_predictions` | 等价重写旧 `update_predictions` 的 slot-0 数据写回，批量取数组后逐帧序列化，避免反复切整张 pandas 表；不改变清洗计算 |

原清洗核心 SHA-256：`b4988835274b71ff7cbe0b0946b76f3c3ac78319ed4eb3f85f27a57aebfcf255`。

历史算法与工程保护分开：新外层额外拒绝错误 schema、多动物、全缺失不能补齐、参数变更后复用旧产物。有效输入的历史兼容性由回归测试核对；不能把拒绝旧脚本曾容忍的坏输入视作算法同一性证明。

已安装 DeepLabCut 3.0.0rc14 的包元数据显示许可证为 `LGPL-3.0-or-later`；该信息仅描述依赖，不自动决定本项目原创/历史脚本的许可证。模型权重、实验数据及各依赖需各自遵守授权条件。

2026-09-11，仓库维护者明确要求将现有代码推送并改为公开仓库。本次公开不添加 MIT 等项目许可证，也不发布真实视频、模型、身份信息或 SSH/API 密钥。模型权重由使用者提供或上游下载，不写进 Git。

**仍待维护者确定**：历史脚本的完整作者署名、授权和再许可范围，以及本项目的开源许可证。公开可见不代表这些事项已完成核定；本文件保留现有来源记录，不推定依赖许可证自动适用于本项目。

SuperAnimal 模型和论文归属上游作者。本项目独立维护，不代表官方项目。引用 Ye et al. 2024 及 DeepLabCut 相关软件论文，见 README 的引用章节。上游当前将 SuperAnimal 模型限定为非商业研究用途，具体以[官方许可说明](https://github.com/DeepLabCut/DeepLabCut#license)与所用模型条款为准。

本地 `.local/` 只存审计副本、测试日志、演示视频与私有配置，并已加入 `.gitignore`；这些不是可公开的仓库素材。
