# 2026-09-10 GPU 恢复后的完整链路补验

## 结论与范围

**通过：固定历史权重的新 GPU 推理 → 清洗 → 10/27 点 H5 → 10 点叠加视频 → 哈希验收 → 缓存续跑。** 使用原始视频的前 200 帧，不是复用已存在的原始预测；未重新训练、未启动完整 22 条 trial。

本次运行代码为 `8a5795a`；补验后的提交仅更新文档，未修改科学算法、代码或参数默认值。既有两条完整历史结果回归和 19 项单元测试仍是独立证据，不能与本次短片推理混淆。

## 配置与设备

| 项目 | 实测/实际配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090，24,564 MiB |
| 容器内存上限 | 128,849,018,880 字节，120 GiB |
| 驱动 | 580.76.05 |
| Python / DLC | 3.11.16 / 安装包元数据 3.0.0rc14 |
| Torch / CUDA runtime | 2.8.0+cu128 / 12.8 |
| 视频 | 200 帧，640×480，24.996 FPS，约 8.001 秒 |
| 模式 | checkpoint，`video_adapt=False`，max_individuals=1 |
| 模型 | topviewmouse / hrnet_w32 / fasterrcnn_resnet50_fpn_v2 |
| 权重 | 历史 4+4 epoch 适配的两个 `004.pt` checkpoint，未改动 |
| 推理 batch size | pose=2，detector=2 |
| 清洗 | 关键点阈值 0.1、bbox 阈值 0.5、5 帧居中均值、原线性补齐规则 |
| 跳变 | off；这段短片未指定切点 |

## 执行与验收

在 tmux 中依次运行 doctor、check、与 `sap run` 相同的 Runner、verify，再执行 CLI `sap run` 与 verify。测量脚本额外记录 CUDA 峰值分配，且断言实际 CUDA 分配大于零，避免只看到显卡存在就声称使用了 GPU。

- Detector 200/200 帧与 pose 200/200 帧完成；新的 raw JSON/H5 通过同源数值与帧数检查。
- 清洗、导出完成，最终 `COMPLETE`；verify 返回 1 trial、200 帧、10 点。
- 再次执行的 prepare、infer、cuts、clean、export 均为 `CACHED`，verify 再次通过，启动脚本退出码 0。
- Runner 端到端耗时 **17.457 秒**，包含本次模型加载/推理、清洗、导出及内部验收；不含输入构造、doctor/check、下载、后续续跑和外部测量前的 Torch 导入。不能外推为长视频稳定吞吐量。
- CUDA 峰值 allocated **1,139,844,096 字节（约 1.06 GiB）**，reserved **1,507,852,288 字节（约 1.40 GiB）**；这是 PyTorch allocator 的统计，不是系统总显存峰值。
- 10 点 H5 从完整 27 点表精确选列，视频帧数保持 200；原始结果与早期 CPU 失败 attempt 保留不覆盖。

## 原始证据与边界

完整配置、输入/权重/代码哈希在私有输出的 `provenance.json`；步骤文件哈希在 `receipts/`。交付机器的私有证据位置：

- `.local/gpu-smoke/run.log`、`exit-code.txt`、`gpu-measurement.json`；
- `.local/gpu-smoke-output-20260910/verification.json`；
- `.local/gpu-smoke-output-20260910/data/gpu05/`：本次新推理后的 H5、视频与 QC。

这些材料均不进入 GitHub 仓库。短片的技术完成不等于全数据姿态精度已验证，不证明新适配或 zero-shot 效果；从零部署到另一台服务器和整机迁移不在本次补验范围内。
