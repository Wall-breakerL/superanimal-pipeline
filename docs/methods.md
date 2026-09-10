# 方法与数据契约

## 输入和推理

仅接受顶视单鼠：`superanimal_topviewmouse` / `hrnet_w32` / `fasterrcnn_resnet50_fpn_v2`，`max_individuals=1`。模型实例切换、多鼠身份保持、其他物种不在第一版范围。单鼠视频直接输入；裁剪参数为像素 x/y/width/height，按每行 manifest 显式给定，不推断分组。原生 FPS 保留，容器声明与解码差异记录在 `video_info.json`；较大差异拒绝继续。预先计算的 JSON/H5 必须与同一视频、同一坐标空间对应；不支持再裁剪。

## 历史兼容清洗

1. bbox 负值/无效尺寸或置信度不足，bbox 自身的 x/y/w/h/score 置空。关键点坐标负值或 likelihood 不足，其 x/y/likelihood 置空。低 bbox 分数不自动把仍有效的关键点全部删除。
2. bbox、关键点沿时间做 `interpolate(method='linear', axis=0, limit_direction='both')`。没有最大 gap 长度；开头/结尾按 pandas 该方法补齐。likelihood 会被插值，不能再把它理解为原始逐帧模型置信度。
3. x/y 做 `rolling(window=5, center=True, min_periods=1).mean()`。bbox 和 likelihood 不平滑。候选边界仅隔开这个均值窗口，插值仍跨整个视频执行。
4. 无法补齐的全缺失列报错，失败 attempt 保存 gaps/mask，不宣布技术完成。

`legacy.py` 保留旧代码原文；新的外层增加输入 schema、有限值、全缺失失败、哈希和视频对齐检查。合法且可补齐的数据应与旧核心数值一致。未增加 median、速度删点、长段截断或坐标裁切。

## 跳变边界

`cuts.mode=manual` 读取 `{trial_id: [frame_index, ...]}`；必须列出所有 trial，空列表表示无边界。frame_index 从 0 开始，表示该帧是新段第一帧；不能填第 0 帧或超出帧数。

`cuts.mode=paired` 需要每对两行、相同 `pair_id`、匹配 FPS/帧数，并由使用者保证是同一源时间线。计算 bbox 中心的相邻位移，各自取 99.9% 分位数；两侧同一转场均达到阈值时记为候选。和历史方法一样，静止到分位数为零时可能产生大量候选，故必须看审计记录，不能解释为真实剪切检测精度。单独视频默认关闭，不用“异常高速”替代同步判据。

## 输出

原始 SuperAnimal JSON 和四层 H5 位于 infer attempt。清洗 H5 保留输入索引、存储数值精度和列契约：`scorer/individuals/bodyparts/coords`，key=`df_with_missing`。仅支持单个 scorer、animal0、按历史顺序的 27 点，未知 schema 失败，避免猜映射。

10 点按顺序：nose、left_ear_tip、right_ear_tip、neck、mouse_center、tail_base、left_shoulder、right_shoulder、left_hip、right_hip。保留 x/y/likelihood，无重新插值或滤波。所有点以像素为单位。

标注视频是新的轻量 OpenCV 可视化外层，黄色点、橙色躯干骨架，阈值默认 0.2。与旧 DLC 27 点彩色画法可能不同；可视化样式不改变 H5。选择 27 点时画全部点，连线仅为固定躯干骨架。不是生物学行为分类视频。

## 验收边界

结构有效、无 NaN、文件哈希一致、正常解码，不意味着逐点定位正确。长插值、镜头跳切、左右语义混淆、场地标定和人工真值仍需独立检查。这里不输出动物疾病判断，不暗中删除不利 trial，不执行 MoSeq。
