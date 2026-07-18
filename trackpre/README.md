# trackpre — 无人机多目标追踪与轨迹预测模块

基于 FRED 数据集的无人机检测后处理流程，复现论文中的跟踪、预测benchmark，实现从检测框到目标跟踪、再到未来位置预测的完整链路。
跟踪使用ByteTracker；预测使用transformer预测器，融合event特征。


## 环境配置

创建环境：conda env create -f environment.yml
激活环境：conda activate ami26


## 目录结构

```
trackpre/
├── track.py                 # ByteTracker 多目标跟踪（卡尔曼滤波 + 两阶段关联）
├── gt_to_tracks.py          # FRED GT 标注 → tracks.txt 格式转换（供训练用）
├── train_transformer.py     # Transformer 轨迹预测模型训练
├── predict_trajectory.py    # 滑动窗口轨迹预测推理
├── trackpre_pipeline.py     # 端到端流水线（跟踪 + 预测一键运行）
├── trackpre_visualize.py    # 跟踪与预测结果联合可视化
│
├── gt_tracks/               # GT 标注转换后的轨迹文件（训练数据）
│   └── seq_{N}/tracks.txt
├── track_result/            # ByteTracker 跟踪输出
│   └── seq{N}/tracks.txt
├── prediction_result/       # 轨迹预测输出
│   └── {N}/predictions_future_*.txt
├── pipeline_result/         # 流水线完整输出（跟踪 + 预测）
│   └── seq{N}/
│       ├── tracks.txt
│       └── predictions_future.txt
├── premodel/                # 已训练的 Transformer checkpoint
│   └── transformer_*_best.pt
└── vis_result/              # 可视化图片输出
    └── {N}/*.jpg
```


## 数据格式

### 输入：检测结果 `predictions_fused.txt`

格式：
<filename> <x1> <y1> <x2> <y2> <conf> <cls_id>
例：`seq13_frame_002274.jpg 120.0 45.5 200.0 90.0 0.87 0`

### 中间：轨迹文件 `tracks.txt`
格式：
<timestamp>,<track_id>,<x>,<y>,<w>,<h>
- `timestamp`：帧相对于序列起点的秒数（6 位小数）
- `x, y`：目标框左上角像素坐标
- `w, h`：目标框宽高

### 输出：预测结果 `predictions_future.txt`

格式
<ts_obs_end> <track_id> <step> <pred_x1> <pred_y1> <pred_x2> <pred_y2>

- `ts_obs_end`：观测窗口末帧时间戳（秒）
- `step`：预测步（1 … N_f），对应未来 1/30s 到 0.8s
- `pred_*`：预测的 bounding box（xyxy，像素坐标）


## 模型结构（`train_transformer.py`）

观测窗口: N_p = 12 帧 (0.4 s @ 30 FPS)
预测窗口: N_f = 24 帧 (0.8 s @ 30 FPS)

输入:
  bbox_seq  [B, N_p, 4]       归一化 cxcywh 坐标
  crop_seq  [B, N_p, C, S, S] 对应 Event 帧中的目标裁剪区域

结构:
  EventCNN (3层 Conv) → 128 维事件特征
  input_proj: [4 + 128] → d_model=256
  TransformerEncoder (4层, 8 头)
  TransformerDecoder (4层, 8 头, N_p+N_f 个可学习查询)
  out_head → 归一化 cxcywh

输出:
  past_pred   [B, N_p, 4]   过去轨迹重建（辅助损失）
  future_pred [B, N_f, 4]   未来轨迹预测

损失: L = L_future + 0.5 × L_past  (MSE)


## 快速使用

### 1. 准备训练数据（GT → tracks.txt）（已完成，直接使用结果）

将 FRED 数据集的 `coordinates.txt` 转换为 `tracks.txt` 格式：

```bash
# 自动扫描并处理所有序列
python gt_to_tracks.py

# 只处理指定序列
python gt_to_tracks.py --seqs 0 13 14 15
```

输出位于 `gt_tracks/seq_{N}/tracks.txt`，并自动创建指向 Event 帧的软链接。

### 2. 训练轨迹预测模型（已完成，直接使用结果）

```bash
python train_transformer.py
```

训练配置（在 `train_transformer.py` 顶部修改）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `FPS` | 30 | 视频帧率 |
| `N_p` | 12 | 观测帧数（0.4 s） |
| `N_f` | 24 | 预测帧数（0.8 s） |
| `D_MODEL` | 256 | Transformer 维度 |
| `NHEAD` | 8 | 注意力头数 |
| `NUM_LAYERS` | 4 | Encoder/Decoder 层数 |
| `EVENT_FEAT_DIM` | 128 | CNN 事件特征维度 |
| `CROP_SIZE` | 64 | Event 帧裁剪尺寸 |

模型自动保存至 `premodel/transformer_{时间戳}_best.pt`（验证集最优）。

### 3. 单独运行跟踪

```bash
python track.py
```

读取 `predictions_fused.txt`，对每条序列运行 ByteTracker，输出至 `track_result/{seq_id}/tracks.txt`。

跟踪器参数（`ByteTracker`）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `high_thresh` | 0.4 | 高置信度检测框阈值 |
| `low_thresh` | 0.1 | 低置信度检测框阈值 |
| `iou_thresh` | 0.3 | IoU 匹配阈值 |
| `max_lost` | 30 | 轨迹最大丢失帧数 |
| `min_hits` | 3 | 轨迹确认所需最少命中帧数 |

### 4. 单独运行轨迹预测

```bash
python predict_trajectory.py \
    --tracks  trackpre/track_result/seq15/tracks.txt \
    --model   trackpre/premodel/transformer_*_best.pt \
    --events_root  ori_dataset/
```

对 `tracks.txt` 中每条轨迹做滑动窗口预测，每移动一帧输出 N_f 步预测框，结果保存至 `prediction_result/{seq_num}/predictions_future_{时间戳}.txt`。

### 5. 跟踪 + 预测 流水线

```bash
python trackpre_pipeline.py \
    --fused       predictions_fused.txt \
    --model       trackpre/premodel/transformer_*_best.pt \
    --events_root ori_dataset/ \
    --out_dir     trackpre/pipeline_result/
```

自动完成 Step 1（跟踪）→ Step 2（预测），所有结果按序列写入 `pipeline_result/{seq_id}/`。

### 6. 可视化

修改 `trackpre_visualize.py` 顶部的路径变量后运行：

```bash
python trackpre_visualize.py
```

每帧输出一张叠加图：
- **粗彩色框**：当前帧的跟踪框（按 track_id 配色）
- **细绿→红渐变框**：未来 N_f 步预测框（绿色=近未来，红色=远未来）

结果保存至 `vis_result/{seq_id}/`。

---

## 数据流总览

```
predictions_fused.txt
        │
        ▼
   track.py (ByteTracker)
        │
        ▼
track_result/{seq_id}/tracks.txt
        │
        ├──────────────────────────┐
        │                          │
        ▼                          ▼
predict_trajectory.py      trackpre_visualize.py
        │                          │
        ▼                          ▼
prediction_result/        vis_result/{seq_id}/*.jpg
  {seq_num}/
  predictions_future.txt


训练侧:
ori_dataset/{N}/coordinates.txt
        │
        ▼
   gt_to_tracks.py
        │
        ▼
gt_tracks/seq_{N}/tracks.txt
        │
        ▼
  train_transformer.py
        │
        ▼
premodel/transformer_*_best.pt
```

---

## 参考

- Magrini et al., "FRED: The Florence RGB-Event Drone Dataset", 2025
- ByteTrack: Multi-Object Tracking by Associating Every Detection Box (Zhang et al., 2022)
