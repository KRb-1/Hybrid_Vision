"""
对 tracks.txt 中的每条轨迹进行滑动窗口预测：
  - 观测窗口: N_p 帧 (0.4 s)
  - 预测序列: N_f 帧 (0.8 s)

输入:  <seq_dir>/tracks.txt  +  transformer_best.pt
输出:  predictions_future.txt
输出格式 (空格分隔):
    ts_obs_end  track_id  step  pred_x1  pred_y1  pred_x2  pred_y2
      ts_obs_end : 观测窗口末帧的时间戳 (s)
      step       : 1 … N_f
      pred_*     : 预测 bounding box (xyxy, 像素坐标)
"""

import os
import sys
from datetime import datetime
import numpy as np
import torch
from collections import defaultdict, deque

sys.path.insert(0, os.path.dirname(__file__))
from train_transformer import (
    TrajectoryTransformer,
    build_event_index,
    find_event_frame,
    load_event_crop,
    cxcywh_norm_to_xyxy,
    N_p, N_f, FPS, IMG_W, IMG_H,
    EVENT_CHANNELS, CROP_SIZE, EVENT_FEAT_DIM,
    D_MODEL, NHEAD, NUM_LAYERS,
)


INPUT_TRACK = '/root/autodl-tmp/HybridVision/trackpre/track_result/seq9/tracks.txt'
MODEL_PATH = '/root/autodl-tmp/HybridVision/trackpre/premodel/transformer_20260705_034901_best.pt'
OUTPUT_DIR = '/root/autodl-tmp/HybridVision/trackpre/prediction_result/'
DEFAULT_EVENT_ROOT = '/root/autodl-tmp/HybridVision/ori_dataset/test'   # 父目录，子目录名 = 序列号

def load_model(model_path):
    """
    从 checkpoint 恢复模型及其超参数。
    返回 (model, ckpt_params_dict)。
    """
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)

    params = {
        'd_model':        ckpt.get('d_model',        D_MODEL),
        'nhead':          ckpt.get('nhead',           NHEAD),
        'num_layers':     ckpt.get('num_layers',      NUM_LAYERS),
        'event_feat_dim': ckpt.get('event_feat_dim',  EVENT_FEAT_DIM),
        'n_p':            ckpt.get('n_p',             N_p),
        'n_f':            ckpt.get('n_f',             N_f),
        'fps':            ckpt.get('fps',             FPS),
        'img_w':          ckpt.get('img_w',           IMG_W),
        'img_h':          ckpt.get('img_h',           IMG_H),
        'crop_size':      ckpt.get('crop_size',       CROP_SIZE),
        'event_channels': ckpt.get('event_channels',  EVENT_CHANNELS),
    }

    model = TrajectoryTransformer(
        d_model=params['d_model'],
        nhead=params['nhead'],
        num_layers=params['num_layers'],
        event_feat_dim=params['event_feat_dim'],
    )
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, params


# ─── 单次预测 ──────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_sequence(model, past_frames, params, device):
    """
    past_frames: list[N_p] of (timestamp, box_xyxy, event_path|None)
    返回:        ndarray [N_f, 4]  xyxy 像素坐标
    """
    n_p    = params['n_p']
    img_w  = params['img_w']
    img_h  = params['img_h']
    c_size = params['crop_size']
    e_ch   = params['event_channels']

    enc_boxes = []
    enc_crops = []
    for _, box_xyxy, evt_path in past_frames:
        x1, y1, x2, y2 = box_xyxy
        enc_boxes.append([
            (x1 + x2) / 2 / img_w,
            (y1 + y2) / 2 / img_h,
            (x2 - x1)      / img_w,
            (y2 - y1)      / img_h,
        ])
        if evt_path is not None:
            crop = load_event_crop(evt_path, box_xyxy)
        else:
            crop = torch.zeros(e_ch, c_size, c_size)
        enc_crops.append(crop)

    bbox_seq = torch.tensor(enc_boxes, dtype=torch.float32).unsqueeze(0).to(device)
    crop_seq = torch.stack(enc_crops).unsqueeze(0).to(device)

    _, future_pred = model(bbox_seq, crop_seq)          # [1, N_f, 4] normalized cxcywh

    future_pred = future_pred[0].cpu().numpy()          # [N_f, 4]
    results = []
    for pred in future_pred:
        results.append(cxcywh_norm_to_xyxy(pred, img_w, img_h))
    return np.array(results)                            # [N_f, 4] xyxy


def main(
    tracks_file   = INPUT_TRACK,
    model_path    = MODEL_PATH,
    event_seq_dir = None,                # None → 从 tracks_file 路径自动推导
    events_root   = DEFAULT_EVENT_ROOT,  # 仅在 event_seq_dir=None 时使用
    output_file   = None,                # None → OUTPUT_DIR/{seq_num}/predictions_future_{ts}.txt
):
    # 从 tracks 路径提取序列号（供 event 推导和输出目录命名共用）
    seq_id  = os.path.basename(os.path.dirname(tracks_file))
    seq_num = seq_id[3:].lstrip('_')     # "seq0"→"0", "seq_0"→"0"

    # 自动推导 event 目录
    if event_seq_dir is None:
        event_seq_dir = os.path.join(events_root, seq_num)
        print(f"[自动推导] event 目录: {event_seq_dir}  (seq_id={seq_id!r})")

    # 自动生成输出路径：prediction_result/{seq_num}/predictions_future_{YYYYMMDD_HHMMSS}.txt
    if output_file is None:
        run_ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir  = os.path.join(OUTPUT_DIR, seq_num)
        os.makedirs(out_dir, exist_ok=True)
        output_file = os.path.join(out_dir, f'predictions_future_{run_ts}.txt')
        print(f"[自动推导] 输出路径: {output_file}")
    else:
        out_dir = os.path.dirname(output_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    # 加载模型
    model, params = load_model(model_path)
    n_p  = params['n_p']
    n_f  = params['n_f']
    fps  = params['fps']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f"模型已加载: N_p={n_p} 帧, N_f={n_f} 帧  ({n_p/fps:.1f}s → {n_f/fps:.1f}s)")
    print(f"device: {device}")
    print(f"tracks 文件: {tracks_file}")
    print(f"event 来源:  {event_seq_dir}")

    # 构建 event 帧索引（来自 GT 序列目录）
    ts_list, path_list = build_event_index(event_seq_dir)
    if not ts_list:
        print(f"[警告] 在 {event_seq_dir} 下未找到 Event/Frames，event 特征将为零向量")

    # 解析 tracks.txt，格式: timestamp,track_id,x,y,w,h  (x,y = top-left)
    track_data = defaultdict(list)
    with open(tracks_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) != 6:
                continue
            ts, track_id, x, y, w, h = parts
            ts = float(ts)
            x, y, w, h = float(x), float(y), float(w), float(h)
            xyxy = [x, y, x + w, y + h]
            evt_path = find_event_frame(ts, ts_list, path_list)
            track_data[track_id.strip()].append((ts, xyxy, evt_path))

    # 滑动窗口预测
    results = []
    total_windows = 0

    for track_id, frames in sorted(track_data.items()):
        frames.sort(key=lambda r: r[0])
        total = len(frames)

        if total < n_p:
            continue

        # 滑窗: 每移动一帧就生成一组 N_f 预测
        buf = deque(maxlen=n_p)
        for frame in frames:
            buf.append(frame)
            if len(buf) < n_p:
                continue

            past = list(buf)
            ts_obs_end = past[-1][0]
            preds = predict_sequence(model, past, params, device)  # [N_f, 4]

            for step, pred_box in enumerate(preds, start=1):
                results.append(
                    f"{ts_obs_end:.6f} {track_id} {step} "
                    f"{pred_box[0]:.3f} {pred_box[1]:.3f} "
                    f"{pred_box[2]:.3f} {pred_box[3]:.3f}"
                )
            total_windows += 1

    # 保存
    with open(output_file, 'w') as f:
        f.write(
            "# ts_obs_end  track_id  step  "
            "pred_x1  pred_y1  pred_x2  pred_y2\n"
        )
        for line in results:
            f.write(line + '\n')

    print(f"\n预测完成！")
    print(f"  轨迹数:    {len(track_data)}")
    print(f"  预测窗口:  {total_windows}")
    print(f"  输出行数:  {len(results)}  (每窗口 {n_f} 行)")
    print(f"  结果保存至: {output_file}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Sliding-window trajectory prediction from tracks.txt'
    )
    parser.add_argument('--tracks',      default=INPUT_TRACK,
                        help='输入 tracks.txt 路径')
    parser.add_argument('--model',       default=MODEL_PATH,
                        help='transformer checkpoint 路径')
    parser.add_argument('--events_root', default=DEFAULT_EVENT_ROOT,
                        help='event 数据父目录（子目录名 = 序列号），'
                             '默认从 tracks 路径自动推导具体序列')
    parser.add_argument('--out',         default=None,
                        help='输出文件路径（默认: OUTPUT_DIR/{seq_num}/predictions_future_{时间戳}.txt）')
    args = parser.parse_args()

    main(
        tracks_file  = args.tracks,
        model_path   = args.model,
        events_root  = args.events_root,
        output_file  = args.out,
    )
