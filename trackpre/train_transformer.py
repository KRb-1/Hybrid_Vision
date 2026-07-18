"""
Transformer encoder-decoder for drone trajectory prediction.
Based on the FRED paper (Magrini et al., 2025) baseline.

Observation:  N_p = 12 frames (0.4 s @ 30 FPS)
Prediction:   N_f = 24 frames (0.8 s @ 30 FPS)
Decoder outputs: N_p past reconstruction + N_f future prediction
Loss:         L_future + λ * L_past  (λ = 0.5)
Event features: 3-layer CNN on cropped event frames

Data format (tracks.txt): timestamp,track_id,x,y,w,h  (x,y = top-left pixel)
Event frames: <seq>/Event/Frames/Video_N_frame_{timestamp_us}.png
"""

import bisect
import glob as _glob
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


FPS      = 30
H_OBS    = 0.4                     # observation window (s)
H_PRED   = 0.8                     # prediction horizon (s)
N_p      = round(FPS * H_OBS)      # 12
N_f      = round(FPS * H_PRED)     # 24
LAMBDA   = 0.5                     # weight for past reconstruction loss

IMG_W    = 1280
IMG_H    = 720
CROP_SIZE      = 64                # resize all event crops to this
EVENT_CHANNELS = 1                 # 1 = grayscale; set 3 for RGB event frames
EVENT_FEAT_DIM = 128               # CNN output dimension
D_MODEL        = 256
NHEAD          = 8
NUM_LAYERS     = 4

BASE_DIR           = '/root/autodl-tmp/HybridVision/trackpre/gt_tracks'
OUTPUT_DIR         = '/root/autodl-tmp/HybridVision/trackpre/premodel'
DEFAULT_EVENT_ROOT = '/root/autodl-tmp/HybridVision/ori_dataset/train'


def xyxy_to_cxcywh_norm(boxes, img_w=IMG_W, img_h=IMG_H):
    """[N,4] xyxy → [N,4] normalized cxcywh"""
    cx = (boxes[:, 0] + boxes[:, 2]) / 2 / img_w
    cy = (boxes[:, 1] + boxes[:, 3]) / 2 / img_h
    w  = (boxes[:, 2] - boxes[:, 0]) / img_w
    h  = (boxes[:, 3] - boxes[:, 1]) / img_h
    return np.stack([cx, cy, w, h], axis=1)


def cxcywh_norm_to_xyxy(pred, img_w=IMG_W, img_h=IMG_H):
    """[4] normalized cxcywh → [4] xyxy"""
    cx, cy, w, h = pred
    return np.array([
        (cx - w / 2) * img_w,
        (cy - h / 2) * img_h,
        (cx + w / 2) * img_w,
        (cy + h / 2) * img_h,
    ])


# ─── Event frame utilities ─────────────────────────────────────────────────
def build_event_index(seq_dir):
    """
    Scan <seq_num>/Event/Frames/*.png.
    Returns two parallel sorted lists:
        ts_list:   [int]  timestamps in µs
        path_list: [str]  corresponding file paths

    Supports two filename conventions:
      - Original FRED:  Video_N_frame_{timestamp_us}.png
                        → last '_'-segment is already µs
      - FRED_ready GT:  frame_{frame_number}.png
                        → frame number converted to µs via frame_num / FPS * 1e6
    """
    frames_dir = os.path.join(seq_dir, 'Event', 'Frames')
    if not os.path.isdir(frames_dir):
        return [], []

    entries = []
    for fname in os.listdir(frames_dir):
        if not fname.endswith('.png'):
            continue
        ts_str = fname.rsplit('_', 1)[-1].replace('.png', '')
        try:
            raw = int(ts_str)
        except ValueError:
            continue
        # Distinguish frame-number files (frame_XXXXXX.png, raw ≤ ~1e6)
        # from native µs timestamps (typically > 1e9).
        if raw < 10_000_000:          # treat as frame number → convert to µs
            ts_us = int(raw / FPS * 1e6)
        else:                          # already µs
            ts_us = raw
        entries.append((ts_us, os.path.join(frames_dir, fname)))
    entries.sort()
    ts_list   = [e[0] for e in entries]
    path_list = [e[1] for e in entries]
    return ts_list, path_list


def find_event_frame(ts_sec, ts_list, path_list):
    """
    Binary search for the event frame closest to ts_sec (seconds).
    Returns file path, or None if the index is empty.
    """
    if not ts_list:
        return None
    ts_us = int(ts_sec * 1e6)
    idx = bisect.bisect_left(ts_list, ts_us)
    # Pick closer of left/right neighbor
    if idx == len(ts_list):
        idx -= 1
    elif idx > 0 and (ts_us - ts_list[idx - 1]) < (ts_list[idx] - ts_us):
        idx -= 1
    return path_list[idx]


from functools import lru_cache

@lru_cache(maxsize=512)
def _load_image_cached(frame_path):
    """Load and cache a full event frame (shared across crops from the same frame)."""
    mode = 'L' if EVENT_CHANNELS == 1 else 'RGB'
    return Image.open(frame_path).convert(mode)


def load_event_crop(frame_path, box_xyxy):
    """
    Load a grayscale (or RGB) event frame, crop to the bounding box,
    resize to CROP_SIZE × CROP_SIZE.
    box_xyxy: [x1, y1, x2, y2] in pixels.
    Returns float32 tensor [EVENT_CHANNELS, CROP_SIZE, CROP_SIZE] in [0, 1].
    """
    img = _load_image_cached(frame_path)

    x1, y1, x2, y2 = box_xyxy
    x1 = max(0, int(x1));  y1 = max(0, int(y1))
    x2 = min(img.width,  max(int(x2), x1 + 1))
    y2 = min(img.height, max(int(y2), y1 + 1))

    crop = img.crop((x1, y1, x2, y2)).resize(
        (CROP_SIZE, CROP_SIZE), Image.BILINEAR
    )
    arr = np.array(crop, dtype=np.float32) / 255.0
    if EVENT_CHANNELS == 1:
        arr = arr[np.newaxis]           # [1, H, W]
    else:
        arr = arr.transpose(2, 0, 1)   # [3, H, W]
    return torch.tensor(arr)


# ─── Dataset ───────────────────────────────────────────────────────────────
class TrajectoryDataset(Dataset):
    """
    Each raw sample stores metadata (no images loaded at init time):
        past_meta:    list of N_p × (box_xyxy [4], event_path or None)
        future_xyxy:  list of N_f × [x1, y1, x2, y2]

    __getitem__ loads event crops on the fly and returns:
        enc_boxes:   [N_p, 4]              normalized cxcywh  (encoder input)
        enc_crops:   [N_p, C, S, S]        event crops        (encoder input)
        tgt_past:    [N_p, 4]              normalized cxcywh  (past GT)
        tgt_future:  [N_f, 4]             normalized cxcywh  (future GT)
    """

    def __init__(self, raw_samples):
        self.samples = raw_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        past_meta, future_xyxy = self.samples[idx]

        enc_boxes = []
        enc_crops = []
        for box_xyxy, evt_path in past_meta:
            x1, y1, x2, y2 = box_xyxy
            enc_boxes.append([
                (x1 + x2) / 2 / IMG_W,
                (y1 + y2) / 2 / IMG_H,
                (x2 - x1)      / IMG_W,
                (y2 - y1)      / IMG_H,
            ])
            if evt_path is not None:
                crop = load_event_crop(evt_path, box_xyxy)
            else:
                crop = torch.zeros(EVENT_CHANNELS, CROP_SIZE, CROP_SIZE)
            enc_crops.append(crop)

        past_xyxy = np.array([m[0] for m in past_meta], dtype=np.float32)
        tgt_past   = xyxy_to_cxcywh_norm(past_xyxy)
        tgt_future = xyxy_to_cxcywh_norm(
            np.array(future_xyxy, dtype=np.float32)
        )

        return (
            torch.tensor(enc_boxes, dtype=torch.float32),   # [N_p, 4]
            torch.stack(enc_crops),                           # [N_p, C, S, S]
            torch.tensor(tgt_past,   dtype=torch.float32),  # [N_p, 4]
            torch.tensor(tgt_future, dtype=torch.float32),  # [N_f, 4]
        )


# ─── Model ─────────────────────────────────────────────────────────────────
class EventCNN(nn.Module):
    """3-layer CNN: event crop → feature vector."""

    def __init__(self, in_ch=EVENT_CHANNELS, out_dim=EVENT_FEAT_DIM):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x):
        return self.fc(self.backbone(x).flatten(1))


class TrajectoryTransformer(nn.Module):
    """
    Encoder: observes N_p past frames (bbox coords + event CNN features).
    Decoder: dual output head.
      - Queries 0 … N_p-1  → past reconstruction
      - Queries N_p … N_p+N_f-1 → future prediction
    """

    def __init__(self, d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS,
                 event_feat_dim=EVENT_FEAT_DIM, dropout=0.1):
        super().__init__()
        self.cnn = EventCNN(out_dim=event_feat_dim)

        # Fuse bbox (4) + event features → d_model
        self.input_proj = nn.Linear(4 + event_feat_dim, d_model)

        # Learnable positional encoding for encoder sequence
        self.enc_pos = nn.Parameter(torch.randn(1, N_p, d_model))

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers, norm=nn.LayerNorm(d_model)
        )

        # Learnable decoder queries: N_p past + N_f future
        self.dec_queries = nn.Parameter(torch.randn(1, N_p + N_f, d_model))

        # Transformer decoder
        dec_layer = nn.TransformerDecoderLayer(
            d_model, nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            dec_layer, num_layers, norm=nn.LayerNorm(d_model)
        )

        # Output head → normalized cxcywh
        self.out_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 4),
        )

    def forward(self, bbox_seq, crop_seq):
        """
        bbox_seq:  [B, N_p, 4]             normalized cxcywh
        crop_seq:  [B, N_p, C, S, S]       event crops
        Returns:
            past_pred:   [B, N_p, 4]
            future_pred: [B, N_f, 4]
        """
        B = bbox_seq.size(0)

        # Event CNN (merge batch and time dimensions)
        crops = crop_seq.view(B * N_p, EVENT_CHANNELS, CROP_SIZE, CROP_SIZE)
        evt_feats = self.cnn(crops).view(B, N_p, EVENT_FEAT_DIM)  # [B, N_p, E]

        # Encoder input
        src = self.input_proj(
            torch.cat([bbox_seq, evt_feats], dim=-1)
        )                                                # [B, N_p, d]
        src = src + self.enc_pos

        memory = self.encoder(src)                       # [B, N_p, d]

        # Decoder
        queries = self.dec_queries.expand(B, -1, -1)    # [B, N_p+N_f, d]
        out = self.decoder(queries, memory)              # [B, N_p+N_f, d]

        preds = self.out_head(out)                       # [B, N_p+N_f, 4]
        return preds[:, :N_p], preds[:, N_p:]           # past, future


def load_trajectory_samples(base_dir, events_root=None, min_len=N_p + N_f):
    """
    Scan all <base_dir>/*/tracks.txt sequences.
    Returns a list of raw samples:
        (past_meta, future_xyxy)
    where
        past_meta:   list[N_p] of (box_xyxy, event_frame_path|None)
        future_xyxy: list[N_f] of [x1, y1, x2, y2]

    Event 查找优先级:
      1. seq_dir/Event/Frames/  (gt_to_tracks.py 创建的软链接)
      2. events_root/{seq_num}/  (自动从 seq_id 推导，如 "seq_0"→"0")
    """
    raw_samples = []

    for fpath in sorted(_glob.glob(os.path.join(base_dir, '*', 'tracks.txt'))):
        seq_dir = os.path.dirname(fpath)
        seq_id  = os.path.basename(seq_dir)
        seq_num = seq_id[3:].lstrip('_')   # "seq_0"→"0", "seq0"→"0", "seq13"→"13"

        # 优先用 seq_dir 下的软链接，找不到则回落到 events_root/{seq_num}
        ts_list, path_list = build_event_index(seq_dir)
        if not ts_list and events_root:
            fallback = os.path.join(events_root, seq_num)
            ts_list, path_list = build_event_index(fallback)
            if ts_list:
                print(f"[{seq_id}] event 来自 {fallback}")
            else:
                print(f"[warning] {seq_id}: 未找到 Event/Frames，event特征将为零")
        elif not ts_list:
            print(f"[warning] {seq_id}: 未找到 Event/Frames，event特征将为零")

        # Parse tracks.txt → group by track_id
        track_data = defaultdict(list)
        with open(fpath, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) != 6:
                    continue
                ts, track_id, x, y, w, h = parts
                ts = float(ts)
                x, y, w, h = float(x), float(y), float(w), float(h)
                xyxy = [x, y, x + w, y + h]
                evt_path = find_event_frame(ts, ts_list, path_list)
                track_data[f"{seq_id}_{track_id.strip()}"].append(
                    (ts, xyxy, evt_path)
                )

        # Build sliding-window samples
        for key, frames in track_data.items():
            frames.sort(key=lambda r: r[0])
            total = len(frames)
            if total < min_len:
                continue

            for i in range(total - N_p - N_f + 1):
                past   = frames[i       : i + N_p]
                future = frames[i + N_p : i + N_p + N_f]

                past_meta   = [(f[1], f[2]) for f in past]    # (xyxy, path)
                future_xyxy = [f[1] for f in future]

                raw_samples.append((past_meta, future_xyxy))

    return raw_samples


def train(
    base_dir    = BASE_DIR,
    events_root = DEFAULT_EVENT_ROOT,
    epochs      = 1,
    batch_size  = 4,
    lr          = 1e-4,
    num_workers = 4,
    save_path   = None,
):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if save_path is None:
        run_ts    = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(OUTPUT_DIR, f'transformer_{run_ts}_best.pt')

    print("加载轨迹数据 …")
    print(f"  tracks 目录:  {base_dir}")
    print(f"  events 根目录: {events_root}")
    raw = load_trajectory_samples(base_dir, events_root=events_root)
    print(f"有效滑动窗口样本: {len(raw)}  (Np={N_p}, Nf={N_f})")

    if not raw:
        print("错误: 无有效样本，请检查数据路径或轨迹最小长度。")
        return

    # Train / val split by sample (80/20)
    np.random.seed(42)
    idx   = np.random.permutation(len(raw))
    split = max(1, int(len(raw) * 0.8))
    train_raw = [raw[i] for i in idx[:split]]
    val_raw   = [raw[i] for i in idx[split:]]

    train_ds = TrajectoryDataset(train_raw)
    val_ds   = TrajectoryDataset(val_raw)
    print(f"train: {len(train_ds)}  val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=4,
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}")

    model     = TrajectoryTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_val = float('inf')
    scale    = (IMG_W + IMG_H) / 2
    scaler   = torch.cuda.amp.GradScaler()

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        tr_loss_f = tr_loss_p = 0.0

        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:4d}/{epochs} [Train]",
            leave=False,
            dynamic_ncols=True,
            disable=not sys.stdout.isatty(),
        )
        for step, (enc_boxes, enc_crops, tgt_past, tgt_future) in enumerate(train_bar, 1):
            enc_boxes  = enc_boxes.to(device)
            enc_crops  = enc_crops.to(device)
            tgt_past   = tgt_past.to(device)
            tgt_future = tgt_future.to(device)

            with torch.cuda.amp.autocast():
                past_pred, future_pred = model(enc_boxes, enc_crops)
                loss_f = criterion(future_pred, tgt_future)
                loss_p = criterion(past_pred,   tgt_past)
                loss   = loss_f + LAMBDA * loss_p

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            tr_loss_f += loss_f.item()
            tr_loss_p += loss_p.item()

            train_bar.set_postfix(
                loss_f=f"{tr_loss_f/step:.5f}",
                loss_p=f"{tr_loss_p/step:.5f}",
            )

        tr_loss_f /= len(train_loader)
        tr_loss_p /= len(train_loader)

        # ── Validate ──
        model.eval()
        val_loss_f = val_loss_p = 0.0

        val_bar = tqdm(
            val_loader,
            desc=f"Epoch {epoch:4d}/{epochs} [ Val ]",
            leave=False,
            dynamic_ncols=True,
            disable=not sys.stdout.isatty(),
        )
        with torch.no_grad():
            for step, (enc_boxes, enc_crops, tgt_past, tgt_future) in enumerate(val_bar, 1):
                enc_boxes  = enc_boxes.to(device)
                enc_crops  = enc_crops.to(device)
                tgt_past   = tgt_past.to(device)
                tgt_future = tgt_future.to(device)

                past_pred, future_pred = model(enc_boxes, enc_crops)
                val_loss_f += criterion(future_pred, tgt_future).item()
                val_loss_p += criterion(past_pred,   tgt_past).item()

                val_bar.set_postfix(
                    loss_f=f"{val_loss_f/step:.5f}",
                    loss_p=f"{val_loss_p/step:.5f}",
                )

        val_loss_f /= max(len(val_loader), 1)
        val_loss_p /= max(len(val_loader), 1)
        val_total   = val_loss_f + LAMBDA * val_loss_p

        scheduler.step()

        # 每个 epoch 打印一行汇总
        saved_mark = ""
        if val_total < best_val:
            best_val   = val_total
            saved_mark = "  [saved]"
            torch.save({
                'state_dict': model.state_dict(),
                'img_w': IMG_W, 'img_h': IMG_H,
                'n_p': N_p, 'n_f': N_f, 'fps': FPS,
                'crop_size': CROP_SIZE,
                'event_channels': EVENT_CHANNELS,
                'event_feat_dim': EVENT_FEAT_DIM,
                'd_model': D_MODEL, 'nhead': NHEAD, 'num_layers': NUM_LAYERS,
            }, save_path)

        lr_now = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch:4d}/{epochs}  "
            f"lr={lr_now:.2e}  "
            f"train_f={tr_loss_f:.5f}  train_p={tr_loss_p:.5f}  "
            f"val_f={val_loss_f:.5f}(≈{val_loss_f**0.5*scale:.1f}px)  "
            f"val_p={val_loss_p:.5f}  val_total={val_total:.5f}"
            f"{saved_mark}"
        )

    print(f"\n训练完成！最佳 val total loss: {best_val:.6f}")
    print(f"  future ADE ≈ {(best_val / (1 + LAMBDA))**0.5 * scale:.1f} px")
    print(f"模型已保存至: {save_path}")


if __name__ == '__main__':
    train()
