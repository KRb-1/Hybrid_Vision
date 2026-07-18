"""
pipeline.py — Tracking + Trajectory Prediction Pipeline

Step 1: Read predictions_fused.txt → run ByteTracker → write per-sequence tracks.txt
Step 2: For each sequence, run transformer prediction → write predictions_future.txt

Output layout:
    {out_dir}/{seq_id}/tracks.txt
    {out_dir}/{seq_id}/predictions_future.txt
"""

import argparse
import os
import sys

import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

from track import parse_seq_frame, ByteTracker, Track
from predict_trajectory import main as run_prediction

FPS = 30.0

DEFAULT_FUSED      = '/root/autodl-tmp/HybridVision/teamA_fusion/ami_project_ss26-main/predictions_wbf_fused_official.txt'
DEFAULT_MODEL      = '/root/autodl-tmp/HybridVision/trackpre/premodel/transformer_20260705_034901_best.pt'
DEFAULT_EVENT_ROOT = '/root/autodl-tmp/HybridVision/ori_dataset/test'
DEFAULT_OUT_DIR    = '/root/autodl-tmp/HybridVision/trackpre/pipeline_result/'


# ─── Tracking ─────────────────────────────────────────────────────────

def run_tracking(fused_file, output_dir):
    """
    Read predictions_fused.txt, run ByteTracker per sequence,
    write {output_dir}/{seq_id}/tracks.txt in GT format:
        timestamp,track_id,x1,y1,w,h

    Returns: list of (seq_id, tracks_file_path)
    """
    # Parse fused detections: filename x1 y1 x2 y2 conf cls_id
    raw_preds = defaultdict(list)
    with open(fused_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 7:
                filename, x1, y1, x2, y2, conf, cls_id = parts
                raw_preds[filename].append(
                    [float(x1), float(y1), float(x2), float(y2), float(conf)]
                )

    # Group by sequence, sort by frame number
    seq_frames = defaultdict(list)
    for filename in raw_preds:
        seq_id, frame_num = parse_seq_frame(filename)
        seq_frames[seq_id].append((frame_num, filename))
    for seq_id in seq_frames:
        seq_frames[seq_id].sort(key=lambda r: r[0])

    print(f"[Tracking] {len(seq_frames)} 条序列: {sorted(seq_frames.keys())}")

    tracker = ByteTracker(
        high_thresh=0.4,
        low_thresh=0.1,
        iou_thresh=0.3,
        max_lost=30,
        min_hits=3,
    )

    seq_tracks = []

    for seq_id in sorted(seq_frames.keys()):
        frames = seq_frames[seq_id]
        tracker.reset()          # resets tracks list and Track._next_id = 1

        seq_lines = []

        for frame_num, filename in frames:
            dets = raw_preds.get(filename, [])
            if dets:
                arr    = np.array(dets)
                boxes  = arr[:, :4]
                scores = arr[:, 4]
            else:
                boxes  = np.empty((0, 4))
                scores = np.empty(0)

            results   = tracker.update(boxes, scores)
            timestamp = frame_num / FPS

            for (x1, y1, x2, y2, conf, track_id) in results:
                w = x2 - x1
                h = y2 - y1
                seq_lines.append(
                    f"{timestamp:.6f},{track_id},"
                    f"{int(round(x1))},{int(round(y1))},"
                    f"{int(round(w))},{int(round(h))}"
                )

        seq_out_dir = os.path.join(output_dir, seq_id)
        os.makedirs(seq_out_dir, exist_ok=True)
        tracks_path = os.path.join(seq_out_dir, 'tracks.txt')
        with open(tracks_path, 'w') as f:
            for line in seq_lines:
                f.write(line + '\n')

        print(f"  [{seq_id}] {len(seq_lines)} 条记录 → {tracks_path}")
        seq_tracks.append((seq_id, tracks_path))

    return seq_tracks


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Tracking + Trajectory Prediction Pipeline'
    )
    parser.add_argument('--fused',       default=DEFAULT_FUSED,
                        help='path to predictions_fused.txt')
    parser.add_argument('--model',       default=DEFAULT_MODEL,
                        help='path to transformer_best.pt')
    parser.add_argument('--events_root', default=DEFAULT_EVENT_ROOT,
                        help='parent dir containing per-sequence subfolders '
                             '(e.g. ami_project_ss26/); '
                             'seq_id "seq13" → events_root/13/')
    parser.add_argument('--out_dir',     default=DEFAULT_OUT_DIR,
                        help='output root directory')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Step 1 ──
    print("=" * 60)
    print("Tracking...")
    print(f"  fused: {args.fused}")
    print("=" * 60)
    seq_tracks = run_tracking(args.fused, args.out_dir)

    # ── Step 2 ──
    print()
    print("=" * 60)
    print("Trajectory Predicting...")
    print(f"  model:       {args.model}")
    print(f"  events_root: {args.events_root}")
    print("=" * 60)

    for seq_id, tracks_path in seq_tracks:
        pred_out = os.path.join(args.out_dir, seq_id, 'predictions_future.txt')

        # seq_id 格式: "seq0", "seq13", ... → 去掉 "seq" 前缀得到序列号
        seq_num = seq_id[3:] if seq_id.startswith('seq') else seq_id
        event_seq_dir = os.path.join(args.events_root, seq_num)

        print(f"\n[{seq_id}]  tracks: {tracks_path}")
        print(f"          events: {event_seq_dir}")
        run_prediction(
            tracks_file   = tracks_path,
            model_path    = args.model,
            event_seq_dir = event_seq_dir,
            output_file   = pred_out,
        )

    # ── Summary ──
    print()
    print("=" * 60)
    print("TrackPrediction Pipeline 完成！")
    print(f"输出目录: {args.out_dir}")


if __name__ == '__main__':
    main()
