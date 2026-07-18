"""
evaluate.py — Evaluate tracking and prediction accuracy against ground truth.

Tracking metrics : Precision, Recall, F1, MOTA, MOTP, ID Switches
Prediction metrics: ADE (Average Displacement Error), FDE (Final Displacement Error),
                    per-step mean IoU and mean center-distance

Ground truth format (ori_dataset/<seq_num>/coordinates.txt):
    timestamp: x1, y1, x2, y2, track_id, class_name

Tracking result format (pipeline_result/<seq_id>/tracks.txt):
    timestamp,track_id,x,y,w,h  (xywh, top-left origin)

Prediction result format (pipeline_result/<seq_id>/predictions_future.txt):
    ts_obs_end track_id step pred_x1 pred_y1 pred_x2 pred_y2
"""

import argparse
import json
import os
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment

FPS = 30.0
TS_TOL = 1.0 / FPS  # match within ±1 frame (~0.0333 s)


def load_gt(gt_file: str) -> dict:
    """
    Parse coordinates.txt.
    Returns dict: timestamp (float) → list of [x1, y1, x2, y2, track_id].
    """
    frames = {}
    with open(gt_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ts_str, rest = line.split(':', 1)
            ts = float(ts_str.strip())
            parts = [p.strip() for p in rest.split(',')]
            box = [float(parts[0]), float(parts[1]),
                   float(parts[2]), float(parts[3]),
                   int(parts[4])]
            frames.setdefault(ts, []).append(box)
    return frames


def load_tracks(tracks_file: str) -> dict:
    """
    Parse tracks.txt  (timestamp,track_id,x,y,w,h).
    Returns dict: timestamp (float) → list of [x1, y1, x2, y2, track_id].
    """
    frames = {}
    with open(tracks_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            ts = float(parts[0])
            tid = int(parts[1])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            frames.setdefault(ts, []).append([x, y, x + w, y + h, tid])
    return frames


def load_predictions(pred_file: str) -> list:
    """
    Parse predictions_future.txt.
    Returns list of dicts with keys:
        ts_obs_end, track_id, step, x1, y1, x2, y2, future_ts
    """
    preds = []
    with open(pred_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            p = line.split()
            ts_obs = float(p[0])
            step = int(p[2])
            preds.append({
                'ts_obs_end': ts_obs,
                'track_id': int(p[1]),
                'step': step,
                'x1': float(p[3]),
                'y1': float(p[4]),
                'x2': float(p[5]),
                'y2': float(p[6]),
                'future_ts': ts_obs + step / FPS,
            })
    return preds


def box_iou(a, b) -> float:
    """IoU between two [x1, y1, x2, y2] boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def center_dist(a, b) -> float:
    """Euclidean distance between box centers."""
    cx_a = (a[0] + a[2]) / 2
    cy_a = (a[1] + a[3]) / 2
    cx_b = (b[0] + b[2]) / 2
    cy_b = (b[1] + b[3]) / 2
    return float(np.hypot(cx_a - cx_b, cy_a - cy_b))


def build_iou_matrix(gt_boxes, pred_boxes) -> np.ndarray:
    mat = np.zeros((len(gt_boxes), len(pred_boxes)))
    for i, g in enumerate(gt_boxes):
        for j, p in enumerate(pred_boxes):
            mat[i, j] = box_iou(g, p)
    return mat


def nearest_ts(target: float, sorted_ts: np.ndarray):
    """
    Return (nearest_timestamp, diff) if within TS_TOL, else (None, inf).
    Uses binary search for O(log n) lookup.
    """
    idx = np.searchsorted(sorted_ts, target)
    candidates = []
    if idx < len(sorted_ts):
        candidates.append(sorted_ts[idx])
    if idx > 0:
        candidates.append(sorted_ts[idx - 1])
    if not candidates:
        return None, float('inf')
    best = min(candidates, key=lambda t: abs(t - target))
    diff = abs(best - target)
    return (float(best), diff) if diff <= TS_TOL else (None, float('inf'))


def evaluate_tracking(gt_frames: dict, track_frames: dict,
                      iou_thr: float = 0.5) -> dict:
    """
    Frame-by-frame Hungarian matching.

    Iterates over track timestamps; for each, finds the nearest GT frame
    within ±1 frame (TS_TOL).  Applies Hungarian matching on IoU.

    Returns a dict with: TP, FP, FN, ID_Switches, Precision, Recall, F1,
                         MOTA, MOTP, Total_GT_Boxes
    """
    gt_ts_arr = np.sort(np.array(list(gt_frames.keys())))
    track_ts_sorted = sorted(track_frames.keys())

    tp = fp = fn = id_sw = 0
    motp_iou_sum = 0.0
    prev_gt_to_pred: dict = {}   # gt_track_id → pred_track_id at previous frame

    for t in track_ts_sorted:
        pred_list = track_frames[t]               # [[x1,y1,x2,y2,tid], ...]
        near_gt_ts, _ = nearest_ts(t, gt_ts_arr)

        if near_gt_ts is None:
            # No GT close enough → all predictions are FP
            fp += len(pred_list)
            continue

        gt_list = gt_frames[near_gt_ts]           # [[x1,y1,x2,y2,tid], ...]
        n_gt   = len(gt_list)
        n_pred = len(pred_list)

        if n_gt == 0:
            fp += n_pred
            continue
        if n_pred == 0:
            fn += n_gt
            continue

        gt_boxes   = np.array([b[:4] for b in gt_list])
        pred_boxes = np.array([b[:4] for b in pred_list])
        gt_ids     = [b[4] for b in gt_list]
        pred_ids   = [b[4] for b in pred_list]

        iou_mat = build_iou_matrix(gt_boxes, pred_boxes)
        row_ind, col_ind = linear_sum_assignment(1.0 - iou_mat)

        matched_gt   = set()
        matched_pred = set()
        curr_gt_to_pred: dict = {}

        for r, c in zip(row_ind, col_ind):
            if iou_mat[r, c] >= iou_thr:
                matched_gt.add(r)
                matched_pred.add(c)
                tp += 1
                motp_iou_sum += iou_mat[r, c]
                gid = int(gt_ids[r])
                pid = int(pred_ids[c])
                curr_gt_to_pred[gid] = pid
                if gid in prev_gt_to_pred and prev_gt_to_pred[gid] != pid:
                    id_sw += 1

        fn += n_gt - len(matched_gt)
        fp += n_pred - len(matched_pred)
        prev_gt_to_pred = curr_gt_to_pred

    total_gt = tp + fn
    prec  = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec   = tp / total_gt  if total_gt  > 0 else 0.0
    f1    = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    mota  = 1.0 - (fp + fn + id_sw) / total_gt if total_gt > 0 else 0.0
    motp  = motp_iou_sum / tp if tp > 0 else 0.0

    return dict(
        TP=tp, FP=fp, FN=fn, ID_Switches=id_sw,
        Total_GT_Boxes=total_gt,
        Precision=round(prec, 6),
        Recall=round(rec, 6),
        F1=round(f1, 6),
        MOTA=round(mota, 6),
        MOTP=round(motp, 6),
    )


def evaluate_predictions(gt_frames: dict, predictions: list) -> dict:
    """
    For each predicted future box, find the nearest GT frame (within ±1 frame).
    Among all GT boxes at that frame, pick the one with highest IoU (ties broken
    by center distance).

    Returns a dict with: ADE, FDE, n_matched, n_trajectories,
                         per_step (dict step → mean_IoU, mean_dist, n_samples)
    """
    gt_ts_arr = np.sort(np.array(list(gt_frames.keys())))

    per_step_iou  = {s: [] for s in range(1, 25)}
    per_step_dist = {s: [] for s in range(1, 25)}
    # traj_key → {step → center_dist}  (for ADE/FDE per trajectory)
    traj_dists: dict = {}

    for pred in predictions:
        future_ts = pred['future_ts']
        step      = pred['step']
        pred_box  = [pred['x1'], pred['y1'], pred['x2'], pred['y2']]

        near_gt_ts, _ = nearest_ts(future_ts, gt_ts_arr)
        if near_gt_ts is None:
            continue

        gt_list = gt_frames[near_gt_ts]
        if not gt_list:
            continue

        # Pick the best-matching GT box (highest IoU; tie → nearest center)
        best_iou  = -1.0
        best_dist = float('inf')
        for gt_b in gt_list:
            iou_val  = box_iou(pred_box, gt_b[:4])
            dist_val = center_dist(pred_box, gt_b[:4])
            if iou_val > best_iou or (iou_val == best_iou and dist_val < best_dist):
                best_iou  = iou_val
                best_dist = dist_val

        per_step_iou[step].append(max(0.0, best_iou))
        per_step_dist[step].append(best_dist)

        traj_key = (pred['ts_obs_end'], pred['track_id'])
        traj_dists.setdefault(traj_key, {})[step] = best_dist

    # ADE: mean center distance across ALL matched predictions
    all_dists = [d for dlist in per_step_dist.values() for d in dlist]
    ade = float(np.mean(all_dists)) if all_dists else float('nan')

    # FDE: center distance at the LAST available step for each trajectory
    fde_vals = []
    for step_map in traj_dists.values():
        if step_map:
            last_step = max(step_map)
            fde_vals.append(step_map[last_step])
    fde = float(np.mean(fde_vals)) if fde_vals else float('nan')

    per_step = {}
    for s in range(1, 25):
        per_step[s] = dict(
            mean_IoU  = round(float(np.mean(per_step_iou[s])),  6) if per_step_iou[s]  else None,
            mean_dist = round(float(np.mean(per_step_dist[s])), 4) if per_step_dist[s] else None,
            n_samples = len(per_step_iou[s]),
        )

    return dict(
        ADE=round(ade, 4),
        FDE=round(fde, 4),
        n_matched=len(all_dists),
        n_trajectories=len(traj_dists),
        per_step=per_step,
    )


def report_tracking(res: dict, seq: str):
    print(f"\n{'─'*58}")
    print(f"  Tracking Evaluation — {seq}")
    print(f"{'─'*58}")
    print(f"  Total GT boxes  : {res['Total_GT_Boxes']}")
    print(f"  TP / FP / FN    : {res['TP']} / {res['FP']} / {res['FN']}")
    print(f"  ID Switches     : {res['ID_Switches']}")
    print(f"  Precision       : {res['Precision']:.4f}")
    print(f"  Recall          : {res['Recall']:.4f}")
    print(f"  F1              : {res['F1']:.4f}")
    print(f"  MOTA            : {res['MOTA']:.4f}  (1 - (FP+FN+IDSW)/GT, higher=better)")
    print(f"  MOTP            : {res['MOTP']:.4f}  (mean IoU of matched pairs, higher=better)")


def report_predictions(res: dict, seq: str):
    print(f"\n{'─'*58}")
    print(f"  Prediction Evaluation — {seq}")
    print(f"{'─'*58}")
    print(f"  Matched predictions : {res['n_matched']}")
    print(f"  Trajectories        : {res['n_trajectories']}")
    print(f"  ADE (all steps)     : {res['ADE']:.2f} px  (Average Displacement Error)")
    print(f"  FDE (last step)     : {res['FDE']:.2f} px  (Final Displacement Error)")
    print()
    print(f"  Per-step breakdown  (1 step = 1/30 s ≈ 33 ms):")
    print(f"  {'Step':>4}  {'t (s)':>6}  {'Mean IoU':>9}  {'Mean dist (px)':>14}  {'Samples':>8}")
    print(f"  {'─'*4}  {'─'*6}  {'─'*9}  {'─'*14}  {'─'*8}")
    for s in range(1, 25):
        m = res['per_step'][s]
        t_s = s / FPS
        iou_str  = f"{m['mean_IoU']:.4f}"  if m['mean_IoU']  is not None else '  N/A  '
        dist_str = f"{m['mean_dist']:.2f}" if m['mean_dist'] is not None else '   N/A '
        print(f"  {s:>4}  {t_s:>6.3f}  {iou_str:>9}  {dist_str:>14}  {m['n_samples']:>8}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate tracking and trajectory prediction accuracy'
    )
    parser.add_argument('--tracks',      required=True,
                        help='Path to tracks.txt')
    parser.add_argument('--predictions', required=True,
                        help='Path to predictions_future.txt')
    parser.add_argument('--gt',          required=True,
                        help='Path to coordinates.txt (ground truth)')
    parser.add_argument('--iou_thr', type=float, default=0.5,
                        help='IoU threshold for tracking match (default: 0.5)')
    parser.add_argument('--output', default=None,
                        help='Output JSON file path (optional, defaults to eval_result.json '
                             'in the same directory as tracks.txt)')
    args = parser.parse_args()

    trk_file  = os.path.abspath(args.tracks)
    pred_file = os.path.abspath(args.predictions)
    gt_file   = os.path.abspath(args.gt)

    for path, label in [(gt_file, 'GT'), (trk_file, 'Tracks'), (pred_file, 'Predictions')]:
        if not os.path.exists(path):
            print(f"[Error] {label} file not found: {path}")
            sys.exit(1)

    print(f"Loading data …")
    print(f"  GT          : {gt_file}")
    print(f"  Tracks      : {trk_file}")
    print(f"  Predictions : {pred_file}")
    gt_frames    = load_gt(gt_file)
    track_frames = load_tracks(trk_file)
    predictions  = load_predictions(pred_file)

    gt_ts_sorted  = sorted(gt_frames.keys())
    trk_ts_sorted = sorted(track_frames.keys())
    print(f"  GT          : {sum(len(v) for v in gt_frames.values()):,} boxes, "
          f"t=[{gt_ts_sorted[0]:.3f}, {gt_ts_sorted[-1]:.3f}] s")
    print(f"  Tracks      : {sum(len(v) for v in track_frames.values()):,} boxes, "
          f"t=[{trk_ts_sorted[0]:.3f}, {trk_ts_sorted[-1]:.3f}] s")
    print(f"  Predictions : {len(predictions):,} rows")

    seq_label = os.path.basename(os.path.dirname(trk_file))

    print(f"\nEvaluating tracking  (IoU threshold = {args.iou_thr}) …")
    track_res = evaluate_tracking(gt_frames, track_frames, iou_thr=args.iou_thr)
    report_tracking(track_res, seq_label)

    print(f"\nEvaluating predictions …")
    pred_res = evaluate_predictions(gt_frames, predictions)
    report_predictions(pred_res, seq_label)

    # ── Save ──
    output_file = args.output
    if output_file is None:
        result_dir = os.path.dirname(trk_file)
        output_file = os.path.join(result_dir, 'eval_result.json')

    results = {
        'tracks_file': trk_file,
        'predictions_file': pred_file,
        'gt_file': gt_file,
        'iou_threshold': args.iou_thr,
        'tracking': track_res,
        'prediction': {k: v for k, v in pred_res.items() if k != 'per_step'},
        'prediction_per_step': pred_res['per_step'],
    }
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {output_file}")


if __name__ == '__main__':
    main()
