import cv2
import os
import bisect
from collections import defaultdict

IMAGE_DIR        = '/root/autodl-tmp/HybridVision/ori_dataset/test/51/RGB'
TRACKS_FILE      = '/root/autodl-tmp/HybridVision/trackpre/pipeline_result/seq51/tracks.txt'
PREDICTIONS_FILE = '/root/autodl-tmp/HybridVision/trackpre/pipeline_result/seq51/predictions_future.txt'
_seq_name = os.path.basename(os.path.dirname(TRACKS_FILE))   # "seqXX"
_seq_id   = _seq_name.replace('seq', '')                      # "XX"
OUTPUT_DIR = os.path.join('/root/autodl-tmp/HybridVision/trackpre/vis_result', _seq_id)

# ─── Image Index ──────────────────────────────────────────────────────────────
def parse_image_ts(fname):
    """Parse absolute timestamp (seconds) from:
    Video_N_HH_MM_SS.ffffff.jpg  →  HH*3600 + MM*60 + SS.ffffff
    """
    parts = fname.replace('.jpg', '').split('_')
    hh, mm = int(parts[2]), int(parts[3])
    ss = float(parts[4])
    return hh * 3600 + mm * 60 + ss


def build_image_index(image_dir):
    """Return sorted list of (abs_ts, filename) for all .jpg in image_dir."""
    imgs = sorted(f for f in os.listdir(image_dir) if f.endswith('.jpg'))
    return [(parse_image_ts(f), f) for f in imgs]


_MAX_DRIFT_S = 1 / 30.0  # tolerate at most one frame gap (~33ms)

def find_nearest_image(ts_abs, index):
    """Binary-search the nearest image filename for a given absolute timestamp.
    Returns None if the closest image is more than _MAX_DRIFT_S away (out of range).
    """
    ts_list = [t for t, _ in index]
    pos = bisect.bisect_left(ts_list, ts_abs)
    if pos == 0:
        candidate = index[0]
    elif pos >= len(index):
        candidate = index[-1]
    else:
        before, after = index[pos - 1], index[pos]
        candidate = before if abs(before[0] - ts_abs) <= abs(after[0] - ts_abs) else after
    if abs(candidate[0] - ts_abs) > _MAX_DRIFT_S:
        return None
    return candidate[1]


# ─── Color Helpers ────────────────────────────────────────────────────────────
_TRACK_PALETTE = [
    (255,  80,  80),
    ( 80, 200,  80),
    ( 80,  80, 255),
    (255, 220,  80),
    (200,  80, 255),
    ( 80, 255, 220),
    (180, 120,  60),
    ( 60, 180, 120),
    (120,  60, 180),
    (200, 200,  80),
]

def track_color(track_id):
    """Consistent BGR color per track ID."""
    return _TRACK_PALETTE[int(track_id) % len(_TRACK_PALETTE)]


def step_color(step, n_steps):
    """BGR gradient: green (step=1, near future) → red (step=n_steps, far future)."""
    alpha = (step - 1) / max(n_steps - 1, 1)
    return (0, int(255 * (1 - alpha)), int(255 * alpha))


# ─── Combined Visualization ───────────────────────────────────────────────────
def visualize_combined(image_dir, tracks_file, predictions_file, output_dir):
    """
    One output image per frame, showing both tracking and prediction results.

    tracks_file format (comma-separated, 6 fields):
        timestamp, track_id, x1, y1, w, h

    predictions_file format (space-separated, 7 fields):
        ts_obs_end  track_id  step  pred_x1  pred_y1  pred_x2  pred_y2

    Per-frame drawing:
      - Solid colored box (thick)  : current track observation, colored by track_id
      - Green→red gradient boxes (thin) : N_f predicted future positions for each track
        Labels shown at steps 1, N_f/4, N_f/2, 3*N_f/4, N_f
    """
    os.makedirs(output_dir, exist_ok=True)
    index = build_image_index(image_dir)
    t0 = index[0][0]

    # ── Parse tracks ──
    # frame_tracks[img_name] = [(track_id, x1, y1, x2, y2)]
    # ts_to_img[(track_id, ts_rel_rounded)] = img_name  (links predictions → frame)
    frame_tracks = defaultdict(list)
    ts_to_img    = {}

    with open(tracks_file) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) != 6:
                continue
            ts_rel, track_id, x, y, w, h = parts
            ts_rel_f = float(ts_rel)
            img_name = find_nearest_image(t0 + ts_rel_f, index)
            if img_name is None:
                continue
            x, y, w, h = float(x), float(y), float(w), float(h)
            tid = track_id.strip()
            frame_tracks[img_name].append((tid, int(x), int(y), int(x + w), int(y + h)))
            ts_to_img[(tid, round(ts_rel_f, 6))] = img_name

    # ── Parse predictions ──
    # frame_preds[img_name][track_id] = [(step, x1, y1, x2, y2), ...]
    frame_preds = defaultdict(lambda: defaultdict(list))

    with open(predictions_file) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) != 7:
                continue
            ts_end, tid, step, px1, py1, px2, py2 = parts
            tid = tid.strip()
            img_name = ts_to_img.get((tid, round(float(ts_end), 6)))
            if img_name is None:
                continue
            frame_preds[img_name][tid].append(
                (int(step), float(px1), float(py1), float(px2), float(py2))
            )

    # ── Draw per frame ──
    all_frames = sorted(set(frame_tracks) | set(frame_preds))
    print(f"[Combined] Drawing {len(all_frames)} frames "
          f"({len(frame_tracks)} with tracks, {len(frame_preds)} with predictions)...")

    for img_name in all_frames:
        img_path = os.path.join(image_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"  [skip] Cannot load: {img_path}")
            continue

        # Draw track boxes (solid, thick border)
        for (track_id, x1, y1, x2, y2) in frame_tracks.get(img_name, []):
            color = track_color(track_id)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"ID:{track_id}", (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw predicted future boxes (thin, green→red gradient)
        for track_id, preds in frame_preds.get(img_name, {}).items():
            preds.sort(key=lambda p: p[0])
            n_steps = preds[-1][0]
            label_steps = {1, n_steps // 4, n_steps // 2, n_steps * 3 // 4, n_steps}
            for (step, px1, py1, px2, py2) in preds:
                color = step_color(step, n_steps)
                ix1, iy1, ix2, iy2 = int(px1), int(py1), int(px2), int(py2)
                cv2.rectangle(img, (ix1, iy1), (ix2, iy2), color, 1)
                if step in label_steps:
                    cv2.putText(img, f"t+{step}", (ix1, iy2 + 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        cv2.imwrite(os.path.join(output_dir, img_name), img)

    print(f"[Combined] Done. {len(all_frames)} images saved to: {output_dir}")


if __name__ == '__main__':
    visualize_combined(IMAGE_DIR, TRACKS_FILE, PREDICTIONS_FILE, OUTPUT_DIR)
