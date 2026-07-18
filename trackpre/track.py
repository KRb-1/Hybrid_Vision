import numpy as np
from collections import defaultdict

INPUT_FILE   = '/root/autodl-tmp/HybridVision/teamA_fusion/ami_project_ss26-main/predictions_fused.txt'
OUTPUT_DIR   = '/root/autodl-tmp/HybridVision/trackpre/track_result'              # 每条序列的 tracks.txt 存放目录
FPS          = 30.0

# 文件名格式: seq13_frame_002274.jpg
def parse_seq_frame(filename):
    """返回 (seq_id, frame_number)"""
    base = filename.replace('.jpg', '')
    parts = base.split('_frame_')
    seq_id = parts[0]              # e.g. 'seq13'
    frame_num = int(parts[1])      # e.g. 2274
    return seq_id, frame_num


class KalmanFilter:
    """跟踪 [cx, cy, s, r] (中心坐标, 面积, 宽高比)，速度为隐状态。"""

    def __init__(self):
        dt = 1.0
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i + 4] = dt

        self.H = np.zeros((4, 8))
        self.H[:4, :4] = np.eye(4)

        self.Q = np.eye(8) * 1.0
        self.Q[4:, 4:] *= 0.01

        self.R = np.eye(4) * 1.0

    def initiate(self, measurement):
        x = np.zeros(8)
        x[:4] = measurement
        P = np.eye(8) * 10.0
        return x, P

    def predict(self, x, P):
        x = self.F @ x
        P = self.F @ P @ self.F.T + self.Q
        return x, P

    def update(self, x, P, z):
        S = self.H @ P @ self.H.T + self.R
        K = P @ self.H.T @ np.linalg.inv(S)
        y = z - self.H @ x
        x = x + K @ y
        P = (np.eye(8) - K @ self.H) @ P
        return x, P


def xyxy_to_csr(box):
    """[x1,y1,x2,y2] → [cx,cy,s,r]"""
    w = box[2] - box[0]
    h = box[3] - box[1]
    cx = box[0] + w / 2
    cy = box[1] + h / 2
    s = w * h
    r = w / float(h) if h > 0 else 1.0
    return np.array([cx, cy, s, r])


def csr_to_xyxy(x):
    """[cx,cy,s,r] → [x1,y1,x2,y2]"""
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w if w > 0 else 0
    return np.array([
        x[0] - w / 2, x[1] - h / 2,
        x[0] + w / 2, x[1] + h / 2
    ])


def iou_matrix(dets, trks):
    """计算 [N,4] × [M,4] 的 IoU 矩阵"""
    n, m = len(dets), len(trks)
    if n == 0 or m == 0:
        return np.zeros((n, m))

    dets = np.array(dets)
    trks = np.array(trks)

    area_d = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
    area_t = (trks[:, 2] - trks[:, 0]) * (trks[:, 3] - trks[:, 1])

    iou = np.zeros((n, m))
    for i in range(n):
        xx1 = np.maximum(dets[i, 0], trks[:, 0])
        yy1 = np.maximum(dets[i, 1], trks[:, 1])
        xx2 = np.minimum(dets[i, 2], trks[:, 2])
        yy2 = np.minimum(dets[i, 3], trks[:, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou[i] = inter / (area_d[i] + area_t - inter + 1e-6)
    return iou


def greedy_match(cost, thresh):
    """贪心匹配，返回 [(det_idx, trk_idx)] 匹配对"""
    matched = []
    used_d = set()
    used_t = set()
    # 按相似度从高到低排序
    pairs = sorted(
        [(cost[i, j], i, j) for i in range(cost.shape[0]) for j in range(cost.shape[1])],
        key=lambda x: -x[0]
    )
    for val, i, j in pairs:
        if val < thresh:
            break
        if i not in used_d and j not in used_t:
            matched.append((i, j))
            used_d.add(i)
            used_t.add(j)
    return matched


class Track:
    _next_id = 1

    def __init__(self, box, score):
        self.id = Track._next_id
        Track._next_id += 1
        self.kf = KalmanFilter()
        self.x, self.P = self.kf.initiate(xyxy_to_csr(box))
        self.score = score
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.state = 'tentative'  # tentative / confirmed / lost

    def predict(self):
        self.x, self.P = self.kf.predict(self.x, self.P)
        self.age += 1
        self.time_since_update += 1

    def update(self, box, score):
        z = xyxy_to_csr(box)
        self.x, self.P = self.kf.update(self.x, self.P, z)
        self.score = score
        self.hits += 1
        self.time_since_update = 0
        if self.hits >= 3:
            self.state = 'confirmed'

    def get_box(self):
        return csr_to_xyxy(self.x)


class ByteTracker:
    """ByteTrack 风格的两阶段关联追踪器"""

    def __init__(self, high_thresh=0.5, low_thresh=0.1, iou_thresh=0.3,
                 max_lost=30, min_hits=3):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.iou_thresh = iou_thresh
        self.max_lost = max_lost
        self.min_hits = min_hits
        self.tracks = []

    def reset(self):
        self.tracks = []
        Track._next_id = 1

    def update(self, boxes, scores):
        """
        boxes: [N, 4] xyxy
        scores: [N]
        返回: [(x1,y1,x2,y2, conf, track_id), ...]
        """
        # 预测所有 track 的下一步
        for t in self.tracks:
            t.predict()

        # 分高低分检测框
        high_mask = scores >= self.high_thresh
        low_mask = (scores >= self.low_thresh) & ~high_mask

        dets_high = boxes[high_mask]
        scores_high = scores[high_mask]
        dets_low = boxes[low_mask]
        scores_low = scores[low_mask]

        trk_boxes = np.array([t.get_box() for t in self.tracks]) if self.tracks else np.empty((0, 4))

        # 阶段一：高分检测框与 confirmed/tentative 轨迹匹配
        if len(dets_high) > 0 and len(trk_boxes) > 0:
            iou = iou_matrix(dets_high, trk_boxes)
            matched = greedy_match(iou, self.iou_thresh)
        else:
            matched = []

        matched_d = {d for d, _ in matched}
        matched_t = {t for _, t in matched}

        for d, t in matched:
            self.tracks[t].update(dets_high[d], scores_high[d])

        # 阶段二：低分检测框与未匹配的 lost 轨迹匹配
        unmatched_trks = [i for i in range(len(self.tracks)) if i not in matched_t]
        lost_trks_idx = [i for i in unmatched_trks if self.tracks[i].state == 'confirmed']

        if len(dets_low) > 0 and len(lost_trks_idx) > 0:
            lost_boxes = np.array([self.tracks[i].get_box() for i in lost_trks_idx])
            iou2 = iou_matrix(dets_low, lost_boxes)
            matched2 = greedy_match(iou2, self.iou_thresh)
            for d, ti in matched2:
                self.tracks[lost_trks_idx[ti]].update(dets_low[d], scores_low[d])
                matched_t.add(lost_trks_idx[ti])

        # 未匹配的高分检测框 → 新建轨迹
        for d in range(len(dets_high)):
            if d not in matched_d:
                self.tracks.append(Track(dets_high[d], scores_high[d]))

        # 删除长期 lost 的轨迹
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_lost]

        # 输出已确认且当前被更新的轨迹
        results = []
        for t in self.tracks:
            if t.state == 'confirmed' and t.time_since_update == 0:
                b = t.get_box()
                results.append((b[0], b[1], b[2], b[3], t.score, t.id))
        return results


def main():
    import os

    # 1. read detection results
    raw_preds = defaultdict(list)
    with open(INPUT_FILE, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 7:
                filename, x1, y1, x2, y2, conf, cls_id = parts
                raw_preds[filename].append(
                    [float(x1), float(y1), float(x2), float(y2), float(conf)]
                )

    # 2. 按序列分组，帧内排序
    seq_frames = defaultdict(list)
    for filename in raw_preds:
        seq_id, frame_num = parse_seq_frame(filename)
        seq_frames[seq_id].append((frame_num, filename))
    for seq_id in seq_frames:
        seq_frames[seq_id].sort(key=lambda x: x[0])

    print(f"共检测到 {len(seq_frames)} 条序列: {sorted(seq_frames.keys())}")

    # 3. 对每条序列运行 ByteTracker
    tracker = ByteTracker(
        high_thresh=0.4,
        low_thresh=0.1,
        iou_thresh=0.3,
        max_lost=30,
        min_hits=3,
    )

    all_tracked_flat = []   # 合并文件（供 train_lstm.py 使用，空格分隔 xyxy）
    total_frames = 0

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for seq_id in sorted(seq_frames.keys()):
        frames = seq_frames[seq_id]
        tracker.reset()

        seq_lines = []  # 当前序列的 GT 格式行

        for frame_num, filename in frames:
            dets = raw_preds.get(filename, [])
            if len(dets) == 0:
                boxes = np.empty((0, 4))
                scores = np.empty(0)
            else:
                arr = np.array(dets)
                boxes = arr[:, :4]
                scores = arr[:, 4]

            results = tracker.update(boxes, scores)

            # timestamp = 序列内相对时间（秒）
            timestamp = frame_num / FPS

            for (x1, y1, x2, y2, conf, track_id) in results:
                w = x2 - x1
                h = y2 - y1

                # GT 格式: timestamp,track_id,x1,y1,w,h
                seq_lines.append(
                    f"{timestamp:.6f},{track_id},{int(round(x1))},{int(round(y1))},{int(round(w))},{int(round(h))}"
                )

                # 合并文件格式（供 train_lstm.py）: filename x1 y1 x2 y2 conf cls_id track_id
                all_tracked_flat.append(
                    f"{filename} {x1:.3f} {y1:.3f} {x2:.3f} {y2:.3f} {conf:.4f} 0 {track_id}"
                )

        # 按序列写入单独的 tracks.txt（与 GT 文件夹结构一致）
        seq_out_dir = os.path.join(OUTPUT_DIR, seq_id)
        os.makedirs(seq_out_dir, exist_ok=True)
        seq_out_path = os.path.join(seq_out_dir, 'tracks.txt')
        with open(seq_out_path, 'w') as f:
            for line in seq_lines:
                f.write(line + '\n')
        print(f" [{seq_id}] {len(seq_lines)} 条记录 → {seq_out_path}")

        total_frames += len(frames)

    print(f"Tracking finished！")
    print(f"processed frames: {total_frames}")
    print(f"output bounding boxes: {len(all_tracked_flat)}")
    print(f"output directory: {OUTPUT_DIR}/{{seq_id}}/tracks.txt")


if __name__ == '__main__':
    main()
