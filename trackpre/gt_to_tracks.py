"""
gt_to_tracks.py

FRED 数据集 coordinates.txt → tracks.txt，供 train_transformer.py 使用。

输入格式 (coordinates.txt / coordinates_rgb.txt):
    time: x1, y1, x2, y2, id, class
      time  → 录制开始后的时间（秒.微秒）
      x1,y1 → 左上角像素坐标
      x2,y2 → 右下角像素坐标
      id    → 目标唯一标识（跨帧一致）
      class → 无人机型号

策略：直接输出原始标注帧，不做插值
  - 对每个 drone id，按时间排序后检测断点：
      相邻两帧时间差 ≤ MAX_GAP_SEC → 连续，保持同一 track_id
      相邻两帧时间差 > MAX_GAP_SEC → 断开，分配新 track_id

输出格式 (tracks.txt): timestamp,track_id,x,y,w,h
  timestamp → 原始时间（秒，6 位小数）
  x,y       → 左上角像素坐标（四舍五入取整）
  w,h       → 宽高（像素，四舍五入取整）

同时在输出序列目录中创建 Event 软链接，供 train_transformer.py 使用。

用法:
  python gt_to_tracks.py                  # 自动处理 BASE 下所有序列
  python gt_to_tracks.py --seqs 0 1 2    # 仅处理指定序列
  python gt_to_tracks.py --use_rgb       # 使用 coordinates_rgb.txt
"""

import argparse
import os
from collections import defaultdict

BASE        = '/root/autodl-tmp/HybridVision/ori_dataset/train'
OUTPUT_ROOT = '/root/autodl-tmp/HybridVision/trackpre/gt_tracks'
MAX_GAP_SEC = 1.0   # 超过此秒数的空缺视为轨迹断点（约 30 帧）


def parse_coordinates(txt_path):
    """
    解析 coordinates.txt / coordinates_rgb.txt。
    格式: "time: x1, y1, x2, y2, id, class"
    返回 list of (time, x1, y1, x2, y2, drone_id)，按时间升序。
    """
    entries = []
    with open(txt_path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                time_str, rest = line.split(':', 1)
                t = float(time_str.strip())
                parts = rest.split(',', 5)
                x1 = float(parts[0].strip())
                y1 = float(parts[1].strip())
                x2 = float(parts[2].strip())
                y2 = float(parts[3].strip())
                drone_id = int(float(parts[4].strip()))
                entries.append((t, x1, y1, x2, y2, drone_id))
            except (ValueError, IndexError) as e:
                print(f"  [跳过 line {lineno}] {line!r}: {e}")
    entries.sort(key=lambda e: e[0])
    return entries


def process_sequence(seq_num, use_rgb=False, global_track_id_start=1):
    """
    处理 ami_project_ss26/{seq_num}/ 序列。
    返回 (输出路径 | None, 下一个可用 global_track_id)。
    """
    seq_dir  = os.path.join(BASE, str(seq_num))
    ann_name = 'coordinates_rgb.txt' if use_rgb else 'coordinates.txt'
    ann_path = os.path.join(seq_dir, ann_name)

    if not os.path.isfile(ann_path):
        print(f'[跳过] {ann_path} 不存在')
        return None, global_track_id_start

    print(f'\n[seq {seq_num}] 读取 {ann_name} ...')
    entries = parse_coordinates(ann_path)
    if not entries:
        print('  无有效标注')
        return None, global_track_id_start

    # 按 drone_id 分组
    by_id = defaultdict(list)
    for t, x1, y1, x2, y2, drone_id in entries:
        by_id[drone_id].append((t, x1, y1, x2, y2))

    print(f'  标注总数: {len(entries)}，无人机 ID: {sorted(by_id.keys())}')

    # 输出目录
    seq_out = os.path.join(OUTPUT_ROOT, f'seq_{seq_num}')
    os.makedirs(seq_out, exist_ok=True)

    # 创建 Event 软链接，供 build_event_index 使用
    evt_src  = os.path.join(seq_dir, 'Event')
    evt_link = os.path.join(seq_out, 'Event')
    if os.path.islink(evt_link):
        os.unlink(evt_link)
    if not os.path.exists(evt_link) and os.path.isdir(evt_src):
        os.symlink(evt_src, evt_link)

    lines           = []
    global_track_id = global_track_id_start
    total_wins      = 0

    for drone_id in sorted(by_id.keys()):
        drone_frames = sorted(by_id[drone_id], key=lambda r: r[0])

        # 按 MAX_GAP_SEC 切分为连续轨迹段
        segments  = []
        seg_start = 0
        for i in range(1, len(drone_frames)):
            if drone_frames[i][0] - drone_frames[i - 1][0] > MAX_GAP_SEC:
                segments.append(drone_frames[seg_start:i])
                seg_start = i
        segments.append(drone_frames[seg_start:])

        print(f'  drone_id={drone_id}: {len(drone_frames)} 帧，'
              f'{len(segments)} 个轨迹段')

        for seg in segments:
            if len(seg) < 2:
                global_track_id += 1
                continue

            for t, x1, y1, x2, y2 in seg:
                lines.append(
                    f'{t:.6f},{global_track_id},'
                    f'{int(round(x1))},{int(round(y1))},'
                    f'{int(round(x2 - x1))},{int(round(y2 - y1))}'
                )

            seg_frames = len(seg)
            wins = max(0, seg_frames - 35)
            total_wins += wins
            print(f'    track {global_track_id}: '
                  f't={seg[0][0]:.3f}~{seg[-1][0]:.3f}s '
                  f'({seg_frames} 帧), 窗口数={wins}')
            global_track_id += 1

    out_path = os.path.join(seq_out, 'tracks.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + ('\n' if lines else ''))

    print(f'  [{seq_num}] 合计记录: {len(lines)} 行, '
          f'轨迹段: {global_track_id - global_track_id_start} 条, '
          f'训练窗口: {total_wins}')
    print(f'  → {out_path}')
    return out_path, global_track_id


def discover_sequences():
    """
    扫描 BASE 目录，返回所有包含坐标标注文件的子目录名列表（排序后）。
    """
    seqs = []
    if not os.path.isdir(BASE):
        print(f'[错误] BASE 目录不存在: {BASE}')
        return seqs
    for entry in sorted(os.listdir(BASE)):
        seq_dir = os.path.join(BASE, entry)
        if not os.path.isdir(seq_dir):
            continue
        if (os.path.isfile(os.path.join(seq_dir, 'coordinates.txt')) or
                os.path.isfile(os.path.join(seq_dir, 'coordinates_rgb.txt'))):
            seqs.append(entry)
    return seqs


def main():
    parser = argparse.ArgumentParser(
        description='Convert FRED coordinates.txt to tracks.txt for training'
    )
    parser.add_argument(
        '--seqs', nargs='+', default=None,
        help='要处理的序列编号列表（默认自动发现 BASE 下所有序列）'
    )
    parser.add_argument(
        '--use_rgb', action='store_true',
        help='使用 coordinates_rgb.txt（仅 RGB 区域框）而非 coordinates.txt'
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    if args.seqs:
        seq_list = args.seqs
        print(f'处理指定序列: {seq_list}')
    else:
        seq_list = discover_sequences()
        if not seq_list:
            print(f'[警告] 在 {BASE} 下未找到任何含标注文件的序列目录')
            return
        print(f'自动发现 {len(seq_list)} 个序列: {seq_list}')

    next_id = 1
    for seq_num in seq_list:
        _, next_id = process_sequence(
            seq_num,
            use_rgb=args.use_rgb,
            global_track_id_start=next_id,
        )

    print(f'\n完成！输出目录: {OUTPUT_ROOT}')


if __name__ == '__main__':
    main()
