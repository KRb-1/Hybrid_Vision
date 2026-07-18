import os
import shutil
from datetime import datetime


def run_detection(rgb_path, event_path, output_folder):
    """
    Input:
        rgb_path: path to uploaded RGB video
        event_path: path to uploaded event file or event frame video
        output_folder: folder to save output video

    Output:
        output_path: detection result video path
        event_vis_path: event visualization video path
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_path = os.path.join(output_folder, f"detection_result_{timestamp}.mp4")

    # Placeholder:
    # 目前还没有接模型，所以先把 RGB 视频复制成结果视频
    shutil.copy(rgb_path, output_path)

    # 如果 event 本身是视频，就直接返回它
    # 如果 event 是 npy/h5/txt/csv，后面需要在这里生成 event visualization video
    event_ext = event_path.rsplit(".", 1)[-1].lower()

    if event_ext in ["mp4", "avi", "mov", "mkv"]:
        event_vis_path = event_path
    else:
        event_vis_path = None

    return output_path, event_vis_path

# def run_detection(rgb_path, event_path, output_folder):
    # 1. 读取 RGB 视频
    # 2. 读取 event 数据
    # 3. 生成 event frame 或 event visualization
    # 4. 时间同步 RGB frame 和 event frame
    # 5. 输入 hybrid detection model
    # 6. 得到 bounding box, class label, confidence
    # 7. 用 OpenCV 把检测框画到 RGB 视频上
    # 8. 保存 result.mp4
    # 9. 返回 output_path 和 event_vis_path