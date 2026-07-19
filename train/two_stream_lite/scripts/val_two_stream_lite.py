from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fred_two_stream.ultralytics_patch import register_rgb_event_stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a trained RGB+Event two-stream-lite model.")
    parser.add_argument("--weights", required=True, help="Path to best.pt or last.pt.")
    parser.add_argument("--data", type=Path, default=Path("data/fred_fusion_6ch/fred_fusion_6ch.yaml"))
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default=0)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--project", default="runs/val")
    parser.add_argument("--name", default="two_stream_lite_val")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    register_rgb_event_stem()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    metrics = model.val(
        data=str(args.data),
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        split=args.split,
        conf=args.conf,
        iou=args.iou,
        project=args.project,
        name=args.name,
    )
    print(metrics)


if __name__ == "__main__":
    main()
