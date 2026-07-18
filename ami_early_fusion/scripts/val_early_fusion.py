from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a trained 6-channel early-fusion YOLO model.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=0)
    return parser.parse_args()


def main() -> None:
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: install ultralytics and torch before validation.") from exc
    args = parse_args()
    model = YOLO(str(args.weights))
    print(model.val(data=str(args.data), imgsz=args.imgsz, batch=args.batch, device=args.device))


if __name__ == "__main__":
    main()
