from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from fred_fusion_utils import save_json, write_six_channel_tiff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict with a 6-channel FRED early-fusion YOLO model.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--event", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=Path("reports/pred_examples"))
    return parser.parse_args()


def main() -> None:
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: install ultralytics and torch before prediction.") from exc

    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))

    with tempfile.TemporaryDirectory() as tmp:
        fused = Path(tmp) / f"{args.rgb.stem}_fusion.tiff"
        write_six_channel_tiff(args.rgb, args.event, fused)
        results = model.predict(source=str(fused), imgsz=args.imgsz, conf=args.conf, device=args.device, save=False)

    raw = []
    for result in results:
        result.save(filename=str(args.out / f"{args.rgb.stem}_prediction.jpg"))
        boxes = []
        if result.boxes is not None:
            for box in result.boxes:
                boxes.append(
                    {
                        "xyxy": [float(x) for x in box.xyxy[0].tolist()],
                        "conf": float(box.conf[0]),
                        "cls": int(box.cls[0]),
                    }
                )
        raw.append({"image": str(args.rgb), "event": str(args.event), "boxes": boxes})
    save_json(args.out / f"{args.rgb.stem}_prediction.json", raw)
    print(f"wrote predictions to {args.out}")


if __name__ == "__main__":
    main()
