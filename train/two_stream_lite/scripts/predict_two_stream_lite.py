from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fred_two_stream.ultralytics_patch import register_rgb_event_stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict with a trained RGB+Event two-stream-lite model.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--event", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--out", type=Path, default=Path("reports/two_stream_predictions"))
    return parser.parse_args()


def load_pair(rgb_path: Path, event_path: Path) -> tuple[np.ndarray, Image.Image]:
    rgb = Image.open(rgb_path).convert("RGB")
    event = Image.open(event_path).convert("RGB")
    if event.size != rgb.size:
        event = event.resize(rgb.size, Image.BILINEAR)
    fused = np.concatenate([np.asarray(rgb), np.asarray(event)], axis=2)
    return fused, rgb


def draw_predictions(rgb: Image.Image, boxes: list[dict], out_path: Path) -> None:
    image = rgb.copy()
    draw = ImageDraw.Draw(image)
    for box in boxes:
        x1, y1, x2, y2 = box["xyxy"]
        conf = box["confidence"]
        draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=3)
        draw.text((x1 + 2, max(0, y1 - 14)), f"{conf:.2f}", fill=(255, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def main() -> None:
    args = parse_args()
    register_rgb_event_stem()

    from ultralytics import YOLO

    fused, rgb = load_pair(args.rgb, args.event)
    model = YOLO(args.weights)
    start = time.perf_counter()
    results = model.predict(source=[fused], imgsz=args.imgsz, conf=args.conf, verbose=False)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    boxes: list[dict] = []
    if results:
        result = results[0]
        for item in result.boxes:
            boxes.append(
                {
                    "xyxy": [float(v) for v in item.xyxy[0].tolist()],
                    "confidence": float(item.conf[0]),
                    "class_id": int(item.cls[0]),
                }
            )

    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.rgb.stem
    draw_predictions(rgb, boxes, args.out / f"{stem}_two_stream_bbox.jpg")
    payload = {"rgb": str(args.rgb), "event": str(args.event), "inference_time_ms": elapsed_ms, "boxes": boxes}
    (args.out / f"{stem}_predictions.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (args.out / f"{stem}_predictions.txt").open("w", encoding="utf-8") as file:
        for box in boxes:
            file.write(
                f"{box['class_id']} {box['confidence']:.6f} "
                + " ".join(f"{v:.2f}" for v in box["xyxy"])
                + "\n"
            )
    print(f"predictions: {len(boxes)}")
    print(f"inference_time_ms: {elapsed_ms:.2f}")


if __name__ == "__main__":
    main()
