from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fred_data_utils import find_by_stem, image_files, load_data_yaml, read_rgb, six_channel_to_rgb_event


PROJECT_ROOT = ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch inference with a two-stream-lite RGB+Event YOLO model.")
    parser.add_argument("--weights", required=True, help="Path to best.pt or last.pt.")
    parser.add_argument("--rgb-dir", type=Path, default=PROJECT_ROOT / "FRED_ready" / "test" / "images")
    parser.add_argument("--event-dir", type=Path, default=PROJECT_ROOT / "FRED_ready" / "test" / "events")
    parser.add_argument("--data", type=Path, default=None, help="Optional fallback path to a TIFF fred_fusion_6ch.yaml.")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=Path("predictions_two_stream_lite.txt"))
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for quick checks.")
    return parser.parse_args()


def fused_samples(data: dict, split: str) -> list[Path]:
    root = data["_root"]
    image_rel = data.get(split)
    if not image_rel:
        return []
    image_dir = Path(image_rel)
    if not image_dir.is_absolute():
        image_dir = root / image_dir
    return image_files(image_dir)


def raw_pairs(rgb_dir: Path, event_dir: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for rgb in image_files(rgb_dir):
        event = find_by_stem(event_dir, rgb.stem)
        if event is not None:
            pairs.append((rgb, event))
    return pairs


def load_six_channel_array(path: Path) -> np.ndarray:
    rgb, event = six_channel_to_rgb_event(path)
    if event.size != rgb.size:
        event = event.resize(rgb.size)
    return np.concatenate([np.asarray(rgb), np.asarray(event)], axis=2)


def load_pair_array(rgb_path: Path, event_path: Path) -> np.ndarray:
    rgb = read_rgb(rgb_path)
    event = read_rgb(event_path)
    if event.size != rgb.size:
        event = event.resize(rgb.size)
    return np.concatenate([np.asarray(rgb), np.asarray(event)], axis=2)


def main() -> None:
    args = parse_args()
    try:
        from fred_two_stream.ultralytics_patch import register_rgb_event_stem
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: install torch and ultralytics before inference.") from exc

    register_rgb_event_stem()
    use_raw_pairs = args.rgb_dir.exists() and args.event_dir.exists()
    if use_raw_pairs:
        samples = raw_pairs(args.rgb_dir, args.event_dir)
        source_desc = f"{args.rgb_dir} + {args.event_dir}"
    elif args.data is not None:
        samples = fused_samples(load_data_yaml(args.data), args.split)
        source_desc = str(args.data)
    else:
        raise SystemExit(
            "No RGB/Event image directories found. Pass --rgb-dir and --event-dir, "
            "or pass --data for an existing fused TIFF dataset."
        )
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        raise SystemExit(f"No samples found from {source_desc}")

    model = YOLO(args.weights)
    lines: list[str] = []
    for index, sample in enumerate(samples, start=1):
        if use_raw_pairs:
            rgb_path, event_path = sample
            fused = load_pair_array(rgb_path, event_path)
            output_name = rgb_path.name
        else:
            fused_path = sample
            fused = load_six_channel_array(fused_path)
            output_name = f"{fused_path.stem}.jpg"
        results = model.predict(
            source=[fused],
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            verbose=False,
        )
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                lines.append(f"{output_name} {x1} {y1} {x2} {y2} {conf} {cls_id}")
        if index % 500 == 0 or index == len(samples):
            print(f"inference: {index}/{len(samples)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"wrote {len(lines)} predictions to {args.out}")


if __name__ == "__main__":
    main()
