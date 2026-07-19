from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from fred_fusion_utils import (
    Sample,
    collect_fused_split,
    find_by_stem,
    image_files,
    load_data_yaml,
    write_six_channel_tiff,
)


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch inference with a 6-channel early-fusion YOLO model.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to best.pt or last.pt.")
    parser.add_argument("--rgb-dir", type=Path, default=PROJECT_ROOT / "FRED_ready" / "test" / "images")
    parser.add_argument("--event-dir", type=Path, default=PROJECT_ROOT / "FRED_ready" / "test" / "events")
    parser.add_argument("--data", type=Path, default=None, help="Optional fallback path to a TIFF fred_fusion_6ch.yaml.")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=Path("predictions_early_fusion.txt"))
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for quick checks.")
    return parser.parse_args()


def collect_raw_pairs(rgb_dir: Path, event_dir: Path) -> list[Sample]:
    samples: list[Sample] = []
    for rgb in image_files(rgb_dir):
        event = find_by_stem(event_dir, rgb.stem)
        if event is not None:
            samples.append(Sample(rgb.stem, rgb, event, Path()))
    return samples


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: install ultralytics before inference.") from exc

    use_raw_pairs = args.rgb_dir.exists() and args.event_dir.exists()
    if use_raw_pairs:
        samples = collect_raw_pairs(args.rgb_dir, args.event_dir)
        source_desc = f"{args.rgb_dir} + {args.event_dir}"
    elif args.data is not None:
        data = load_data_yaml(args.data)
        samples = collect_fused_split(data, args.split)
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

    model = YOLO(str(args.weights))
    lines: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for index, sample in enumerate(samples, start=1):
            if use_raw_pairs:
                source = tmp_dir / f"{sample.stem}_fusion.tiff"
                write_six_channel_tiff(sample.rgb, sample.event, source)
                output_name = sample.rgb.name
            else:
                source = sample.rgb
                output_name = f"{sample.rgb.stem}.jpg"
            results = model.predict(
                source=str(source),
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
