from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fred_data_utils import find_by_stem, image_files, load_data_yaml, six_channel_to_rgb_event


PROJECT_ROOT = ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw two-stream-lite prediction boxes on RGB jpg images.")
    parser.add_argument("--image-dir", type=Path, default=PROJECT_ROOT / "FRED_ready" / "test" / "images")
    parser.add_argument("--data", type=Path, default=None, help="Optional fallback path to a TIFF fred_fusion_6ch.yaml.")
    parser.add_argument("--predictions", type=Path, default=Path("predictions_two_stream_lite.txt"))
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--out", type=Path, default=Path("reports/two_stream_lite_predictions"))
    parser.add_argument("--limit", type=int, default=None, help="Optional number of predicted images to draw.")
    return parser.parse_args()


def fused_samples(data: dict, split: str) -> dict[str, Path]:
    root = data["_root"]
    image_rel = data.get(split)
    if not image_rel:
        return {}
    image_dir = Path(image_rel)
    if not image_dir.is_absolute():
        image_dir = root / image_dir
    return {path.name: path for path in image_files(image_dir)}


def read_predictions(path: Path) -> dict[str, list[tuple[float, float, float, float, float, int]]]:
    grouped: dict[str, list[tuple[float, float, float, float, float, int]]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if len(parts) != 7:
            continue
        filename, x1, y1, x2, y2, conf, cls_id = parts
        grouped.setdefault(filename, []).append(
            (float(x1), float(y1), float(x2), float(y2), float(conf), int(float(cls_id)))
        )
    return grouped


def draw_boxes(image, boxes: list[tuple[float, float, float, float, float, int]]) -> None:
    draw = ImageDraw.Draw(image)
    for x1, y1, x2, y2, conf, cls_id in boxes:
        draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=3)
        label = f"{cls_id}:{conf:.2f}"
        draw.text((x1 + 2, max(0, y1 - 14)), label, fill=(255, 0, 0))


def main() -> None:
    args = parse_args()
    image_by_name = {path.name: path for path in image_files(args.image_dir)}
    if args.data is not None:
        samples = fused_samples(load_data_yaml(args.data), args.split)
        samples.update({f"{path.stem}.jpg": path for path in samples.values()})
    else:
        samples = {}
    predictions = read_predictions(args.predictions)
    args.out.mkdir(parents=True, exist_ok=True)

    written = 0
    for filename, boxes in predictions.items():
        image_path = image_by_name.get(filename)
        if image_path is None:
            image_path = find_by_stem(args.image_dir, Path(filename).stem)
        if image_path is not None:
            image = Image.open(image_path).convert("RGB")
        elif filename in samples:
            rgb, _ = six_channel_to_rgb_event(samples[filename])
            image = rgb.convert("RGB")
        else:
            print(f"skip missing RGB image for prediction: {filename}")
            continue
        draw_boxes(image, boxes)
        image.save(args.out / f"{Path(filename).stem}_two_stream_lite_bbox.jpg", quality=95)
        written += 1
        if args.limit is not None and written >= args.limit:
            break

    print(f"wrote {written} visualizations to {args.out}")


if __name__ == "__main__":
    main()
