from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from fred_fusion_utils import collect_fused_split, find_by_stem, image_files, load_data_yaml, six_channel_to_rgb_event


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw early-fusion prediction boxes on RGB jpg images.")
    parser.add_argument("--image-dir", type=Path, default=PROJECT_ROOT / "FRED_ready" / "test" / "images")
    parser.add_argument("--data", type=Path, default=None, help="Optional fallback path to a TIFF fred_fusion_6ch.yaml.")
    parser.add_argument("--predictions", type=Path, default=Path("predictions_early_fusion.txt"))
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--out", type=Path, default=Path("reports/early_fusion_predictions"))
    parser.add_argument("--limit", type=int, default=None, help="Optional number of predicted images to draw.")
    return parser.parse_args()


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
        draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=3)
        label = f"{cls_id}:{conf:.2f}"
        draw.text((x1 + 2, max(0, y1 - 14)), label, fill=(0, 255, 0))


def main() -> None:
    args = parse_args()
    image_by_name = {path.name: path for path in image_files(args.image_dir)}
    if args.data is not None:
        data = load_data_yaml(args.data)
        fused_by_name = {}
        for sample in collect_fused_split(data, args.split):
            fused_by_name[sample.rgb.name] = sample.rgb
            fused_by_name[f"{sample.rgb.stem}.jpg"] = sample.rgb
    else:
        fused_by_name = {}
    predictions = read_predictions(args.predictions)
    args.out.mkdir(parents=True, exist_ok=True)

    written = 0
    for filename, boxes in predictions.items():
        image_path = image_by_name.get(filename)
        if image_path is None:
            image_path = find_by_stem(args.image_dir, Path(filename).stem)
        if image_path is not None:
            image = Image.open(image_path).convert("RGB")
        elif filename in fused_by_name:
            image, _ = six_channel_to_rgb_event(fused_by_name[filename])
            image = image.convert("RGB")
        else:
            print(f"skip missing RGB image for prediction: {filename}")
            continue
        draw_boxes(image, boxes)
        image.save(args.out / f"{Path(filename).stem}_early_fusion_bbox.jpg", quality=95)
        written += 1
        if args.limit is not None and written >= args.limit:
            break

    print(f"wrote {written} visualizations to {args.out}")


if __name__ == "__main__":
    main()
