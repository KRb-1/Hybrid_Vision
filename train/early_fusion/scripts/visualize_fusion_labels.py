from __future__ import annotations

import argparse
import random
from pathlib import Path

from fred_fusion_utils import collect_fused_split, draw_boxes, load_data_yaml, parse_label_file, six_channel_to_rgb_event


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize labels on RGB and Event channels from fused TIFFs.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--num", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("reports/vis_check"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_data_yaml(args.data)
    samples = collect_fused_split(data, args.split)
    rng = random.Random(args.seed)
    rng.shuffle(samples)
    chosen = samples[: args.num]
    args.out.mkdir(parents=True, exist_ok=True)

    for sample in chosen:
        boxes, errors = parse_label_file(sample.label)
        if errors:
            print(f"skip {sample.stem}: {errors[0]}")
            continue
        rgb, event = six_channel_to_rgb_event(sample.rgb)
        draw_boxes(rgb, boxes).save(args.out / f"{sample.stem}_rgb_bbox.jpg", quality=95)
        draw_boxes(event, boxes).save(args.out / f"{sample.stem}_event_bbox.jpg", quality=95)
        draw_boxes(rgb, boxes).save(args.out / f"{sample.stem}_fusion_rgb_channels_bbox.jpg", quality=95)
        draw_boxes(event, boxes).save(args.out / f"{sample.stem}_fusion_event_channels_bbox.jpg", quality=95)
    print(f"wrote {len(chosen)} sample visualizations to {args.out}")


if __name__ == "__main__":
    main()
