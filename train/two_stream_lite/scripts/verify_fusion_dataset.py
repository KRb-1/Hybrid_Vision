from __future__ import annotations

import argparse
from pathlib import Path

from fred_data_utils import SPLITS, image_files, load_data_yaml, read_six_channel_tiff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify FRED 6-channel TIFF dataset.")
    parser.add_argument("--data", type=Path, default=Path("data/fred_fusion_6ch/fred_fusion_6ch.yaml"))
    parser.add_argument("--max-files", type=int, default=64)
    parser.add_argument("--report", type=Path, default=Path("reports/dataset_check.md"))
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--skip-image-check", action="store_true", help="Only check labels and file pairing; do not open TIFFs.")
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--print-errors", type=int, default=50, help="Print up to N errors to stdout when checks fail.")
    return parser.parse_args()


def parse_yolo_label(path: Path, nc: int) -> tuple[int, list[str]]:
    errors: list[str] = []
    boxes = 0
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split()
        if len(parts) != 5:
            errors.append(f"{path}:{line_no}: expected 5 columns, got {len(parts)}")
            continue
        try:
            cls = int(float(parts[0]))
            xywh = [float(value) for value in parts[1:]]
        except ValueError:
            errors.append(f"{path}:{line_no}: non-numeric label values")
            continue
        if cls < 0 or cls >= nc:
            errors.append(f"{path}:{line_no}: class id {cls} outside [0, {nc - 1}]")
        if any(value < 0.0 or value > 1.0 for value in xywh):
            errors.append(f"{path}:{line_no}: bbox values outside [0, 1]: {xywh}")
        if xywh[2] <= 0.0 or xywh[3] <= 0.0:
            errors.append(f"{path}:{line_no}: bbox width/height must be > 0: {xywh}")
        boxes += 1
    return boxes, errors


def main() -> None:
    args = parse_args()
    data = load_data_yaml(args.data)
    root = data["_root"]
    lines = ["# Dataset Check", ""]
    errors: list[str] = []
    total_images = 0
    total_boxes = 0
    empty_labels = 0
    nc = len(data.get("names", {}))

    for split in SPLITS:
        image_dir = root / data.get(split, f"images/{split}")
        label_dir = root / "labels" / split
        images = image_files(image_dir)
        total_images += len(images)
        split_boxes = 0
        split_empty_labels = 0
        lines.append(f"- {split}: {len(images)} images")
        checked_images = images if args.max_files <= 0 else images[: args.max_files]
        print(f"{split}: checking {len(checked_images)}/{len(images)} images")
        for index, image_path in enumerate(checked_images, start=1):
            if not args.skip_image_check:
                channels, size = read_six_channel_tiff(image_path)
                if len(channels) != 6:
                    errors.append(f"{image_path}: expected 6 channels, got {len(channels)}")
                if size[0] <= 0 or size[1] <= 0:
                    errors.append(f"{image_path}: invalid size {size}")
            label = label_dir / f"{image_path.stem}.txt"
            if not label.exists():
                errors.append(f"{image_path}: missing label {label}")
                continue
            boxes, label_errors = parse_yolo_label(label, nc)
            errors.extend(label_errors)
            split_boxes += boxes
            if boxes == 0:
                split_empty_labels += 1
            if args.progress_every > 0 and (index % args.progress_every == 0 or index == len(checked_images)):
                print(f"{split}: {index}/{len(checked_images)} checked")
        total_boxes += split_boxes
        empty_labels += split_empty_labels
        lines.append(f"  - checked labels: {len(checked_images)}")
        lines.append(f"  - boxes in checked labels: {split_boxes}")
        lines.append(f"  - empty checked labels: {split_empty_labels}")
    if total_images == 0 and not args.allow_empty:
        errors.append("dataset contains 0 images; check the build step and --raw-root path")

    lines.extend(["", "## Summary", f"- nc: {nc}", f"- total images: {total_images}", f"- boxes checked: {total_boxes}", f"- empty labels checked: {empty_labels}"])
    lines.extend(["", "## Errors"])
    lines.extend(f"- {error}" for error in errors) if errors else lines.append("- none")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.report)
    if errors:
        print(f"dataset check failed with {len(errors)} errors")
        for error in errors[: args.print_errors]:
            print(f"- {error}")
        if len(errors) > args.print_errors:
            print(f"... {len(errors) - args.print_errors} more errors in {args.report}")
        raise SystemExit(f"dataset check failed with {len(errors)} errors")


if __name__ == "__main__":
    main()
