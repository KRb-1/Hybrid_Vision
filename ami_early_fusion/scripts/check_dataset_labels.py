from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from fred_fusion_utils import SPLITS, collect_fused_split, load_data_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check fused FRED labels and images for NaN/Inf and invalid values.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=Path("reports/dataset_nan_check.md"))
    parser.add_argument("--max-report", type=int, default=100)
    parser.add_argument("--progress-every", type=int, default=10000)
    return parser.parse_args()


def image_size(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as image:
            return image.size
    except Exception:
        return None


def check_label(path: Path, num_classes: int) -> tuple[int, list[str]]:
    errors: list[str] = []
    boxes = 0
    if not path.exists():
        return boxes, [f"missing label: {path}"]

    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split()
        if len(parts) != 5:
            errors.append(f"{path}:{lineno} has {len(parts)} columns")
            continue
        try:
            cls_float = float(parts[0])
            cls = int(cls_float)
            vals = [float(x) for x in parts[1:]]
        except ValueError:
            errors.append(f"{path}:{lineno} contains non-numeric values")
            continue

        if not math.isfinite(cls_float) or any(not math.isfinite(v) for v in vals):
            errors.append(f"{path}:{lineno} contains NaN/Inf")
            continue
        if cls_float != cls or cls < 0 or cls >= num_classes:
            errors.append(f"{path}:{lineno} invalid class {parts[0]}")
        cx, cy, width, height = vals
        if any(v < 0.0 or v > 1.0 for v in vals):
            errors.append(f"{path}:{lineno} bbox values outside [0,1]")
        if width <= 0.0 or height <= 0.0:
            errors.append(f"{path}:{lineno} bbox width/height must be > 0")
        boxes += 1

    return boxes, errors


def main() -> None:
    args = parse_args()
    data = load_data_yaml(args.data)
    names = data.get("names", {0: "drone"})
    num_classes = len(names)
    errors: list[str] = []
    split_counts: dict[str, int] = {}
    box_counts: dict[str, int] = {}
    empty_counts: dict[str, int] = {}
    size_counts: dict[str, Counter] = {}
    split_stems: dict[str, set[str]] = defaultdict(set)

    for split in SPLITS:
        samples = collect_fused_split(data, split)
        split_counts[split] = len(samples)
        box_counts[split] = 0
        empty_counts[split] = 0
        size_counts[split] = Counter()
        print(f"{split}: checking {len(samples)} samples")
        for i, sample in enumerate(samples, start=1):
            split_stems[split].add(sample.stem)
            size = image_size(sample.rgb)
            if size is None:
                errors.append(f"unreadable image: {sample.rgb}")
            else:
                width, height = size
                if width <= 0 or height <= 0:
                    errors.append(f"invalid image size {width}x{height}: {sample.rgb}")
                size_counts[split][size] += 1

            boxes, label_errors = check_label(sample.label, num_classes)
            box_counts[split] += boxes
            errors.extend(label_errors)
            if sample.label.exists() and sample.label.read_text(encoding="utf-8").strip() == "":
                empty_counts[split] += 1

            if args.progress_every and i % args.progress_every == 0:
                print(f"{split}: checked {i}/{len(samples)}")

    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        leaked = split_stems[a] & split_stems[b]
        if leaked:
            errors.append(f"{a}/{b} leakage: {len(leaked)} duplicate stems, e.g. {sorted(leaked)[:5]}")

    lines = ["# FRED Dataset NaN/Label Check", ""]
    lines.append(f"- data yaml: `{args.data}`")
    lines.append(f"- root: `{data['_root']}`")
    lines.append(f"- classes: `{num_classes}`")
    lines.append("")
    lines.append("## Summary")
    for split in SPLITS:
        common_sizes = ", ".join(f"{w}x{h}: {n}" for (w, h), n in size_counts[split].most_common(5))
        lines.append(
            f"- {split}: {split_counts.get(split, 0)} images, {box_counts.get(split, 0)} boxes, "
            f"{empty_counts.get(split, 0)} empty labels, sizes: {common_sizes or 'n/a'}"
        )

    lines.append("")
    lines.append("## Result")
    if errors:
        lines.append(f"FAILED with {len(errors)} issue(s).")
        lines.append("")
        for error in errors[: args.max_report]:
            lines.append(f"- {error}")
        if len(errors) > args.max_report:
            lines.append(f"- ... {len(errors) - args.max_report} more")
    else:
        lines.append("PASSED.")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
