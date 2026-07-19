from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

from fred_fusion_utils import SPLITS, collect_fused_split, load_data_yaml, parse_label_file, read_six_channel_tiff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify a FRED 6-channel TIFF YOLO dataset.")
    parser.add_argument("--data", type=Path, required=True, help="Path to fred_fusion_6ch.yaml.")
    parser.add_argument("--report", type=Path, default=Path("reports/dataset_check.md"))
    parser.add_argument("--max-errors", type=int, default=80)
    parser.add_argument("--max-samples-per-split", type=int, default=None, help="Randomly verify at most this many samples per split.")
    parser.add_argument("--progress-every", type=int, default=5000, help="Print progress every N checked samples. Use 0 to disable.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_data_yaml(args.data)
    errors: list[str] = []
    warnings: list[str] = []
    counts: dict[str, int] = {}
    empty_labels: dict[str, int] = {}
    sizes: dict[str, Counter] = {}
    split_stems: dict[str, set[str]] = defaultdict(set)

    if int(data.get("channels", 0)) != 6:
        errors.append("dataset yaml must contain channels: 6")

    rng = random.Random(args.seed)
    checked_counts: dict[str, int] = {}
    for split in SPLITS:
        all_samples = collect_fused_split(data, split)
        counts[split] = len(all_samples)
        samples = all_samples
        if args.max_samples_per_split and len(samples) > args.max_samples_per_split:
            samples = sorted(rng.sample(samples, args.max_samples_per_split), key=lambda s: s.stem)
        checked_counts[split] = len(samples)
        empty = 0
        size_counter: Counter = Counter()
        print(f"{split}: checking {len(samples)}/{len(all_samples)} samples")
        for i, sample in enumerate(samples, start=1):
            split_stems[split].add(sample.stem)
            pages, size = read_six_channel_tiff(sample.rgb)
            if len(pages) != 6:
                errors.append(f"{sample.rgb} has {len(pages)} channels, expected 6")
            if any(page.mode != "L" for page in pages):
                errors.append(f"{sample.rgb} contains non-uint8/non-L TIFF pages")
            if len({page.size for page in pages}) > 1:
                errors.append(f"{sample.rgb} has inconsistent channel sizes")
            if size != (0, 0):
                size_counter[size] += 1

            boxes, label_errors = parse_label_file(sample.label)
            errors.extend(label_errors)
            if sample.label.exists() and not boxes and sample.label.read_text(encoding="utf-8").strip() == "":
                empty += 1
            if args.progress_every and i % args.progress_every == 0:
                print(f"{split}: checked {i}/{len(samples)}")
        empty_labels[split] = empty
        sizes[split] = size_counter

    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        leaked = split_stems[a] & split_stems[b]
        if leaked:
            errors.append(f"{a}/{b} leakage: {len(leaked)} duplicate stems, e.g. {sorted(leaked)[:5]}")

    if not counts.get("train"):
        errors.append("train split is empty")
    if not counts.get("val"):
        errors.append("val split is empty")

    lines = ["# FRED Fusion Dataset Check", ""]
    lines.append(f"- data yaml: `{args.data}`")
    lines.append(f"- root: `{data['_root']}`")
    lines.append(f"- channels: `{data.get('channels')}`")
    if args.max_samples_per_split:
        lines.append(f"- sampled check: at most `{args.max_samples_per_split}` samples per split")
    lines.append("")
    lines.append("## Split Summary")
    for split in SPLITS:
        common_sizes = ", ".join(f"{w}x{h}: {n}" for (w, h), n in sizes[split].most_common(5))
        lines.append(
            f"- {split}: {counts.get(split, 0)} samples, checked {checked_counts.get(split, 0)}, "
            f"{empty_labels.get(split, 0)} empty labels in checked set, sizes: {common_sizes or 'n/a'}"
        )
    lines.append("")
    lines.append("## Result")
    if errors:
        lines.append(f"FAILED with {len(errors)} error(s).")
        lines.append("")
        for err in errors[: args.max_errors]:
            lines.append(f"- {err}")
        if len(errors) > args.max_errors:
            lines.append(f"- ... {len(errors) - args.max_errors} more")
    else:
        lines.append("PASSED.")
    if warnings:
        lines.append("")
        lines.append("## Warnings")
        lines.extend(f"- {w}" for w in warnings)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
