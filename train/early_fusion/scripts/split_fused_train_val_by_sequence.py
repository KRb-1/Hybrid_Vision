from __future__ import annotations

import argparse
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

from fred_fusion_utils import load_data_yaml


SEQ_RE = re.compile(r"^(seq[^_]+)_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Move full sequences from fused train split to val split.")
    parser.add_argument("--data", type=Path, required=True, help="Path to fred_fusion_6ch.yaml.")
    parser.add_argument("--val-frac", type=float, default=0.15, help="Approximate fraction of train images to move to val.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seqs", nargs="*", default=None, help="Explicit sequence ids to move, e.g. seq8 seq9.")
    parser.add_argument("--apply", action="store_true", help="Actually move files. Without this, only prints a dry-run plan.")
    return parser.parse_args()


def seq_id(stem: str) -> str:
    match = SEQ_RE.match(stem)
    if not match:
        raise ValueError(f"Cannot parse sequence id from stem: {stem}")
    return match.group(1)


def main() -> None:
    args = parse_args()
    data = load_data_yaml(args.data)
    root = data["_root"]
    train_images = root / "images" / "train"
    train_labels = root / "labels" / "train"
    val_images = root / "images" / "val"
    val_labels = root / "labels" / "val"

    images = sorted(train_images.glob("*.tiff"))
    by_seq: dict[str, list[Path]] = defaultdict(list)
    for image in images:
        by_seq[seq_id(image.stem)].append(image)

    if args.seqs:
        selected = sorted(set(args.seqs))
    else:
        target = max(1, int(len(images) * args.val_frac))
        seqs = sorted(by_seq)
        rng = random.Random(args.seed)
        rng.shuffle(seqs)
        selected = []
        moved_count = 0
        for seq in seqs:
            selected.append(seq)
            moved_count += len(by_seq[seq])
            if moved_count >= target:
                break
        selected = sorted(selected)

    missing = [seq for seq in selected if seq not in by_seq]
    if missing:
        raise SystemExit(f"Selected sequence(s) not found in train split: {missing}")

    selected_images = [image for seq in selected for image in by_seq[seq]]
    selected_labels = [train_labels / f"{image.stem}.txt" for image in selected_images]

    print(f"dataset root: {root}")
    print(f"train images found: {len(images)}")
    print(f"sequences found: {len(by_seq)}")
    print(f"selected val sequences: {', '.join(selected)}")
    print(f"images to move train -> val: {len(selected_images)}")
    print(f"mode: {'APPLY' if args.apply else 'DRY RUN'}")

    if not args.apply:
        print("Re-run with --apply to move files.")
        return

    val_images.mkdir(parents=True, exist_ok=True)
    val_labels.mkdir(parents=True, exist_ok=True)
    for image, label in zip(selected_images, selected_labels):
        target_image = val_images / image.name
        target_label = val_labels / label.name
        if target_image.exists() or target_label.exists():
            raise SystemExit(f"Refusing to overwrite existing val file for stem {image.stem}")
        shutil.move(str(image), target_image)
        if label.exists():
            shutil.move(str(label), target_label)
    print("done")


if __name__ == "__main__":
    main()
