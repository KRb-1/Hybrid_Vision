from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from fred_data_utils import SPLITS, image_files, load_data_yaml
from verify_fusion_dataset import parse_yolo_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove invalid YOLO label rows from a FRED fused dataset.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=(*SPLITS, "all"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--backup-suffix", default=".bak")
    return parser.parse_args()


def valid_label_line(raw: str, nc: int) -> bool:
    raw = raw.strip()
    if not raw:
        return False
    parts = raw.split()
    if len(parts) != 5:
        return False
    try:
        cls = int(float(parts[0]))
        xywh = [float(value) for value in parts[1:]]
    except ValueError:
        return False
    return 0 <= cls < nc and all(0.0 <= value <= 1.0 for value in xywh) and xywh[2] > 0.0 and xywh[3] > 0.0


def main() -> None:
    args = parse_args()
    data = load_data_yaml(args.data)
    root = data["_root"]
    nc = len(data.get("names", {}))
    splits = SPLITS if args.split == "all" else (args.split,)
    changed = 0
    removed_rows = 0

    for split in splits:
        image_dir = root / data.get(split, f"images/{split}")
        label_dir = root / "labels" / split
        for image_path in image_files(image_dir):
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            _, errors = parse_yolo_label(label_path, nc)
            if not errors:
                continue
            lines = label_path.read_text(encoding="utf-8").splitlines()
            kept = [line for line in lines if valid_label_line(line, nc)]
            removed = len([line for line in lines if line.strip()]) - len(kept)
            removed_rows += removed
            changed += 1
            print(f"{label_path}: remove {removed} invalid rows, keep {len(kept)} rows")
            if not args.dry_run:
                backup_path = label_path.with_name(label_path.name + args.backup_suffix)
                if not backup_path.exists():
                    shutil.copy2(label_path, backup_path)
                label_path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")

    print(f"changed label files: {changed}")
    print(f"removed invalid rows: {removed_rows}")
    if args.dry_run:
        print("dry run only; no files were modified")


if __name__ == "__main__":
    main()
