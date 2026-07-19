from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from fred_fusion_utils import SPLITS, collect_raw_split, limit_samples, write_data_yaml, write_six_channel_tiff


COMPRESSION_MAP = {
    "deflate": "tiff_deflate",
    "lzw": "tiff_lzw",
    "none": None,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 6-channel RGB+Event TIFFs for FRED early fusion.")
    parser.add_argument("--raw-root", type=Path, default=Path("FRED_ready"), help="Root with train/val/test images/events/labels.")
    parser.add_argument("--out-root", type=Path, default=Path("data/fred_fusion_6ch"), help="Output dataset root.")
    parser.add_argument("--limit-train", type=int, default=None, help="Optional train sample limit for smoke tests.")
    parser.add_argument("--limit-val", type=int, default=None, help="Optional val sample limit for smoke tests.")
    parser.add_argument("--limit-test", type=int, default=None, help="Optional test sample limit for smoke tests.")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=list(SPLITS),
        help="Dataset splits to build. Use '--splits train val' to postpone test conversion.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="Remove existing output root first.")
    parser.add_argument(
        "--rebuild-existing",
        action="store_true",
        help="Rebuild samples even when output TIFF and label already exist. By default existing samples are skipped.",
    )
    parser.add_argument("--workers", type=int, default=1, help="Parallel image conversion workers. Use 8-16 on a 24-vCPU server.")
    parser.add_argument(
        "--compression",
        choices=tuple(COMPRESSION_MAP),
        default="deflate",
        help="TIFF compression. 'none' is fastest but uses much more disk.",
    )
    return parser.parse_args()


def sample_done(sample, image_out: Path, label_out: Path) -> bool:
    image_path = image_out / f"{sample.stem}.tiff"
    label_path = label_out / f"{sample.stem}.txt"
    return image_path.exists() and image_path.stat().st_size > 0 and label_path.exists()


def build_one(sample, image_out: Path, label_out: Path, compression: str | None) -> str:
    write_six_channel_tiff(sample.rgb, sample.event, image_out / f"{sample.stem}.tiff", compression=compression)
    shutil.copy2(sample.label, label_out / f"{sample.stem}.txt")
    return sample.stem


def warn_if_no_samples(raw_root: Path, split: str) -> None:
    split_root = raw_root / split
    dirs = {
        "images": split_root / "images",
        "events": split_root / "events",
        "labels": split_root / "labels",
    }
    print(f"{split}: found 0 matched samples under {split_root}")
    for name, directory in dirs.items():
        if directory.exists():
            file_count = sum(1 for p in directory.iterdir() if p.is_file())
            print(f"{split}: {name} dir exists with {file_count} files: {directory}")
        else:
            print(f"{split}: missing {name} dir: {directory}")


def main() -> None:
    args = parse_args()
    if args.overwrite and args.out_root.exists():
        shutil.rmtree(args.out_root)

    limits = {"train": args.limit_train, "val": args.limit_val, "test": args.limit_test}
    total = 0
    for split in args.splits:
        samples = collect_raw_split(args.raw_root, split)
        if not samples:
            warn_if_no_samples(args.raw_root, split)
        samples = limit_samples(samples, limits[split], args.seed)
        image_out = args.out_root / "images" / split
        label_out = args.out_root / "labels" / split
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)
        compression = COMPRESSION_MAP[args.compression]

        if not args.rebuild_existing:
            before = len(samples)
            samples = [sample for sample in samples if not sample_done(sample, image_out, label_out)]
            skipped = before - len(samples)
            if skipped:
                print(f"{split}: skipped {skipped} existing samples")

        if args.workers <= 1:
            for i, sample in enumerate(samples, start=1):
                build_one(sample, image_out, label_out, compression)
                if i % 500 == 0:
                    print(f"{split}: {i}/{len(samples)}")
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(build_one, sample, image_out, label_out, compression) for sample in samples]
                for i, future in enumerate(as_completed(futures), start=1):
                    future.result()
                    if i % 500 == 0:
                        print(f"{split}: {i}/{len(samples)}")

        print(f"{split}: wrote {len(samples)} samples")
        total += len(samples)

    yaml_path = write_data_yaml(args.out_root)
    print(f"dataset yaml: {yaml_path}")
    print(f"total samples: {total}")


if __name__ == "__main__":
    main()
