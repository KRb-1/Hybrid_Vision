from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil
from pathlib import Path

from fred_data_utils import (
    SPLITS,
    collect_raw_split,
    describe_expected_raw_layouts,
    limit_samples,
    write_data_yaml,
    write_six_channel_tiff,
)


def convert_one_sample(task: tuple[object, Path, Path, str | None, bool]) -> bool:
    sample, image_out, label_out, compression, skip_existing = task
    image_path = image_out / f"{sample.stem}.tiff"
    label_path = label_out / f"{sample.stem}.txt"
    if skip_existing and image_path.exists() and label_path.exists():
        return False
    write_six_channel_tiff(sample.rgb, sample.event, image_path, compression=compression)
    shutil.copy2(sample.label, label_path)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 6-channel RGB+Event TIFFs for FRED two-stream training.")
    parser.add_argument("--raw-root", type=Path, default=Path("FRED_ready"))
    parser.add_argument("--out-root", type=Path, default=Path("data/fred_fusion_6ch"))
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", help="Skip samples that already have image and label outputs.")
    parser.add_argument("--allow-empty", action="store_true", help="Write yaml even when no samples are found.")
    parser.add_argument("--progress-every", type=int, default=50, help="Print progress every N converted samples.")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel conversion workers.")
    parser.add_argument(
        "--compression",
        default="tiff_deflate",
        choices=("tiff_deflate", "none"),
        help="TIFF compression. Use 'none' for faster writing but larger files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"raw root: {args.raw_root.resolve()}")
    print(f"output root: {args.out_root.resolve()}")
    if args.overwrite and args.out_root.exists():
        print(f"removing existing output root: {args.out_root}")
        shutil.rmtree(args.out_root)

    compression = None if args.compression == "none" else args.compression
    limits = {"train": args.limit_train, "val": args.limit_val, "test": args.limit_test}
    total = 0
    total_written = 0
    for split in SPLITS:
        all_samples = collect_raw_split(args.raw_root, split)
        samples = limit_samples(all_samples, limits[split], args.seed)
        print(f"{split}: found {len(all_samples)} matched samples, converting {len(samples)}")
        image_out = args.out_root / "images" / split
        label_out = args.out_root / "labels" / split
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)

        tasks = [(sample, image_out, label_out, compression, args.skip_existing) for sample in samples]
        written = 0
        done = 0
        if args.workers <= 1:
            for task in tasks:
                written += int(convert_one_sample(task))
                done += 1
                if args.progress_every > 0 and (done % args.progress_every == 0 or done == len(samples)):
                    print(f"{split}: {done}/{len(samples)} processed, {written} written")
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(convert_one_sample, task) for task in tasks]
                for future in as_completed(futures):
                    written += int(future.result())
                    done += 1
                    if args.progress_every > 0 and (done % args.progress_every == 0 or done == len(samples)):
                        print(f"{split}: {done}/{len(samples)} processed, {written} written")

        print(f"{split}: wrote {written} samples")
        total += len(samples)
        total_written += written

    yaml_path = write_data_yaml(args.out_root)
    print(f"dataset yaml: {yaml_path}")
    print(f"total samples: {total}")
    print(f"total written this run: {total_written}")
    if total == 0 and not args.allow_empty:
        raise SystemExit(
            "No samples were found. Check --raw-root and the dataset layout.\n"
            "On Linux, use '../FRED_ready_official_filtered', not '..\\FRED_ready_official_filtered'.\n\n"
            + describe_expected_raw_layouts(args.raw_root)
        )


if __name__ == "__main__":
    main()
