from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backup useful files from a run that later produced NaN.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory containing weights/ and results.csv.")
    parser.add_argument("--tag", default="epoch1_safe")
    return parser.parse_args()


def copy_once(src: Path, dst: Path) -> None:
    if not src.exists():
        print(f"missing, skip: {src}")
        return
    if dst.exists():
        print(f"exists, not overwriting: {dst}")
        return
    shutil.copy2(src, dst)
    print(f"copied: {src} -> {dst}")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    copy_once(run_dir / "weights" / "best.pt", run_dir / "weights" / f"best_{args.tag}.pt")
    copy_once(run_dir / "results.csv", run_dir / f"results_before_nan_{args.tag}.csv")
    last = run_dir / "weights" / "last.pt"
    if last.exists():
        print(f"NOTE: {last} exists but may be NaN-contaminated. Do not resume it until finite checks pass.")


if __name__ == "__main__":
    main()
