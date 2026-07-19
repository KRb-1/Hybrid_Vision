from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backup a finite safe run before starting new AMP experiments.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--tag", default="safe_2ep")
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
    weights_dir = run_dir / "weights"
    copy_once(weights_dir / "best.pt", weights_dir / f"best_{args.tag}.pt")
    copy_once(weights_dir / "last.pt", weights_dir / f"last_{args.tag}.pt")
    copy_once(run_dir / "results.csv", run_dir / f"results_{args.tag}.csv")
    print("Use the backed-up best checkpoint as --weights for new experiments. Do not use resume=True when changing AMP/LR.")


if __name__ == "__main__":
    main()
