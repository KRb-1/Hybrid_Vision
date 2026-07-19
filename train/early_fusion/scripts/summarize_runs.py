from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


DEFAULT_RUNS = [
    "early_fusion_yolov8m_1280_e30",
    "early_fusion_m1280_safe_3ep",
    "early_fusion_from_safe2_amp_lr1e4_2ep",
    "early_fusion_from_safe2_amp_lr1e4_biaslr1e4_2ep",
    "early_fusion_from_safe2_amp_lr1e4_b16_2ep",
    "early_fusion_from_safe2_amp_lr1e4_biaslr1e4_b16_2ep",
    "early_fusion_from_amp1_b16_biaslr1e4_e27",
    "early_fusion_from_safe2_amp_lr1e4_e15",
    "early_fusion_from_safe2_amp_lr1e4_biaslr1e4_e15",
    "early_fusion_from_safe2_m960_amp_e20",
    "early_fusion_m960_to_m1280_finetune_e5",
]

FIELDS = [
    "run_name",
    "epoch",
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "val/box_loss",
    "val/cls_loss",
    "val/dfl_loss",
    "nan_occurred",
    "best_epoch",
    "best_mAP50",
    "best_mAP50-95",
    "notes",
]


NOTES = {
    "early_fusion_yolov8m_1280_e30": "old_nan_run epoch 2 invalid due to NaN losses; do not use last.pt.",
    "early_fusion_m1280_safe_3ep": (
        "safe_amp_false_3ep: finite but too slow for full training. Epoch 2 mAP drop is early training fluctuation; "
        "best.pt still preserves best validation epoch."
    ),
    "early_fusion_from_safe2_amp_lr1e4_2ep": "AMP low-LR stability test; main candidate path if finite.",
    "early_fusion_from_safe2_amp_lr1e4_biaslr1e4_2ep": "AMP low-LR stability test with warmup_bias_lr fixed to lr0.",
    "early_fusion_from_safe2_amp_lr1e4_b16_2ep": "AMP low-LR batch16 run; can continue to AMP epoch 28 for about 30 total epochs including safe2.",
    "early_fusion_from_safe2_amp_lr1e4_biaslr1e4_b16_2ep": "AMP low-LR batch16 run with warmup_bias_lr fixed to lr0; can continue to AMP epoch 28.",
    "early_fusion_from_amp1_b16_biaslr1e4_e27": "New batch16 AMP run from finite AMP1 weights with warmup_bias_lr fixed to lr0; targets about 30 total epochs including safe2+AMP1.",
    "early_fusion_from_safe2_amp_lr1e4_e15": "AMP low-LR main candidate if stable.",
    "early_fusion_from_safe2_amp_lr1e4_biaslr1e4_e15": "AMP low-LR main candidate with warmup_bias_lr fixed to lr0.",
    "early_fusion_from_safe2_m960_amp_e20": "Fallback if 1280 AMP low-LR is unstable.",
    "early_fusion_m960_to_m1280_finetune_e5": "Final 1280 fine-tune after stable 960 AMP fallback.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize YOLO run results into reports/experiment_results.csv.")
    parser.add_argument("--runs-root", type=Path, default=Path("runs/fred"))
    parser.add_argument("--out", type=Path, default=Path("reports/experiment_results.csv"))
    parser.add_argument("--runs", nargs="*", default=DEFAULT_RUNS)
    return parser.parse_args()


def as_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def finite(value: str) -> bool:
    number = as_float(value)
    return math.isfinite(number)


def read_results(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append({(key.strip() if key else key): value for key, value in row.items()})
    return rows


def best_metrics(rows: list[dict[str, str]]) -> tuple[str, str, str]:
    best_row: dict[str, str] | None = None
    best_map = -math.inf
    for row in rows:
        value = as_float(row.get("metrics/mAP50(B)", ""))
        if math.isfinite(value) and value > best_map:
            best_map = value
            best_row = row
    if best_row is None:
        return "", "", ""
    return (
        best_row.get("epoch", ""),
        best_row.get("metrics/mAP50(B)", ""),
        best_row.get("metrics/mAP50-95(B)", ""),
    )


def row_has_nan(row: dict[str, str]) -> bool:
    numeric_cols = [
        "train/box_loss",
        "train/cls_loss",
        "train/dfl_loss",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "val/box_loss",
        "val/cls_loss",
        "val/dfl_loss",
    ]
    return any(col in row and row[col] != "" and not finite(row[col]) for col in numeric_cols)


def main() -> None:
    args = parse_args()
    output_rows: list[dict[str, str]] = []
    for run_name in args.runs:
        results_path = args.runs_root / run_name / "results.csv"
        if not results_path.exists():
            output_rows.append(
                {
                    "run_name": run_name,
                    "epoch": "",
                    "nan_occurred": "",
                    "best_epoch": "",
                    "best_mAP50": "",
                    "best_mAP50-95": "",
                    "notes": f"missing results.csv at {results_path}",
                }
            )
            continue

        rows = read_results(results_path)
        best_epoch, best_map50, best_map5095 = best_metrics(rows)
        for row in rows:
            nan_occurred = row_has_nan(row)
            output_rows.append(
                {
                    "run_name": run_name,
                    "epoch": row.get("epoch", ""),
                    "train/box_loss": row.get("train/box_loss", ""),
                    "train/cls_loss": row.get("train/cls_loss", ""),
                    "train/dfl_loss": row.get("train/dfl_loss", ""),
                    "metrics/precision(B)": row.get("metrics/precision(B)", ""),
                    "metrics/recall(B)": row.get("metrics/recall(B)", ""),
                    "metrics/mAP50(B)": row.get("metrics/mAP50(B)", ""),
                    "metrics/mAP50-95(B)": row.get("metrics/mAP50-95(B)", ""),
                    "val/box_loss": row.get("val/box_loss", ""),
                    "val/cls_loss": row.get("val/cls_loss", ""),
                    "val/dfl_loss": row.get("val/dfl_loss", ""),
                    "nan_occurred": str(nan_occurred),
                    "best_epoch": best_epoch,
                    "best_mAP50": best_map50,
                    "best_mAP50-95": best_map5095,
                    "notes": NOTES.get(run_name, ""),
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
