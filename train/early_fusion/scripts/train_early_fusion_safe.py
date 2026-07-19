from __future__ import annotations

import argparse
import csv
from pathlib import Path

from check_checkpoint_finite import check_checkpoint
from train_early_fusion import first_conv, patch_first_conv_to_6ch, str2bool, write_env_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NaN-resistant YOLOv8 training for 6-channel FRED early fusion.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", default="yolov8m.pt")
    parser.add_argument("--resume", default=None, help="Resume from a safe run last.pt. Do not use a NaN-contaminated last.pt.")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default=0)
    parser.add_argument("--project", default="runs/fred")
    parser.add_argument("--name", default="early_fusion_m1280_safe_3ep")
    parser.add_argument("--amp", type=str2bool, default=False)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=0.0005)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--warmup-epochs", type=float, default=5.0)
    parser.add_argument("--warmup-bias-lr", type=float, default=None)
    parser.add_argument("--mosaic", type=float, default=0.0)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--hsv-h", type=float, default=0.0)
    parser.add_argument("--hsv-s", type=float, default=0.0)
    parser.add_argument("--hsv-v", type=float, default=0.1)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--save-period", type=int, default=1)
    parser.add_argument("--patch-strategy", default="scratch_event", choices=("scratch_event", "repeat", "zero"))
    parser.add_argument("--no-patch", action="store_true")
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def check_lr_groups(save_dir: Path) -> None:
    results_csv = save_dir / "results.csv"
    if not results_csv.exists():
        return
    with results_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        cleaned = {key.strip(): value for key, value in row.items()}
        try:
            pg0 = float(cleaned.get("lr/pg0", "nan"))
            pg2 = float(cleaned.get("lr/pg2", "nan"))
        except ValueError:
            continue
        if pg0 > 0 and pg2 > 10 * pg0:
            epoch = cleaned.get("epoch", "?")
            print(
                "WARNING: lr/pg2 is more than 10x lr/pg0 "
                f"at epoch {epoch}: lr/pg0={pg0}, lr/pg2={pg2}. "
                "Check warmup_bias_lr."
            )


def append_experiment_row(path: Path, args: argparse.Namespace, save_dir: Path, nan_occurred: bool, notes: str) -> None:
    fields = [
        "experiment_name",
        "weights",
        "imgsz",
        "epochs",
        "batch",
        "amp",
        "optimizer",
        "lr0",
        "mosaic",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "mAP50",
        "mAP50-95",
        "precision",
        "recall",
        "box_loss_final",
        "cls_loss_final",
        "dfl_loss_final",
        "nan_occurred",
        "train_time",
        "notes",
    ]
    row = {
        "experiment_name": args.name,
        "weights": args.resume or args.weights,
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "amp": args.amp,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "mosaic": args.mosaic,
        "hsv_h": args.hsv_h,
        "hsv_s": args.hsv_s,
        "hsv_v": args.hsv_v,
        "mAP50": "",
        "mAP50-95": "",
        "precision": "",
        "recall": "",
        "box_loss_final": "",
        "cls_loss_final": "",
        "dfl_loss_final": "",
        "nan_occurred": nan_occurred,
        "train_time": "",
        "notes": notes or args.notes,
    }

    results_csv = save_dir / "results.csv"
    if results_csv.exists():
        with results_csv.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows:
            last = rows[-1]
            row["mAP50"] = last.get("metrics/mAP50(B)", "")
            row["mAP50-95"] = last.get("metrics/mAP50-95(B)", "")
            row["precision"] = last.get("metrics/precision(B)", "")
            row["recall"] = last.get("metrics/recall(B)", "")
            row["box_loss_final"] = last.get("train/box_loss", "")
            row["cls_loss_final"] = last.get("train/cls_loss", "")
            row["dfl_loss_final"] = last.get("train/dfl_loss", "")

    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    if args.warmup_bias_lr is None:
        args.warmup_bias_lr = args.lr0
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: install ultralytics and torch before training.") from exc

    write_env_info(Path("reports/env_info.txt"))
    model = YOLO(args.resume or args.weights)
    if not args.resume and not args.no_patch:
        patch_first_conv_to_6ch(model, args.patch_strategy)

    conv = first_conv(model.model)
    print(f"first conv before train: {tuple(conv.weight.shape)}")
    if conv.in_channels != 6:
        raise RuntimeError(f"Model first conv is {conv.in_channels} channels, expected 6")

    project = Path(args.project)
    if not project.is_absolute():
        project = project.resolve()
    if args.resume:
        save_dir = Path(args.resume).resolve().parents[1]
    else:
        save_dir = project / args.name

    train_kwargs = {
        "data": str(args.data),
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        "amp": args.amp,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "warmup_bias_lr": args.warmup_bias_lr,
        "mosaic": args.mosaic,
        "mixup": args.mixup,
        "hsv_h": args.hsv_h,
        "hsv_s": args.hsv_s,
        "hsv_v": args.hsv_v,
        "fliplr": args.fliplr,
        "patience": args.patience,
        "save_period": args.save_period,
        "project": str(project),
        "name": args.name,
    }
    if args.resume:
        train_kwargs["resume"] = True

    print("training config:")
    print(f"  lr0: {args.lr0}")
    print(f"  warmup_epochs: {args.warmup_epochs}")
    print(f"  warmup_bias_lr: {args.warmup_bias_lr}")
    print(f"  amp: {args.amp}")
    print(f"  optimizer: {args.optimizer}")
    print(f"  imgsz: {args.imgsz}")
    print(f"  batch: {args.batch}")

    nan_occurred = False
    notes = args.notes
    try:
        results = model.train(**train_kwargs)
        print(results)
    finally:
        check_lr_groups(save_dir)
        for weight_name in ("best.pt", "last.pt"):
            weight_path = save_dir / "weights" / weight_name
            if weight_path.exists():
                print(f"checking checkpoint finite: {weight_path}")
                ok = check_checkpoint(weight_path)
                if not ok:
                    nan_occurred = True
                    print("WARNING: checkpoint contains NaN/Inf. Do not use this model for validation, prediction, or resume.")
            else:
                print(f"checkpoint not found: {weight_path}")

        append_experiment_row(Path("reports/experiment_results.csv"), args, save_dir, nan_occurred, notes)


if __name__ == "__main__":
    main()
