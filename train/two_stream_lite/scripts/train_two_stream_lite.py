from __future__ import annotations

import argparse
import csv
import platform
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fred_two_stream.pretrained import first_rgb_event_stem, load_matching_pretrained
from fred_two_stream.ultralytics_patch import register_rgb_event_stem


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "y", "on"}


def parse_cache(value: str | None) -> str | bool | None:
    if value is None or value.lower() in {"none", "off", "false", "0", "no"}:
        return None
    if value.lower() in {"true", "1", "yes", "on"}:
        return True
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 two-stream-lite RGB+Event detector.")
    parser.add_argument("--data", type=Path, default=Path("data/fred_fusion_6ch/fred_fusion_6ch.yaml"))
    parser.add_argument("--model", type=Path, default=Path("configs/models/yolov8m_rgb_event_stem.yaml"))
    parser.add_argument("--pretrained", default="yolov8m.pt")
    parser.add_argument("--weights", default=None, help="Start a new run from an existing checkpoint without resuming optimizer state.")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default=0)
    parser.add_argument("--amp", type=str2bool, default=True)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=0.003)
    parser.add_argument("--lrf", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--cos-lr", type=str2bool, default=True)
    parser.add_argument("--warmup-epochs", type=float, default=None)
    parser.add_argument("--warmup-bias-lr", type=float, default=None)
    parser.add_argument("--hsv-h", type=float, default=None)
    parser.add_argument("--hsv-s", type=float, default=None)
    parser.add_argument("--hsv-v", type=float, default=None)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--save-period", type=int, default=None)
    parser.add_argument("--cache", default=None, help="Ultralytics cache mode: ram, disk, true, or none.")
    parser.add_argument("--project", default="runs/fred")
    parser.add_argument("--name", default="two_stream_lite_yolov8m_960_e50")
    parser.add_argument("--skip-pretrained", action="store_true")
    return parser.parse_args()


def write_env_info(path: Path) -> None:
    import torch
    import ultralytics

    lines = [
        f"python: {platform.python_version()}",
        f"platform: {platform.platform()}",
        f"torch: {torch.__version__}",
        f"cuda available: {torch.cuda.is_available()}",
        f"cuda version: {torch.version.cuda}",
        f"ultralytics: {ultralytics.__version__}",
    ]
    if torch.cuda.is_available():
        lines.append(f"gpu: {torch.cuda.get_device_name(0)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_experiment_row(path: Path, args: argparse.Namespace, weights: Path | None) -> None:
    fieldnames = [
        "experiment_name",
        "fusion_type",
        "model",
        "weights",
        "imgsz",
        "epochs",
        "batch",
        "optimizer",
        "lr0",
        "mAP50",
        "mAP50-95",
        "precision",
        "recall",
        "F1",
        "inference_time_ms",
        "params",
        "FLOPs",
        "train_time",
        "GPU",
        "commit_hash",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(
            {
                "experiment_name": args.name,
                "fusion_type": "two_stream_lite",
                "model": str(args.model),
                "weights": str(weights or ""),
                "imgsz": args.imgsz,
                "epochs": args.epochs,
                "batch": args.batch,
                "optimizer": args.optimizer,
                "lr0": args.lr0,
                "notes": "Fill metrics from Ultralytics results.csv after full run.",
            }
        )


def main() -> None:
    args = parse_args()
    if args.resume and args.weights:
        raise SystemExit("Use either --resume or --weights, not both. --weights starts a new run; --resume keeps optimizer state.")
    register_rgb_event_stem()

    from ultralytics import YOLO

    write_env_info(Path("reports/env_info.txt"))
    if args.resume:
        model = YOLO(args.resume)
    elif args.weights:
        model = YOLO(args.weights)
    else:
        model = YOLO(str(args.model))
        stem = first_rgb_event_stem(model.model)
        if stem is None:
            raise RuntimeError("Model was built without RGBEventStem")
        if not args.skip_pretrained:
            report = load_matching_pretrained(model, args.pretrained)
        else:
            report = load_matching_pretrained(model, None)
        Path("reports").mkdir(exist_ok=True)
        Path("reports/two_stream_pretrained_report.txt").write_text(report.format() + "\n", encoding="utf-8")
        print(report.format())

    info_path = Path("reports/two_stream_model_info.txt")
    with info_path.open("w", encoding="utf-8") as file:
        model.info(detailed=False, verbose=True)
        file.write(str(model.model) + "\n")

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
        "cos_lr": args.cos_lr,
        "close_mosaic": args.close_mosaic,
        "patience": args.patience,
        "project": args.project,
        "name": args.name,
    }
    cache = parse_cache(args.cache)
    if cache is not None:
        train_kwargs["cache"] = cache
    if args.weight_decay is not None:
        train_kwargs["weight_decay"] = args.weight_decay
    if args.lrf is not None:
        train_kwargs["lrf"] = args.lrf
    if args.warmup_epochs is not None:
        train_kwargs["warmup_epochs"] = args.warmup_epochs
    if args.warmup_bias_lr is not None:
        train_kwargs["warmup_bias_lr"] = args.warmup_bias_lr
    if args.hsv_h is not None:
        train_kwargs["hsv_h"] = args.hsv_h
    if args.hsv_s is not None:
        train_kwargs["hsv_s"] = args.hsv_s
    if args.hsv_v is not None:
        train_kwargs["hsv_v"] = args.hsv_v
    if args.save_period is not None:
        train_kwargs["save_period"] = args.save_period
    if args.resume:
        train_kwargs["resume"] = True

    results = model.train(**train_kwargs)
    print(results)

    run_dir = Path(args.project) / args.name
    best = run_dir / "weights" / "best.pt"
    if best.exists():
        reloaded = YOLO(str(best))
        print(f"reloaded best.pt: {best}")
        print(reloaded.info(detailed=False, verbose=False))
    append_experiment_row(Path("reports/experiment_results.csv"), args, best if best.exists() else None)


if __name__ == "__main__":
    main()
