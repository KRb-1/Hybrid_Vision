from __future__ import annotations

import argparse
import platform
from pathlib import Path


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "y", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 on 6-channel FRED RGB+Event early fusion data.")
    parser.add_argument("--data", type=Path, default=Path("data/fred_fusion_6ch/fred_fusion_6ch.yaml"))
    parser.add_argument("--weights", default="yolov8m.pt")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default=0)
    parser.add_argument("--amp", type=str2bool, default=True)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=0.003)
    parser.add_argument("--cos-lr", type=str2bool, default=True)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--project", default="runs/fred")
    parser.add_argument("--name", default="early_fusion_yolov8m_960_e50")
    parser.add_argument("--patch-strategy", default="scratch_event", choices=("scratch_event", "repeat", "zero"))
    parser.add_argument("--no-patch", action="store_true", help="Only rely on Ultralytics automatic channel handling.")
    return parser.parse_args()


def first_conv(module):
    import torch.nn as nn

    for submodule in module.modules():
        if isinstance(submodule, nn.Conv2d):
            return submodule
    raise RuntimeError("No Conv2d layer found in model")


def patch_first_conv_to_6ch(yolo_model, strategy: str = "scratch_event") -> None:
    import torch
    import torch.nn as nn

    conv = first_conv(yolo_model.model)
    old_weight = conv.weight.detach().clone()
    old_shape = tuple(old_weight.shape)
    if conv.in_channels == 6:
        print(f"first conv already has 6 input channels: {old_shape}")
        return
    if conv.in_channels != 3:
        raise RuntimeError(f"Expected first conv to have 3 or 6 input channels, got {conv.in_channels}")

    new_conv = nn.Conv2d(
        in_channels=6,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    ).to(device=conv.weight.device, dtype=conv.weight.dtype)

    with torch.no_grad():
        new_conv.weight[:, :3] = old_weight
        if strategy == "scratch_event":
            nn.init.kaiming_normal_(new_conv.weight[:, 3:], mode="fan_out", nonlinearity="relu")
        elif strategy == "repeat":
            new_conv.weight[:, 3:] = old_weight
        elif strategy == "zero":
            new_conv.weight[:, 3:].zero_()
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)

    parent = None
    child_name = None
    for module in yolo_model.model.modules():
        for name, child in module.named_children():
            if child is conv:
                parent = module
                child_name = name
                break
        if parent is not None:
            break
    if parent is None or child_name is None:
        raise RuntimeError("Could not locate first Conv2d parent module")
    setattr(parent, child_name, new_conv)
    print(f"old first conv shape: {old_shape}")
    print(f"new first conv shape: {tuple(new_conv.weight.shape)}")
    print("RGB weight copied: yes")
    print(f"Event init strategy: {strategy}")


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


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: install ultralytics and torch before training.") from exc

    write_env_info(Path("reports/env_info.txt"))
    model = YOLO(args.resume or args.weights)
    if not args.resume and not args.no_patch:
        patch_first_conv_to_6ch(model, args.patch_strategy)

    project = Path(args.project)
    if not project.is_absolute():
        project = project.resolve()

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
        "project": str(project),
        "name": args.name,
    }
    if args.resume:
        train_kwargs["resume"] = True

    conv = first_conv(model.model)
    print(f"first conv before train: {tuple(conv.weight.shape)}")
    if conv.in_channels != 6:
        raise RuntimeError(f"Model first conv is {conv.in_channels} channels, expected 6")
    results = model.train(**train_kwargs)
    print(results)


if __name__ == "__main__":
    main()
