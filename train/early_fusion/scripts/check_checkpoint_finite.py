from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check a YOLO checkpoint for NaN/Inf tensors.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--max-report", type=int, default=30)
    return parser.parse_args()


def check_state_dict(name: str, module, max_report: int = 30) -> bool:
    import torch

    if module is None:
        print(f"{name}: None")
        return True

    if hasattr(module, "float"):
        state_dict = module.float().state_dict()
    elif isinstance(module, dict):
        state_dict = module
    else:
        state_dict = module.state_dict()

    bad: list[str] = []
    for key, value in state_dict.items():
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            bad.append(key)

    if bad:
        print(f"{name}: BAD, contains NaN/Inf:")
        for key in bad[:max_report]:
            print(f"  {key}")
        if len(bad) > max_report:
            print(f"  ... {len(bad) - max_report} more")
        return False

    print(f"{name}: OK, all finite")
    return True


def check_checkpoint(weights: Path, max_report: int = 30) -> bool:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: install torch before checking checkpoints.") from exc

    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        model_ok = check_state_dict("model", checkpoint.get("model"), max_report)
        ema_ok = check_state_dict("ema", checkpoint.get("ema"), max_report)
        return model_ok and ema_ok

    return check_state_dict("checkpoint", checkpoint, max_report)


def main() -> None:
    args = parse_args()
    ok = check_checkpoint(args.weights, args.max_report)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
