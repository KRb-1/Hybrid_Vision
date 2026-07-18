from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .rgb_event_stem import RGBEventStem


@dataclass
class LoadReport:
    total_parameters: int
    loaded_parameters: int
    skipped_parameters: int
    loaded_tensors: int
    skipped_tensors: int
    rgb_branch_from_pretrained: bool
    event_branch_init: str = "kaiming"
    fusion_conv_init: str = "kaiming"

    def format(self) -> str:
        return "\n".join(
            [
                f"Total parameters: {self.total_parameters}",
                f"Loaded pretrained parameters: {self.loaded_parameters}",
                f"Skipped parameters: {self.skipped_parameters}",
                f"Loaded tensors: {self.loaded_tensors}",
                f"Skipped tensors: {self.skipped_tensors}",
                "RGB branch initialized from pretrained first conv: "
                f"{'yes' if self.rgb_branch_from_pretrained else 'no'}",
                f"Event branch initialized: {self.event_branch_init}",
                f"Fusion conv initialized: {self.fusion_conv_init}",
            ]
        )


def _conv2d_from_ultralytics_conv(module: nn.Module) -> nn.Conv2d | None:
    conv = getattr(module, "conv", None)
    return conv if isinstance(conv, nn.Conv2d) else None


def first_model_conv(module: nn.Module) -> nn.Conv2d | None:
    for submodule in module.modules():
        if isinstance(submodule, nn.Conv2d):
            return submodule
    return None


def first_rgb_event_stem(module: nn.Module) -> RGBEventStem | None:
    for submodule in module.modules():
        if isinstance(submodule, RGBEventStem):
            return submodule
    return None


def kaiming_init_conv(conv: nn.Conv2d) -> None:
    nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


def init_stem_from_pretrained(target_model: nn.Module, pretrained_model: nn.Module) -> bool:
    stem = first_rgb_event_stem(target_model)
    pretrained_conv = first_model_conv(pretrained_model)
    if stem is None:
        raise RuntimeError("Target model does not contain RGBEventStem")
    if pretrained_conv is None:
        return False

    rgb_conv = _conv2d_from_ultralytics_conv(stem.rgb)
    event_conv = _conv2d_from_ultralytics_conv(stem.event)
    fuse_conv = _conv2d_from_ultralytics_conv(stem.fuse)
    if rgb_conv is None or event_conv is None or fuse_conv is None:
        raise RuntimeError("RGBEventStem branches do not expose expected Conv.conv modules")

    with torch.no_grad():
        copy_out = min(rgb_conv.weight.shape[0], pretrained_conv.weight.shape[0])
        compatible = (
            pretrained_conv.weight.ndim == 4
            and pretrained_conv.weight.shape[1] == 3
            and pretrained_conv.weight.shape[2:] == rgb_conv.weight.shape[2:]
        )
        if compatible and copy_out > 0:
            rgb_conv.weight[:copy_out].copy_(pretrained_conv.weight[:copy_out].to(rgb_conv.weight))
            if rgb_conv.bias is not None and pretrained_conv.bias is not None:
                rgb_conv.bias[:copy_out].copy_(pretrained_conv.bias[:copy_out].to(rgb_conv.bias))
            if copy_out < rgb_conv.weight.shape[0]:
                nn.init.kaiming_normal_(rgb_conv.weight[copy_out:], mode="fan_out", nonlinearity="relu")
        else:
            kaiming_init_conv(rgb_conv)

        kaiming_init_conv(event_conv)
        kaiming_init_conv(fuse_conv)
    return bool(compatible and copy_out > 0)


def load_matching_pretrained(target_yolo, pretrained: str | Path | None) -> LoadReport:
    from ultralytics import YOLO

    target_model = target_yolo.model
    total = sum(p.numel() for p in target_model.parameters())
    if not pretrained:
        init_stem_from_pretrained(target_model, target_model)
        return LoadReport(total, 0, total, 0, len(target_model.state_dict()), False)

    pretrained_yolo = YOLO(str(pretrained))
    pretrained_model = pretrained_yolo.model
    rgb_init = init_stem_from_pretrained(target_model, pretrained_model)

    target_state = target_model.state_dict()
    pretrained_state = pretrained_model.state_dict()
    update = {}
    loaded_params = 0
    skipped_params = 0
    skipped_tensors = 0

    for key, value in target_state.items():
        source = pretrained_state.get(key)
        if source is not None and tuple(source.shape) == tuple(value.shape):
            update[key] = source.to(value.device, dtype=value.dtype)
            loaded_params += value.numel()
        else:
            skipped_params += value.numel()
            skipped_tensors += 1

    target_state.update(update)
    target_model.load_state_dict(target_state, strict=True)
    return LoadReport(total, loaded_params, skipped_params, len(update), skipped_tensors, rgb_init)
