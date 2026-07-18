from __future__ import annotations

import torch
from torch import nn


try:
    from ultralytics.nn.modules import Conv
except ModuleNotFoundError as exc:  # pragma: no cover - dependency check path
    raise ModuleNotFoundError(
        "RGBEventStem requires ultralytics. Install backend dependencies first."
    ) from exc


class RGBEventStem(nn.Module):
    """Two-stream shallow stem for 6-channel RGB+Event input.

    Channel convention:
      - x[:, 0:3]: RGB
      - x[:, 3:6]: Event
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 3,
        s: int = 2,
        p: int | None = None,
        g: int = 1,
        act=True,
    ):
        super().__init__()
        if c1 != 6:
            raise ValueError(f"RGBEventStem expects 6 input channels, got {c1}")
        if c2 % 2 != 0:
            raise ValueError(f"RGBEventStem output channels must be even, got {c2}")
        if g != 1:
            raise ValueError("RGBEventStem branch convolutions require groups=1")

        c_branch = c2 // 2
        self.in_channels = c1
        self.out_channels = c2
        self.rgb = Conv(3, c_branch, k, s, p, 1, act=act)
        self.event = Conv(3, c_branch, k, s, p, 1, act=act)
        self.fuse = Conv(c2, c2, 1, 1, None, 1, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"RGBEventStem expects BCHW input, got shape {tuple(x.shape)}")
        if x.shape[1] == 3:
            # Ultralytics may run a 3-channel dummy forward while inferring strides.
            # Real FRED inference data remains 6-channel; the dummy event stream is zero-filled.
            x = torch.cat((x, torch.zeros_like(x)), dim=1)
        if x.shape[1] != 6:
            raise ValueError(f"RGBEventStem expects 6 input channels, got {x.shape[1]}")
        rgb = x[:, 0:3, :, :]
        event = x[:, 3:6, :, :]
        return self.fuse(torch.cat((self.rgb(rgb), self.event(event)), dim=1))
