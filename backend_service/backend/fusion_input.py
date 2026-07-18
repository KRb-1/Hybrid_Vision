from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def read_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_six_channel_array(rgb_path: Path, event_path: Path) -> np.ndarray:
    """Return a HWC uint8 array with channels [R, G, B, E_R, E_G, E_B]."""

    rgb = read_rgb(rgb_path)
    event = read_rgb(event_path)
    if event.size != rgb.size:
        event = event.resize(rgb.size, Image.BILINEAR)
    return np.concatenate([np.asarray(rgb), np.asarray(event)], axis=2)


def write_six_channel_tiff(
    rgb_path: Path,
    event_path: Path,
    out_path: Path,
    compression: str | None = "tiff_deflate",
) -> Path:
    """Write RGB and Event images as a 6-page TIFF expected by early_fusion."""

    rgb = read_rgb(rgb_path)
    event = read_rgb(event_path)
    if event.size != rgb.size:
        event = event.resize(rgb.size, Image.BILINEAR)

    pages = list(rgb.split()) + list(event.split())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f"{out_path.stem}.tmp{out_path.suffix}")
    save_kwargs = {"save_all": True, "append_images": pages[1:]}
    if compression:
        save_kwargs["compression"] = compression
    try:
        pages[0].save(tmp_path, **save_kwargs)
        tmp_path.replace(out_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return out_path
