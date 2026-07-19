from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import yaml
from PIL import Image


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Sample:
    stem: str
    rgb: Path
    event: Path
    label: Path


def image_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def find_by_stem(directory: Path, stem: str, exts: Sequence[str] = IMAGE_EXTS) -> Path | None:
    if not directory.exists():
        return None
    for ext in exts:
        path = directory / f"{stem}{ext}"
        if path.exists():
            return path
    matches = sorted(path for path in directory.iterdir() if path.is_file() and path.stem == stem)
    return matches[0] if matches else None


def raw_split_layouts(raw_root: Path, split: str) -> list[tuple[Path, Path, Path]]:
    return [
        (raw_root / split / "images", raw_root / split / "events", raw_root / split / "labels"),
        (raw_root / "images" / split, raw_root / "events" / split, raw_root / "labels" / split),
    ]


def collect_raw_split(raw_root: Path, split: str) -> list[Sample]:
    image_dir, event_dir, label_dir = raw_split_layouts(raw_root, split)[0]
    for candidate in raw_split_layouts(raw_root, split):
        if candidate[0].exists() and candidate[1].exists() and candidate[2].exists():
            image_dir, event_dir, label_dir = candidate
            break

    samples: list[Sample] = []
    for rgb in image_files(image_dir):
        event = find_by_stem(event_dir, rgb.stem)
        label = label_dir / f"{rgb.stem}.txt"
        if event is not None and label.exists():
            samples.append(Sample(rgb.stem, rgb, event, label))
    return samples


def describe_expected_raw_layouts(raw_root: Path) -> str:
    lines = [f"raw root: {raw_root}"]
    for split in SPLITS:
        lines.append(f"{split}:")
        for image_dir, event_dir, label_dir in raw_split_layouts(raw_root, split):
            lines.append(f"  images={image_dir} exists={image_dir.exists()}")
            lines.append(f"  events={event_dir} exists={event_dir.exists()}")
            lines.append(f"  labels={label_dir} exists={label_dir.exists()}")
    return "\n".join(lines)


def limit_samples(samples: list[Sample], limit: int | None, seed: int) -> list[Sample]:
    if limit is None or limit <= 0 or limit >= len(samples):
        return samples
    rng = random.Random(seed)
    selected = samples[:]
    rng.shuffle(selected)
    return sorted(selected[:limit], key=lambda item: item.stem)


def read_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def write_six_channel_tiff(
    rgb_path: Path,
    event_path: Path,
    out_path: Path,
    compression: str | None = "tiff_deflate",
) -> None:
    rgb = read_rgb(rgb_path)
    event = read_rgb(event_path)
    if event.size != rgb.size:
        event = event.resize(rgb.size, Image.BILINEAR)

    pages = list(rgb.split()) + list(event.split())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"save_all": True, "append_images": pages[1:]}
    if compression:
        save_kwargs["compression"] = compression
    pages[0].save(out_path, **save_kwargs)


def read_six_channel_tiff(path: Path) -> tuple[list[Image.Image], tuple[int, int]]:
    img = Image.open(path)
    pages: list[Image.Image] = []
    try:
        frame_count = getattr(img, "n_frames", 1)
        for index in range(frame_count):
            img.seek(index)
            pages.append(img.convert("L").copy())
    finally:
        img.close()
    if len(pages) == 1 and pages[0].mode == "RGB":
        channels = list(pages[0].split())
        return channels, channels[0].size
    return pages, pages[0].size if pages else (0, 0)


def six_channel_to_rgb_event(path: Path) -> tuple[Image.Image, Image.Image]:
    channels, _ = read_six_channel_tiff(path)
    if len(channels) != 6:
        raise ValueError(f"{path} has {len(channels)} channels, expected 6")
    return Image.merge("RGB", channels[:3]), Image.merge("RGB", channels[3:6])


def write_data_yaml(out_root: Path) -> Path:
    yaml_path = out_root / "fred_fusion_6ch.yaml"
    data = {
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "channels": 6,
        "names": {0: "drone"},
    }
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return yaml_path


def load_data_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    root = Path(data.get("path") or path.parent)
    if not root.is_absolute():
        root = path.parent / root
    data["_root"] = root
    data["_yaml_path"] = path
    return data
