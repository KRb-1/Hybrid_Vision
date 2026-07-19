from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import yaml
from PIL import Image, ImageDraw


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Sample:
    stem: str
    rgb: Path
    event: Path
    label: Path


def resolve_path(path: str | Path, base: Path | None = None) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (base or Path.cwd()) / path


def image_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def find_by_stem(directory: Path, stem: str, exts: Sequence[str] = IMAGE_EXTS) -> Path | None:
    for ext in exts:
        p = directory / f"{stem}{ext}"
        if p.exists():
            return p
    lower_matches = sorted(p for p in directory.iterdir() if p.is_file() and p.stem == stem)
    return lower_matches[0] if lower_matches else None


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


def read_six_channel_tiff(path: Path) -> tuple[list[Image.Image], tuple[int, int]]:
    img = Image.open(path)
    pages: list[Image.Image] = []
    try:
        frame_count = getattr(img, "n_frames", 1)
        for i in range(frame_count):
            img.seek(i)
            pages.append(img.convert("L").copy())
    finally:
        img.close()
    if len(pages) == 1 and pages[0].mode == "RGB":
        channels = list(pages[0].split())
        return channels, channels[0].size
    if pages:
        return pages, pages[0].size
    return [], (0, 0)


def six_channel_to_rgb_event(path: Path) -> tuple[Image.Image, Image.Image]:
    pages, _ = read_six_channel_tiff(path)
    if len(pages) != 6:
        raise ValueError(f"{path} has {len(pages)} channels, expected 6")
    return Image.merge("RGB", pages[:3]), Image.merge("RGB", pages[3:6])


def load_data_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    dataset_root = resolve_path(data["path"]) if data.get("path") else path.parent
    data["_yaml_path"] = path
    data["_root"] = dataset_root
    return data


def write_data_yaml(out_root: Path) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    yaml_path = out_root / "fred_fusion_6ch.yaml"
    data = {
        "path": str(out_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "channels": 6,
        "names": {0: "drone"},
    }
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return yaml_path


def collect_raw_split(raw_root: Path, split: str) -> list[Sample]:
    split_root = raw_root / split
    rgb_dir = split_root / "images"
    event_dir = split_root / "events"
    label_dir = split_root / "labels"
    samples: list[Sample] = []
    for rgb in image_files(rgb_dir):
        event = find_by_stem(event_dir, rgb.stem)
        label = label_dir / f"{rgb.stem}.txt"
        if event and label.exists():
            samples.append(Sample(rgb.stem, rgb, event, label))
    return samples


def collect_fused_split(data: dict, split: str) -> list[Sample]:
    root = data["_root"]
    image_rel = data.get(split)
    if not image_rel:
        return []
    image_dir = resolve_path(image_rel, root)
    label_dir = root / "labels" / split
    samples: list[Sample] = []
    for fused in image_files(image_dir):
        label = label_dir / f"{fused.stem}.txt"
        samples.append(Sample(fused.stem, fused, fused, label))
    return samples


def limit_samples(samples: list[Sample], limit: int | None, seed: int) -> list[Sample]:
    if not limit or limit >= len(samples):
        return samples
    rng = random.Random(seed)
    selected = samples[:]
    rng.shuffle(selected)
    return sorted(selected[:limit], key=lambda s: s.stem)


def parse_label_file(path: Path) -> tuple[list[tuple[int, float, float, float, float]], list[str]]:
    errors: list[str] = []
    boxes: list[tuple[int, float, float, float, float]] = []
    if not path.exists():
        return boxes, [f"missing label: {path}"]
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split()
        if len(parts) != 5:
            errors.append(f"{path}:{lineno} has {len(parts)} columns")
            continue
        try:
            cls = int(float(parts[0]))
            vals = tuple(float(x) for x in parts[1:])
        except ValueError:
            errors.append(f"{path}:{lineno} contains non-numeric values")
            continue
        if any(v < 0.0 or v > 1.0 for v in vals):
            errors.append(f"{path}:{lineno} bbox values outside [0,1]")
        boxes.append((cls, *vals))
    return boxes, errors


def draw_boxes(image: Image.Image, boxes: Iterable[tuple[int, float, float, float, float]]) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    for cls, cx, cy, bw, bh in boxes:
        x1 = (cx - bw / 2.0) * w
        y1 = (cy - bh / 2.0) * h
        x2 = (cx + bw / 2.0) * w
        y2 = (cy + bh / 2.0) * h
        draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=3)
        draw.text((x1 + 2, max(0, y1 - 12)), str(cls), fill=(255, 0, 0))
    return out


def save_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
