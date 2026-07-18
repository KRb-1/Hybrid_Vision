from __future__ import annotations

import re
import zipfile
from pathlib import Path

import cv2
import shutil
import subprocess


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def natural_key(path: Path):
    """
    Make frame_2.png come before frame_10.png.
    """
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def extract_zip(zip_path: Path, output_dir: Path) -> None:
    """
    Safely extract a zip file.
    Supports zips with or without a top-level folder.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_root = output_dir.resolve()

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for member in zip_ref.infolist():
            if member.is_dir():
                continue

            member_name = member.filename

            if member_name.startswith("__MACOSX/") or member_name.endswith(".DS_Store"):
                continue

            target_path = (output_dir / member_name).resolve()

            if not str(target_path).startswith(str(output_root)):
                raise ValueError("Unsafe zip file path detected.")

            target_path.parent.mkdir(parents=True, exist_ok=True)

            with zip_ref.open(member) as source, target_path.open("wb") as target:
                target.write(source.read())


def list_image_frames(frame_dir: Path) -> list[Path]:
    """
    Recursively collect image frames from a folder and sort them naturally.
    """
    frames = [
        path
        for path in frame_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    return sorted(frames, key=natural_key)


def write_video_from_frames(frame_paths: list[Path], output_path: Path, fps: int = 30) -> None:
    if not frame_paths:
        raise ValueError("No frames found for video writing.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_dir = output_path.parent / f"{output_path.stem}_tmp_frames"
    temp_dir.mkdir(parents=True, exist_ok=True)

    first_valid_frame = None

    for index, frame_path in enumerate(frame_paths):
        frame = cv2.imread(str(frame_path))

        if frame is None:
            continue

        height, width = frame.shape[:2]

        # H.264 + yuv420p 要求宽高最好是偶数
        if width % 2 != 0:
            frame = frame[:, :-1]
        if height % 2 != 0:
            frame = frame[:-1, :]

        temp_frame_path = temp_dir / f"{index:06d}.jpg"
        cv2.imwrite(str(temp_frame_path), frame)

        if first_valid_frame is None:
            first_valid_frame = temp_frame_path

    if first_valid_frame is None:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise ValueError("No readable frames found for video writing.")

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-i", str(temp_dir / "%06d.jpg"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path),
    ]

    subprocess.run(ffmpeg_cmd, check=True)

    shutil.rmtree(temp_dir, ignore_errors=True)