from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.baseline import get_precomputed_summary
from backend.config import (
    B_SAMPLE_DIR,
    DEFAULT_CONFIDENCE,
    EARLY_FUSION_WEIGHTS,
    INFERENCE_DEVICE,
    LATE_FUSION_EVENT_WEIGHTS,
    LATE_FUSION_RGB_WEIGHTS,
    OUTPUT_DIR,
    TWO_FUSION_WEIGHTS,
    UPLOAD_DIR,
    ensure_runtime_dirs,
)
from backend.inference import run_detection
from backend.model_registry import models
from backend.schemas import DetectResponse
from backend.visualization import draw_detections
from backend.video_io import extract_zip, list_image_frames, write_video_from_frames


ensure_runtime_dirs()
app = FastAPI(title="AMI Hybrid Vision API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/results", StaticFiles(directory=OUTPUT_DIR), name="results")
if B_SAMPLE_DIR.exists():
    app.mount("/team-b-samples", StaticFiles(directory=B_SAMPLE_DIR), name="team-b-samples")

ImageUpload = Annotated[UploadFile, File()]
Method = Literal["early_fusion", "two_fusion", "late_fusion"]


def _save_upload(upload: UploadFile, destination: Path) -> Path:
    suffix = Path(upload.filename or "image.jpg").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}:
        raise HTTPException(400, "This version accepts image files only.")
    destination = destination.with_suffix(suffix)
    with destination.open("wb") as handle:
        shutil.copyfileobj(upload.file, handle)
    return destination

def _save_zip_upload(upload: UploadFile, destination: Path) -> Path:
    suffix = Path(upload.filename or "frames.zip").suffix.lower()

    if suffix != ".zip":
        raise HTTPException(400, "This endpoint accepts .zip files only.")

    destination = destination.with_suffix(".zip")
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("wb") as handle:
        shutil.copyfileobj(upload.file, handle)

    return destination


def _run_pair(
    rgb_path: Path,
    event_path: Path,
    method: str,
    confidence: float,
    request_id: str,
):
    return run_detection(rgb_path, event_path, method, confidence, request_id=request_id), rgb_path


@app.get("/api/health")
def health() -> dict:
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "unavailable"
    return {
        "status": "ok",
        "device_available": device,
        "inference_device_setting": INFERENCE_DEVICE,
        "models_loaded": models.loaded,
        "weights_available": {
            "early_fusion": EARLY_FUSION_WEIGHTS.exists(),
            "two_fusion": TWO_FUSION_WEIGHTS.exists(),
            "late_fusion": LATE_FUSION_RGB_WEIGHTS.exists() and LATE_FUSION_EVENT_WEIGHTS.exists(),
        },
        "last_model_error": models.last_error,
        "team_b_online_available": False,
    }


@app.post("/api/detect", response_model=DetectResponse)
def detect(
    rgb_file: ImageUpload,
    event_file: ImageUpload,
    method: Annotated[Method, Form()] = "two_fusion",
    use_tracking: Annotated[bool, Form()] = False,
    confidence: Annotated[float, Form(ge=0.0, le=1.0)] = DEFAULT_CONFIDENCE,
):
    if use_tracking:
        raise HTTPException(501, "Tracking has not been integrated by Team D yet.")
    request_id = uuid.uuid4().hex
    upload_dir = UPLOAD_DIR / request_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    rgb_path = _save_upload(rgb_file, upload_dir / "rgb")
    event_path = _save_upload(event_file, upload_dir / "event")
    try:
        detections, display_path = _run_pair(rgb_path, event_path, method, confidence, request_id)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Inference failed: {exc}") from exc
    result_path = OUTPUT_DIR / request_id / "detection.jpg"
    draw_detections(display_path, detections, result_path)
    return {
        "request_id": request_id,
        "method": method,
        "execution_mode": "online",
        "detections": detections,
        "tracks": None,
        "visualization_url": f"/results/{request_id}/detection.jpg",
        "note": "Team A online hybrid model. Team B reconstruction baselines are exposed via /api/compare.",
    }

@app.post("/api/detect-sequence")
def detect_sequence(
    rgb_zip: Annotated[UploadFile, File()],
    event_zip: Annotated[UploadFile, File()],
    method: Annotated[Method, Form()] = "two_fusion",
    confidence: Annotated[float, Form(ge=0.0, le=1.0)] = DEFAULT_CONFIDENCE,
    fps: Annotated[int, Form(ge=1, le=120)] = 30,
    max_frames: Annotated[int, Form(ge=1, le=3000)] = 300,
):

    request_id = uuid.uuid4().hex

    upload_dir = UPLOAD_DIR / request_id
    output_dir = OUTPUT_DIR / request_id

    rgb_extract_dir = upload_dir / "rgb_frames"
    event_extract_dir = upload_dir / "event_frames"
    detection_frame_dir = output_dir / "detection_frames"

    upload_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    detection_frame_dir.mkdir(parents=True, exist_ok=True)

    rgb_zip_path = _save_zip_upload(rgb_zip, upload_dir / "rgb_frames")
    event_zip_path = _save_zip_upload(event_zip, upload_dir / "event_frames")

    try:
        extract_zip(rgb_zip_path, rgb_extract_dir)
        extract_zip(event_zip_path, event_extract_dir)

        rgb_frames = list_image_frames(rgb_extract_dir)
        event_frames = list_image_frames(event_extract_dir)

        if not rgb_frames:
            raise HTTPException(400, "No RGB frames found in rgb_zip.")

        if not event_frames:
            raise HTTPException(400, "No Event frames found in event_zip.")

        frame_count = min(len(rgb_frames), len(event_frames), max_frames)

        if frame_count <= 0:
            raise HTTPException(400, "No synchronized frame pairs found.")

        rgb_frames = rgb_frames[:frame_count]
        event_frames = event_frames[:frame_count]

        detections_per_frame = []
        detection_frame_paths = []

        for frame_index, (rgb_frame, event_frame) in enumerate(zip(rgb_frames, event_frames)):
            try:
                detections = run_detection(
                    rgb_path=rgb_frame,
                    event_path=event_frame,
                    method=method,
                    confidence=confidence,
                    request_id=request_id,
                )
            except Exception as exc:
                raise HTTPException(
                    500,
                    f"Inference failed at frame {frame_index}: {exc}",
                ) from exc

            for detection in detections:
                detection["frame"] = frame_index
                detection["timestamp"] = frame_index / fps

            detection_frame_path = detection_frame_dir / f"{frame_index:06d}.jpg"
            draw_detections(rgb_frame, detections, detection_frame_path)

            detection_frame_paths.append(detection_frame_path)

            detections_per_frame.append(
                {
                    "frame": frame_index,
                    "timestamp": frame_index / fps,
                    "detections": detections,
                }
            )

        rgb_video_path = output_dir / "rgb_input.mp4"
        event_video_path = output_dir / "event_input.mp4"
        output_video_path = output_dir / "output_detection.mp4"

        write_video_from_frames(rgb_frames, rgb_video_path, fps=fps)
        write_video_from_frames(event_frames, event_video_path, fps=fps)
        write_video_from_frames(detection_frame_paths, output_video_path, fps=fps)

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Sequence detection failed: {exc}") from exc

    total_detections = sum(len(item["detections"]) for item in detections_per_frame)

    return {
        "status": "success",
        "request_id": request_id,
        "method": method,
        "execution_mode": "online_sequence",
        "frame_count": frame_count,
        "rgb_frame_count": len(rgb_frames),
        "event_frame_count": len(event_frames),
        "fps": fps,
        "total_detections": total_detections,
        "rgb_video_url": f"/results/{request_id}/rgb_input.mp4",
        "event_video_url": f"/results/{request_id}/event_input.mp4",
        "output_video_url": f"/results/{request_id}/output_detection.mp4",
        "detections_per_frame": detections_per_frame,
        "note": "Frame sequence detection using synchronized RGB/Event frame pairs.",
    }


@app.get("/api/compare")
@app.post("/api/compare")
def compare() -> dict:
    return {
        "status": "success",
        "team_a": {
            "online_methods": ["early_fusion", "two_fusion", "late_fusion"],
            "default_method": "two_fusion",
            "hybrid_type": "RGB/Event hybrid fusion models plus late-fusion baseline",
        },
        "team_b": get_precomputed_summary(),
    }
