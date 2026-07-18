from pathlib import Path
from threading import Lock

import torch

from backend.config import (
    DEFAULT_CONFIDENCE,
    DEFAULT_IMAGE_SIZE,
    INFERENCE_DEVICE,
    LATE_FUSION_IOU,
    OUTPUT_DIR,
)
from backend.fusion import nms_late_fusion
from backend.fusion_input import load_six_channel_array, write_six_channel_tiff
from backend.model_registry import models


_inference_lock = Lock()


def _xywh(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]


def _is_out_of_memory(exc: Exception) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _predict(model, source, confidence: float, device: str):
    try:
        return model.predict(
            source=source,
            conf=confidence,
            imgsz=DEFAULT_IMAGE_SIZE,
            device=device,
            half=device != "cpu",
            verbose=False,
        )
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if not _is_out_of_memory(exc):
            raise
        model.to("cpu")
        torch.cuda.empty_cache()
        return model.predict(
            source=source,
            conf=confidence,
            imgsz=DEFAULT_IMAGE_SIZE,
            device="cpu",
            half=False,
            verbose=False,
        )


def _detections_from_results(results, source: str) -> list[dict]:
    detections: list[dict] = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            coords = [float(value) for value in box.xyxy[0].cpu().tolist()]
            class_id = int(box.cls[0].cpu().item())
            name = result.names.get(class_id, "drone")
            detections.append(
                {
                    "frame": 0,
                    "bbox_xyxy": coords,
                    "bbox_xywh": _xywh(coords),
                    "class_id": class_id,
                    "class_name": str(name),
                    "confidence": float(box.conf[0].cpu().item()),
                    "source": source,
                }
            )
    return detections


def run_detection(
    rgb_path: Path,
    event_path: Path,
    method: str,
    confidence: float = DEFAULT_CONFIDENCE,
    request_id: str | None = None,
) -> list[dict]:
    model = models.get(method)
    device = INFERENCE_DEVICE
    if device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"

    if method == "early_fusion":
        fused_path = OUTPUT_DIR / (request_id or "tmp") / "early_fusion_input.tiff"
        source = str(write_six_channel_tiff(rgb_path, event_path, fused_path))
    elif method == "two_fusion":
        source = [load_six_channel_array(rgb_path, event_path)]
    elif method == "late_fusion":
        source = None
    else:
        raise ValueError(f"Unsupported method: {method}")

    with _inference_lock:
        if method == "late_fusion":
            rgb_results = _predict(model["rgb"], str(rgb_path), confidence, device)
            event_results = _predict(model["event"], str(event_path), confidence, device)
            detections = nms_late_fusion(
                _detections_from_results(rgb_results, "rgb"),
                _detections_from_results(event_results, "event"),
                iou_threshold=LATE_FUSION_IOU,
                source="late_fusion",
            )
            del rgb_results, event_results
        else:
            results = _predict(model, source, confidence, device)
            detections = _detections_from_results(results, method)
            del results
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return detections
