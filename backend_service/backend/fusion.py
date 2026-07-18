from __future__ import annotations


def _iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / (area_a + area_b - intersection + 1e-9)


def nms_late_fusion(
    rgb: list[dict], event: list[dict], iou_threshold: float = 0.45, source: str = "fused"
) -> list[dict]:
    """Class-aware NMS over RGB and Event detections."""
    candidates = [dict(item) for item in rgb + event]
    candidates.sort(key=lambda item: item["confidence"], reverse=True)
    kept: list[dict] = []
    while candidates:
        best = candidates.pop(0)
        best["source"] = source
        kept.append(best)
        candidates = [
            item
            for item in candidates
            if item["class_id"] != best["class_id"]
            or _iou(item["bbox_xyxy"], best["bbox_xyxy"]) <= iou_threshold
        ]
    return kept
