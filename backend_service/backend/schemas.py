from typing import Literal

from pydantic import BaseModel, Field


class Detection(BaseModel):
    frame: int = 0
    bbox_xyxy: list[float] = Field(min_length=4, max_length=4)
    bbox_xywh: list[float] = Field(min_length=4, max_length=4)
    class_id: int = 0
    class_name: str = "drone"
    confidence: float
    source: Literal["early_fusion", "two_fusion", "late_fusion", "rgb", "event", "fused"]


class DetectResponse(BaseModel):
    status: Literal["success"] = "success"
    request_id: str
    method: str
    execution_mode: Literal["online", "precomputed"]
    detections: list[Detection]
    tracks: list | None = None
    visualization_url: str | None = None
    note: str | None = None
