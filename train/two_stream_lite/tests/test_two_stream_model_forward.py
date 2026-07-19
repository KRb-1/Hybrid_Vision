from __future__ import annotations

from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")

from fred_two_stream.pretrained import first_rgb_event_stem
from fred_two_stream.ultralytics_patch import register_rgb_event_stem


def test_two_stream_yolo_model_forward() -> None:
    from ultralytics import YOLO

    register_rgb_event_stem()
    model = YOLO(str(Path("configs/models/yolov8m_rgb_event_stem.yaml")))
    assert first_rgb_event_stem(model.model) is not None
    x = torch.randn(1, 6, 320, 320)
    with torch.no_grad():
        out = model.model(x)
    assert out is not None
