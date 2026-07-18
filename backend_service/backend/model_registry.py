from __future__ import annotations

from threading import Lock

from backend.config import (
    EARLY_FUSION_WEIGHTS,
    LATE_FUSION_EVENT_WEIGHTS,
    LATE_FUSION_RGB_WEIGHTS,
    TWO_FUSION_WEIGHTS,
)


SUPPORTED_METHODS = ("early_fusion", "two_fusion", "late_fusion")


class ModelRegistry:
    """Loads each YOLO model once, on first use."""

    def __init__(self) -> None:
        self._models: dict[str, object] = {}
        self._lock = Lock()
        self.last_error: str | None = None

    def get(self, method: str):
        if method not in SUPPORTED_METHODS:
            raise ValueError(f"Unsupported method: {method}")
        if method in self._models:
            return self._models[method]
        with self._lock:
            if method not in self._models:
                try:
                    from ultralytics import YOLO

                    if method == "two_fusion":
                        from fred_two_stream.ultralytics_patch import register_rgb_event_stem

                        register_rgb_event_stem()

                    if method == "late_fusion":
                        if not LATE_FUSION_RGB_WEIGHTS.exists():
                            raise FileNotFoundError(f"Weight file not found: {LATE_FUSION_RGB_WEIGHTS}")
                        if not LATE_FUSION_EVENT_WEIGHTS.exists():
                            raise FileNotFoundError(f"Weight file not found: {LATE_FUSION_EVENT_WEIGHTS}")
                        self._models[method] = {
                            "rgb": YOLO(str(LATE_FUSION_RGB_WEIGHTS)),
                            "event": YOLO(str(LATE_FUSION_EVENT_WEIGHTS)),
                        }
                    else:
                        path = EARLY_FUSION_WEIGHTS if method == "early_fusion" else TWO_FUSION_WEIGHTS
                        if not path.exists():
                            raise FileNotFoundError(f"Weight file not found: {path}")
                        self._models[method] = YOLO(str(path))
                    self.last_error = None
                except Exception as exc:
                    self.last_error = str(exc)
                    raise
        return self._models[method]

    @property
    def loaded(self) -> list[str]:
        return sorted(self._models)


models = ModelRegistry()
