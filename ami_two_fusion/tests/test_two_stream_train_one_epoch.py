from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


pytest.importorskip("torch")
pytest.importorskip("ultralytics")


def test_two_stream_train_one_epoch_smoke() -> None:
    if os.environ.get("RUN_TRAIN_SMOKE") != "1":
        pytest.skip("Set RUN_TRAIN_SMOKE=1 to run the one-epoch training smoke test.")
    if not Path("FRED_ready").exists():
        pytest.skip("FRED_ready dataset is not present.")

    subprocess.run(
        [
            sys.executable,
            "scripts/build_fusion_tiff.py",
            "--raw-root",
            "FRED_ready",
            "--out-root",
            "data/fred_fusion_6ch_debug",
            "--limit-train",
            "4",
            "--limit-val",
            "2",
            "--limit-test",
            "2",
            "--overwrite",
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/train_two_stream_lite.py",
            "--data",
            "data/fred_fusion_6ch_debug/fred_fusion_6ch.yaml",
            "--model",
            "configs/models/yolov8m_rgb_event_stem.yaml",
            "--skip-pretrained",
            "--imgsz",
            "320",
            "--epochs",
            "1",
            "--batch",
            "1",
            "--workers",
            "0",
            "--device",
            "cpu",
            "--project",
            "runs/debug",
            "--name",
            "two_stream_lite_smoke",
        ],
        check=True,
    )
    assert Path("runs/debug/two_stream_lite_smoke/weights/best.pt").exists()
