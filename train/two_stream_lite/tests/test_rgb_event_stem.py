from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")

from fred_two_stream.rgb_event_stem import RGBEventStem


def test_rgb_event_stem_forward_shape() -> None:
    stem = RGBEventStem(6, 48, 3, 2)
    x = torch.randn(2, 6, 320, 320)
    y = stem(x)
    assert y.shape == (2, 48, 160, 160)


def test_rgb_event_stem_rejects_wrong_channels() -> None:
    stem = RGBEventStem(6, 48, 3, 2)
    with pytest.raises(ValueError):
        stem(torch.randn(1, 3, 320, 320))
