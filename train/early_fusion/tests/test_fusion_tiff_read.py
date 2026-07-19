from pathlib import Path

from PIL import Image

from scripts.fred_fusion_utils import read_six_channel_tiff, write_six_channel_tiff


def test_fusion_tiff_has_six_uint8_pages(tmp_path: Path) -> None:
    rgb = tmp_path / "rgb.jpg"
    event = tmp_path / "event.png"
    out = tmp_path / "fusion.tiff"
    Image.new("RGB", (16, 12), (10, 20, 30)).save(rgb)
    Image.new("RGB", (16, 12), (40, 50, 60)).save(event)

    write_six_channel_tiff(rgb, event, out)
    pages, size = read_six_channel_tiff(out)

    assert len(pages) == 6
    assert size == (16, 12)
    assert all(page.mode == "L" for page in pages)
