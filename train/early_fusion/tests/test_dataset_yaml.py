from pathlib import Path

from scripts.fred_fusion_utils import load_data_yaml, write_data_yaml


def test_dataset_yaml_contains_six_channels(tmp_path: Path) -> None:
    yaml_path = write_data_yaml(tmp_path / "fred_fusion_6ch")
    data = load_data_yaml(yaml_path)

    assert data["channels"] == 6
    assert data["names"][0] == "drone"
    assert data["train"] == "images/train"
