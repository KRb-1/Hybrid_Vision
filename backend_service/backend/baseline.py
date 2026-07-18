import csv
from collections import defaultdict

from backend.config import B_SAMPLE_DIR, B_SUMMARY_CSV


def get_precomputed_summary() -> dict:
    methods: dict[str, dict] = defaultdict(
        lambda: {"sequences": 0, "reconstructed_frames": 0, "detected_frames": 0}
    )
    if B_SUMMARY_CSV.exists():
        with B_SUMMARY_CSV.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                item = methods[row["method"]]
                item["sequences"] += 1
                item["reconstructed_frames"] += int(row["num_output_images"])
                item["detected_frames"] += int(row["num_detection_txt"])
    for method, item in methods.items():
        total = item["reconstructed_frames"]
        item["detection_coverage"] = item["detected_frames"] / total if total else 0.0
        samples = sorted((B_SAMPLE_DIR / method).glob("*.jpg"))
        item["sample_images"] = [f"/team-b-samples/{method}/{p.name}" for p in samples]
        item["metric_note"] = "Detection coverage is not mAP, precision, or recall."
    return {
        "execution_mode": "precomputed",
        "online_inference_available": False,
        "methods": dict(methods),
    }


def run_reconstruction_baseline(*_args, **_kwargs):
    raise NotImplementedError(
        "Team B did not provide a locally runnable E2VID/FireNet pipeline and weights."
    )

