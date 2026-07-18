from pathlib import Path
import os


ROOT_DIR = Path(__file__).resolve().parent.parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT_DIR / ".ultralytics"))
MODEL_DIR = ROOT_DIR / "models"
TEAM_B_DIR = ROOT_DIR / "team_b"
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", ROOT_DIR / "uploads"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", ROOT_DIR / "outputs"))
EARLY_FUSION_WEIGHTS = Path(
    os.getenv("EARLY_FUSION_WEIGHTS", MODEL_DIR / "early_fusion_best.pt")
)
TWO_FUSION_WEIGHTS = Path(
    os.getenv("TWO_FUSION_WEIGHTS", MODEL_DIR / "two_fusion_best.pt")
)
LATE_FUSION_RGB_WEIGHTS = Path(
    os.getenv("LATE_FUSION_RGB_WEIGHTS", MODEL_DIR / "late_fusion_rgb_best.pt")
)
LATE_FUSION_EVENT_WEIGHTS = Path(
    os.getenv("LATE_FUSION_EVENT_WEIGHTS", MODEL_DIR / "late_fusion_event_best.pt")
)
LATE_FUSION_IOU = float(os.getenv("LATE_FUSION_IOU", "0.45"))
B_SUMMARY_CSV = TEAM_B_DIR / "results_summary" / "detection_summary.csv"
B_SAMPLE_DIR = TEAM_B_DIR / "sample_detection_images"
DEFAULT_CONFIDENCE = float(os.getenv("CONFIDENCE", "0.25"))
DEFAULT_IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "640"))
INFERENCE_DEVICE = os.getenv("INFERENCE_DEVICE", "auto")


def ensure_runtime_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
