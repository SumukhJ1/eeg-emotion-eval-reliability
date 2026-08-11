from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

SAMPLING_RATE = 128
WINDOW_SECONDS = 2
WINDOW_SAMPLES = SAMPLING_RATE * WINDOW_SECONDS

LABEL_MAP = {
    "boring": 0,
    "calm": 1,
    "horror": 2,
    "funny": 3,
}
