from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"

IMAGE_SIZE = (128, 128)

CLASS_NAMES = [
    "pizza",
    "steak"
]