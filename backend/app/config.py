from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "ml" / "models" / "final_model.keras"

CLASS_NAMES_PATH = BASE_DIR / "ml" / "models" / "class_names.json"

IMAGE_SIZE = (224, 224)