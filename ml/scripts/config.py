from pathlib import Path

#Project Paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
DATASET_DIR = DATA_DIR / "tomato"

MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

#Image Configuration
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)

CHANNELS = 3
BATCH_SIZE = 32

#Dataset Configuration
VALIDATION_SPLIT = 0.20
SEED = 42

#Training Configuration
HEAD_EPOCHS = 15
FINETUNE_EPOCHS = 5

INITIAL_LEARNING_RATE = 1e-3
FINETUNE_LEARNING_RATE = 1e-5

DROPOUT_RATE = 0.30

#Model Configuration
MODEL_NAME = "EfficientNetB0"

BEST_MODEL_PATH = MODELS_DIR / "best_model.keras"
FINAL_MODEL_PATH = MODELS_DIR / "final_model.keras"

CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"
METADATA_PATH = MODELS_DIR / "metadata.json"

#Evaluation Outputs
CLASSIFICATION_REPORT_PATH = OUTPUTS_DIR / "classification_report.txt"

CONFUSION_MATRIX_PATH = OUTPUTS_DIR / "confusion_matrix.png"

SAMPLE_PREDICTIONS_PATH = OUTPUTS_DIR / "sample_predictions.png"

METRICS_PATH = OUTPUTS_DIR / "metrics.json"