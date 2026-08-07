from pathlib import Path
import tensorflow as tf

from config import FINAL_MODEL_PATH

BASE_DIR = Path(__file__).resolve().parent.parent

EXPORT_DIR = BASE_DIR / "saved_model"

print("Loading model...")

model = tf.keras.models.load_model(FINAL_MODEL_PATH)

print("Exporting TensorFlow SavedModel...")

model.export(str(EXPORT_DIR))

print("SavedModel exported successfully.")

print(f"Location : {EXPORT_DIR}")