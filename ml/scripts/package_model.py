import tarfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SAVED_MODEL_DIR = BASE_DIR / "saved_model"

OUTPUT_FILE = BASE_DIR / "model.tar.gz"

print("Packaging TensorFlow SavedModel")

with tarfile.open(OUTPUT_FILE, "w:gz") as tar:

    for item in SAVED_MODEL_DIR.iterdir():

        tar.add(
            item,
            arcname=item.name
        )

print("Packaging Complete!")

print(f"Output : {OUTPUT_FILE}")