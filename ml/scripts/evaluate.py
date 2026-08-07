import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from preprocess import (
    load_dataset,
    prepare_dataset
)

DATASET_PATH = r"D:\MLPortfolioProjects\PlantDiseasePredictionDL\ml\data\tomato"

MODEL_DIR = Path("../models")

OUTPUT_DIR = Path("../outputs")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

#Load Dataset
print("Loading validation dataset")
train_dataset, validation_dataset, class_names = load_dataset(DATASET_PATH)
validation_dataset = prepare_dataset(validation_dataset)

#Load Model
print("Loading trained model...")

model = tf.keras.models.load_model(
    MODEL_DIR / "best_model.keras"
)

print("Model Loaded Successfully")

#Predict

print("Generating predictions...")

y_true = []

y_pred = []

for images, labels in validation_dataset:

    predictions = model.predict(images, verbose=0)

    predicted_labels = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())

    y_pred.extend(predicted_labels)

#Classification Report

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names
)

print(report)

with open(
    OUTPUT_DIR / "classification_report.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(report)

#Confusion Matrix

cm = confusion_matrix(
    y_true,
    y_pred
)

#Plot Confusion Matrix
plt.figure(figsize=(18,18))

sns.heatmap(
    cm,
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "confusion_matrix.png",
    dpi=300
)

plt.show()

#Sample Predictions

plt.figure(figsize=(12,12))

for images, labels in validation_dataset.take(1):

    predictions = model.predict(images, verbose=0)

    predicted = np.argmax(predictions, axis=1)

    for i in range(9):

        ax = plt.subplot(3,3,i+1)

        plt.imshow(images[i].numpy().astype("uint8"))

        plt.title(
            f"True: {class_names[labels[i]]}\nPred: {class_names[predicted[i]]}",
            fontsize=8
        )

        plt.axis("off")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "sample_predictions.png",
    dpi=300
)

plt.show()

#Save Metrics

accuracy = np.mean(np.array(y_true) == np.array(y_pred))

metrics = {
    "accuracy": float(accuracy)
}

with open(
    OUTPUT_DIR / "metrics.json",
    "w"
) as f:

    json.dump(metrics, f, indent=4)

print("Evaluation Completed Successfully")

print(f"Accuracy : {accuracy:.4f}")