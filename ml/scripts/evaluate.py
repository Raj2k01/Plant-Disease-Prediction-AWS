import json

import matplotlib.pyplot as plt
import numpy as np
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

from config import *

#Load Dataset
print("Evaluation Started")

print(f"Dataset : {DATASET_DIR}")
print(f"Model   : {BEST_MODEL_PATH}")
print(f"Outputs : {OUTPUTS_DIR}")

train_dataset, validation_dataset, class_names = load_dataset(DATASET_DIR)

validation_dataset = prepare_dataset(validation_dataset)

#Load Model
print("\nLoading model")

model = tf.keras.models.load_model(BEST_MODEL_PATH)

print("Model Loaded Successfully")

#Prediction
print("Generating predictions")

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
    target_names=class_names,
    zero_division=0
)

print(report)

with open(CLASSIFICATION_REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(report)

#Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12,10))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(CONFUSION_MATRIX_PATH, dpi=300)

plt.close()

print("Confusion Matrix Saved")

#Sample Predictions
plt.figure(figsize=(12,12))

for images, labels in validation_dataset.take(1):

    predictions = model.predict(images, verbose=0)

    predicted = np.argmax(predictions, axis=1)

    for i in range(min(9, len(images))):

        plt.subplot(3,3,i+1)

        plt.imshow(images[i].numpy().astype("uint8"))

        plt.title(
            f"True : {class_names[labels[i]]}\nPred : {class_names[predicted[i]]}",
            fontsize=8
        )

        plt.axis("off")

plt.tight_layout()

plt.savefig(SAMPLE_PREDICTIONS_PATH, dpi=300)

plt.close()

print("Sample Predictions Saved")

#Save Metrics
accuracy = np.mean(np.array(y_true) == np.array(y_pred))

metrics = {

    "model": MODEL_NAME,
    "dataset": "PlantVillage Tomato",
    "framework": "TensorFlow",
    "accuracy": float(accuracy),
    "num_classes": len(class_names)
}

with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=4)

print("Evaluation Completed Successfully")
print(f"Validation Accuracy : {accuracy:.4f}")