import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from pathlib import Path
import json

from preprocess import (
    load_dataset,
    prepare_dataset,
    data_augmentation,
    normalization,
)

DATASET_PATH = r"D:\MLPortfolioProjects\PlantDiseasePredictionDL\ml\data\tomato"

MODEL_DIR = Path("../models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (224, 224)

print("Loading dataset")

train_dataset, validation_dataset, class_names = load_dataset(DATASET_PATH)

train_dataset = prepare_dataset(train_dataset)
validation_dataset = prepare_dataset(validation_dataset)

NUM_CLASSES = len(class_names)

print(f"Classes : {NUM_CLASSES}")

print("Building model...")

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224,224,3)
)

base_model.trainable = False


inputs = tf.keras.Input(shape=(224,224,3))

x = data_augmentation(inputs)

x = normalization(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)

model = Model(inputs, outputs)


model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ModelCheckpoint(
        filepath=MODEL_DIR / "best_model.keras",
        monitor="val_accuracy",
        save_best_only=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        verbose=1
    )

]
#(due to No gpu using less epoch for testing)
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=15,
    callbacks=callbacks
)

model.save(MODEL_DIR / "final_model.keras")

print("Training Complete")
print("Model Saved Successfully")

#saving class names
with open(MODEL_DIR / "class_names.json", "w") as f:
    json.dump(class_names, f, indent=4)

#save metadata
metadata = {
    "model": "EfficientNetB0",
    "framework": "TensorFlow",
    "image_size": 224,
    "num_classes": NUM_CLASSES
}

with open(MODEL_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)