import json
import time

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

from preprocess import (
    load_dataset,
    prepare_dataset,
    data_augmentation,
)

from config import *

start_time = time.time()

#Load Dataset
print("Loading Dataset")

train_dataset, validation_dataset, class_names = load_dataset(DATASET_DIR)

train_dataset = prepare_dataset(train_dataset)
validation_dataset = prepare_dataset(validation_dataset)

NUM_CLASSES = len(class_names)

print(f"Classes : {NUM_CLASSES}")
print(class_names)

#Build Model
print("Building EfficientNetB0")

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(*IMAGE_SIZE, CHANNELS)
)

base_model.trainable = False

inputs = tf.keras.Input(shape=(*IMAGE_SIZE, CHANNELS))

x = data_augmentation(inputs)

x = preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(DROPOUT_RATE)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)

model = tf.keras.Model(inputs, outputs)

#Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=INITIAL_LEARNING_RATE
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

#Callbacks
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=BEST_MODEL_PATH,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)

callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),

    checkpoint,

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        verbose=1
    )
]

#Training
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=HEAD_EPOCHS,
    callbacks=callbacks
)

#Fine-Tuning
print("Fine-Tuning")

base_model.trainable = True

for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=FINETUNE_LEARNING_RATE
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_finetune = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    initial_epoch=HEAD_EPOCHS,
    epochs=HEAD_EPOCHS + FINETUNE_EPOCHS,
    callbacks=callbacks
)

#Save Final Model
best_model = tf.keras.models.load_model(BEST_MODEL_PATH)

best_model.save(FINAL_MODEL_PATH)

#Save Class Names
with open(CLASS_NAMES_PATH, "w") as f:
    json.dump(class_names, f, indent=4)

#Save Metadata
metadata = {
    "model": MODEL_NAME,
    "framework": "TensorFlow",
    "image_size": IMAGE_HEIGHT,
    "num_classes": NUM_CLASSES
}

with open(METADATA_PATH, "w") as f:
    json.dump(metadata, f, indent=4)

end_time = time.time()

print("Training Completed Successfully")
print(f"Training Time : {(end_time-start_time)/60:.2f} minutes")
print("Artifacts Saved Successfully")