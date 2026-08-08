import json

import numpy as np
import tensorflow as tf

from app.config import (
    MODEL_PATH,
    CLASS_NAMES_PATH
)


print("Loading Model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model Loaded")


with open(CLASS_NAMES_PATH) as f:

    CLASS_NAMES = json.load(f)


def predict(image):

    prediction = model.predict(
        image,
        verbose=0
    )[0]

    predicted_index = int(np.argmax(prediction))

    confidence = float(prediction[predicted_index])

    return {

        "prediction": CLASS_NAMES[predicted_index],

        "confidence": round(confidence, 4)
    }