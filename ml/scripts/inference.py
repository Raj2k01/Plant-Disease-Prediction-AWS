import json
from io import BytesIO

import numpy as np
import tensorflow as tf

from PIL import Image

from config import (
    BEST_MODEL_PATH,
    CLASS_NAMES_PATH,
    IMAGE_SIZE
)

#Load Class Names
with open(CLASS_NAMES_PATH, "r") as f:
    CLASS_NAMES = json.load(f)

#Model Loader
def model_fn(model_dir=None):
   
    print("Loading model")

    model = tf.keras.models.load_model(BEST_MODEL_PATH)

    print("Model Loaded Successfully")

    return model

#Input Processing
def input_fn(request_body, content_type):

    if content_type != "application/x-image":

        raise ValueError(
            f"Unsupported Content-Type: {content_type}"
        )

    image = Image.open(
        BytesIO(request_body)
    ).convert("RGB")

    image = image.resize(IMAGE_SIZE)

    image = np.array(image)

    image = np.expand_dims(
        image,
        axis=0
    )

    return image


#Prediction
def predict_fn(input_data, model):

    predictions = model.predict(
        input_data,
        verbose=0
    )

    predicted_index = int(
        np.argmax(predictions)
    )

    confidence = float(
        np.max(predictions)
    )

    return {

        "class_index": predicted_index,

        "class_name": CLASS_NAMES[predicted_index],

        "confidence": confidence

    }

#Output Formatter
def output_fn(prediction, accept):

    if accept != "application/json":

        raise ValueError(
            f"Unsupported Accept Type: {accept}"
        )
    return json.dumps(prediction)