import json
from io import BytesIO

import numpy as np
import tensorflow as tf
from PIL import Image


IMAGE_SIZE = (224, 224)

MODEL_PATH = "/opt/ml/model"

CLASS_NAMES_PATH = "/opt/ml/model/class_names.json"


# Load class names
with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    CLASS_NAMES = json.load(f)


# Load SavedModel
def model_fn(model_dir):

    print("Loading TensorFlow SavedModel...")

    model = tf.saved_model.load(model_dir)

    print("SavedModel loaded successfully")

    return model


# Process input image
def input_fn(request_body, content_type):

    if content_type != "application/x-image":
        raise ValueError(
            f"Unsupported Content-Type: {content_type}"
        )

    image = Image.open(
        BytesIO(request_body)
    ).convert("RGB")

    image = image.resize(IMAGE_SIZE)

    image = np.array(image).astype(np.float32)

    image = np.expand_dims(
        image,
        axis=0
    )

    return image


# Prediction
def predict_fn(input_data, model):

    # Get serving signature
    infer = model.signatures["serving_default"]

    predictions = infer(
        tf.constant(input_data)
    )

    # Get output tensor
    output = list(predictions.values())[0]

    probabilities = output.numpy()[0]

    predicted_index = int(
        np.argmax(probabilities)
    )

    confidence = float(
        np.max(probabilities)
    )

    return {
        "class_index": predicted_index,
        "class_name": CLASS_NAMES[predicted_index],
        "confidence": confidence
    }


# Format output
def output_fn(prediction, accept):

    if accept != "application/json":
        raise ValueError(
            f"Unsupported Accept type: {accept}"
        )

    return json.dumps(prediction)