import io
import json
import os

import numpy as np
from PIL import Image


MODEL_DIR = "/opt/ml/model"

CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    CLASS_NAMES = json.load(f)


def input_handler(data, context):
    """
    Convert incoming image into TensorFlow Serving JSON.
    """

    content_type = context.request_content_type

    if content_type not in ["image/jpeg", "image/jpg", "image/png"]:
        raise ValueError(
            f"Unsupported content type: {content_type}"
        )

    # Read image bytes
    image_bytes = data.read()

    # Open image
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Resize
    image = image.resize((224, 224))

    # Convert to numpy
    image_array = np.array(image, dtype=np.float32)

    # Normalize
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # TensorFlow Serving request
    request = {
        "instances": image_array.tolist()
    }

    return json.dumps(request)


def output_handler(data, context):
    """
    Convert TensorFlow Serving response into application response.
    """

    if data.status_code != 200:
        raise ValueError(data.content.decode("utf-8"))

    response = json.loads(data.content.decode("utf-8"))

    predictions = response["predictions"]

    probabilities = np.array(predictions[0])

    class_index = int(np.argmax(probabilities))

    confidence = float(probabilities[class_index])

    class_name = CLASS_NAMES[class_index]

    result = {
        "class_index": class_index,
        "class_name": class_name,
        "confidence": confidence
    }

    return json.dumps(result), "application/json"