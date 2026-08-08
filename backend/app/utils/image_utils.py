import numpy as np

from PIL import Image

from tensorflow.keras.applications.efficientnet import preprocess_input

from app.config import IMAGE_SIZE


def preprocess_image(file):

    image = Image.open(file).convert("RGB")

    image = image.resize(IMAGE_SIZE)

    image = np.array(image).astype(np.float32)

    image = np.expand_dims(image, axis=0)

    image = preprocess_input(image)

    return image