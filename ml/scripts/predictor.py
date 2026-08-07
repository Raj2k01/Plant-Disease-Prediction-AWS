import argparse
import json

from inference import (
    model_fn,
    input_fn,
    predict_fn
)

#Load Model
print("Loading model")

model = model_fn()

print("Model Loaded Successfully")

#Command Line Arguments
parser = argparse.ArgumentParser()

parser.add_argument(
    "image_path",
    help="Path to input image"
)

args = parser.parse_args()

#Read Image
with open(args.image_path, "rb") as f:

    image_bytes = f.read()

#Prediction
image = input_fn(
    image_bytes,
    "application/x-image"
)

prediction = predict_fn(
    image,
    model
)

#Display Result
print("\nPrediction Result")
print(json.dumps(
    prediction,
    indent=4
))