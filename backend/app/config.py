import os

SAGEMAKER_ENDPOINT_NAME = os.getenv(
    "SAGEMAKER_ENDPOINT_NAME",
    "plant-disease-prediction-endpoint-v8"
)

AWS_REGION = os.getenv(
    "AWS_REGION",
    "us-east-1"
)