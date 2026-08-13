import json
import os

import boto3


REGION = os.getenv("AWS_REGION", "us-east-1")

ENDPOINT_NAME = os.getenv(
    "SAGEMAKER_ENDPOINT_NAME",
    "plant-diesase-prediction-endpoint-v8" 
)

runtime = boto3.client(
    "sagemaker-runtime",
    region_name=REGION
)


def predict(image_bytes, content_type="image/png"):

    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        Body=image_bytes,
        ContentType=content_type,
        Accept="application/json"
    )

    result = json.loads(
        response["Body"].read().decode("utf-8")
    )

    return result