from fastapi import APIRouter, UploadFile, File

from app.services.predictor import predict
from app.utils.image_utils import preprocess_image


router = APIRouter()


@router.post("/predict")
async def prediction(image: UploadFile = File(...)):

    processed_image = preprocess_image(image.file)

    result = predict(processed_image)

    return result