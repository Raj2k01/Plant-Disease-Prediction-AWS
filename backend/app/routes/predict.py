from fastapi import APIRouter, UploadFile, File

from app.services.predictor import predict


router = APIRouter()


@router.post("/predict")
async def prediction(image: UploadFile = File(...)):

    image_bytes = await image.read()

    content_type = image.content_type or "image/png"

    result = predict(
        image_bytes,
        content_type
    )

    return result