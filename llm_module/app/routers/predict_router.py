from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
from llm_module.app.services.multimodal_mock import mock_food_classifier
import base64

router = APIRouter()


class ImageInput(BaseModel):
    image_base64: str


@router.post("/predict/image")
async def predict_file(file: UploadFile = File(...)):
    """
    이미지 파일을 받아 WatsonVision에 전달
    """
    contents = await file.read()
    result = mock_food_classifier(image_bytes=contents)
    return {"food_name": result}


class ImageURL(BaseModel):
    url: str


@router.post("/predict/url")
async def predict_from_url(data: ImageURL):
    """
    이미지 URL을 받아 WatsonVision에 전달
    """
    result = mock_food_classifier(image_url=data.url)
    return result


@router.post("/predict")
def predict_food(input_data: ImageInput):
    """
    이미지 base64문자열을 받아 WatsonVision에 전달
    """
    food = mock_food_classifier(image_base64=input_data.image_base64)
    return {"food name": food}
