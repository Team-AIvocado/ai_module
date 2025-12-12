from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel
from inference_module.app.services.inference_service import run_v2
from inference_module.app.services.utils.image_loader import load_image_from_url, ensure_pil
from inference_module.app.services.fallback_service import process_fallback


router = APIRouter()


class InferenceURLRequest(BaseModel):
    image_url: str
    image_id: str


@router.post("/analyze")
async def analyze(image: UploadFile = File(...), image_id: str = Form(...)):
    contents = await image.read()
    await image.seek(0)
    result = await run_v2(image, image_id)
    return await process_fallback(result, image_bytes=contents)


@router.post("/analyze-url")
async def analyze_url(body: InferenceURLRequest):
    img = load_image_from_url(body.image_url)
    result = await run_v2(img, body.image_id)
    return await process_fallback(result, image_url=body.image_url)
