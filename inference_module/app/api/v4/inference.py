from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel
from inference_module.app.services.inference_service import run_v4
from inference_module.app.services.utils.image_loader import load_image_from_url, ensure_pil
from inference_module.app.services.fallback_service import process_fallback


router = APIRouter()


class InferenceURLRequest(BaseModel):
    image_url: str
    image_id: str


@router.post("/analyze")
async def analyze(image: UploadFile = File(...), image_id: str = Form(...)):
    # Read bytes for potential fallback usage
    contents = await image.read()
    
    # Reset cursor for inference service which expects the file at start
    await image.seek(0)
    
    # Run inference (run_v4 handles ensure_pil internally)
    result = await run_v4(image, image_id)
    
    return await process_fallback(result, image_bytes=contents)


@router.post("/anaylze-url")
async def analyze_url(body: InferenceURLRequest):
    img = load_image_from_url(body.image_url)
    result = await run_v4(img, body.image_id)
    return await process_fallback(result, image_url=body.image_url)
