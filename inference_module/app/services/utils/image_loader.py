import io
from PIL import Image
import requests
from fastapi import UploadFile
from starlette.datastructures import UploadFile as StarletteUploadFile


def ensure_pil(image):
    # UploadFile
    if isinstance(image, UploadFile) or isinstance(image, StarletteUploadFile):
        image.file.seek(0)
        img_bytes = image.file.read()
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # PIL.Image
    if isinstance(image, Image.Image):
        return image

    raise ValueError(f"Unsupported image type: {type(image)}")


def load_image_from_uploadfile(upload_file):
    file_bytes = upload_file.file.read()
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")


def load_image_from_url(url: str) -> Image.Image:
    # S3 등 외부 이미지 URL -> PIL.Image 변환
    response = requests.get(url)
    response.raise_for_status()

    img_bytes = io.BytesIO(response.content)
    return Image.open(img_bytes).convert("RGB")
