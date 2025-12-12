from inference_module.app.services.utils.image_loader import ensure_pil
from .pipelines.v1_yolo import predict_v1
from .pipelines.v2_effnet import predict_v2
from .pipelines.v3_det_yolo import predict_v3
from .pipelines.v4_det_effnet import predict_v4


async def run_v1(image, image_id):
    img = ensure_pil(image)
    return predict_v1(img, image_id)


async def run_v2(image, image_id):
    img = ensure_pil(image)
    return predict_v2(img, image_id)


async def run_v3(image, image_id):
    img = ensure_pil(image)
    return predict_v3(img, image_id)


async def run_v4(image, image_id):
    img = ensure_pil(image)
    return predict_v4(img, image_id)
