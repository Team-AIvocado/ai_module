from inference_module.app.services.model_loader import yolo_cls
from inference_module.app.services.utils.image_loader import ensure_pil


def predict_v1(img, image_id):

    img = ensure_pil(img)

    label, conf, res = yolo_cls.predict(img)

    top5_idx = res.probs.top5
    top5_conf = res.probs.top5conf

    top3_idx = top5_idx[:3]
    top3_conf = [float(top5_conf[i]) for i in range(3)]
    top3_labels = [res.names[i] for i in top3_idx]

    return {
        "image_id": image_id,
        "food_name": top3_labels[0],
        "candidates": [
            {"label": top3_labels[0], "confidence": top3_conf[0]},
            {"label": top3_labels[1], "confidence": top3_conf[1]},
            {"label": top3_labels[1], "confidence": top3_conf[2]},
        ],
    }
