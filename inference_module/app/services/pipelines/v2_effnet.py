from inference_module.app.services.model_loader import effnet_cls
import torch.nn.functional as F


def predict_v2(img, image_id):
    logits = effnet_cls.model(
        effnet_cls.transform(img).unsqueeze(0).to(effnet_cls.device)
    )
    probs = F.softmax(logits, dim=1)

    top3_conf, top3_idx = probs.topk(3, dim=1)
    top3_idx = top3_idx[0].tolist()
    top3_conf = top3_conf[0].tolist()

    labels = [effnet_cls.class_names[i] for i in top3_idx]

    return {
        "image_id": image_id,
        "food_name": labels[0],
        "candidates": [
            {"label": labels[i], "confidence": float(top3_conf[i])}
            for i in range(len(labels))
        ],
    }
