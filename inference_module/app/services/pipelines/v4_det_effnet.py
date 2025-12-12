from inference_module.app.services.model_loader import effnet_cls, detector
import torch.nn.functional as F


def predict_v4(img, image_id):

    # original
    x = effnet_cls.transform(img).unsqueeze(0).to(effnet_cls.device)
    logits_o = effnet_cls.model(x)
    probs_o = F.softmax(logits_o, dim=1)
    conf_o, idx_o = probs_o.max(dim=1)
    label_o = effnet_cls.class_names[idx_o.item()]

    # crop
    crop, det_conf = detector.detect_and_crop(img)
    x_c = effnet_cls.transform(crop).unsqueeze(0).to(effnet_cls.device)
    logits_c = effnet_cls.model(x_c)
    probs_c = F.softmax(logits_c, dim=1)
    conf_c, idx_c = probs_c.max(dim=1)
    label_c = effnet_cls.class_names[idx_c.item()]

    # smart selection
    if conf_c.item() > conf_o.item():
        chosen_probs = probs_c
    else:
        chosen_probs = probs_o

    # top3
    top3_conf, top3_idx = chosen_probs.topk(3, dim=1)
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
