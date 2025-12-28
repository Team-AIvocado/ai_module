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

    # Magic Number Protocol Logic moved to fallback_service.py
    # We only ensure clean labels and deduplication here.
    
    final_candidates = []
    seen_labels = set()

    for i in range(len(labels)):
        # 1. Clean Label (Remove dots and spaces for deduplication)
        raw_label = str(labels[i])
        cleaned_label = raw_label.replace(".", "").strip()
        
        conf = float(top3_conf[i])
        
        # 2. Deduplication
        if cleaned_label in seen_labels:
            continue
        seen_labels.add(cleaned_label)
        
        final_candidates.append({"label": cleaned_label, "confidence": conf})

    return {
        "image_id": image_id,
        "food_name": final_candidates[0]["label"] if final_candidates else "Unknown",
        "candidates": final_candidates,
    }
