from inference_module.app.services.model_loader import yolo_cls
from inference_module.app.services.utils.image_loader import ensure_pil


def predict_v1(img, image_id):

    img = ensure_pil(img)

    label, conf, res = yolo_cls.predict(img)

    top5_idx = res.probs.top5
    top5_conf = res.probs.top5conf

    top3_idx = top5_idx[:3]
    top3_conf = [float(top5_conf[i]) for i in range(3)]

    class_names = yolo_cls.class_names if yolo_cls.class_names else res.names
    print(f"DEBUG v1: class_names len={len(class_names) if class_names else 0}")

    top3_labels = []
    for i in top3_idx:
        idx_val = int(i)  # Tensor -> int
        if class_names and isinstance(class_names, list) and idx_val < len(class_names):
            top3_labels.append(class_names[idx_val])
        elif class_names and isinstance(class_names, dict) and idx_val in class_names:
            top3_labels.append(class_names[idx_val])
        else:
            print(f"DEBUG v1: Fallback to res.names for idx {idx_val}")
            top3_labels.append(res.names[idx_val])

    return {
        "image_id": image_id,
        "food_name": top3_labels[0],
        "candidates": [
            {"label": top3_labels[0], "confidence": top3_conf[0]},
            {"label": top3_labels[1], "confidence": top3_conf[1]},
            {"label": top3_labels[1], "confidence": top3_conf[2]},
        ],
    }
