from inference_module.app.services.model_loader import detector, yolo_cls


def predict_v3(img, image_id):

    # Original
    label_o, conf_o, res_o = yolo_cls.predict(img)
    # idx_o = res_o.probs.top1    <-- Removed redundant lines
    # conf_o = float(...)         <-- Removed redundant lines
    # label_o = res_o.names[...]  <-- Removed overwriting

    # Detector → Crop
    crop, det_conf = detector.detect_and_crop(img)
    label_c, conf_c, res_c = yolo_cls.predict(crop)
    # idx_c, conf_c, label_c overwrite removed

    # Smart Selection
    if conf_c > conf_o:
        chosen = res_c
        final_label = label_c
        final_conf = conf_c
    else:
        chosen = res_o
        final_label = label_o
        final_conf = conf_o

    top5_idx = chosen.probs.top5[:3]
    top5_conf = chosen.probs.top5conf[:3]

    # Use translated names for candidates if available
    class_names = yolo_cls.class_names if yolo_cls.class_names else chosen.names
    labels = [class_names[i] for i in top5_idx]
    confs = [float(top5_conf[i]) for i in range(3)]

    return {
        "image_id": image_id,
        "food_name": final_label,
        "candidates": [{"label": labels[i], "confidence": confs[i]} for i in range(3)],
    }
