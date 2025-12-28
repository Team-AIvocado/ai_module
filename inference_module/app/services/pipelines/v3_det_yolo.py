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
    classes = yolo_cls.class_names if yolo_cls.class_names else chosen.names
    
    # Cleaning & Deduplication (Same logic as V4)
    final_candidates = []
    seen_labels = set()

    # top5까지 순회하면서 중복 제거 후 3개 채우기
    for i in range(len(top5_idx)):
        # 1. Clean Label
        raw_label = str(classes[top5_idx[i]])
        cleaned_label = raw_label.replace(".", "").strip()
        conf = float(top5_conf[i])

        # 2. Deduplication
        if cleaned_label in seen_labels:
            continue
        seen_labels.add(cleaned_label)

        final_candidates.append({"label": cleaned_label, "confidence": conf})
        
        if len(final_candidates) >= 3:
            break

    return {
        "image_id": image_id,
        "food_name": final_candidates[0]["label"] if final_candidates else "Unknown",
        "candidates": final_candidates,
    }
