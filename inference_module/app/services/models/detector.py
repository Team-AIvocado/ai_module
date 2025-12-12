from PIL import Image
from ultralytics import YOLO


class Detector:
    def __init__(self, weight_path, device="cpu"):
        self.device = device
        self.model = YOLO(weight_path)
        self.model.to(device)

    def detect_and_crop(self, img: Image.Image, conf_threshold=0.15):
        res = self.model(img, conf=conf_threshold, verbose=False)[0]

        if len(res.boxes) == 0:
            # 탐지 실패시 원본 이미지와 conf=0 반환
            return img, 0.0

        # conf가 가장 높은 박스 선택
        best_box = res.boxes[0]

        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
        det_conf = float(best_box.conf)

        # padding 동일하게 반영
        w, h = img.size
        pad = int((x2 - x1) * 0.1)
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)

        crop = img.crop((x1, y1, x2, y2))

        return crop, det_conf
