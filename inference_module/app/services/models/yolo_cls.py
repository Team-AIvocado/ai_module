from PIL import Image
from ultralytics import YOLO
import numpy as np


class YOLOClassifier:
    def __init__(self, weight_path, device="cpu"):
        self.device = device
        self.model = YOLO(weight_path)
        self.model.to(device)

    def predict(self, img: Image.Image):
        img_np = np.array(img)
        res = self.model(img_np, verbose=False)[0]

        label = res.names[res.probs.top1]
        conf = float(res.probs.top1conf)

        return label, conf, res
