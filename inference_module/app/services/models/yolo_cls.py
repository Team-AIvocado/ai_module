from PIL import Image
from ultralytics import YOLO
import numpy as np


class YOLOClassifier:
    def __init__(self, weight_path, device="cpu", class_file=None):
        self.device = device
        self.model = YOLO(weight_path)
        self.model.to(device)

        self.class_names = None
        if class_file:
            print(f"Loading class names from {class_file}...")
            try:
                with open(class_file, "r", encoding="utf-8") as f:
                    self.class_names = [
                        line.strip() for line in f.readlines() if line.strip()
                    ]
                print(f"Loaded {len(self.class_names)} class names.")
            except Exception as e:
                print(f"Failed to load class file: {e}")
                self.class_names = None

    def predict(self, img: Image.Image):
        img_np = np.array(img)
        res = self.model(img_np, verbose=False)[0]

        idx = int(res.probs.top1)
        conf = float(res.probs.top1conf)

        if self.class_names:
            try:
                label = self.class_names[idx]
            except IndexError:
                print(
                    f"IndexError: idx {idx} out of range for class_names (len {len(self.class_names)})"
                )
                label = res.names[idx]
        else:
            label = res.names[idx]

        return label, conf, res
