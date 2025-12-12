from inference_module.app.services.models.yolo_cls import YOLOClassifier
from inference_module.app.services.models.effnet_cls import EffnetClassifier
from inference_module.app.services.models.detector import Detector
from inference_module.configs import config
import torch

# 디바이스 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# weight 경로
DETECTOR_PATH = (
    config.WEIGHTS_DIR / "detector/kfood_integrated_detector/weights/best.pt"
)
EFFNET_PATH = config.WEIGHTS_DIR / "classifier_effnet_finetuned/best_effnet_b0.pt"
YOLO_CLS_PATH = config.WEIGHTS_DIR / "classifier_yolo_finetuned/weights/best.pt"
CLASS_fILE = config.WEIGHTS_DIR / "classes.txt"

# 싱글톤 인스턴스
# cls 모델
yolo_cls = YOLOClassifier(weight_path=YOLO_CLS_PATH, device=DEVICE)
effnet_cls = EffnetClassifier(
    weight_path=EFFNET_PATH, device=DEVICE, class_file=CLASS_fILE
)

# detector
detector = Detector(weight_path=DETECTOR_PATH, device=DEVICE)
