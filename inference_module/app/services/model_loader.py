from inference_module.app.services.models.yolo_cls import YOLOClassifier
from inference_module.app.services.models.effnet_cls import EffnetClassifier
from inference_module.app.services.models.detector import Detector
from inference_module.configs import config
import torch
import json
import logging
from pathlib import Path

# 로거 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 디바이스 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_active_model_config():
    """active_model.json 파일에서 활성 모델 설정을 로드합니다."""
    # config.py가 있는 폴더(configs)와 같은 위치에 active_model.json이 있음
    config_path = Path(config.__file__).resolve().parent / "active_model.json"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            model_config = json.load(f)
            logger.info(f"모델 설정을 성공적으로 로드했습니다: {config_path}")
            return model_config
    except FileNotFoundError:
        logger.error(f"설정 파일을 찾을 수 없습니다: {config_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"설정 파일 형식이 올바르지 않습니다: {config_path}")
        raise


# 모델 설정 로드
active_config = load_active_model_config()


# 경로 설정 함수
def get_model_path(model_key, config_data):
    """설정 데이터에서 모델 가중치 경로를 반환합니다."""
    rel_path = config_data[model_key]["weights_path"]
    return config.WEIGHTS_DIR / rel_path


def get_class_file(model_key, config_data):
    """설정 데이터에서 클래스 파일 경로를 반환합니다."""
    rel_path = config_data[model_key].get("class_file")
    if rel_path:
        return config.WEIGHTS_DIR / rel_path
    return None


# Weight 경로 및 클래스 파일 경로 로드
DETECTOR_PATH = get_model_path("detector", active_config)
EFFNET_PATH = get_model_path("efficientnet-b0-cls", active_config)
YOLO_CLS_PATH = get_model_path("yolo-cls", active_config)

# 각 모델별 클래스 파일 (설정에 있는 경우)
EFFNET_CLASS_FILE = get_class_file("efficientnet-b0-cls", active_config)
YOLO_CLASS_FILE = get_class_file("yolo-cls", active_config)

# 싱글톤 인스턴스 생성
# Classifiers
yolo_cls = YOLOClassifier(
    weight_path=YOLO_CLS_PATH, device=DEVICE, class_file=YOLO_CLASS_FILE
)
effnet_cls = EffnetClassifier(
    weight_path=EFFNET_PATH, device=DEVICE, class_file=EFFNET_CLASS_FILE
)

# Detector
detector = Detector(weight_path=DETECTOR_PATH, device=DEVICE)

logger.info("모든 AI 모델이 성공적으로 초기화되었습니다.")
