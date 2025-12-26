from inference_module.app.services.models.yolo_cls import YOLOClassifier
from inference_module.app.services.models.effnet_cls import EffnetClassifier
from inference_module.app.services.models.detector import Detector
from inference_module.configs import config
from training.registry import ModelRegistry
import torch
import json
import logging
import os
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


def ensure_file_download(config_data, model_key, local_path, is_class_file=False):
    """
    로컬 파일이 없으면 S3에서 다운로드를 시도합니다.
    """
    target_path = Path(local_path)
    if target_path.exists():
        return target_path

    # 파일이 없고 S3 버킷 설정이 확인되면 다운로드 시도
    bucket_name = os.environ.get("MODEL_REGISTRY_BUCKET")
    if not bucket_name:
        logger.warning(
            f"모델 파일이 없음: {target_path}. MODEL_REGISTRY_BUCKET이 설정되지 않아 다운로드를 건너뜁니다."
        )
        return target_path  # 다운로드 불가, 에러는 나중에 발생할 것임

    try:
        logger.info(f"S3에서 다운로드 시도 중: {model_key} -> {target_path}")
        registry = ModelRegistry(bucket_name=bucket_name)

        version = config_data[model_key]["version"]

        # S3 키 구성: {model_key}/{version}/{filename}
        # 로컬 경로의 파일명을 그대로 사용한다고 가정
        filename = target_path.name

        # 다운로드 실행
        downloaded_path = registry.download_model(
            model_type=model_key,
            version=version,
            filename=filename,
            target_dir=str(target_path.parent),
        )
        logger.info(f"다운로드 완료: {downloaded_path}")
        return Path(downloaded_path)

    except Exception as e:
        logger.error(f"S3 다운로드 실패: {e}")
        return target_path


# 경로 설정 함수
def get_model_path(model_key, config_data):
    """설정 데이터에서 모델 가중치 경로를 반환하며, 필요시 다운로드합니다."""
    rel_path = config_data[model_key]["weights_path"]
    abs_path = config.WEIGHTS_DIR / rel_path

    return ensure_file_download(config_data, model_key, abs_path)


def get_class_file(model_key, config_data):
    """설정 데이터에서 클래스 파일 경로를 반환하며, 필요시 다운로드합니다."""
    rel_path = config_data[model_key].get("class_file")
    if rel_path:
        abs_path = config.WEIGHTS_DIR / rel_path
        return ensure_file_download(
            config_data, model_key, abs_path, is_class_file=True
        )
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
