import os
import boto3
from pathlib import Path
from typing import Optional

class ModelRegistry:
    def __init__(self, bucket_name: Optional[str] = None):
        self.bucket_name = bucket_name or os.environ.get("MODEL_REGISTRY_BUCKET")
        if not self.bucket_name:
            raise ValueError("MODEL_REGISTRY_BUCKET environment variable is not set")
        
        self.s3 = boto3.client("s3")

    def upload_model(self, local_path: str, model_type: str, version: str) -> str:
        """
        모델 파일을 S3에 업로드합니다.
        경로 형식: {model_type}/{version}/model.pt
        계획대로 진행: 'efficientnet-b0-cls/v5.0.0/model.pt'
        """
        # TODO: 기존 다운로드 스크립트와 매칭 필요 - 'models/' 
        file_path = Path(local_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found: {local_path}")

        # S3 키 생성
        # 예시: efficientnet-b0-cls/v5.0.0/model.pt
        s3_key = f"{model_type}/{version}/{file_path.name}"
        
        print(f"Uploading {local_path} to s3://{self.bucket_name}/{s3_key}")
        self.s3.upload_file(str(file_path), self.bucket_name, s3_key)
        
        return f"s3://{self.bucket_name}/{s3_key}"

    def download_model(self, model_type: str, version: str, filename: str, target_dir: str):
        """
        S3에서 모델 다운로드
        """
        s3_key = f"{model_type}/{version}/{filename}"
        target_path = Path(target_dir) / filename
        
        print(f"Downloading s3://{self.bucket_name}/{s3_key} to {target_path}")
        
        target_path.parent.mkdir(parents=True, exist_ok=True)
        self.s3.download_file(self.bucket_name, s3_key, str(target_path))
        
        return str(target_path)

    def list_versions(self, model_type: str):
        """
        해당 모델의 사용 가능한 버전들 반환
        """
        response = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=f"{model_type}/")
        if 'Contents' not in response:
            return []
            
        keys = [obj['Key'] for obj in response['Contents']]
        # 버전 추출 로직이 복잡할 수 있으므로, 일단 키 리스트를 반환
        return keys
