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

    def upload_artifact(
        self,
        local_path: str,
        artifact_type: str,
        version: str,
        filename: Optional[str] = None,
    ) -> str:
        """
        일반 아티팩트(파일)를 S3에 업로드합니다.
        경로 형식: {artifact_type}/{version}/{filename}
        """
        file_path = Path(local_path)
        if not file_path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {local_path}")

        target_filename = filename or file_path.name
        s3_key = f"{artifact_type}/{version}/{target_filename}"

        print(
            f"[ModelRegistry] 업로드 시작: {local_path} -> s3://{self.bucket_name}/{s3_key}"
        )
        self.s3.upload_file(str(file_path), self.bucket_name, s3_key)

        return f"s3://{self.bucket_name}/{s3_key}"

    def upload_model(self, local_path: str, model_type: str, version: str) -> str:
        """
        모델 파일을 S3에 업로드합니다. (upload_artifact 래퍼)
        """
        return self.upload_artifact(local_path, model_type, version)

    def download_model(
        self, model_type: str, version: str, filename: str, target_dir: str
    ):
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
        response = self.s3.list_objects_v2(
            Bucket=self.bucket_name, Prefix=f"{model_type}/"
        )
        if "Contents" not in response:
            return []

        keys = [obj["Key"] for obj in response["Contents"]]
        # 버전 추출 로직이 복잡할 수 있으므로, 일단 키 리스트를 반환
        return keys
