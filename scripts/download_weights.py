import os
import boto3
from pathlib import Path

def download_weights():
    bucket_name = os.environ.get("MODEL_REGISTRY_BUCKET")
    if not bucket_name:
        print("Warning: MODEL_REGISTRY_BUCKET not set. Skipping download.")
        return

    s3 = boto3.client("s3")
    local_dir = Path("ai_module/inference_module/weights") # 컨테이너 내부 절대 경로 기준 확인 필요
    
    # Docker WORKDIR /app 기준 -> ai_module 폴더가 어떻게 복사되는지 중요
    # 현재 구조: Dockerfile COPY . . -> /app/main.py, /app/ai_module/... (X)
    # 현재 Dockerfile: COPY . . -> /app/inference_module, /app/main.py 등
    
    # config.py: PROJECT_ROOT = Path(__file__).resolve().parent.parent (inference_module)
    # config.py: WEIGHTS_DIR = PROJECT_ROOT / "weights" (inference_module/weights)

    # Dockerfile을 보면 ai_module 내부 파일을 /app 으로 복사함.
    # 즉 /app/inference_module/weights 가 되어야 함.
    local_dir = Path("inference_module/weights")

    print(f"Downloading weights from s3://{bucket_name} to {local_dir}")
    local_dir.mkdir(parents=True, exist_ok=True)

    # List objects
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name)

    for page in pages:
        if "Contents" not in page:
            continue
        for obj in page["Contents"]:
            key = obj["Key"]
            # S3 Key: v1/weights.pt  -> Local: inference_module/weights/v1/weights.pt
            # 만약 S3가 폴더 구조를 그대로 가지고 있다면 그대로 다운로드
            
            # Skip folders
            if key.endswith("/"):
                continue

            local_file_path = local_dir / key
            local_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            print(f"Downloading {key}...")
            s3.download_file(bucket_name, key, str(local_file_path))

    print("Download complete.")

if __name__ == "__main__":
    download_weights()
