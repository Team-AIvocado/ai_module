import os
import boto3
import pandas as pd
from PIL import Image
from io import BytesIO
from torch.utils.data import Dataset
from pathlib import Path

class KFoodDataset(Dataset):
    def __init__(self, csv_path: str, cache_dir: str = "data/images", transform=None):
        """
        Args:
            csv_path (str): 데이터셋 CSV 파일 경로 (dataset_*.csv).
            cache_dir (str): 다운로드한 이미지를 캐시할 로컬 디렉토리.
            transform (callable, optional): 샘플에 적용할 선택적 변환.
        """
        self.data_frame = pd.read_csv(csv_path)
        self.cache_dir = Path(cache_dir)
        self.transform = transform
        self.s3_client = boto3.client("s3")
        
        # 필터링: S3 URL이 있는 행만 사용
        self.data_frame = self.data_frame[self.data_frame['s3_url'].notna()]
        
        # Cache Directory 생성
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[KFoodDataset] {csv_path}에서 {len(self.data_frame)}개 샘플 로드 완료")

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if idx >= len(self.data_frame):
            raise IndexError
            
        row = self.data_frame.iloc[idx]
        s3_url = row['s3_url']
        # Ground Truth (Label) - 실제 학습시엔 Label Encoding 필요
        label_text = row['ground_truth'] 
        
        # 1. 로컬 캐시 확인
        # URL에서 Key 추출 -> 파일명으로 사용
        # https://.../meals/kfood_dataset/Food/uuid.jpg -> meals_kfood_dataset_Food_uuid.jpg (캐시용 평탄화 구조)
        # 또는 그냥 uuid.jpg 사용 (UUID는 유니크하므로)
        
        try:
            # S3 Key 파싱
            # 형식: https://{bucket}.s3.{region}.amazonaws.com/{key}
            parts = s3_url.split(".amazonaws.com/")
            if len(parts) < 2:
                # 잘못된 URL?
                raise ValueError(f"Invalid S3 URL: {s3_url}")
            
            full_key = parts[1] # meals/kfood_dataset/Kimchi/uuid.jpg
            bucket_name = parts[0].split("://")[1].split(".s3")[0]
            
            # 로컬 파일명: 마지막 부분(uuid.jpg)을 사용하여 단순하게 유지하거나,
            # 전체 경로 구조를 복제하여 확인할 수 있게 함.
            # UUID 유일성을 확인하거나 폴더 구조를 복제하는 것이 안전함.
            # "Food/Images" 확인을 위해 폴더 구조 복제 방식 사용.
            
            local_path = self.cache_dir / full_key
            
            if not local_path.exists():
                # S3에서 다운로드
                local_path.parent.mkdir(parents=True, exist_ok=True)
                # print(f"Downloading {full_key}...") # __getitem__에서는 너무 시끄러움
                self.s3_client.download_file(bucket_name, full_key, str(local_path))
            
            # 2. 이미지 로드
            image = Image.open(local_path).convert("RGB")
            
            if self.transform:
                image = self.transform(image)
                
            return image, label_text
            
        except Exception as e:
            print(f"인덱스 {idx} 로드 오류 ({s3_url}): {e}")
            # None을 반환하거나 에러를 처리? PyTorch DataLoader는 None을 싫어함.
            # 보통 특수한 에러 항목을 반환하거나 건너뜀.
            # 이 MVP에서는 루프가 멈추지 않도록 빈 텐서나 랜덤 텐서를 반환하지만,
            # 이상적으로는 손상된 이미지를 미리 필터링해야 함.
            # 기본 플레이스홀더 반환:
            if self.transform:
                # 검은 이미지 생성
                placeholder = Image.new('RGB', (224, 224), (0, 0, 0))
                return self.transform(placeholder), "ERROR"
            else:
                return Image.new('RGB', (224, 224), (0, 0, 0)), "ERROR"
