import os
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
from datetime import datetime

# Local imports
from .data_loader import KFoodDataset
from .registry import ModelRegistry

def train_pipeline(
    csv_path: str,
    class_file: str,
    epochs: int = 1,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    upload_s3: bool = True
):
    print(f"--- MLOps 학습 파이프라인 시작 ---")
    
    # 1. 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"장치: {device}")
    
    # 2. 클래스 정의 로드 (Static Source of Truth)
    print(f"[1/5] 클래스 불러오는 중 ({class_file})...")
    with open(class_file, 'r', encoding='utf-8') as f:
        # 빈 줄 필터링
        class_names = [line.strip() for line in f.readlines() if line.strip()]
    
    num_classes = len(class_names)
    print(f"{num_classes}개 클래스 로드 완료 (Static).")
    
    # 3. 데이터 준비
    print("[2/5] 데이터 준비 중...")
    
    # 간단한 변환 (Transform)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = KFoodDataset(csv_path=csv_path, transform=transform)
    
    # Train/Validation Split (8:2)
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"[2/5] 데이터 준비 완료: 학습 {train_size}건, 검증 {val_size}건")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 데이터셋의 라벨 vs 정적 클래스 파일 검증 (전체 데이터셋 기준)
    dataset_labels = set(dataset.data_frame['ground_truth'].unique())
    static_labels = set(class_names)
    
    # 새로운 클래스 감지 (MLOps 알림)
    new_classes = dataset_labels - static_labels
    if new_classes:
        print(f"   경고: 데이터셋에서 클래스 파일에 없는 {len(new_classes)}개의 새로운 클래스 발견!")
        print(f"   예시: {list(new_classes)[:5]}")
        print("   이 샘플들은 학습에서 제외되거나 에러를 유발할 수 있습니다.")
    
    # 4. 모델 초기화 (EfficientNet-B0)
    print(f"[3/5] 모델 초기화 중 (EfficientNet-B0), 클래스 수: {num_classes}...")
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=num_classes)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 5. 학습 루프
    print(f"[4/5] {epochs} 에폭 학습 시작...")
    
    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        max_batches = 5 # 데모용 제한 (전체 학습 시 제거 필요)
        
        for i, (inputs, labels_text) in enumerate(train_loader):
            if i >= max_batches:
                break
                
            inputs = inputs.to(device)
            
            # 견고한 라벨 인코딩
            targets = []
            valid_indices = []
            for idx, text in enumerate(labels_text):
                if text in class_names:
                    targets.append(class_names.index(text))
                    valid_indices.append(idx)
            
            if not targets:
                continue
                
            targets = torch.tensor(targets, dtype=torch.long).to(device)
            inputs = inputs[valid_indices]
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_train_loss = running_loss / min(len(train_loader), max_batches)
        
        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i, (inputs, labels_text) in enumerate(val_loader):
                if i >= max_batches: # 데모용 제한
                    break
                    
                inputs = inputs.to(device)
                
                targets = []
                valid_indices = []
                for idx, text in enumerate(labels_text):
                    if text in class_names:
                        targets.append(class_names.index(text))
                        valid_indices.append(idx)
                
                if not targets:
                    continue
                    
                targets = torch.tensor(targets, dtype=torch.long).to(device)
                inputs = inputs[valid_indices]
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        avg_val_loss = val_loss / min(len(val_loader), max_batches)
        val_acc = 100 * correct / total if total > 0 else 0
        
        print(f"  에폭 [{epoch+1}/{epochs}] | 학습 손실: {avg_train_loss:.4f} | 검증 손실: {avg_val_loss:.4f} | 검증 정확도: {val_acc:.2f}%")
            
    print("학습 완료.")
    
    # 6. 모델 저장
    print(f"[5/5] 모델 아티팩트 저장 중...")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    version = f"v5.0.0-{timestamp}"
    
    save_dir = Path("output/models")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model_filename = "model.pt"
    local_model_path = save_dir / model_filename
    
    torch.save(model.state_dict(), local_model_path)
    print(f"로컬 저장 완료: {local_model_path}")
    
    # 7. 레지스트리에 업로드
    if upload_s3:
        print(f"레지스트리에 업로드 중...")
        try:
            registry = ModelRegistry()
            # v5로 업로드 (시뮬레이션된 미래 버전)
            s3_path = registry.upload_model(
                local_path=str(local_model_path),
                model_type="efficientnet-b0-cls",
                version=version
            )
            print(f"모델 등록 완료: {s3_path}")
        except Exception as e:
            print(f"업로드 실패: {e}")
            
if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="MLOps Training Pipeline")
    parser.add_argument("csv_path", type=str, help="Path to dataset CSV")
    parser.add_argument("--class-file", type=str, default="inference_module/weights/classes_v2.txt", help="Path to classes.txt")
    
    args = parser.parse_args()
    
    train_pipeline(args.csv_path, args.class_file)
