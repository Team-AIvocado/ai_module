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

import matplotlib.pyplot as plt
import numpy as np

def train_pipeline(
    csv_path: str,
    class_file: str,
    epochs: int = 1,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    upload_s3: bool = True
):
    print(f"--- MLOps 학습 파이프라인 시작 (Level 2.5) ---")
    
    # 설정: Quality Gate 기준
    BASELINE_ACC = 50.0 
    
    # 1. 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"장치: {device}")
    
    # 2. 클래스 정의 로드
    print(f"[1/5] 클래스 불러오는 중 ({class_file})...")
    with open(class_file, 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]
    
    num_classes = len(class_names)
    print(f"{num_classes}개 클래스 로드 완료.")
    
    # 3. 데이터 준비
    print("[2/5] 데이터 준비 중...")
    
    # S3 경로 처리
    if csv_path.startswith("s3://"):
        import boto3
        from urllib.parse import urlparse
        
        print(f"S3 경로 감지: {csv_path}")
        parsed = urlparse(csv_path)
        bucket = parsed.netloc
        key = parsed.path.lstrip('/')
        
        local_csv_path = "dataset.csv"
        print(f"다운로드 중... (s3://{bucket}/{key} -> {local_csv_path})")
        
        s3 = boto3.client('s3')
        s3.download_file(bucket, key, local_csv_path)
        csv_path = local_csv_path # 로컬 경로로 교체
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = KFoodDataset(csv_path=csv_path, transform=transform)
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"[2/5] 데이터 준비 완료: 학습 {train_size}건, 검증 {val_size}건")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 4. 모델 초기화
    print(f"[3/5] 모델 초기화 중...")
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=num_classes)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Metrics Storage for Visualization
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # 5. 학습 루프
    print(f"[4/5] {epochs} 에폭 학습 시작...")
    
    for epoch in range(epochs):
        # --- Training ---
        model.train()
        running_loss = 0.0
        for i, (inputs, labels_text) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = []
            valid_indices = []
            for idx, text in enumerate(labels_text):
                if text in class_names:
                    targets.append(class_names.index(text))
                    valid_indices.append(idx)
            
            if not targets: continue
            
            targets = torch.tensor(targets, dtype=torch.long).to(device)
            inputs = inputs[valid_indices]
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            # Progress Log
            if (i + 1) % 10 == 0:
                print(f"  > Epoch {epoch+1} Progress: {i+1} batches")

        avg_train_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0
        
        # --- Validation ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels_text in val_loader:
                inputs = inputs.to(device)
                targets = []
                valid_indices = []
                for idx, text in enumerate(labels_text):
                    if text in class_names:
                        targets.append(class_names.index(text))
                        valid_indices.append(idx)
                if not targets: continue
                
                targets = torch.tensor(targets, dtype=torch.long).to(device)
                inputs = inputs[valid_indices]
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_acc = 100 * correct / total if total > 0 else 0
        
        print(f"  Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Store metrics
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
            
    print("학습 완료.")
    
    # 6. Quality Gate (Level 2.5)
    final_acc = history['val_acc'][-1] if history['val_acc'] else 0.0
    print(f"[Quality Gate] 검증 정확도: {final_acc:.2f}% (기준: {BASELINE_ACC}%)")
    
    if final_acc < BASELINE_ACC:
        print(f"!!! 품질 기준 미달 !!! 학습 모델을 저장하지 않습니다.")
        if upload_s3:
            print("S3 업로드를 스킵합니다.")
            return # Exit pipeline
            
    # 7. 모델 저장 (Pass시)
    print(f"[5/5] 모델 아티팩트 저장 중...")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    version = f"v5.0.0-{timestamp}"
    
    save_dir = Path("output/models")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = save_dir / "model.pt"
    graph_path = save_dir / "training_graph.png"
    
    torch.save(model.state_dict(), model_path)
    print(f"로컬 모델 저장: {model_path}")
    
    # 8. 시각화 그래프 생성
    try:
        plt.figure(figsize=(12, 5))
        
        # Loss Plot
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title('Loss History')
        plt.legend()
        
        # Acc Plot
        plt.subplot(1, 2, 2)
        plt.plot(history['val_acc'], label='Val Accuracy')
        plt.axhline(y=BASELINE_ACC, color='r', linestyle='--', label='Baseline')
        plt.title('Accuracy History')
        plt.legend()
        
        plt.savefig(graph_path)
        print(f"학습 그래프 저장: {graph_path}")
        plt.close()
    except Exception as e:
        print(f"그래프 생성 실패: {e}")

    # 9. 레지스트리 업로드
    if upload_s3:
        print(f"레지스트리에 업로드 중 ({version})...")
        try:
            registry = ModelRegistry()
            # Model
            model_s3 = registry.upload_model(str(model_path), "efficientnet-b0-cls", version)
            print(f" - Model: {model_s3}")
            
            # Graph
            graph_key = f"efficientnet-b0-cls/{version}/graph.png"
            registry.s3.upload_file(str(graph_path), registry.bucket_name, graph_key)
            print(f" - Graph: s3://{registry.bucket_name}/{graph_key}")
            
        except Exception as e:
            print(f"업로드 실패: {e}")
            
if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="MLOps Training Pipeline")
    parser.add_argument("csv_path", type=str, help="Path to dataset CSV")
    parser.add_argument("--class-file", type=str, default="inference_module/weights/classes_v2.txt", help="Path to classes.txt")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train")
    parser.add_argument("--no-upload", action="store_true", help="Skip uploading to S3 Registry")
    
    args = parser.parse_args()
    
    train_pipeline(args.csv_path, args.class_file, epochs=args.epochs, upload_s3=not args.no_upload)
