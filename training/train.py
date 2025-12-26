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
    upload_s3: bool = True,
):
    print(f"--- MLOps 학습 파이프라인 시작 ---")

    # 1. 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"장치: {device}")

    # 2. 클래스 정의 로드 (Static Source of Truth)
    print(f"[1/5] 클래스 불러오는 중 ({class_file})...")
    with open(class_file, "r", encoding="utf-8") as f:
        # 빈 줄 필터링
        class_names = [line.strip() for line in f.readlines() if line.strip()]

    num_classes = len(class_names)
    print(f"{num_classes}개 클래스 로드 완료 (Static).")

    # 3. 데이터 준비
    print("[2/5] 데이터 준비 중...")

    # 간단한 변환 (Transform)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = KFoodDataset(csv_path=csv_path, transform=transform)

    # Train/Validation Split (8:2)
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    print(f"[2/5] 데이터 준비 완료: 학습 {train_size}건, 검증 {val_size}건")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 데이터셋의 라벨 vs 정적 클래스 파일 검증 (전체 데이터셋 기준)
    dataset_labels = set(dataset.data_frame["ground_truth"].unique())
    static_labels = set(class_names)

    # 새로운 클래스 감지 (MLOps 알림)
    new_classes = dataset_labels - static_labels
    if new_classes:
        print(
            f"   경고: 데이터셋에서 클래스 파일에 없는 {len(new_classes)}개의 새로운 클래스 발견!"
        )
        print(f"   예시: {list(new_classes)[:5]}")
        print("   이 샘플들은 학습에서 제외되거나 에러를 유발할 수 있습니다.")

    # 4. 모델 초기화 (EfficientNet-B0)
    print(f"[3/5] 모델 초기화 중 (EfficientNet-B0), 클래스 수: {num_classes}...")
    model = timm.create_model(
        "efficientnet_b0", pretrained=True, num_classes=num_classes
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 5. 학습 루프
    print(f"[4/5] {epochs} 에폭 학습 시작...")

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        max_batches = 5  # 데모용 제한 (전체 학습 시 제거 필요)

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
                if i >= max_batches:  # 데모용 제한
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

        print(
            f"  에폭 [{epoch+1}/{epochs}] | 학습 손실: {avg_train_loss:.4f} | 검증 손실: {avg_val_loss:.4f} | 검증 정확도: {val_acc:.2f}%"
        )

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
        print(f"[MLOps] 레지스트리에 업로드 중...")
        try:
            registry = ModelRegistry()  # 환경변수 MODEL_REGISTRY_BUCKET 사용

            # 모델 업로드
            s3_model_path = registry.upload_artifact(
                local_path=str(local_model_path),
                artifact_type="efficientnet-b0-cls",
                version=version,
                filename="model.pt",
            )
            print(f"모델 등록 완료: {s3_model_path}")

            # 클래스 파일 업로드 (번들링)
            s3_class_path = registry.upload_artifact(
                local_path=class_file,
                artifact_type="efficientnet-b0-cls",
                version=version,
                filename="classes.txt",
            )
            print(f"클래스 파일 등록 완료: {s3_class_path}")

            # 8. 설정 파일 자동 업데이트 (선택)
            # 주의: 로컬 학습이므로 weights_path는 로컬 경로로 설정 (S3 경로는 배포용)
            update_config_file = "inference_module/configs/active_model.json"
            if os.path.exists(update_config_file):
                print(f"[MLOps] 설정 파일({update_config_file}) 업데이트 중...")
                import json

                with open(update_config_file, "r", encoding="utf-8") as f:
                    config_data = json.load(f)

                if "efficientnet-b0-cls" in config_data:
                    config_data["efficientnet-b0-cls"]["version"] = version
                    # 로컬 테스트 편의를 위해 방금 생성된 모델 경로로 변경
                    # (실제 배포 시에는 로직이 다를 수 있음)
                    # config_data["efficientnet-b0-cls"]["weights_path"] = str(local_model_path.relative_to(Path("inference_module/weights").parent)) # 경로 계산 복잡함
                    # 간단히 절대 경로 혹은 상대 경로 매핑을 위해 output/models 가 weights_dir 하위가 아니라는 점 고려
                    # 여기서는 버전 정보만 업데이트하고 경로는 수동 관리하거나,
                    # 또는 output/models 를 weights 폴더 안으로 이동시키는 전략이 필요함.
                    # 현재 구조상 output/models는 ai_module/output/models 이므로 inference_module 외부임.
                    # 따라서 weights_path 업데이트는 건너뛰고 version만 기록하거나,
                    # 학습된 모델을 inference_module/weights/custom_trained/ 로 복사해야 함.

                    print(f"  - 버전 업데이트: {version}")

                    # TODO: 안전하게 가중치 파일 이동/복사 로직 추가 필요

                with open(update_config_file, "w", encoding="utf-8") as f:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
                print(f"설정 파일 저장 완료.")

        except Exception as e:
            print(f"[MLOps] 업로드 또는 설정 업데이트 실패: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="MLOps Training Pipeline")
    parser.add_argument("csv_path", type=str, help="데이터셋 CSV 경로")
    parser.add_argument(
        "--class-file",
        type=str,
        default="inference_module/weights/classes_v2.txt",
        help="classes.txt 경로",
    )
    parser.add_argument("--epochs", type=int, default=1, help="학습 Epoch 수")
    parser.add_argument(
        "--upload", action="store_true", help="S3 업로드 및 설정 업데이트 활성화"
    )
    # parser.add_argument("--no-upload", action="store_true", help="업로드 비활성화 (deprecated)")

    args = parser.parse_args()

    train_pipeline(
        csv_path=args.csv_path,
        class_file=args.class_file,
        epochs=args.epochs,
        upload_s3=args.upload,
    )
